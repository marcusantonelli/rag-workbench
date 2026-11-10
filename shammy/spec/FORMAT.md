# The `.shammy` file format, version 1.0

All integers and floats are little-endian. Offsets are from the start of the file. The reference
implementation of everything on this page is `python/shammy_ref.py`; the engine is tested against it.

## Layout

```text
0            Header     64 bytes
4096         Section    every section starts on a 4096-byte boundary (zero padding between)
...          Section
...          Footer     the section directory; starts wherever the last section ended
...          Trailer    32 bytes, immediately after the footer
[next commit: more sections, a new footer, a new trailer]
```

A **commit** is: append sections, append a footer that lists every section of the file (old and
new), append a trailer, fsync. Earlier footers stay in the file and are ignored.

To open a file, scan backwards from the end for the 8 trailer magic bytes; accept the first
trailer whose `footer_offset + footer_length` equals the trailer's own offset and whose checksum
matches the footer bytes. Bytes after that trailer are an interrupted write: readers ignore
them, and a writer truncates them before appending.

## Header (64 bytes)

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 8 | magic | `53 48 41 4D 4D 59 00 01` ("SHAMMY\0\1") |
| 8 | 2 | format_major | 1. A reader refuses any other major version |
| 10 | 2 | format_minor | 0. A reader skips section types it doesn't know |
| 12 | 4 | dims | A power of two, 16 or more |
| 16 | 4 | sketch_dims | dims/4 or dims/2 |
| 20 | 4 | flags | bit 0: vectors are unit length (always set); bit 1: lossless tier present |
| 24 | 8 | rotation_seed | Default `0x5348414D4D593031`. Files merge by copying only if this matches |
| 32 | 16 | file_uuid | Random; identifies the file's lineage |
| 48 | 8 | reserved | Zero |
| 56 | 8 | header_checksum | XXH3-64 (seed 0) of bytes 0-55 |

## Trailer (32 bytes)

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | footer_offset |
| 8 | 8 | footer_length |
| 16 | 8 | footer_checksum: XXH3-64 of the footer bytes |
| 24 | 8 | magic `53 48 46 4F 4F 54 00 01` ("SHFOOT\0\1") |

## Footer

Strings are a u16 byte length followed by UTF-8.

| Field | Type | Notes |
|---|---|---|
| footer_version | u32 | 1 |
| generation | u64 | +1 per commit. Mirrored in `rag.db` so the two files can be matched |
| row_count | u64 | All rows, deleted ones included |
| live_row_count | u64 | Rows not deleted |
| segment_count | u32 | Segments are numbered 0..segment_count |
| instruction_hash | u64 | XXH3-64 of the query instruction text; 0 if none |
| model | string | Embedding model name |
| file_type_count | u16 | |
| file_types | string × count | A row's `ftype` byte is an index into this list |
| section_count | u32 | |
| sections | 48 bytes × count | Below |

Section entry (48 bytes):

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | segment |
| 4 | 2 | type |
| 6 | 2 | codec (0 = raw; nothing else is defined) |
| 8 | 8 | offset (a multiple of 4096) |
| 16 | 8 | length in bytes |
| 24 | 8 | rows |
| 32 | 8 | checksum: XXH3-64 of the section bytes |
| 40 | 8 | block_checksums_offset (0: none; reserved for per-megabyte checksums) |

## Sections

A segment is a set of sections with the same row count; row *i* of each describes the same chunk.
Segments are immutable. With D = dims and S = sketch_dims:

| Type | Name | Bytes per row | Contents |
|---|---|---|---|
| 1 | ids | 8 | u64 chunk id |
| 2 | doc | 4 | u32 document id |
| 3 | ftype | 1 | index into `file_types` |
| 4 | sketch | S/8 | sign bits of rotated dimensions 0..S |
| 5 | bits | (D-S)/8 | sign bits of rotated dimensions S..D |
| 6 | corr | 8 | f32 scale over dimensions 0..S, f32 scale over 0..D |
| 7 | q8 | D | i8 per rotated dimension |
| 8 | q8corr | 8 | f32 factor, f32 reserved (0) |
| 9 | f16 | 2D | the unit vector, unrotated, as IEEE float16 (only with flag bit 1) |
| 10 | tomb | 1 bit | deletion bitmap, (rows+7)/8 bytes; when a segment has several, the last in the directory wins |
| 20 | clusters | | reserved |
| 30-39 | keyword index | | reserved |

Bit *i* of a bit array is bit `i % 8` (least significant first) of byte `i / 8`. A set sign bit
means the value is negative. A set tombstone bit means the row is deleted.

## Encoding a vector

1. **Normalise** to unit length: multiply each component by `(float32)(1 / sqrt(sum of squares in float64))`.
   An all-zero vector stays zero. This unit vector is what `f16` stores.
2. **Rotate.** Split the dimensions into three blocks: [0, D/4), [D/4, D/2), [D/2, D). For each
   block, three times: multiply by that round's signs, then apply the orthonormal Walsh-Hadamard
   transform (butterflies `a+b, a-b` with strides 1, 2, 4, ...; then multiply by
   `(float32)(1/sqrt(n))`). All in float32.
   The signs come from SplitMix64 seeded with `rotation_seed`: for each block in order, for each
   round in order, one 64-bit output per 64 dimensions; bit `i % 64` set means -1.
3. **sketch, bits**: bit set where the rotated value is below zero.
4. **corr**: for the range 0..S and for 0..D, `sum of squares / sum of absolute values`
   (accumulated in float64, stored as float32; 0 if the sum of absolute values is 0).
5. **q8**: with `peak` the largest absolute rotated value, for each scale f in
   `1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.25, 0.18`: `code[i] = clamp(round_half_even(v[i] / (peak*f) * 127), -127, 127)`
   in float32. Keep the code with the highest cosine to v (float64; the first scale wins a tie).
   `q8corr.factor = (v . v) / (code . v)`; a zero vector has code 0 and factor 0.

There is no centroid: on Qwen3-Embedding-8B vectors, subtracting the mean made the sketch worse
and the other codes no better (milestone 0).

## Searching

With the query normalised to `u` and rotated to `q`:

| Stage | Over | Score | Keep |
|---|---|---|---|
| 1 | every live row that passes the filter | minus the Hamming distance between sketch bits of the row and of `q` | stage1 (default: 1% of live rows, at least 4096) |
| 2 | stage-1 survivors | `corr.full x sum(sign_i x q_i)` over all D dimensions | stage2 (default 256) |
| 3 | stage-2 survivors | `q8corr.factor x sum(code_i x q_i)` | finish (default 50); k if there is no lossless tier |
| finish | stage-3 survivors, lossless files only | `sum(f16_i x u_i)` | k |

Sums are accumulated in float64 and scores stored as float32. Ties are broken by segment then
row at every stage, and results are ordered by score, then id, so a search is deterministic.
Every count is raised to at least k.
