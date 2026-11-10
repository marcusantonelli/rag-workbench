//! The hot loops, each with a plain version and CPU-specific versions chosen at run time.
//! Every version must return exactly what the plain one returns.

/// Hamming distance from `query` to each `width`-byte row of `codes`, written to `out`.
pub fn hamming_rows(codes: &[u8], width: usize, query: &[u8], out: &mut [u16]) {
    debug_assert_eq!(query.len(), width);
    debug_assert_eq!(codes.len(), width * out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if width % 64 == 0 && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            // Safety: the features were just detected, and the lengths are checked above.
            return unsafe { x86::hamming_rows_avx512(codes, width, query, out) };
        }
        if width % 32 == 0 && is_x86_feature_detected!("avx2") {
            return unsafe { x86::hamming_rows_avx2(codes, width, query, out) };
        }
    }
    hamming_rows_plain(codes, width, query, out)
}

/// Which version `hamming_rows` uses for this width on this machine (for `shammy bench`).
pub fn hamming_kernel_name(width: usize) -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if width % 64 == 0 && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            return "AVX-512 VPOPCNTDQ";
        }
        if width % 32 == 0 && is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
    }
    let _ = width;
    "plain"
}

pub fn hamming_rows_plain(codes: &[u8], width: usize, query: &[u8], out: &mut [u16]) {
    for (row, o) in codes.chunks_exact(width).zip(out.iter_mut()) {
        let mut n = 0u32;
        let (mut a, mut b) = (row.chunks_exact(8), query.chunks_exact(8));
        for (x, y) in (&mut a).zip(&mut b) {
            n += (u64::from_le_bytes(x.try_into().unwrap()) ^ u64::from_le_bytes(y.try_into().unwrap())).count_ones();
        }
        for (x, y) in a.remainder().iter().zip(b.remainder()) {
            n += (x ^ y).count_ones();
        }
        *o = n as u16;
    }
}

/// Sums in stages 2 and 3 are kept in 16 float32 lanes (dimension i goes to lane i % 16) and the
/// lanes are added up in order in float64. Fixing that order is what lets the plain, AVX2 and
/// AVX-512 versions agree to the last bit.
pub const LANES: usize = 16;

pub fn reduce(acc: &[f32; LANES]) -> f64 {
    acc.iter().map(|x| *x as f64).sum()
}

fn wide(start: usize, len: usize) -> bool {
    start % LANES == 0 && len % LANES == 0 && len > 0
}

/// acc[lane] += +q[i] or -q[i] (bit i of `bits` set means minus). `start` is the dimension index
/// of q[0], so a vector stored in two pieces can be summed piece by piece.
pub fn sign_dot(bits: &[u8], q: &[f32], start: usize, acc: &mut [f32; LANES]) {
    debug_assert_eq!(bits.len() * 8, q.len());
    #[cfg(target_arch = "x86_64")]
    if wide(start, q.len()) {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86::sign_dot_avx512(bits, q, acc) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86::sign_dot_avx2(bits, q, acc) };
        }
    }
    sign_dot_plain(bits, q, start, acc)
}

pub fn sign_dot_plain(bits: &[u8], q: &[f32], start: usize, acc: &mut [f32; LANES]) {
    for (i, x) in q.iter().enumerate() {
        let minus = bits[i / 8] >> (i % 8) & 1 == 1;
        acc[(start + i) % LANES] += if minus { -*x } else { *x };
    }
}

/// acc[lane] += code[i] (as a signed byte) x q[i], multiplied and added in float32.
pub fn i8_dot(code: &[u8], q: &[f32], acc: &mut [f32; LANES]) {
    debug_assert_eq!(code.len(), q.len());
    #[cfg(target_arch = "x86_64")]
    if wide(0, q.len()) {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86::i8_dot_avx512(code, q, acc) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86::i8_dot_avx2(code, q, acc) };
        }
    }
    i8_dot_plain(code, q, acc)
}

pub fn i8_dot_plain(code: &[u8], q: &[f32], acc: &mut [f32; LANES]) {
    for (i, (c, x)) in code.iter().zip(q).enumerate() {
        acc[i % LANES] += (*c as i8) as f32 * *x;
    }
}

/// Which version the stage 2 and 3 kernels use on this machine.
pub fn dot_kernel_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512";
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
    }
    "plain"
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub unsafe fn hamming_rows_avx512(codes: &[u8], width: usize, query: &[u8], out: &mut [u16]) {
        let words = width / 64;
        let mut q = [_mm512_setzero_si512(); 16];
        let q = if words <= 16 { &mut q[..words] } else { return super::hamming_rows_plain(codes, width, query, out) };
        for (i, w) in q.iter_mut().enumerate() {
            *w = _mm512_loadu_si512(query.as_ptr().add(i * 64) as *const _);
        }
        let mut p = codes.as_ptr();
        for o in out.iter_mut() {
            let mut acc = _mm512_setzero_si512();
            for w in q.iter() {
                let x = _mm512_xor_si512(_mm512_loadu_si512(p as *const _), *w);
                acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(x));
                p = p.add(64);
            }
            *o = _mm512_reduce_add_epi64(acc) as u16;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn hamming_rows_avx2(codes: &[u8], width: usize, query: &[u8], out: &mut [u16]) {
        let lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        let low = _mm256_set1_epi8(0x0f);
        let words = width / 32;
        let mut p = codes.as_ptr();
        for o in out.iter_mut() {
            let mut acc = _mm256_setzero_si256();
            for i in 0..words {
                let x = _mm256_xor_si256(_mm256_loadu_si256(p as *const _),
                                         _mm256_loadu_si256(query.as_ptr().add(i * 32) as *const _));
                let lo = _mm256_shuffle_epi8(lookup, _mm256_and_si256(x, low));
                let hi = _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi16(x, 4), low));
                // per-byte counts (at most 8) summed into four 64-bit lanes
                acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_add_epi8(lo, hi), _mm256_setzero_si256()));
                p = p.add(32);
            }
            let s = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            *o = (_mm_cvtsi128_si64(s) + _mm_extract_epi64(s, 1)) as u16;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn sign_dot_avx512(bits: &[u8], q: &[f32], acc: &mut [f32; 16]) {
        let mut a = _mm512_loadu_ps(acc.as_ptr());
        let zero = _mm512_setzero_ps();
        for (g, chunk) in q.chunks_exact(16).enumerate() {
            let mask = u16::from_le_bytes([*bits.get_unchecked(g * 2), *bits.get_unchecked(g * 2 + 1)]);
            let v = _mm512_loadu_ps(chunk.as_ptr());
            a = _mm512_add_ps(a, _mm512_mask_sub_ps(v, mask, zero, v));
        }
        _mm512_storeu_ps(acc.as_mut_ptr(), a);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn sign_dot_avx2(bits: &[u8], q: &[f32], acc: &mut [f32; 16]) {
        let mut lo = _mm256_loadu_ps(acc.as_ptr());
        let mut hi = _mm256_loadu_ps(acc.as_ptr().add(8));
        let pick = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let sign = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
        let flip = |byte: u8, v: __m256| {
            let set = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(byte as i32), pick), pick);
            _mm256_xor_ps(v, _mm256_and_ps(_mm256_castsi256_ps(set), sign))
        };
        for (g, chunk) in q.chunks_exact(16).enumerate() {
            lo = _mm256_add_ps(lo, flip(*bits.get_unchecked(g * 2), _mm256_loadu_ps(chunk.as_ptr())));
            hi = _mm256_add_ps(hi, flip(*bits.get_unchecked(g * 2 + 1), _mm256_loadu_ps(chunk.as_ptr().add(8))));
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), lo);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), hi);
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn i8_dot_avx512(code: &[u8], q: &[f32], acc: &mut [f32; 16]) {
        let mut a = _mm512_loadu_ps(acc.as_ptr());
        for (c, x) in code.chunks_exact(16).zip(q.chunks_exact(16)) {
            let v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(c.as_ptr() as *const _)));
            a = _mm512_add_ps(a, _mm512_mul_ps(v, _mm512_loadu_ps(x.as_ptr())));
        }
        _mm512_storeu_ps(acc.as_mut_ptr(), a);
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn i8_dot_avx2(code: &[u8], q: &[f32], acc: &mut [f32; 16]) {
        let mut lo = _mm256_loadu_ps(acc.as_ptr());
        let mut hi = _mm256_loadu_ps(acc.as_ptr().add(8));
        for (c, x) in code.chunks_exact(16).zip(q.chunks_exact(16)) {
            let a = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(c.as_ptr() as *const _)));
            let b = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(c.as_ptr().add(8) as *const _)));
            lo = _mm256_add_ps(lo, _mm256_mul_ps(a, _mm256_loadu_ps(x.as_ptr())));
            hi = _mm256_add_ps(hi, _mm256_mul_ps(b, _mm256_loadu_ps(x.as_ptr().add(8))));
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), lo);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), hi);
    }

    #[cfg(test)]
    pub fn run_all(codes: &[u8], width: usize, query: &[u8]) -> Vec<(&'static str, Vec<u16>)> {
        let mut v = Vec::new();
        let n = codes.len() / width;
        if width % 64 == 0 && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            let mut o = vec![0u16; n];
            unsafe { hamming_rows_avx512(codes, width, query, &mut o) };
            v.push(("avx512", o));
        }
        if width % 32 == 0 && is_x86_feature_detected!("avx2") {
            let mut o = vec![0u16; n];
            unsafe { hamming_rows_avx2(codes, width, query, &mut o) };
            v.push(("avx2", o));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotate::SplitMix64;

    #[test]
    fn every_hamming_kernel_matches_the_plain_one() {
        let mut rng = SplitMix64(11);
        for width in [2usize, 8, 32, 64, 96, 128, 256, 512] {
            let rows = 257;
            let mut codes: Vec<u8> = (0..rows * width).map(|_| rng.next() as u8).collect();
            let query: Vec<u8> = (0..width).map(|_| rng.next() as u8).collect();
            codes[..width].copy_from_slice(&query); // distance 0
            for b in codes[width..2 * width].iter_mut().zip(&query) {
                *b.0 = !*b.1; // every bit differs
            }
            let mut plain = vec![0u16; rows];
            hamming_rows_plain(&codes, width, &query, &mut plain);
            assert_eq!((plain[0], plain[1] as usize), (0, width * 8));
            let mut picked = vec![0u16; rows];
            hamming_rows(&codes, width, &query, &mut picked);
            assert_eq!(picked, plain, "dispatch, width {width}");
            #[cfg(target_arch = "x86_64")]
            for (name, got) in x86::run_all(&codes, width, &query) {
                assert_eq!(got, plain, "{name}, width {width}");
            }
        }
    }

    #[test]
    fn every_dot_kernel_matches_the_plain_one() {
        let mut rng = SplitMix64(23);
        for dims in [64usize, 256, 1024, 4096] {
            let q: Vec<f32> = (0..dims).map(|_| (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect();
            let bits: Vec<u8> = (0..dims / 8).map(|_| rng.next() as u8).collect();
            let code: Vec<u8> = (0..dims).map(|_| rng.next() as u8).collect();
            let start = [0.25f32; LANES];

            let mut want = start;
            sign_dot_plain(&bits, &q, 0, &mut want);
            let mut got = start;
            sign_dot(&bits, &q, 0, &mut got);
            assert_eq!(got, want, "sign_dot dispatch, {dims} dims");
            // in two pieces, as the sketch and the remaining bits are stored
            let mut pieces = start;
            sign_dot(&bits[..dims / 32], &q[..dims / 4], 0, &mut pieces);
            sign_dot(&bits[dims / 32..], &q[dims / 4..], dims / 4, &mut pieces);
            assert_eq!(pieces, want, "sign_dot in two pieces, {dims} dims");

            let mut want8 = start;
            i8_dot_plain(&code, &q, &mut want8);
            let mut got8 = start;
            i8_dot(&code, &q, &mut got8);
            assert_eq!(got8, want8, "i8_dot dispatch, {dims} dims");

            #[cfg(target_arch = "x86_64")]
            unsafe {
                if is_x86_feature_detected!("avx2") {
                    let (mut a, mut b) = (start, start);
                    x86::sign_dot_avx2(&bits, &q, &mut a);
                    x86::i8_dot_avx2(&code, &q, &mut b);
                    assert_eq!((a, b), (want, want8), "avx2, {dims} dims");
                }
                if is_x86_feature_detected!("avx512f") {
                    let (mut a, mut b) = (start, start);
                    x86::sign_dot_avx512(&bits, &q, &mut a);
                    x86::i8_dot_avx512(&code, &q, &mut b);
                    assert_eq!((a, b), (want, want8), "avx512, {dims} dims");
                }
            }
        }
    }
}
