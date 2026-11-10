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
}
