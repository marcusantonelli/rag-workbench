//! The rotation every vector and query goes through before it is quantized.
//!
//! Dimensions are split into three blocks, [0, D/4), [D/4, D/2) and [D/2, D), so the Matryoshka
//! prefixes D/4 and D/2 stay prefixes. Each block gets three rounds of (seeded sign flips,
//! orthonormal Walsh-Hadamard transform). Nothing is stored: the signs come from the seed.

use crate::error::{Error, Result};

pub const ROUNDS: usize = 3;

/// SplitMix64: the specification's only random source.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    #[inline]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

pub fn blocks(dims: usize) -> [(usize, usize); 3] {
    [(0, dims / 4), (dims / 4, dims / 2), (dims / 2, dims)]
}

pub struct Rotation {
    dims: usize,
    /// signs[round][dimension] as +1.0 / -1.0
    signs: Vec<Vec<f32>>,
}

impl Rotation {
    /// Signs are drawn block by block, round by round, dimension by dimension: one SplitMix64
    /// word per 64 dimensions, least significant bit first, a set bit meaning -1.
    pub fn new(dims: usize, seed: u64) -> Result<Self> {
        if dims < 64 || !dims.is_power_of_two() {
            return Err(Error::Invalid(format!(
                "{dims} dimensions: this format version needs a power of two, 64 or more"
            )));
        }
        let mut rng = SplitMix64(seed);
        let mut signs = vec![vec![1.0f32; dims]; ROUNDS];
        for (a, b) in blocks(dims) {
            for round in signs.iter_mut() {
                let mut word = 0u64;
                for (i, s) in round[a..b].iter_mut().enumerate() {
                    if i % 64 == 0 {
                        word = rng.next();
                    }
                    *s = if (word >> (i % 64)) & 1 == 1 { -1.0 } else { 1.0 };
                }
            }
        }
        Ok(Self { dims, signs })
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Rotate in place.
    pub fn apply(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.dims);
        for (a, b) in blocks(self.dims) {
            for round in &self.signs {
                for (x, s) in v[a..b].iter_mut().zip(&round[a..b]) {
                    *x *= *s;
                }
                fwht(&mut v[a..b]);
            }
        }
    }
}

/// Orthonormal Walsh-Hadamard transform, in place. The length must be a power of two.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let scale = (1.0 / (n as f64).sqrt()) as f32;
    for v in x.iter_mut() {
        *v *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_keeps_length_and_block_prefixes() {
        let rot = Rotation::new(64, 7).unwrap();
        let mut v: Vec<f32> = (0..64).map(|i| ((i * 37 % 11) as f32 - 5.0) / 7.0).collect();
        let before = v.clone();
        rot.apply(&mut v);
        let norm = |s: &[f32]| s.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        for (a, b) in blocks(64) {
            assert!((norm(&v[a..b]) - norm(&before[a..b])).abs() < 1e-4);
        }
        assert_ne!(v, before);
    }

    #[test]
    fn splitmix_matches_the_reference_values() {
        let mut r = SplitMix64(0);
        assert_eq!(r.next(), 0xE220A8397B1DCDAF);
        assert_eq!(r.next(), 0x6E789E6AA1B965F4);
    }

    #[test]
    fn rejects_dimensions_that_are_not_a_power_of_two() {
        assert!(Rotation::new(768, 1).is_err());
    }
}
