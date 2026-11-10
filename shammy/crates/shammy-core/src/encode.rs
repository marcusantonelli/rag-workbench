//! Turning one vector into its stored codes. See spec/FORMAT.md, "Encoding".

use crate::rotate::Rotation;

/// Clipping scales tried for the 8-bit code, as fractions of the vector's largest component.
pub const Q8_SCALES: [f32; 8] = [1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.25, 0.18];
pub const Q8_LEVELS: f32 = 127.0;

pub struct Encoded {
    /// Sign bits of all rotated dimensions, bit i of byte i/8 (least significant first); set = negative.
    pub bits: Vec<u8>,
    /// Estimator scale over the sketch dimensions and over all dimensions.
    pub corr: [f32; 2],
    pub q8: Vec<i8>,
    /// Multiply (code . rotated query) by this to estimate the dot product.
    pub q8_factor: f32,
}

/// Scale a vector to unit length (left unchanged if it is all zeros).
pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm > 0.0 {
        let inv = (1.0 / norm) as f32;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

pub fn pack_bits(rotated: &[f32]) -> Vec<u8> {
    let mut bits = vec![0u8; rotated.len() / 8];
    for (i, x) in rotated.iter().enumerate() {
        if *x < 0.0 {
            bits[i / 8] |= 1 << (i % 8);
        }
    }
    bits
}

/// |v|^2 / |v|_1 over a range: turns (signs . query) into an estimate of (v . query).
fn scale(v: &[f32]) -> f32 {
    let (mut l1, mut l2) = (0f64, 0f64);
    for x in v {
        l1 += (*x as f64).abs();
        l2 += (*x as f64) * (*x as f64);
    }
    if l1 > 0.0 { (l2 / l1) as f32 } else { 0.0 }
}

/// Encode a unit-length vector. `scratch` is reused between calls.
pub fn encode(unit: &[f32], rot: &Rotation, sketch_dims: usize, scratch: &mut Vec<f32>) -> Encoded {
    scratch.clear();
    scratch.extend_from_slice(unit);
    rot.apply(scratch);
    let v = &scratch[..];
    let bits = pack_bits(v);
    let corr = [scale(&v[..sketch_dims]), scale(v)];

    let peak = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let norm2: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    let mut best: Vec<i8> = vec![0; v.len()];
    let mut best_cos = -1f64;
    let mut best_dot = 0f64;
    if peak > 0.0 {
        let mut code = vec![0i8; v.len()];
        for f in Q8_SCALES {
            let denom = peak * f;
            let (mut dot, mut cn) = (0f64, 0f64);
            for (c, x) in code.iter_mut().zip(v) {
                let q = (*x / denom * Q8_LEVELS).round_ties_even().clamp(-Q8_LEVELS, Q8_LEVELS);
                *c = q as i8;
                dot += (q as f64) * (*x as f64);
                cn += (q as f64) * (q as f64);
            }
            let cos = dot / (cn.sqrt() * norm2.sqrt());
            if cos > best_cos {
                best_cos = cos;
                best_dot = dot;
                best.copy_from_slice(&code);
            }
        }
    }
    let q8_factor = if best_dot > 0.0 { (norm2 / best_dot) as f32 } else { 0.0 };
    Encoded { bits, corr, q8: best, q8_factor }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotate::SplitMix64;

    fn random_unit(dims: usize, rng: &mut SplitMix64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dims).map(|_| (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect();
        normalize(&mut v);
        v
    }

    #[test]
    fn codes_estimate_the_dot_product() {
        let dims = 256;
        let rot = Rotation::new(dims, 1).unwrap();
        let mut rng = SplitMix64(42);
        let mut scratch = Vec::new();
        let (mut err8, mut err1) = (0f64, 0f64);
        for _ in 0..50 {
            let a = random_unit(dims, &mut rng);
            let mut q = random_unit(dims, &mut rng);
            for (x, y) in q.iter_mut().zip(&a) {
                *x = 0.6 * *x + 0.4 * *y; // correlated, like a real query
            }
            normalize(&mut q);
            let exact: f64 = a.iter().zip(&q).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
            let e = encode(&a, &rot, dims / 4, &mut scratch);
            let mut qr = q.clone();
            rot.apply(&mut qr);
            let d8: f64 = e.q8.iter().zip(&qr).map(|(c, y)| (*c as f64) * (*y as f64)).sum();
            err8 += (d8 * e.q8_factor as f64 - exact).abs();
            let d1: f64 = qr.iter().enumerate()
                .map(|(i, y)| if e.bits[i / 8] >> (i % 8) & 1 == 1 { -(*y as f64) } else { *y as f64 }).sum();
            err1 += (d1 * e.corr[1] as f64 - exact).abs();
        }
        assert!(err8 / 50.0 < 0.005, "8-bit error {}", err8 / 50.0);
        assert!(err1 / 50.0 < 0.08, "1-bit error {}", err1 / 50.0);
    }
}
