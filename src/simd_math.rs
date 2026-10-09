//! Transcendentals on `fearless_simd`'s native-width `f64` vectors.
//!
//! `fearless_simd` has no transcendentals, and a libm call per lane would
//! leave the vector code. These are branch-free: both sides of every case
//! are computed and a mask selects one. NaN in, NaN out everywhere, and a few
//! ulp of accuracy, which is what the LM operators need to agree with JAX to
//! `1e-10`.

use std::f64::consts::LN_2;

use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

/// `1 / ln 2`.
const INV_LN2: f64 = std::f64::consts::LOG2_E;
/// `ln 2` split so that `LN2_HI` has 21 trailing zero bits and `k * LN2_HI`
/// is exact for every `|k|` that occurs here (Cody-Waite).
const LN2_HI: f64 = 6.931_471_803_691_238e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
/// `2^52`: in `[2^52, 2^53)` the mantissa bits are the integer offset.
const TWO52: f64 = 4_503_599_627_370_496.0;
/// The bits of `sqrt(2) / 2`.
const SQRT1_2_BITS: u64 = 0x3fe6_a09e_667f_3bcd;
/// Above this, `asinh(x) = ln(2x)` to full precision.
const ASINH_LARGE: f64 = 268_435_456.0;

/// `1/2! ..= 1/13!`, for `expm1(r) = r + r^2 P(r)` on `|r| <= ln2/2`.
/// Truncating after `r^13` leaves `|r|^14/14! < 1.1e-17`.
const EXP_C: [f64; 12] = [
    5.000_000_000_000_000e-1,
    1.666_666_666_666_666_6e-1,
    4.166_666_666_666_666_4e-2,
    8.333_333_333_333_333e-3,
    1.388_888_888_888_889e-3,
    1.984_126_984_126_984e-4,
    2.480_158_730_158_73e-5,
    2.755_731_922_398_589_3e-6,
    2.755_731_922_398_589e-7,
    2.505_210_838_544_172e-8,
    2.087_675_698_786_81e-9,
    1.605_904_383_682_161_3e-10,
];

/// fdlibm's `__ieee754_log` coefficients `Lg1..Lg7`, for
/// `log(1+f) = 2s + s R(s^2)`, `s = f/(2+f)`, `|s| <= 0.1716`.
const LG: [f64; 7] = [
    6.666_666_666_666_735_1e-1,
    3.999_999_999_940_941_9e-1,
    2.857_142_874_366_239_1e-1,
    2.222_219_843_214_978_4e-1,
    1.818_357_216_161_805e-1,
    1.531_383_769_920_937_3e-1,
    1.479_819_860_511_658_6e-1,
];

/// `x` clamped to `[low, high]`, keeping NaN, which `max`/`min` would drop.
#[simd]
fn clamp<S: Simd>(simd: S, x: S::f64s, low: f64, high: f64) -> S::f64s {
    let x = x.simd_gt(high).select(S::f64s::splat(simd, high), x);
    x.simd_lt(low).select(S::f64s::splat(simd, low), x)
}

/// `2^k` for an integral `k` in `[-1022, 1023]`.
#[simd]
fn pow2<S: Simd>(_: S, k: S::f64s) -> S::f64s {
    let biased: S::u64s = (k + (1023.0 + TWO52)).bitcast();
    ((biased - TWO52.to_bits()) << 52).bitcast()
}

/// `x = k ln2 + r` with `|r| <= ln2/2`: `(k, expm1(r))`.
#[simd]
fn reduce<S: Simd>(simd: S, x: S::f64s) -> (S::f64s, S::f64s) {
    let k = (x * INV_LN2).round_ties_even();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut poly = S::f64s::splat(simd, EXP_C[11]);
    for &c in EXP_C[..11].iter().rev() {
        poly = poly * r + c;
    }
    (k, r * r * poly + r)
}

/// `exp(x)`, overflowing to `inf` and underflowing through the denormals.
#[simd]
pub(crate) fn exp<S: Simd>(simd: S, x: S::f64s) -> S::f64s {
    let (k, tail) = reduce(simd, clamp(simd, x, -746.0, 710.0));
    // `2^k` in two halves, so that neither leaves the normal range.
    let k1 = (k * 0.5).round_ties_even();
    (tail + 1.0) * pow2(simd, k - k1) * pow2(simd, k1)
}

/// `exp(x) - 1`; saturates at `|x| = 50`, which is all its callers need.
#[simd]
fn expm1<S: Simd>(simd: S, x: S::f64s) -> S::f64s {
    let (k, tail) = reduce(simd, clamp(simd, x, -50.0, 50.0));
    let p = pow2(simd, k);
    let wide = p * tail + (p - 1.0);
    k.simd_eq(0.0).select(tail, wide)
}

/// `ln(u)` for `u >= 1`, including `+inf`.
#[simd]
fn ln_ge1<S: Simd>(simd: S, u: S::f64s) -> S::f64s {
    // Lanes outside the domain are computed at 1 and selected away below.
    let safe = u.simd_ge(1.0).select(u, S::f64s::splat(simd, 1.0));
    // `safe = 2^e m` with `m` in `[sqrt(2)/2, sqrt(2))`.
    let bits: S::u64s = safe.bitcast();
    let e_bits = (bits - SQRT1_2_BITS) >> 52;
    let m: S::f64s = (bits - (e_bits << 52)).bitcast();
    let e = (e_bits | TWO52.to_bits()).bitcast::<S::f64s>() - TWO52;

    let f = m - 1.0;
    let s = f / (f + 2.0);
    let z = s * s;
    let w = z * z;
    let even = w * ((w * (w * LG[5] + LG[3])) + LG[1]);
    let odd = z * ((w * ((w * (w * LG[6] + LG[4])) + LG[2])) + LG[0]);
    let half_square = f * f * 0.5;
    let out = e * LN2_HI - ((half_square - (s * (half_square + even + odd) + e * LN2_LO)) - f);
    u.simd_lt(f64::INFINITY).select(out, u)
}

/// `ln(1 + x)` for `x >= 0`, including `+inf`.
#[simd]
pub(crate) fn ln_1p<S: Simd>(simd: S, x: S::f64s) -> S::f64s {
    let u = x + 1.0;
    // Goldberg: the rounding of `1 + x` cancels in the ratio.
    let corrected = ln_ge1(simd, u) * (x / (u - 1.0));
    let out = u.simd_eq(1.0).select(x, corrected);
    x.simd_lt(f64::INFINITY).select(out, x)
}

/// `sqrt(1 + x^2)`, or `|x|` where the square would overflow.
#[simd]
pub(crate) fn hypot_one<S: Simd>(_: S, x: S::f64s) -> S::f64s {
    let root = (x * x + 1.0).sqrt();
    let a = x.abs();
    a.simd_gt(1e150).select(a, root)
}

#[simd]
pub(crate) fn asinh<S: Simd>(simd: S, x: S::f64s) -> S::f64s {
    let b = x.abs();
    let square = b * b;
    let small = ln_1p(simd, b + square / ((square + 1.0).sqrt() + 1.0));
    let large = ln_ge1(simd, b) + LN_2;
    b.simd_gt(ASINH_LARGE).select(large, small).copysign(x)
}

/// `(sinh(x), cosh(x))`.
#[simd]
pub(crate) fn sinh_cosh<S: Simd>(simd: S, x: S::f64s) -> (S::f64s, S::f64s) {
    let one = S::f64s::splat(simd, 1.0);
    let b = x.abs();
    let em1 = expm1(simd, b);
    let e = em1 + 1.0;
    let sinh_small = (em1 + em1 / e) * 0.5;
    let cosh_small = (e + one / e) * 0.5;
    // Squaring `exp(b/2)` reaches the largest finite results.
    let half = exp(simd, b * 0.5);
    let large = (half * 0.5) * half;
    let small = b.simd_lt(22.0);
    let s = small.select(sinh_small, large);
    let c = small.select(cosh_small, large);
    (s.copysign(x), c)
}

/// `log(cosh(x))` in `_log_cosh`'s form, `|x| + log1p(exp(-2|x|)) - ln 2`,
/// and `tanh(x)`, from one `expm1`.
#[simd]
pub(crate) fn log_cosh_tanh<S: Simd>(simd: S, x: S::f64s) -> (S::f64s, S::f64s) {
    let b = x.abs();
    let em1 = expm1(simd, b * -2.0);
    let value = b + ln_1p(simd, em1 + 1.0) - LN_2;
    let tanh = (-em1 / (em1 + 2.0)).copysign(x);
    (value, tanh)
}

/// Softplus and its first two derivatives.
#[simd]
pub(crate) fn softplus<S: Simd>(simd: S, a: S::f64s) -> (S::f64s, S::f64s, S::f64s) {
    let one = S::f64s::splat(simd, 1.0);
    let e = exp(simd, -a.abs());
    let sigmoid = a.simd_ge(0.0).select(one / (e + 1.0), e / (e + 1.0));
    let positive = a.simd_gt(0.0).select(a, S::f64s::splat(simd, 0.0));
    (
        ln_1p(simd, e) + positive,
        sigmoid,
        sigmoid * (one - sigmoid),
    )
}

/// `jax.nn.gelu`'s default, the tanh approximation `a (1 + tanh(u)) / 2`
/// with `u = c (a + k a^3)`, and its first two derivatives.
#[simd]
pub(crate) fn gelu_tanh<S: Simd>(simd: S, a: S::f64s) -> (S::f64s, S::f64s, S::f64s) {
    const C: f64 = 0.797_884_560_802_865_4;
    const K: f64 = 0.044_715;
    let one = S::f64s::splat(simd, 1.0);
    let a2 = a * a;
    let u = (a2 * K + 1.0) * a * C;
    let (_, t) = log_cosh_tanh(simd, u);
    // `sech^2(u)`, and `u'`, `u''` in `a`.
    let s = one - t * t;
    let du = (a2 * (3.0 * K) + 1.0) * C;
    let d2u = a * (6.0 * K * C);
    let half_a = a * 0.5;
    (
        half_a * (t + 1.0),
        (t + 1.0) * 0.5 + half_a * s * du,
        s * du + half_a * s * (d2u - t * du * du * 2.0),
    )
}

#[cfg(test)]
mod tests {
    use fearless_simd::{dispatch, Level};

    use super::*;

    fn grid() -> Vec<f64> {
        let mut xs = vec![0.0, -0.0, 1e-300, 1e-20, 1e-8, 0.3, 0.5, 1.0, 2.0, 20.0];
        xs.extend([21.9, 22.1, 30.0, 300.0, 700.0, 709.7, 1e10, 3e8, 1e200]);
        let mut x = 1e-6;
        while x < 800.0 {
            xs.push(x);
            x *= 1.13;
        }
        xs.extend([f64::INFINITY, f64::NAN]);
        let negated: Vec<f64> = xs.iter().map(|x| -x).collect();
        xs.extend(negated);
        while xs.len() % 8 != 0 {
            xs.push(1.0);
        }
        xs
    }

    const NAMES: [&str; 8] = [
        "exp", "asinh", "sinh", "cosh", "tanh", "log_cosh", "ln_1p", "softplus",
    ];

    /// `out[f][i]`: function `NAMES[f]` at `xs[i]`.
    #[simd]
    fn evaluate<S: Simd>(simd: S, xs: &[f64], out: &mut [Vec<f64>; 8]) {
        for (start, chunk) in xs.chunks_exact(S::f64s::LEN).enumerate() {
            let x = S::f64s::from_slice(simd, chunk);
            let (s, c) = sinh_cosh(simd, x);
            let (log_cosh, tanh) = log_cosh_tanh(simd, x);
            let values = [
                exp(simd, x),
                asinh(simd, x),
                s,
                c,
                tanh,
                log_cosh,
                ln_1p(simd, x.abs()),
                softplus(simd, x).0,
            ];
            let range = start * S::f64s::LEN..(start + 1) * S::f64s::LEN;
            for (out, value) in out.iter_mut().zip(values) {
                value.store_slice(&mut out[range.clone()]);
            }
        }
    }

    /// `gelu_tanh` against the scalar formula, and its derivatives against
    /// central differences of the value and of the first derivative.
    #[test]
    fn gelu_tanh_and_its_derivatives() {
        #[simd]
        fn eval<S: Simd>(simd: S, x: f64) -> (f64, f64, f64) {
            let (f, d1, d2) = gelu_tanh(simd, S::f64s::splat(simd, x));
            (f.as_slice()[0], d1.as_slice()[0], d2.as_slice()[0])
        }
        let level = Level::new();
        let at = |x: f64| dispatch!(level, simd => eval(simd, x));
        let reference = |x: f64| {
            0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh())
        };
        let h = 1e-5;
        for k in -60..=60 {
            let x = 0.1 * k as f64;
            let (f, d1, d2) = at(x);
            assert!(
                (f - reference(x)).abs() <= 1e-15 * (1.0 + x.abs()),
                "f({x})"
            );
            let fd1 = (reference(x + h) - reference(x - h)) / (2.0 * h);
            let fd2 = (at(x + h).1 - at(x - h).1) / (2.0 * h);
            assert!((d1 - fd1).abs() <= 1e-8, "f'({x}): {d1} != {fd1}");
            assert!((d2 - fd2).abs() <= 1e-8, "f''({x}): {d2} != {fd2}");
        }
        // Saturated: exactly the identity and zero, no NaN.
        assert_eq!(at(40.0), (40.0, 1.0, 0.0));
        let (f, d1, d2) = at(-40.0);
        assert!(f == 0.0 && d1 == 0.0 && d2 == 0.0, "{f} {d1} {d2}");
    }

    #[test]
    fn match_libm() {
        let xs = grid();
        let mut out: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; xs.len()]);
        let level = Level::new();
        dispatch!(level, simd => evaluate(simd, &xs, &mut out));
        for (name, values) in NAMES.iter().zip(&out) {
            for (&x, &ours) in xs.iter().zip(values) {
                let a = x.abs();
                let (reference, tol) = match *name {
                    "exp" => (x.exp(), 4.0 * x.exp().abs()),
                    "asinh" => (x.asinh(), 4.0 * x.asinh().abs()),
                    "sinh" => (x.sinh(), 8.0 * x.sinh().abs()),
                    "cosh" => (x.cosh(), 8.0 * x.cosh().abs()),
                    "tanh" => (x.tanh(), 4.0 * x.tanh().abs()),
                    "ln_1p" => (a.ln_1p(), 4.0 * a.ln_1p()),
                    // Absolute: the form cancels near zero.
                    "log_cosh" => (a + (-2.0 * a).exp().ln_1p() - LN_2, 4.0 * (1.0 + a)),
                    _ => {
                        let reference = (-a).exp().ln_1p() + x.max(0.0);
                        (reference, 4.0 * reference.abs())
                    }
                };
                let tol = f64::EPSILON * tol.max(f64::MIN_POSITIVE);
                let ok = ours == reference
                    || (ours - reference).abs() <= tol
                    || (ours.is_nan() && reference.is_nan());
                assert!(ok, "{name}({x:e}): {ours:e} != {reference:e}");
            }
        }
    }
}
