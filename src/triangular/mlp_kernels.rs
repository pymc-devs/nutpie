//! The per-draw MLP kernels: one conditioner MLP of any depth, forward and
//! backward, vectorized with `pulp` across its widths. Shared by the
//! sampler's layout model and the MLP conditioner's `_draw` methods.

use anyhow::Result;
use numpy::{PyArray1, PyReadonlyArray1};
use pulp::{Arch, WithSimd};
use pyo3::prelude::*;
use serde::Deserialize;
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Activation {
    GeluTanh,
    Relu,
    Softplus,
    Silu,
    Tanh,
}

impl Activation {
    #[inline(always)]
    fn apply(self, v: f64) -> f64 {
        match self {
            // jax.nn.gelu's default, the tanh approximation.
            Activation::GeluTanh => {
                0.5 * v * (1.0 + (0.797_884_560_802_865_4 * (v + 0.044_715 * v * v * v)).tanh())
            }
            // `f64::max` returns the non-NaN operand; `jnp.maximum` propagates.
            Activation::Relu => {
                if v.is_nan() {
                    v
                } else {
                    v.max(0.0)
                }
            }
            Activation::Softplus => (-v.abs()).exp().ln_1p() + v.max(0.0),
            Activation::Silu => v / (1.0 + (-v).exp()),
            Activation::Tanh => v.tanh(),
        }
    }

    /// The activation and its derivative. NaN propagates through both.
    #[inline(always)]
    fn apply_with_derivative(self, v: f64) -> (f64, f64) {
        match self {
            Activation::GeluTanh => {
                const C: f64 = 0.797_884_560_802_865_4;
                let inner = C * (v + 0.044_715 * v * v * v);
                let t = inner.tanh();
                let d_inner = C * (1.0 + 3.0 * 0.044_715 * v * v);
                (
                    0.5 * v * (1.0 + t),
                    0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * d_inner,
                )
            }
            Activation::Relu => {
                if v.is_nan() {
                    (v, v)
                } else if v > 0.0 {
                    (v, 1.0)
                } else {
                    (0.0, 0.0)
                }
            }
            Activation::Softplus => {
                let u = (-v.abs()).exp();
                let sigmoid = if v >= 0.0 {
                    1.0 / (1.0 + u)
                } else {
                    u / (1.0 + u)
                };
                (u.ln_1p() + v.max(0.0), sigmoid)
            }
            Activation::Silu => {
                let s = 1.0 / (1.0 + (-v).exp());
                (v * s, s + v * s * (1.0 - s))
            }
            Activation::Tanh => {
                let t = v.tanh();
                (t, 1.0 - t * t)
            }
        }
    }
}

/// Vectorized `exp` and `log1p`, specialized to the ranges softplus needs.
///
/// Softplus is the largest single cost in a conditioner evaluation -- two libm
/// calls per hidden unit, so 32 per variable at the default width of 16 -- and
/// it is the one part of the kernel that is already a batch of independent
/// values sitting in a contiguous buffer. pulp ships no transcendentals, so
/// they are written out here.
///
/// The domains are much narrower than a general implementation must handle,
/// which is what keeps this short: `exp` is only ever called on `(-inf, 0]`,
/// and `log1p` only on `[0, 1]`.
mod softplus {
    use pulp::Simd;

    /// `1 / ln 2`.
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    /// `ln 2` split so that `LN2_HI` has trailing zero bits and `k * LN2_HI` is
    /// exact for the `|k|` that occur here (Cody-Waite).
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    /// `0x1.8p52`. Adding this to a value with `|x| < 2^51` rounds it to an
    /// integer and leaves that integer in the low mantissa bits, so the
    /// rounding and the integer extraction come out of one add.
    const SHIFTER: f64 = 6_755_399_441_055_744.0;
    /// `exp` of anything below this is zero in f64, so clamping here costs
    /// nothing and bounds `k`.
    const MIN_EXP_ARG: f64 = -746.0;
    /// `2^k` is built by writing `k + 1023` into an exponent field, which only
    /// works while that stays positive -- and `k` reaches -1076 at
    /// `MIN_EXP_ARG`. Biasing by a further `2^SCALE` and dividing it back out
    /// afterwards keeps the constructed power normal and lets the result
    /// underflow gracefully through the denormals instead of off a cliff.
    const SCALE: u64 = 537;
    const UNSCALE: f64 = f64::from_bits((1023 - SCALE) << 52);

    /// `1/2! ..= 1/13!`. The polynomial computes `exp(r) - 1 - r`, which is at
    /// most 0.07 over the reduced range, so its own rounding error enters the
    /// result scaled down by that factor; adding the `1 + r` back at the end is
    /// where the accuracy comes from. Truncating after `r^13` leaves
    /// `|r|^14/14! < 1.1e-17` over `|r| <= ln2/2`.
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

    /// fdlibm's `__ieee754_log` minimax coefficients for
    /// `log(1+f) = 2s + s*R(s*s)`, `s = f/(2+f)`, valid to better than 2^-58
    /// for `|s| <= 0.1716`, i.e. `f` in `[-0.2929, 0.4142]`.
    const LG: [f64; 7] = [
        6.666_666_666_666_735_1e-1,
        3.999_999_999_940_941_9e-1,
        2.857_142_874_366_239_1e-1,
        2.222_219_843_214_978_4e-1,
        1.818_357_216_161_805e-1,
        1.531_383_769_920_937_3e-1,
        1.479_819_860_511_658_6e-1,
    ];

    /// `sqrt(2) - 1`, the split that keeps `|s|` inside `LG`'s range.
    const SQRT2_MINUS_1: f64 = 0.414_213_562_373_095_03;

    /// `exp(a)` for `a <= 0`.
    #[inline(always)]
    fn exp_nonpositive<S: Simd>(simd: S, a: S::f64s) -> S::f64s {
        let a = simd.max_f64s(a, simd.splat_f64s(MIN_EXP_ARG));
        let shifter = simd.splat_f64s(SHIFTER);

        // `t = round(a / ln2) + SHIFTER`, so `k_f` is the rounded quotient and
        // the bits of `t` hold that same integer offset by `SHIFTER`'s bits.
        let t = simd.mul_add_e_f64s(a, simd.splat_f64s(INV_LN2), shifter);
        let k = simd.sub_f64s(t, shifter);

        // `2^k`, built by putting `k + 1023` into the exponent field. pulp has
        // no 64-bit shift, but the shift is a multiply by `2^52`.
        // pulp names these output-first: `transmute_u64s_f64s` reads f64 bits
        // as u64, and `transmute_f64s_u64s` goes back.
        let biased = simd.add_u64s(
            simd.transmute_u64s_f64s(t),
            simd.splat_u64s((1023 + SCALE).wrapping_sub(SHIFTER.to_bits())),
        );
        let two_k = simd.transmute_f64s_u64s(simd.mul_u64s(biased, simd.splat_u64s(1 << 52)));

        // `r = a - k*ln2`, in two steps so the cancellation stays exact.
        let r = simd.negate_mul_add_e_f64s(k, simd.splat_f64s(LN2_HI), a);
        let r = simd.negate_mul_add_e_f64s(k, simd.splat_f64s(LN2_LO), r);

        let mut poly = simd.splat_f64s(EXP_C[11]);
        for coefficient in EXP_C[..11].iter().rev() {
            poly = simd.mul_add_e_f64s(poly, r, simd.splat_f64s(*coefficient));
        }
        // exp(r) = 1 + (r + r*r*poly)
        let tail = simd.mul_add_e_f64s(simd.mul_f64s(r, r), poly, r);
        let exp_r = simd.add_f64s(simd.splat_f64s(1.0), tail);

        // `a <= 0` so `k <= 0` and the scaled product cannot overflow; undoing
        // the bias last is what lets small results decay into the denormals.
        simd.mul_f64s(simd.mul_f64s(two_k, exp_r), simd.splat_f64s(UNSCALE))
    }

    /// `log(1 + u)` for `u` in `[0, 1]`.
    #[inline(always)]
    fn log1p_unit<S: Simd>(simd: S, u: S::f64s) -> S::f64s {
        let one = simd.splat_f64s(1.0);
        // Halve the upper part so the reduced argument lands in the interval
        // `LG` was fitted on, and pay for it with one `ln 2` at the end.
        // `u - 1` is exact for `u >= 0.5` (Sterbenz) and loses at most one
        // ulp on the narrow strip below that.
        let upper = simd.greater_than_f64s(u, simd.splat_f64s(SQRT2_MINUS_1));
        let f = simd.select_f64s(
            upper,
            simd.mul_f64s(simd.sub_f64s(u, one), simd.splat_f64s(0.5)),
            u,
        );

        let s = simd.div_f64s(f, simd.add_f64s(simd.splat_f64s(2.0), f));
        let z = simd.mul_f64s(s, s);
        let w = simd.mul_f64s(z, z);

        let mut even = simd.splat_f64s(LG[5]);
        even = simd.mul_add_e_f64s(even, w, simd.splat_f64s(LG[3]));
        even = simd.mul_add_e_f64s(even, w, simd.splat_f64s(LG[1]));
        let mut odd = simd.splat_f64s(LG[6]);
        odd = simd.mul_add_e_f64s(odd, w, simd.splat_f64s(LG[4]));
        odd = simd.mul_add_e_f64s(odd, w, simd.splat_f64s(LG[2]));
        odd = simd.mul_add_e_f64s(odd, w, simd.splat_f64s(LG[0]));
        let poly = simd.mul_add_e_f64s(simd.mul_f64s(even, w), one, simd.mul_f64s(odd, z));

        let half_square = simd.mul_f64s(simd.splat_f64s(0.5), simd.mul_f64s(f, f));
        // log(1+f) = f - (hfsq - s*(hfsq + R))
        let log_f = simd.sub_f64s(
            f,
            simd.sub_f64s(
                half_square,
                simd.mul_f64s(s, simd.add_f64s(half_square, poly)),
            ),
        );

        let zero = simd.splat_f64s(0.0);
        let hi = simd.select_f64s(upper, simd.splat_f64s(LN2_HI), zero);
        let lo = simd.select_f64s(upper, simd.splat_f64s(LN2_LO), zero);
        simd.add_f64s(simd.add_f64s(hi, lo), log_f)
    }

    /// `log(1 + exp(v))`, in the same stable arrangement as the scalar form:
    /// `max(v, 0) + log1p(exp(-|v|))`.
    #[inline(always)]
    pub fn softplus<S: Simd>(simd: S, v: S::f64s) -> S::f64s {
        let u = exp_nonpositive(simd, simd.neg_f64s(simd.abs_f64s(v)));
        let out = simd.add_f64s(simd.max_f64s(v, simd.splat_f64s(0.0)), log1p_unit(simd, u));
        // Both `max` calls above -- the `max(v, 0)` here and the clamp inside
        // `exp_nonpositive` -- return the *other* operand when given a NaN,
        // which would turn a diverged trajectory back into a finite number.
        // `jnp.maximum` propagates instead, and so must this.
        simd.select_f64s(simd.equal_f64s(v, v), out, v)
    }

    /// Softplus and its derivative, which is `sigmoid`.
    ///
    /// The derivative comes out of the forward pass for a select and a divide:
    /// `u = exp(-|v|)` is already computed, and `sigmoid(v)` is `1/(1 + u)` for
    /// `v >= 0` and `u/(1 + u)` below. No second `exp`.
    #[inline(always)]
    pub fn softplus_with_derivative<S: Simd>(simd: S, v: S::f64s) -> (S::f64s, S::f64s) {
        let zero = simd.splat_f64s(0.0);
        let one = simd.splat_f64s(1.0);
        let u = exp_nonpositive(simd, simd.neg_f64s(simd.abs_f64s(v)));
        let out = simd.add_f64s(simd.max_f64s(v, zero), log1p_unit(simd, u));

        let numerator = simd.select_f64s(simd.greater_than_or_equal_f64s(v, zero), one, u);
        let derivative = simd.div_f64s(numerator, simd.add_f64s(one, u));

        let finite = simd.equal_f64s(v, v);
        (
            simd.select_f64s(finite, out, v),
            simd.select_f64s(finite, derivative, v),
        )
    }
}

/// One conditioner MLP, vectorized across the output width.
///
/// The weights arrive input-major -- one contiguous row of `n_out` values per
/// input, see `TriangularLayout.blob` -- so a layer is an AXPY into `n_out`
/// independent accumulators rather than `n_out` dot products. A dot product's
/// `+=` chain cannot be reassociated without fast-math, so it runs at scalar
/// add latency however wide the multiply is vectorized; an AXPY has one
/// dependency chain per accumulator lane and runs at FMA throughput.
///
/// `pulp` supplies the runtime dispatch, so this picks up FMA and the widest
/// available vectors without the whole extension module having to be built for
/// a specific target.
pub(crate) struct EvalMlp<'a> {
    /// This variable's slice of the weight blob.
    pub(crate) weights: &'a [f64],
    pub(crate) layer_out: &'a [usize],
    pub(crate) activation: Activation,
    pub(crate) n_in: usize,
    /// The gathered parent values on entry; reused as scratch for each hidden
    /// layer's activations.
    pub(crate) inputs: &'a mut [f64],
    /// Receives each layer's pre-activations, and finally the transformer
    /// parameters.
    pub(crate) acc: &'a mut [f64],
    /// When recording a tape: receives `activation'(z_l)` for every hidden
    /// layer, concatenated, which is what `BackpropMlp` needs.
    pub(crate) act_derivs: Option<&'a mut [f64]>,
}

/// Number of SIMD accumulators kept live at once.
///
/// Four covers a whole 16-wide hidden layer in one block under AVX2, and gives
/// four independent FMA chains, which is roughly what it takes to keep an FMA
/// unit busy through its own latency.
const ACC_BLOCK: usize = 4;

/// `acc[..N] += splat(inputs[k]) * weights[k * stride ..][..N * lanes]`, summed
/// over every `k`, with the `N` accumulators held in registers.
///
/// `N` is a const parameter so the inner loop unrolls and LLVM can keep the
/// accumulators in registers across the whole `k` loop. Written against a
/// runtime-length slice instead, it reloads and stores each accumulator around
/// every FMA -- the generated code was `vfmadd213pd` straight out of memory
/// followed by `vmovupd` back into it -- and the store-to-load round trip, not
/// the FMA, becomes the dependency chain. That cost about 3x.
///
/// Each accumulator still sums over `k` in increasing order, exactly as a
/// scalar dot product would, so this changes the schedule and not the result.
#[inline(always)]
fn axpy_block<S: pulp::Simd, const N: usize>(
    simd: S,
    acc: &mut [S::f64s],
    weights: &[f64],
    stride: usize,
    inputs: &[f64],
) {
    let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
    let width = N * lanes;

    let mut regs = [acc[0]; N];
    for i in 1..N {
        regs[i] = acc[i];
    }

    for (k, &value) in inputs.iter().enumerate() {
        let splat = simd.splat_f64s(value);
        let (columns, _) = S::as_simd_f64s(&weights[k * stride..k * stride + width]);
        let columns: &[S::f64s; N] = columns[..N].try_into().expect("width is N vectors");
        for i in 0..N {
            regs[i] = simd.mul_add_e_f64s(splat, columns[i], regs[i]);
        }
    }

    acc[..N].copy_from_slice(&regs);
}

/// `dst = activation(src)`, vectorized where it is worth it.
///
/// Softplus and relu go through SIMD; the rest fall back to a scalar loop,
/// which for the transcendental ones is no worse than before.
#[inline(always)]
fn apply_activation<S: pulp::Simd>(simd: S, activation: Activation, src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    match activation {
        Activation::Softplus | Activation::Relu => {
            // Equal lengths, so the two splits agree lane for lane.
            let (src_head, src_tail) = S::as_simd_f64s(src);
            let (dst_head, dst_tail) = S::as_mut_simd_f64s(dst);
            if activation == Activation::Softplus {
                for (out, &pre) in dst_head.iter_mut().zip(src_head) {
                    *out = softplus::softplus(simd, pre);
                }
            } else {
                let zero = simd.splat_f64s(0.0);
                for (out, &pre) in dst_head.iter_mut().zip(src_head) {
                    // See the NaN note in `softplus::softplus`.
                    let clamped = simd.max_f64s(pre, zero);
                    *out = simd.select_f64s(simd.equal_f64s(pre, pre), clamped, pre);
                }
            }
            for (out, &pre) in dst_tail.iter_mut().zip(src_tail) {
                *out = activation.apply(pre);
            }
        }
        other => {
            for (out, &pre) in dst.iter_mut().zip(src) {
                *out = other.apply(pre);
            }
        }
    }
}

/// `dst = activation(src)` and `deriv = activation'(src)`.
///
/// Only softplus takes the SIMD path here, because it is the only one whose
/// derivative falls out of the forward computation; for the rest the scalar
/// loop is what the forward-only version would have done anyway.
#[inline(always)]
fn apply_activation_with_derivative<S: pulp::Simd>(
    simd: S,
    activation: Activation,
    src: &[f64],
    dst: &mut [f64],
    deriv: &mut [f64],
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), deriv.len());
    match activation {
        Activation::Softplus => {
            let (src_head, src_tail) = S::as_simd_f64s(src);
            let (dst_head, dst_tail) = S::as_mut_simd_f64s(dst);
            let (deriv_head, deriv_tail) = S::as_mut_simd_f64s(deriv);
            for ((out, slope), &pre) in dst_head.iter_mut().zip(deriv_head.iter_mut()).zip(src_head)
            {
                let (value, derivative) = softplus::softplus_with_derivative(simd, pre);
                *out = value;
                *slope = derivative;
            }
            for ((out, slope), &pre) in dst_tail.iter_mut().zip(deriv_tail.iter_mut()).zip(src_tail)
            {
                let (value, derivative) = activation.apply_with_derivative(pre);
                *out = value;
                *slope = derivative;
            }
        }
        other => {
            for ((out, slope), &pre) in dst.iter_mut().zip(deriv.iter_mut()).zip(src) {
                let (value, derivative) = other.apply_with_derivative(pre);
                *out = value;
                *slope = derivative;
            }
        }
    }
}

/// Weight rows processed together in the backward pass.
///
/// Each row carries two accumulators (one per cotangent row), so four rows give
/// eight independent FMA chains -- enough to cover the FMA latency -- and eight
/// horizontal reductions that can overlap instead of one stalling the next.
const BACK_BLOCK: usize = 4;

/// `N` dot products against two shared vectors: `out_a[i] = rows[i]·a` and
/// `out_b[i] = rows[i]·b`, for `N` consecutive rows of length `n_out`.
///
/// The transposed weight layout that makes the forward pass an AXPY makes the
/// backward pass a reduction, which is the shape that cannot be reassociated.
/// Doing one row at a time leaves a single accumulator chain per cotangent and
/// pays a horizontal `reduce_sum` per row, so it runs at reduction *latency*:
/// measured at 0.79 cycles per multiply-add against the forward's 0.48.
///
/// Blocking fixes it the same way `axpy_block` does. `N` is a const parameter
/// so the inner loop unrolls and all `2N` accumulators stay in registers, and
/// the vector index runs *outermost* so every one of them is advanced per
/// iteration rather than each chain being drained in turn.
///
/// Each accumulator still sums its own row in the same order, so this is a
/// scheduling change and the result is unchanged bit for bit.
#[inline(always)]
fn dot2_block<S: pulp::Simd, const N: usize>(
    simd: S,
    rows: &[f64],
    n_out: usize,
    a: &[f64],
    b: &[f64],
    out_a: &mut [f64],
    out_b: &mut [f64],
) {
    let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
    let n_vec = n_out / lanes;
    let head = n_vec * lanes;

    let (a_head, a_tail) = S::as_simd_f64s(a);
    let (b_head, b_tail) = S::as_simd_f64s(b);
    let row_vecs: [&[S::f64s]; N] =
        core::array::from_fn(|i| S::as_simd_f64s(&rows[i * n_out..i * n_out + head]).0);

    let zero = simd.splat_f64s(0.0);
    let mut acc_a = [zero; N];
    let mut acc_b = [zero; N];

    for j in 0..n_vec {
        let av = a_head[j];
        let bv = b_head[j];
        for i in 0..N {
            let w = row_vecs[i][j];
            acc_a[i] = simd.mul_add_e_f64s(w, av, acc_a[i]);
            acc_b[i] = simd.mul_add_e_f64s(w, bv, acc_b[i]);
        }
    }

    for i in 0..N {
        let mut sum_a = simd.reduce_sum_f64s(acc_a[i]);
        let mut sum_b = simd.reduce_sum_f64s(acc_b[i]);
        for ((&r, &x), &y) in rows[i * n_out + head..(i + 1) * n_out]
            .iter()
            .zip(a_tail)
            .zip(b_tail)
        {
            sum_a += r * x;
            sum_b += r * y;
        }
        out_a[i] = sum_a;
        out_b[i] = sum_b;
    }
}

/// Backpropagate two cotangent rows through one conditioner MLP.
///
/// The two rows are `d y_i / d theta` and `d log_det_i / d theta` from the
/// transformer; the results are `d y_i / d y_p` and `d log_det_i / d y_p` for
/// each parent `p`, i.e. one row each of the sparse Jacobian `A` and the log
/// det coupling `C`.
pub(crate) struct BackpropMlp<'a> {
    pub(crate) weights: &'a [f64],
    pub(crate) layer_out: &'a [usize],
    pub(crate) n_in: usize,
    /// `activation'(z_l)` for every hidden layer, concatenated in forward order.
    pub(crate) act_derivs: &'a [f64],
    pub(crate) seed_y: &'a mut [f64],
    pub(crate) seed_l: &'a mut [f64],
    pub(crate) next_y: &'a mut [f64],
    pub(crate) next_l: &'a mut [f64],
    pub(crate) out_y: &'a mut [f64],
    pub(crate) out_l: &'a mut [f64],
}

impl<'a> WithSimd for BackpropMlp<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            weights,
            layer_out,
            n_in,
            act_derivs,
            mut seed_y,
            mut seed_l,
            mut next_y,
            mut next_l,
            out_y,
            out_l,
        } = self;

        // Re-walk the layer geometry rather than carrying it from the forward
        // pass: it is a handful of integer ops against a second buffer.
        let n_layers = layer_out.len();
        let mut layers: SmallVec<[(usize, usize, usize, usize); 8]> = SmallVec::new();
        let mut offset = 0usize;
        let mut deriv_offset = 0usize;
        let mut inputs = n_in;
        for (layer, &n_out) in layer_out.iter().enumerate() {
            layers.push((offset, inputs, n_out, deriv_offset));
            offset += inputs * n_out + n_out;
            if layer + 1 < n_layers {
                deriv_offset += n_out;
            }
            inputs = n_out;
        }

        for layer in (0..n_layers).rev() {
            let (offset, n_in_l, n_out, _) = layers[layer];
            let columns = &weights[offset..offset + n_in_l * n_out];
            let (cotangent_y, cotangent_l) = (&seed_y[..n_out], &seed_l[..n_out]);

            // Full blocks, then the remainder; `BACK_BLOCK` is 4 so the arms
            // below cover every leftover width.
            let mut done = 0;
            while done + BACK_BLOCK <= n_in_l {
                dot2_block::<S, BACK_BLOCK>(
                    simd,
                    &columns[done * n_out..],
                    n_out,
                    cotangent_y,
                    cotangent_l,
                    &mut next_y[done..],
                    &mut next_l[done..],
                );
                done += BACK_BLOCK;
            }
            match n_in_l - done {
                0 => {}
                1 => dot2_block::<S, 1>(
                    simd,
                    &columns[done * n_out..],
                    n_out,
                    cotangent_y,
                    cotangent_l,
                    &mut next_y[done..],
                    &mut next_l[done..],
                ),
                2 => dot2_block::<S, 2>(
                    simd,
                    &columns[done * n_out..],
                    n_out,
                    cotangent_y,
                    cotangent_l,
                    &mut next_y[done..],
                    &mut next_l[done..],
                ),
                _ => dot2_block::<S, 3>(
                    simd,
                    &columns[done * n_out..],
                    n_out,
                    cotangent_y,
                    cotangent_l,
                    &mut next_y[done..],
                    &mut next_l[done..],
                ),
            }
            if layer > 0 {
                // The activation feeding this layer sits at the previous
                // layer's output width, which is this layer's input width.
                let deriv_at = layers[layer - 1].3;
                let slopes = &act_derivs[deriv_at..deriv_at + n_in_l];
                for (k, &slope) in slopes.iter().enumerate() {
                    next_y[k] *= slope;
                    next_l[k] *= slope;
                }
            }
            core::mem::swap(&mut seed_y, &mut next_y);
            core::mem::swap(&mut seed_l, &mut next_l);
        }

        out_y.copy_from_slice(&seed_y[..n_in]);
        out_l.copy_from_slice(&seed_l[..n_in]);
    }
}

impl<'a> WithSimd for EvalMlp<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            weights,
            layer_out,
            activation,
            mut n_in,
            inputs,
            acc,
            mut act_derivs,
        } = self;

        let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
        let n_layers = layer_out.len();
        let mut offset = 0usize;
        let mut deriv_offset = 0usize;

        for (layer, &n_out) in layer_out.iter().enumerate() {
            let bias_at = offset + n_in * n_out;
            acc[..n_out].copy_from_slice(&weights[bias_at..bias_at + n_out]);
            let columns = &weights[offset..bias_at];
            let values = &inputs[..n_in];

            // With no inputs -- a variable without parents, or a layer after a
            // zero-width one -- the output is just the bias. The blocked loop
            // below walks the *outputs*, and would slice the then empty
            // `columns` past its end from the second block on.
            if n_in > 0 {
                let (acc_head, acc_tail) = S::as_mut_simd_f64s(&mut acc[..n_out]);

                // Full blocks first, then whatever is left over: a layer is
                // only a few vectors wide, so the remainder arms cover every
                // width that actually occurs (16 is one full block under AVX2,
                // and the 9-wide parameter layer is two vectors plus a scalar
                // tail).
                let mut done = 0;
                while done + ACC_BLOCK <= acc_head.len() {
                    axpy_block::<S, ACC_BLOCK>(
                        simd,
                        &mut acc_head[done..],
                        &columns[done * lanes..],
                        n_out,
                        values,
                    );
                    done += ACC_BLOCK;
                }
                match acc_head.len() - done {
                    0 => {}
                    1 => axpy_block::<S, 1>(
                        simd,
                        &mut acc_head[done..],
                        &columns[done * lanes..],
                        n_out,
                        values,
                    ),
                    2 => axpy_block::<S, 2>(
                        simd,
                        &mut acc_head[done..],
                        &columns[done * lanes..],
                        n_out,
                        values,
                    ),
                    _ => axpy_block::<S, 3>(
                        simd,
                        &mut acc_head[done..],
                        &columns[done * lanes..],
                        n_out,
                        values,
                    ),
                }

                // At most `lanes - 1` outputs, so not worth blocking.
                if !acc_tail.is_empty() {
                    let base = acc_head.len() * lanes;
                    for (k, &value) in values.iter().enumerate() {
                        let row = &columns[k * n_out + base..];
                        for (slot, &w) in acc_tail.iter_mut().zip(row) {
                            *slot = value.mul_add(w, *slot);
                        }
                    }
                }
            }

            offset = bias_at + n_out;
            if layer + 1 < n_layers {
                match &mut act_derivs {
                    Some(derivs) => apply_activation_with_derivative(
                        simd,
                        activation,
                        &acc[..n_out],
                        &mut inputs[..n_out],
                        &mut derivs[deriv_offset..deriv_offset + n_out],
                    ),
                    None => apply_activation(simd, activation, &acc[..n_out], &mut inputs[..n_out]),
                }
                deriv_offset += n_out;
            }
            n_in = n_out;
        }
    }
}

/// Apply one activation through the SIMD path, for accuracy testing.
///
/// The hidden pre-activations a real flow produces are all O(1), so exercising
/// the vectorized `exp`/`log1p` over their full domain needs a direct handle.
#[pyfunction]
#[pyo3(name = "_activation_for_testing")]
pub fn activation_for_testing<'py>(
    py: Python<'py>,
    activation: &str,
    x: PyReadonlyArray1<'py, f64>,
) -> Result<Bound<'py, PyArray1<f64>>> {
    let activation: Activation =
        from_tag(activation).map_err(|_| anyhow::anyhow!("unknown activation {activation:?}"))?;
    let src = x.as_slice()?.to_vec();
    let mut dst = vec![0.0; src.len()];
    Arch::new().dispatch(ApplyActivation {
        activation,
        src: &src,
        dst: &mut dst,
    });
    Ok(PyArray1::from_vec(py, dst))
}

struct ApplyActivation<'a> {
    activation: Activation,
    src: &'a [f64],
    dst: &'a mut [f64],
}

impl<'a> WithSimd for ApplyActivation<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        apply_activation(simd, self.activation, self.src, self.dst)
    }
}

pub(crate) fn from_tag<T: for<'de> Deserialize<'de>>(tag: &str) -> Result<T, serde_json::Error> {
    serde_json::from_value(serde_json::Value::String(tag.to_string()))
}
