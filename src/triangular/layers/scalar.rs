//! The transformer chain on `f64`, forward, with the derivatives the
//! transform's tape needs.

use smallvec::SmallVec;

use super::{Contract2, Layer, LogGammaBound, Param, PositiveAffineSpec, TangentSasSpec};

impl LogGammaBound {
    #[inline(always)]
    fn apply(&self, unbounded: f64) -> f64 {
        self.low + self.width / (1.0 + (-(self.slope * unbounded + self.offset)).exp())
    }

    /// The same value together with `d/d unbounded`.
    #[inline(always)]
    fn apply_with_derivative(&self, unbounded: f64) -> (f64, f64) {
        let sigmoid = 1.0 / (1.0 + (-(self.slope * unbounded + self.offset)).exp());
        (
            self.low + self.width * sigmoid,
            self.width * self.slope * sigmoid * (1.0 - sigmoid),
        )
    }
}

#[inline(always)]
fn param_or(param: Option<Param>, params: &[f64], default: f64) -> f64 {
    param.map_or(default, |p| p.get(params))
}

#[inline(always)]
fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

impl Layer {
    #[inline(always)]
    fn transform(&self, params: &[f64], y: f64) -> (f64, f64) {
        match self {
            Layer::Contract2(layer) => layer.transform(params, y),
            Layer::TangentSas(layer) => layer.transform(params, y),
            Layer::PositiveAffine(layer) => layer.transform(params, y),
        }
    }

    #[inline(always)]
    fn transform_with_tape(&self, params: &[f64], y: f64) -> (f64, f64, LayerTape) {
        match self {
            Layer::Contract2(layer) => layer.transform_with_tape(params, y),
            Layer::TangentSas(layer) => layer.transform_with_tape(params, y),
            Layer::PositiveAffine(layer) => layer.transform_with_tape(params, y),
        }
    }
}

/// `exp(asinh(a))`, which is algebraically `a + sqrt(1 + a*a)`.
///
/// `Contract2` writes this as an `asinh` followed by an `exp` because the
/// direct form cancels catastrophically for `a << 0`. Taking the conjugate,
/// `a + sqrt(1 + a*a) == 1 / (sqrt(1 + a*a) - a)`, which is well conditioned
/// exactly where the direct form is not -- so picking the branch by sign gives
/// the same accuracy with no transcendental at all. That removes two libm
/// calls per `Contract2` layer for `gamma`, and another for `sigma_mod`.
#[inline(always)]
fn exp_asinh(a: f64) -> f64 {
    // `1 + a*a` overflows above ~1e154, where the result is `2|a|` or
    // `1/(2|a|)` to full precision anyway.
    let root = if a.abs() > 1e150 {
        a.abs()
    } else {
        (1.0 + a * a).sqrt()
    };
    if a >= 0.0 {
        a + root
    } else {
        1.0 / (root - a)
    }
}

/// `exp(asinh(a))` and its derivative, which is `exp(asinh(a)) / sqrt(1 + a*a)`.
#[inline(always)]
fn exp_asinh_with_derivative(a: f64) -> (f64, f64) {
    let root = if a.abs() > 1e150 {
        a.abs()
    } else {
        (1.0 + a * a).sqrt()
    };
    let value = if a >= 0.0 { a + root } else { 1.0 / (root - a) };
    (value, value / root)
}

/// `log(cosh(asinh(s)))`, which is `0.5 * log1p(s*s)` since
/// `cosh(asinh(s)) == sqrt(1 + s*s)`.
///
/// The transformer only ever needs `log cosh` of an `asinh`, so this replaces
/// the general stable form -- an `exp` and a `log1p` -- with one `log1p`.
#[inline(always)]
fn log_cosh_asinh(s: f64) -> f64 {
    if s.abs() > 1e150 {
        s.abs().ln()
    } else {
        0.5 * (s * s).ln_1p()
    }
}

/// `log(cosh(v))` given `sinh(v)`, from `cosh^2 == 1 + sinh^2`.
///
/// Reusing the `sinh` the transform already computed removes the `exp` the
/// general form needs. Above `|v| ~ 300` the `log1p` term is exactly zero in
/// f64 -- which is also where `sinh(v)^2` would overflow -- so both branches
/// are exact.
#[inline(always)]
fn log_cosh_from_sinh(sinh_v: f64, v: f64) -> f64 {
    if v.abs() < 300.0 {
        0.5 * (sinh_v * sinh_v).ln_1p()
    } else {
        v.abs() - std::f64::consts::LN_2
    }
}

/// The elementwise transformer chain, mirroring the JAX layers'
/// `transform_and_log_det`.
#[inline]
pub(crate) fn transform_element(layers: &[Layer], params: &[f64], x: f64) -> (f64, f64) {
    let mut y = x;
    let mut log_det = 0.0;
    for layer in layers {
        let (out, ld) = layer.transform(params, y);
        y = out;
        log_det += ld;
    }
    (y, log_det)
}

/// Per-layer partials of one transformer layer, kept for the reverse pass
/// over the chain. `p_*_in` are with respect to the layer's input; `params`
/// holds, for each parameter the layer actually has, its flat index and the
/// layer's two partials with respect to it.
#[derive(Clone, Copy)]
struct LayerTape {
    /// `d y_out / d y_in`, which is also `exp(log_det)` for this layer.
    p_y_in: f64,
    /// `d log_det / d y_in`.
    p_l_in: f64,
    n_params: usize,
    params: [(usize, f64, f64); 5],
}

impl LayerTape {
    #[inline(always)]
    fn new(p_y_in: f64, p_l_in: f64) -> Self {
        Self {
            p_y_in,
            p_l_in,
            n_params: 0,
            params: [(0, 0.0, 0.0); 5],
        }
    }

    #[inline(always)]
    fn push(&mut self, param: Option<Param>, dy: f64, dld: f64) {
        if let Some(param) = param {
            self.params[self.n_params] = (param.index, dy, dld);
            self.n_params += 1;
        }
    }
}

/// `transform_element`, plus every derivative the pullback needs.
///
/// Returns `(y, log_det, dy/dx, dlog_det/dx)` and fills `dy_dtheta` and
/// `dld_dtheta` with the derivatives against the conditioner's outputs.
///
/// `d y_out / d y_in` of each layer is `exp(log_det)` of that layer, so
/// accumulating it along the chain gives the Jacobian diagonal for free, and
/// more accurately than exponentiating a sum of logs.
pub(crate) fn transform_element_with_grads(
    layers: &[Layer],
    params: &[f64],
    x: f64,
    dy_dtheta: &mut [f64],
    dld_dtheta: &mut [f64],
) -> (f64, f64, f64, f64) {
    let mut tapes: SmallVec<[LayerTape; 4]> = SmallVec::new();
    let mut y = x;
    let mut log_det = 0.0;

    for layer in layers {
        let (out, ld, tape) = layer.transform_with_tape(params, y);
        tapes.push(tape);
        y = out;
        log_det += ld;
    }

    // Reverse over the chain. `ay` is `d y_final / d y_k` and `al` is
    // `d (sum of later log dets) / d y_k`, both at the input of the layer about
    // to be processed.
    dy_dtheta.fill(0.0);
    dld_dtheta.fill(0.0);
    let mut ay = 1.0;
    let mut al = 0.0;
    for tape in tapes.iter().rev() {
        for &(index, p_y, p_l) in &tape.params[..tape.n_params] {
            dy_dtheta[index] = ay * p_y;
            dld_dtheta[index] = p_l + al * p_y;
        }
        al = tape.p_l_in + al * tape.p_y_in;
        ay *= tape.p_y_in;
    }

    (y, log_det, ay, al)
}

/// `tanh(v)` given `sinh(v)` and `cosh(v)`. `cosh` overflows past |v| ~ 355,
/// the same place `sinh` does; the ratio is 1 long before that.
#[inline(always)]
fn tanh_from(sinh_v: f64, cosh_v: f64, v: f64) -> f64 {
    if v.abs() < 300.0 {
        sinh_v / cosh_v
    } else {
        v.signum()
    }
}

impl Contract2 {
    /// One layer of `Contract2::transform_and_log_det`.
    ///
    /// Algebraically identical to the Python version, but arranged so that
    /// each layer costs 6 libm calls instead of 11: `log gamma` is never
    /// needed on its own (the `exp(log_sigma - log_gamma)` factor is just
    /// `sigma_mod / gamma`), and both `exp(asinh(.))` and the two `log cosh`es
    /// have closed forms here. See `exp_asinh`, `log_cosh_asinh` and
    /// `log_cosh_from_sinh`.
    #[inline(always)]
    fn transform(&self, params: &[f64], y: f64) -> (f64, f64) {
        let gamma = match (self.alpha, &self.bound) {
            (None, _) => 1.0,
            (Some(alpha), None) => exp_asinh(alpha.get(params)),
            // Bounded `log gamma` is squashed through a sigmoid, so it has to
            // be formed explicitly and exponentiated the long way.
            (Some(alpha), Some(bound)) => bound.apply(alpha.get(params).asinh()).exp(),
        };
        let log_delta = self.beta.map_or(0.0, |p| p.get(params).asinh());
        let (sigma_mod, log_sigma) = match self.sigma {
            None => (1.0, 0.0),
            Some(sigma) => {
                let sigma_mod = exp_asinh(sigma.get(params));
                (sigma_mod, sigma_mod.ln())
            }
        };

        let centred = match self.nu {
            None => y,
            Some(nu) => y - nu.get(params),
        };
        let half = 0.5 * centred;
        let u = half.asinh();
        let arg = gamma * u + 2.0 * log_delta;
        let sinh_arg = arg.sinh();

        let mut out = 2.0 * (sigma_mod / gamma) * sinh_arg;
        if let Some(mu) = self.mu {
            out += mu.get(params);
        }
        (
            out,
            log_sigma + log_cosh_from_sinh(sinh_arg, arg) - log_cosh_asinh(half),
        )
    }

    /// `transform` and its partials.
    ///
    /// `d y_out / d y_in` is `sigma_mod * cosh(arg) / cosh(u)`, and every
    /// partial is rational in quantities the forward pass already formed:
    /// `cosh(u)` is `sqrt(1 + half^2)` and `cosh(arg)` is
    /// `sqrt(1 + sinh(arg)^2)`, so the derivative costs two square roots and
    /// **no new transcendental calls**.
    #[inline(always)]
    fn transform_with_tape(&self, params: &[f64], y: f64) -> (f64, f64, LayerTape) {
        // gamma, and d gamma / d alpha.
        let (gamma, dgamma) = match (self.alpha, &self.bound) {
            (None, _) => (1.0, 0.0),
            (Some(alpha), None) => exp_asinh_with_derivative(alpha.get(params)),
            (Some(alpha), Some(bound)) => {
                let a = alpha.get(params);
                let (log_gamma, dlog_gamma) = bound.apply_with_derivative(a.asinh());
                let gamma = log_gamma.exp();
                (gamma, gamma * dlog_gamma / (1.0 + a * a).sqrt())
            }
        };
        let (log_delta, dlog_delta) = match self.beta {
            None => (0.0, 0.0),
            Some(beta) => {
                let b = beta.get(params);
                (b.asinh(), 1.0 / (1.0 + b * b).sqrt())
            }
        };
        let (sigma_mod, dsigma_mod, log_sigma) = match self.sigma {
            None => (1.0, 0.0, 0.0),
            Some(sigma) => {
                let (m, dm) = exp_asinh_with_derivative(sigma.get(params));
                (m, dm, m.ln())
            }
        };

        let centred = match self.nu {
            None => y,
            Some(nu) => y - nu.get(params),
        };
        let half = 0.5 * centred;
        let cosh_u = (1.0 + half * half).sqrt();
        let u = half.asinh();
        let tanh_u = half / cosh_u;

        let arg = gamma * u + 2.0 * log_delta;
        let sinh_arg = arg.sinh();
        let cosh_arg = (1.0 + sinh_arg * sinh_arg).sqrt();
        let tanh_arg = tanh_from(sinh_arg, cosh_arg, arg);

        let scale = 2.0 * sigma_mod / gamma;
        let y_out = scale * sinh_arg + self.mu.map_or(0.0, |mu| mu.get(params));
        let ld = log_sigma + log_cosh_from_sinh(sinh_arg, arg) - log_cosh_asinh(half);

        let du_dy = 0.5 / cosh_u;
        let mut tape = LayerTape::new(
            sigma_mod * cosh_arg / cosh_u,
            (gamma * tanh_arg - tanh_u) * du_dy,
        );

        // alpha enters only through gamma, which scales `u` inside `arg` and
        // divides the outer amplitude.
        tape.push(
            self.alpha,
            scale * (cosh_arg * u - sinh_arg / gamma) * dgamma,
            tanh_arg * u * dgamma,
        );
        // beta shifts `arg` by `2 log delta`.
        tape.push(
            self.beta,
            scale * cosh_arg * 2.0 * dlog_delta,
            tanh_arg * 2.0 * dlog_delta,
        );
        // sigma is a pure amplitude, so its log det partial is just
        // `d log sigma_mod / d sigma`.
        tape.push(
            self.sigma,
            (2.0 * sinh_arg / gamma) * dsigma_mod,
            dsigma_mod / sigma_mod,
        );
        tape.push(self.mu, 1.0, 0.0);
        // nu shifts the input, so its partials are the input ones negated.
        tape.push(self.nu, -tape.p_y_in, -tape.p_l_in);

        (y_out, ld, tape)
    }
}

impl TangentSasSpec {
    /// One layer of `TangentSAS.transform_and_log_det`: with `w = b (y - nu)`
    /// and `arg = r asinh(w) + eps`,
    ///
    /// ```text
    /// y_out = nu + (sinh(arg) - sinh(eps)) / (r b cosh(eps)),
    /// ld    = logcosh(arg) - logcosh(eps) - 0.5 log1p(w^2).
    /// ```
    #[inline(always)]
    fn transform(&self, params: &[f64], y: f64) -> (f64, f64) {
        let nu = param_or(self.nu, params, 0.0);
        let b = sigmoid(param_or(self.b, params, 0.0));
        let r = self.r.map_or(1.0, |p| exp_asinh(p.get(params)));
        let (eps, sinh_eps, cosh_eps) = match self.eps {
            None => (0.0, 0.0, 1.0),
            Some(p) => {
                let eps = p.get(params);
                (eps, eps.sinh(), eps.cosh())
            }
        };

        let w = b * (y - nu);
        let arg = r * w.asinh() + eps;
        let sinh_arg = arg.sinh();
        let out = nu + (sinh_arg - sinh_eps) / (r * b * cosh_eps);
        let ld = log_cosh_from_sinh(sinh_arg, arg)
            - log_cosh_from_sinh(sinh_eps, eps)
            - log_cosh_asinh(w);
        (out, ld)
    }

    /// `transform` and its partials. With `u = y - nu`, `R = sqrt(1 + w^2)`,
    /// `a = asinh(w)`, `D = r b cosh(eps)` and `S = y_out - nu`:
    ///
    /// ```text
    /// dS/du  = cosh(arg) / (cosh(eps) R),       dld/du  = b (r tanh(arg) - w/R) / R,
    /// dS/deps = (cosh(arg) - cosh(eps)) / D - S tanh(eps),   dld/deps = tanh(arg) - tanh(eps),
    /// dS/db  = (u dS/du - S) / b,               dld/db  = u (dld/du) / b,
    /// dS/dr  = cosh(arg) a / D - S / r,         dld/dr  = tanh(arg) a,
    /// ```
    ///
    /// and `db/db_raw = b (1 - b)` cancels the `1 / b`.
    #[inline(always)]
    fn transform_with_tape(&self, params: &[f64], y: f64) -> (f64, f64, LayerTape) {
        let nu = param_or(self.nu, params, 0.0);
        let b = sigmoid(param_or(self.b, params, 0.0));
        let (r, dr) = self
            .r
            .map_or((1.0, 0.0), |p| exp_asinh_with_derivative(p.get(params)));
        let (eps, sinh_eps, cosh_eps) = match self.eps {
            None => (0.0, 0.0, 1.0),
            Some(p) => {
                let eps = p.get(params);
                (eps, eps.sinh(), eps.cosh())
            }
        };
        let tanh_eps = sinh_eps / cosh_eps;

        let u = y - nu;
        let w = b * u;
        let root = if w.abs() > 1e150 {
            w.abs()
        } else {
            (1.0 + w * w).sqrt()
        };
        let a = w.asinh();
        let arg = r * a + eps;
        let sinh_arg = arg.sinh();
        let cosh_arg = (1.0 + sinh_arg * sinh_arg).sqrt();
        let tanh_arg = tanh_from(sinh_arg, cosh_arg, arg);

        let denominator = r * b * cosh_eps;
        let shape = (sinh_arg - sinh_eps) / denominator;
        let ld = log_cosh_from_sinh(sinh_arg, arg)
            - log_cosh_from_sinh(sinh_eps, eps)
            - log_cosh_asinh(w);

        let p_y_in = cosh_arg / (cosh_eps * root);
        let p_l_in = b * (r * tanh_arg - w / root) / root;
        let mut tape = LayerTape::new(p_y_in, p_l_in);

        // nu shifts both the input and the output.
        tape.push(self.nu, 1.0 - p_y_in, -p_l_in);
        tape.push(
            self.eps,
            (cosh_arg - cosh_eps) / denominator - shape * tanh_eps,
            tanh_arg - tanh_eps,
        );
        tape.push(
            self.b,
            (1.0 - b) * (u * p_y_in - shape),
            (1.0 - b) * u * p_l_in,
        );
        tape.push(
            self.r,
            (cosh_arg * a / denominator - shape / r) * dr,
            tanh_arg * a * dr,
        );

        (nu + shape, ld, tape)
    }
}

impl PositiveAffineSpec {
    /// `y_out = loc + scale_mod * y`, `scale_mod = scale + sqrt(1 + scale^2)`.
    #[inline(always)]
    fn transform(&self, params: &[f64], y: f64) -> (f64, f64) {
        let loc = param_or(self.loc, params, 0.0);
        match self.scale {
            None => (loc + y, 0.0),
            Some(p) => {
                let scale = p.get(params);
                (loc + exp_asinh(scale) * y, scale.asinh())
            }
        }
    }

    #[inline(always)]
    fn transform_with_tape(&self, params: &[f64], y: f64) -> (f64, f64, LayerTape) {
        let loc = param_or(self.loc, params, 0.0);
        let (scale, (scale_mod, dscale_mod)) = match self.scale {
            None => (0.0, (1.0, 0.0)),
            Some(p) => {
                let scale = p.get(params);
                (scale, exp_asinh_with_derivative(scale))
            }
        };
        let mut tape = LayerTape::new(scale_mod, 0.0);
        tape.push(self.loc, 1.0, 0.0);
        // `log scale_mod = asinh(scale)`, whose derivative is
        // `dscale_mod / scale_mod`.
        tape.push(self.scale, y * dscale_mod, dscale_mod / scale_mod);
        (loc + scale_mod * y, scale.asinh(), tape)
    }
}
