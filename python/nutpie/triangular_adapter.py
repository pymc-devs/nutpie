"""The flow adapter for triangular flows built in Rust, and the parts of the
window logic it shares with `nutpie.transform_adapter`: which draws a fit
uses, the Rust LM run and its report.

`TriangularFlowAdapter` is the adapter `make_transform_adapter(rust_flow=True)`
gives the sampler. It does what `transform_adapter.TransformAdapter` does
with ``coupling_type="triangular"`` and ``method="lm-rust"``, but holds a
`nutpie.triangular_flow.TriangularFlow` instead of a JAX flow: no JAX on the
way from the draws to the sampler's native transform. JAX is only needed for
a model's ``logp_fn``, if the sampler asks the adapter to evaluate it.

Importing this module does not import JAX.
"""

from __future__ import annotations

import traceback

import numpy as np

from nutpie.triangular_flow import (
    build_triangular_flow,
    diag_affine,
    diag_from_gradients,
)

_LOG_STOP_VALUE = -3.5
_LOG_SKIP_TRAINING_VALUE = -3.0
# Fewest distinct draws a diagonal fit needs; with fewer, it matches them
# exactly and its scales are arbitrary.
_MIN_DIAG_DRAWS = 5

_BIJECTION_TRACE = []


def _format_log_f(value):
    value = float(value)
    return f"{value:+.2f}" if np.isfinite(value) else "  nan"


def _recent_distinct(positions):
    """``(keep, n_repeats)`` for a diagonal fit: the chronological indices of
    the newest ``len(positions) // 5 + 3`` distinct draws, and how many
    repeats among them were skipped. Repeats (rejected transitions) say
    nothing new about the score, and a fit to a few distinct points matches
    them exactly."""
    size = len(positions)
    target = max(size // 5 + 3, _MIN_DIAG_DRAWS)
    seen, keep, start = set(), [], 0
    for i in range(size - 1, -1, -1):
        key = positions[i].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
            if len(keep) == target:
                start = i
                break
    return keep[::-1], size - start - len(keep)


def _thin(available, max_draws, recency):
    """Sorted indices of `max_draws` distinct ones of `available` draws, at a
    density growing like ``t**recency`` from the oldest (``t=0``) to the
    newest draw (``t=1``); ``recency=0`` thins evenly.

    The density is capped at one per draw, the newest draws are then all
    taken and the rest goes to older ones. Systematic sampling: draw `i` is
    taken where the cumulative density passes a half-integer, which keeps the
    spacing as even as the density allows, and the selection deterministic.
    """
    t = (np.arange(available) + 0.5) / available
    weight = t**recency
    # The scale of the capped density that sums to `max_draws`. At `hi`
    # every draw is capped, and the sum is `available > max_draws`.
    lo, hi = 0.0, 1.0 / weight.min()
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.minimum(1.0, mid * weight).sum() < max_draws:
            lo = mid
        else:
            hi = mid
    density = np.minimum(1.0, hi * weight)
    cumulative = np.cumsum(density)
    taken = np.floor(cumulative + 0.5) > np.floor(cumulative - density + 0.5)
    return np.flatnonzero(taken)[-max_draws:]


def _select_draws(
    n_draws, start, *, max_draws, recency, val_fraction, block_size, multiple, rng
):
    """``(train, val)`` indices into draws ``start:n_draws`` for a flow fit.

    At most `max_draws` draws; when there are more, they are thinned (see
    `_thin`), denser towards the newest draws with a positive `recency`.
    Thinning lowers the autocorrelation, and keeps the selection
    deterministic. A random fraction `val_fraction` of blocks of `block_size`
    consecutive draws is held out for validation; contiguous blocks keep
    NUTS repeats and strongly correlated neighbours on one side of the split.
    Both parts are cut to a multiple of `multiple` (the Rust LM's SIMD width)
    by dropping their oldest draws, and are in chronological order.
    """
    available = n_draws - start
    if available > max_draws:
        idx = start + _thin(available, max_draws, recency)
    else:
        idx = np.arange(start, n_draws)

    n_blocks = -(-len(idx) // block_size)
    n_val_blocks = round(val_fraction * n_blocks)
    if val_fraction > 0 and n_blocks > 1:
        n_val_blocks = min(max(n_val_blocks, 1), n_blocks - 1)
    is_val_block = np.zeros(n_blocks, dtype=bool)
    is_val_block[rng.choice(n_blocks, size=n_val_blocks, replace=False)] = True
    is_val = np.repeat(is_val_block, block_size)[: len(idx)]

    train, val = idx[~is_val], idx[is_val]
    return train[len(train) % multiple :], val[len(val) % multiple :]


def _run_lm(
    problem,
    theta,
    *,
    max_steps,
    rtol,
    linear_steps,
    min_loss,
    patience,
    forcing,
    lam0,
    max_exact_block_size,
    lmp_size,
    lmp_tol,
    verbose,
    should_stop,
    val_problem=None,
    early_stopping=True,
    mlp_ridge=0.0,
    freeze_units=False,
):
    """The Rust LM fit of `problem` (a `FisherResiduals` with data) from
    `theta`, see `nutpie.triangular_lm.fit`.

    `val_problem` holds the held-out draws for the validation loss and,
    with `early_stopping`, the early stopping. `freeze_units` holds the
    hidden units fixed. Returns ``(theta, losses, lam, n_steps,
    n_accepted)``, with the losses at the returned `theta`: ``"objective"``
    (what LM minimised, with the regularizations), ``"train"`` and ``"val"``
    (Fisher divergences, ``val`` `None` without `val_problem`)."""
    from nutpie.triangular_lm import fisher_divergence, fit

    theta, lam, hist = fit(
        problem,
        theta,
        n_steps=max_steps,
        **({} if lam0 is None else {"lam0": lam0}),
        min_loss=min_loss,
        rtol=rtol,
        patience=patience,
        verbose=verbose,
        should_stop=should_stop,
        cg_max=linear_steps,
        forcing=forcing,
        max_block_size=max_exact_block_size,
        lmp_size=lmp_size,
        lmp_tol=lmp_tol,
        mlp_ridge=mlp_ridge,
        frozen=problem.unit_param_mask if freeze_units else None,
        val_problem=val_problem,
        early_stopping=early_stopping,
    )
    theta = np.asarray(theta)

    accepted = [info for info in hist if info["accept"]]
    # Evaluated afresh: early stopping may have returned an earlier `theta`
    # than the last step's.
    n_var = len(problem.param_offsets) - 1
    r = problem.residuals(theta, linearize=False)
    r_fisher = r.reshape(problem.n_draw, problem.n_residuals)[:, :n_var]
    unit_weights = theta[problem.unit_weight_mask]
    ridge = mlp_ridge / problem.n_draw * float(unit_weights @ unit_weights)
    losses = {
        "objective": float(r @ r) + ridge,
        "train": float(np.vdot(r_fisher, r_fisher)),
        "val": None if val_problem is None else fisher_divergence(val_problem, theta),
    }
    # The damping to carry over is the one after the last accepted step:
    # each rejection in a final streak (typical of a patience stop) only
    # multiplied `lam` by a growing `nu`.
    if accepted:
        lam = accepted[-1]["lam_out"]
    return theta, losses, float(lam), len(hist), len(accepted)


def _affine_fisher_divergence(loc, scale, positions, gradients):
    """The mean Fisher divergence of the draws under the diagonal flow
    ``y = loc + scale * z``."""
    z = (positions - loc) / scale
    residual = z + gradients * scale
    return float((residual * residual).sum(1).mean())


def _soft_clip(gradient, clip):
    """``clip * asinh(gradient / clip)``, or `gradient` without a clip."""
    if clip is None:
        return gradient
    return clip * np.arcsinh(gradient / clip)


class TriangularFlowAdapter:
    """The sampler's flow adapter with a triangular flow built in Rust.

    The windows go as in `transform_adapter.TransformAdapter`: the first
    `num_diag_windows` fit a diagonal affine to the newest distinct draws;
    after that, each window fits the triangular flow with the Rust LM on the
    selected draws, and keeps the refit only if it does better on the
    held-out ones. See `make_transform_adapter` for the settings.
    """

    def __init__(
        self,
        seed,
        position,
        gradient,
        chain,
        *,
        logp_fn,
        sparsity,
        order,
        verbose,
        window_size,
        num_diag_windows,
        initial_skip,
        forget_fraction,
        recency,
        val_fraction,
        val_block_size,
        early_stopping,
        linear_first_fit,
        debug_save_bijection,
        stop_event,
        nn_width,
        activation,
        zero_init,
        contract_transformer,
        tangent_sas_transformer,
        tangent_sas_fix_b,
        log_gamma_bounds,
        input_squash,
        max_epochs,
        solver_rtol,
        lm_linear_steps,
        lm_min_loss,
        lm_fisher_regularization,
        lm_mlp_ridge,
        lm_patience,
        lm_forcing,
        lm_max_exact_block_size,
        lm_lmp_size,
        lm_lmp_tol,
    ):
        from nutpie._lib import FisherResiduals

        if sparsity is None:
            raise ValueError(
                "The triangular flow needs a `sparsity` (a boolean Markov-blanket "
                "adjacency matrix of shape (n_dim, n_dim))."
            )
        self._logp_fn = logp_fn
        self._value_and_grad = None
        self._sparsity = sparsity
        self._order = order
        self._chain = chain
        # 0: silent, 1: one line per window, 2: also the LM iterations
        self._verbose = int(verbose)
        self._window_size = window_size
        self._num_diag_windows = num_diag_windows
        self._initial_skip = initial_skip
        self._forget_fraction = forget_fraction
        self._recency = recency
        self._val_fraction = val_fraction
        self._val_block_size = val_block_size
        self._early_stopping = early_stopping
        self._linear_first_fit = linear_first_fit
        self._debug_save_bijection = debug_save_bijection
        # Set by the main thread when sampling is aborted. The training runs
        # in a chain's thread, where Python never raises KeyboardInterrupt.
        self._stop_event = stop_event
        self._flow_settings = {
            "nn_width": nn_width,
            "activation": activation,
            "zero_init": zero_init,
            "contract_transformer": contract_transformer,
            "tangent_sas_transformer": tangent_sas_transformer,
            "tangent_sas_fix_b": tangent_sas_fix_b,
            "log_gamma_bounds": log_gamma_bounds,
            "input_squash": input_squash,
        }
        self._lm_settings = {
            "max_steps": max_epochs,
            "rtol": solver_rtol,
            "linear_steps": lm_linear_steps,
            "min_loss": lm_min_loss,
            "patience": lm_patience,
            "forcing": lm_forcing,
            "max_exact_block_size": lm_max_exact_block_size,
            "lmp_size": lm_lmp_size,
            "lmp_tol": lm_lmp_tol,
            "mlp_ridge": lm_mlp_ridge,
        }
        self._lm_fisher_regularization = lm_fisher_regularization
        # The Rust LM fit needs a multiple of its SIMD width of draws.
        self._draw_multiple = FisherResiduals.simd_width()
        # Damping the previous LM fit ended with, to start the next one from,
        # see `TransformAdapter`.
        self._lm_lam = None

        # The current flow: the triangular flow once one was accepted, before
        # that the diagonal affine alone.
        self._flow = None
        self._loc, self._scale = diag_affine(
            np.asarray(position, dtype=np.float64)[None],
            np.asarray(gradient, dtype=np.float64)[None],
        )
        self.index = 0

    # ----------------------------------------------------------- interface

    @property
    def transformation_id(self):
        return self.index

    def flow_transform_layout(self):
        """The current flow for the sampler's native leapfrog: the
        `TriangularFlow` itself, or the diagonal affine's layout before it
        exists (see `triangular_rust.flow_transform_layout`)."""
        if self._flow is not None:
            return self._flow
        return {"loc": self._loc.copy(), "scale": self._scale.copy()}

    def inv_transform(self, position, gradient):
        """``(log_det, z, grad_z)``: a draw and its gradient in the base
        space, and the forward transform's log determinant there."""
        position = np.asarray(position, dtype=np.float64)
        gradient = np.asarray(gradient, dtype=np.float64)
        if self._flow is not None:
            return self._flow.inv_transform(position, gradient)
        z = (position - self._loc) / self._scale
        return float(np.log(np.abs(self._scale)).sum()), z, gradient * self._scale

    def init_from_transformed_position_part1(self, transformed_position):
        """The draw at a base-space position, and what part 2 needs."""
        z = np.asarray(transformed_position, dtype=np.float64)
        if self._flow is None:
            y = self._loc + self._scale * z
            return y, float(np.log(np.abs(self._scale)).sum())
        native = self._flow.flow_transform()
        y, log_det = native.transform_and_log_det(z)
        return np.asarray(y), (native, log_det)

    def init_from_transformed_position_part2(self, part1, untransformed_gradient):
        """``(log_det, grad_z)``: the gradient at part 1's draw pulled back."""
        gradient = np.asarray(untransformed_gradient, dtype=np.float64)
        if self._flow is None:
            return part1, gradient * self._scale
        native, log_det = part1
        return log_det, np.asarray(native.pullback(gradient))

    def init_from_transformed_position(self, transformed_position, clip):
        """``(logp, log_det, y, grad_y, grad_z)`` at a base-space position,
        with the model's ``logp_fn``."""
        y, part1 = self.init_from_transformed_position_part1(transformed_position)
        logp, grad_y = self._logp_and_grad(y)
        grad_y = _soft_clip(grad_y, clip)
        log_det, grad_z = self.init_from_transformed_position_part2(part1, grad_y)
        return logp, log_det, y, grad_y, grad_z

    def init_from_untransformed_position(self, untransformed_position, clip):
        """``(logp, log_det, grad_y, z, grad_z)`` at a draw, with the model's
        ``logp_fn``."""
        y = np.asarray(untransformed_position, dtype=np.float64)
        logp, grad_y = self._logp_and_grad(y)
        grad_y = _soft_clip(grad_y, clip)
        log_det, z, grad_z = self.inv_transform(y, grad_y)
        return logp, log_det, grad_y, z, grad_z

    def _logp_and_grad(self, y):
        if self._logp_fn is None:
            raise NotImplementedError(
                "This model has no logp_fn for the adapter to evaluate; the "
                "sampler uses the native flow instead."
            )
        if self._value_and_grad is None:
            import jax

            self._value_and_grad = jax.jit(
                jax.value_and_grad(lambda x: self._logp_fn(x)[0])
            )
        logp, grad = self._value_and_grad(y)
        return float(logp), np.asarray(grad, dtype=np.float64)

    # -------------------------------------------------------------- update

    def _report(self, n_draws, message):
        """One line about the current window, for ``verbose >= 1``."""
        if self._verbose:
            print(
                f"flow chain {self._chain} window {self.index:3d} "
                f"({n_draws:5d} draws): {message}"
            )

    def _should_stop(self):
        return self._stop_event is not None and self._stop_event.is_set()

    def _fisher_divergence(self, flow, positions, gradients):
        """The Fisher divergence of the draws under `flow`, or under the
        diagonal affine for `None`."""
        if flow is None:
            return _affine_fisher_divergence(
                self._loc, self._scale, positions, gradients
            )
        return flow.fisher_divergence(positions, gradients)

    def update(self, seed, positions, gradients, logps):
        self.index += 1
        n_draws = len(positions)
        if self._should_stop() or n_draws == 0:
            return
        try:
            positions = np.asarray(positions, dtype=np.float64)
            gradients = np.asarray(gradients, dtype=np.float64)
            if self.index <= self._num_diag_windows:
                self._update_diag(positions, gradients, n_draws)
            else:
                self._update_flow(seed, positions, gradients, n_draws)
        except Exception as e:
            print("update error:", e)
            print(traceback.format_exc())
            raise

    def _update_diag(self, positions, gradients, n_draws):
        keep, n_repeats = _recent_distinct(positions)
        repeats = f", {n_repeats} repeats skipped" if n_repeats else ""
        if len(keep) < _MIN_DIAG_DRAWS:
            # Too few draws for variances: scale by the gradient instead, so
            # that a bad scale from earlier does not keep the chain stuck.
            self._flow = None
            self._loc, self._scale = diag_from_gradients(
                positions[keep], gradients[keep]
            )
            self._report(
                n_draws,
                f"diag from the gradient, only {len(keep)} distinct draws{repeats}",
            )
            return
        positions, gradients = positions[keep], gradients[keep]
        loc, scale = diag_affine(positions, gradients)
        new_loss = np.log(_affine_fisher_divergence(loc, scale, positions, gradients))
        self._report(
            n_draws,
            f"log F  diag {_format_log_f(new_loss)}  ({len(keep)} draws{repeats})",
        )
        if np.isfinite(new_loss):
            self._flow = None
            self._loc, self._scale = loc, scale

    def _update_flow(self, seed, positions, gradients, n_draws):
        # Early draws come from a chain that may not have converged yet:
        # skip a fixed number, and forget the oldest `forget_fraction`.
        start = max(self._initial_skip, int(self._forget_fraction * n_draws))
        train_idx, val_idx = _select_draws(
            n_draws,
            start,
            max_draws=self._window_size,
            recency=self._recency,
            val_fraction=self._val_fraction,
            block_size=self._val_block_size,
            multiple=self._draw_multiple,
            rng=np.random.default_rng(seed),
        )
        if len(train_idx) < 10:
            self._report(
                n_draws,
                f"keep flow, only {len(train_idx)} draws to train on "
                f"(skipping the first {start})",
            )
            return
        used = np.concatenate([train_idx, val_idx])
        if not (
            np.isfinite(positions[used]).all() and np.isfinite(gradients[used]).all()
        ):
            raise ValueError("The draws or their gradients are not finite.")
        train = (positions[train_idx], gradients[train_idx])
        val = (positions[val_idx], gradients[val_idx]) if len(val_idx) else None
        # The draws to compare flows on: held out, if there are any.
        eval_data = train if val is None else val
        draw_counts = f"{len(train_idx)} train, {len(val_idx)} val draws"
        log_f = "log F" if val is None else "log F val"

        fresh = self._flow is None
        # A fresh flow's first fit only moves the linear part of its
        # conditioners, see `make_transform_adapter`.
        freeze_units = fresh and self._linear_first_fit
        if fresh:
            base = build_triangular_flow(
                seed,
                *train,
                sparsity=self._sparsity,
                order=self._order,
                **self._flow_settings,
            )
            self._report(
                n_draws,
                f"new flow: {_describe(base)}"
                + (", first fit without the MLPs" if freeze_units else ""),
            )
            if self._verbose >= 2:
                fresh_loss = np.log(base.fisher_divergence(*eval_data))
                self._report(
                    n_draws, f"{log_f}  fresh flow {_format_log_f(fresh_loss)}"
                )
        else:
            base = self._flow

        old_loss = np.log(self._fisher_divergence(self._flow, *eval_data))
        if (
            np.isfinite(old_loss)
            and old_loss < _LOG_SKIP_TRAINING_VALUE
            and self.index > 10
        ):
            self._report(
                n_draws,
                f"{log_f}  current {_format_log_f(old_loss)}  -> keep flow, "
                f"already below {_format_log_f(_LOG_SKIP_TRAINING_VALUE)}",
            )
            return

        # Header for the LM steps printed below; a fresh flow has its own.
        if self._verbose >= 2 and not fresh:
            self._report(
                n_draws,
                f"{log_f}  current {_format_log_f(old_loss)}  -> refit  ({draw_counts})",
            )

        problem = base.residuals(
            *train, fisher_regularization=self._lm_fisher_regularization
        )
        val_problem = None if val is None else base.residuals(*val)
        theta, losses, lam, n_steps, n_accepted = _run_lm(
            problem,
            base.theta,
            val_problem=val_problem,
            early_stopping=self._early_stopping,
            freeze_units=freeze_units,
            lam0=self._lm_lam,
            verbose=self._verbose >= 2,
            should_stop=self._should_stop,
            **self._lm_settings,
        )
        if self._should_stop():
            # Sampling was aborted, keep the flow as it is.
            return

        # Kept even if the fit is discarded below: the damping scale says
        # something about the problem, whether or not this fit won. But not
        # from a fit that made no progress at all, where every rejected step
        # only raised it.
        no_progress = n_accepted == 0
        if not no_progress:
            self._lm_lam = lam

        fit = base.with_theta(theta)
        new_loss = np.log(fit.fisher_divergence(*eval_data))

        def report(decision):
            self._report(
                n_draws,
                f"{log_f}  current {_format_log_f(old_loss)}  "
                f"refit {_format_log_f(new_loss)}"
                + (
                    ""
                    if val is None
                    else f"  (train {_format_log_f(np.log(losses['train']))})"
                )
                + f"  -> {decision}"
                + f"  ({n_steps} LM step{'' if n_steps == 1 else 's'})",
            )

        if self._debug_save_bijection:
            _BIJECTION_TRACE.append((self.index, fit, train))

        def valid_new_logp():
            log_det, z, grad_z = fit.inv_transform(train[0][-1], train[1][-1])
            return (
                np.isfinite(log_det)
                and np.isfinite(z).all()
                and np.isfinite(grad_z).all()
            )

        if no_progress:
            report("keep flow, no LM step accepted")
            return

        if not np.isfinite(old_loss) and not np.isfinite(new_loss):
            report("reset to a diagonal flow, both are invalid")
            self._flow = None
            self._loc, self._scale = diag_affine(*train)
            return

        if not valid_new_logp():
            report("keep old flow, refit gives an invalid transform")
            return

        if not np.isfinite(new_loss):
            report("keep old flow, refit is invalid")
            return

        if new_loss > old_loss:
            report("keep old flow, refit is worse")
            return

        report("replace flow")
        self._flow = fit


def _describe(flow):
    """Size and structure of a flow, for the verbose output, as
    `transform_adapter._describe_flow`."""
    return (
        f"{flow.n_var} dimensions, {flow.n_params} parameters, "
        f"at most {flow.max_parents} parents, {flow.n_levels} levels"
    )
