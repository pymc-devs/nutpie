import math
from collections.abc import Callable
from functools import partial
from importlib.util import find_spec

if find_spec("flowjax") is None:
    raise ImportError(
        "The 'flowjax' package is required to use normalizing flow adaptation."
    )

import traceback

import equinox as eqx
import flowjax
import flowjax.flows
import flowjax.train
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import optimistix as optx
import tqdm
from flowjax import bijections
from flowjax.train.losses import MaximumLikelihoodLoss, PRNGKeyArray
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)
from jaxtyping import ArrayLike, PyTree
from paramax import NonTrainable, unwrap

from nutpie.normalizing_flow import Coupling, Householder, Scan, extend_flow, make_flow

_BIJECTION_TRACE = []

_LOG_STOP_VALUE = -5
_LOG_SKIP_TRAINING_VALUE = -4
# Fewest distinct draws a diagonal fit needs; with fewer, it matches them
# exactly and its scales are arbitrary.
_MIN_DIAG_DRAWS = 5

# Remat toggle for the per-draw residual, see `FisherLoss.residuals`.
CHECKPOINT_RESIDUAL = False


def fit_to_data(
    key: PRNGKeyArray,
    dist: PyTree,
    x,
    *,
    condition: ArrayLike | None = None,
    loss_fn: Callable | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 128,
    val_prop: float = 0.1,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    return_best: bool = True,
    show_progress: bool = True,
    opt_state=None,
    verbose: bool = False,
    stop_value: float | None = None,
    method: str = "adam",
    solver_rtol: float = 1e-3,
    solver_atol: float = 1e-6,
    lm_linear_steps: int = 300,
    lm_min_loss: float = float(np.exp(-3)),
    lm_probe_batch: int = 32,
    lm_probes: int = 64,
    lm_probe_groups: int | None = None,
    lm_probe_rounds: int = 1,
    lm_fit_affine: bool = False,
    lm_patience: int = 5,
    lm_line_search: bool = False,
    lm_forcing: str = "residual",
    lm_lam0: float | None = None,
    lm_exact_blocks: bool = False,
    lm_max_exact_block_size: int = 256,
    lm_lmp_size: int = 0,
    lm_lmp_tol: float = 1e-8,
    lm_print_blocks: bool = False,
    lm_diagnose: bool = False,
    should_stop: Callable[[], bool] | None = None,
):
    r"""Train a distribution (e.g. a flow) to samples from the target distribution.

    The distribution can be unconditional :math:`p(x)` or conditional
    :math:`p(x|\text{condition})`. Note that the last batch in each epoch is dropped
    if truncated (to avoid recompilation). This function can also be used to fit
    non-distribution pytrees as long as a compatible loss function is provided.

    Args:
        key: Jax random seed.
        dist: The distribution to train.
        x: Samples from target distribution.
        condition: Conditioning variables. Defaults to None.
        loss_fn: Loss function. Defaults to MaximumLikelihoodLoss.
        max_epochs: Maximum number of epochs. Defaults to 100. When ``method`` is
            ``"lbfgs"`` or ``"lm"``, this instead bounds the number of solver steps.
        max_patience: Number of consecutive epochs with no validation loss improvement
            after which training is terminated. Defaults to 5. Unused unless
            ``method`` is ``"adam"``.
        batch_size: Batch size. Defaults to 100. Unused unless ``method`` is
            ``"adam"``.
        val_prop: Proportion of data to use in validation set. Defaults to 0.1.
            Unused unless ``method`` is ``"adam"``.
        learning_rate: Adam learning rate. Defaults to 5e-4.
        optimizer: Optax optimizer. If provided, this overrides the default Adam
            optimizer, and the learning_rate is ignored. Defaults to None.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.
        method: One of ``"adam"`` (stochastic optax updates, the default),
            ``"lbfgs"`` or ``"lm"`` (Levenberg-Marquardt). The latter two use
            full-batch, deterministic solvers from optimistix, run once over all
            of ``x`` rather than in epochs of shuffled mini-batches. ``"lm"``
            requires ``loss_fn`` to expose a ``residuals`` method (as
            ``FisherLoss`` does) and only supports losses that are a sum of
            squared residuals. ``"lm-rust"`` is the same fit in Rust
            (`nutpie.triangular_lm.fit`), for a ``FisherLoss`` on a flow from
            ``make_flow(kind="triangular")`` with softplus depth-1
            conditioners: always exact blocks, the line search and a frozen
            affine, so the probe, ``lm_fit_affine``, ``lm_line_search``,
            ``lm_exact_blocks`` and ``lm_diagnose`` options do not apply.
        solver_rtol: Relative tolerance used by the L-BFGS/LM solver's convergence
            check. Only used when ``method`` is ``"lbfgs"`` or ``"lm"``.
        solver_atol: Absolute tolerance used by the L-BFGS/LM solver's convergence
            check. Only used when ``method`` is ``"lbfgs"`` or ``"lm"``.
        lm_linear_steps: Cap on the matrix-free CG steps used to solve the
            Gauss-Newton normal equations at each LM iteration (the Jacobian is
            far too large to factorize explicitly, so Jacobian-vector-product
            steps are used instead). This is `lmopt.step`'s ``cg_max``: CG stops
            earlier when it converges, so raising it costs nothing on the steps
            that do converge. Steps that hit the cap are solving a system they
            did not finish, and show as ``cg=<n>*`` in the fit log. Only used
            when ``method`` is ``"lm"``.
        lm_probes: Number of Rademacher probes used to estimate the block
            preconditioner at each LM step. The estimate is a Hutchinson
            average, so its noise falls like ``1/sqrt(lm_probes)`` while the
            cost is one reverse pass each -- it is the dominant per-step cost
            once CG is cheap. It also sets where `lmopt.make_plan` splits a
            conditioner into sub-blocks (``q`` is capped relative to it), so
            raising it both sharpens the estimate and keeps large conditioners
            unsplit. Not used with ``lm_exact_blocks``. Only used when
            ``method`` is ``"lm"``.
        lm_probe_batch: How many of the Rademacher probes used to estimate the
            block preconditioner are taken at once. This is the main memory
            knob of an LM step: each concurrent probe carries a full reverse
            pass through the residual function, so peak memory scales with it
            (and multiplies with the residual function's own internal
            batching). Lowering it trades sequential chunks for peak memory at
            no extra FLOPs. Only used when ``method`` is ``"lm"``.
        lm_probe_groups: If given, estimate the block preconditioner from
            per-group probes instead (see `lmopt.build_blocks_grouped`): the
            draws are split into this many groups, each probed separately, so
            one round yields this many samples for about the cost of a single
            ``lm_probes`` probe plus one forward pass. ``lm_probes`` then only
            sets the sub-block size, and ``lm_probe_batch`` counts groups
            rather than probes. ``None`` keeps the all-draws probes. Only used
            when ``method`` is ``"lm"``.
        lm_probe_rounds: Rounds of per-group probes, each an independent set
            of ``lm_probe_groups`` samples. Only used with ``lm_probe_groups``.
        lm_fit_affine: Whether LM also fits the flow's diagonal affine layer.
            By default it stays at its initialization (see
            `lmopt.split_frozen`). Only used when ``method`` is ``"lm"``.
        lm_patience: Stop the LM fit once this many consecutive steps have
            together lowered the loss by less than a fraction
            ``solver_rtol`` of it. Only used when ``method`` is ``"lm"``.
        lm_line_search: Shorten each LM step to the minimizer of a parabola
            fitted along it, against the Gauss-Newton overshoot on
            large-residual fits (see `lmopt.step`). Costs one extra residual
            evaluation on the steps it shortens. Only used when ``method`` is
            ``"lm"``.
        lm_forcing: How the CG tolerance adapts between LM steps:
            ``"residual"`` (Eisenstat-Walker choice 1) or ``"rho"``
            (``|1 - rho|``), which keeps adapting when the loss plateaus well
            above zero (see `lmopt.step`). With ``method="lm-rust"`` also
            ``"model"``: CG stops once an iteration barely lowers the
            quadratic model (Nash-Sofer). Used when ``method`` is ``"lm"``
            or ``"lm-rust"``.
        lm_diagnose: Print, below each LM step, whether geodesic acceleration
            would have helped and how the step splits over the conditioners
            (see `lmopt.describe_diagnostics`). Costs about one more CG solve
            per step. Needs ``verbose``; only used when ``method`` is ``"lm"``.
        lm_lam0: Initial LM damping; ``None`` uses `lmopt.fit`'s default.
            The damping the fit ends with is returned as
            ``losses["lm_lam"]``, so a caller refitting on similar data can
            carry it over. Only used when ``method`` is ``"lm"``.
        lm_exact_blocks: Compute the LM preconditioner's Gauss-Newton blocks
            exactly (see `FisherLoss.gauss_newton_factors`) instead of
            estimating them from ``lm_probes`` Rademacher probes. Needs
            ``lm_fit_affine=False``. Only used when ``method`` is ``"lm"``.
        lm_max_exact_block_size: With ``lm_exact_blocks``, conditioners with
            more parameters are split into sub-blocks of at most this size.
            This only bounds the cost of the preconditioner, roughly
            ``size**2`` memory and ``size**3`` time per sub-block.
        lm_lmp_size: Number of earlier CG search directions the limited-memory
            preconditioner keeps on top of the block preconditioner; ``0``
            disables it. Only used when ``method`` is ``"lm-rust"``.
        lm_lmp_tol: Relative eigenvalue cutoff below which near-dependent
            stored directions are dropped. Only used when ``method`` is
            ``"lm-rust"``.
        lm_min_loss: Stop the LM fit once the Fisher divergence falls below
            this. Note that the divergence is a *sum* over dimensions, so this
            is an absolute, dimension-independent target: it bounds each
            individual direction's misfit regardless of model size, and a
            larger model therefore has to work harder to reach it. Only used
            when ``method`` is ``"lm"``.

    Returns:
        A tuple containing the trained distribution and the losses.
    """
    if not isinstance(x, tuple):
        x = (x,)
    data = x if condition is None else (*x, condition)
    data = tuple(jnp.asarray(a) for a in data)

    if loss_fn is None:
        loss_fn = MaximumLikelihoodLoss()

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )

    if method == "lm-rust":
        params, loss_val, lam, n_steps, n_accepted = _fit_lm_rust(
            params,
            static,
            data,
            loss_fn,
            max_steps=max_epochs,
            rtol=solver_rtol,
            linear_steps=lm_linear_steps,
            min_loss=lm_min_loss,
            patience=lm_patience,
            forcing=lm_forcing,
            lam0=lm_lam0,
            max_exact_block_size=lm_max_exact_block_size,
            lmp_size=lm_lmp_size,
            lmp_tol=lm_lmp_tol,
            verbose=verbose,
            should_stop=should_stop,
        )
        losses = {
            "train": [float(loss_val)],
            "val": [float(loss_val)],
            "lm_lam": lam,
            "lm_steps": n_steps,
            "lm_accepted": n_accepted,
        }
        return eqx.combine(params, static), losses, None

    if method in ("lbfgs", "lm"):
        fit_solver = _fit_lbfgs if method == "lbfgs" else _fit_lm
        extra_kwargs = (
            {
                "linear_steps": lm_linear_steps,
                "min_loss": lm_min_loss,
                "probe_batch": lm_probe_batch,
                "probes": lm_probes,
                "probe_groups": lm_probe_groups,
                "probe_rounds": lm_probe_rounds,
                "fit_affine": lm_fit_affine,
                "patience": lm_patience,
                "line_search": lm_line_search,
                "forcing": lm_forcing,
                "lam0": lm_lam0,
                "exact_blocks": lm_exact_blocks,
                "max_exact_block_size": lm_max_exact_block_size,
                "verbose": verbose,
                "print_blocks": lm_print_blocks,
                "diagnose": lm_diagnose,
                "should_stop": should_stop,
            }
            if method == "lm"
            else {}
        )
        result = fit_solver(
            params,
            static,
            data,
            loss_fn,
            max_steps=max_epochs,
            rtol=solver_rtol,
            atol=solver_atol,
            **extra_kwargs,
        )
        params, loss_val = result[:2]
        losses = {"train": [float(loss_val)], "val": [float(loss_val)]}
        if method == "lm":
            losses["lm_lam"] = result[2]
            losses["lm_steps"] = result[3]
            losses["lm_accepted"] = result[4]
        dist = eqx.combine(params, static)
        return dist, losses, None
    elif method != "adam":
        raise ValueError(
            f"Unknown method {method!r}, expected 'adam', 'lbfgs', 'lm' or 'lm-rust'."
        )

    if optimizer is None:
        optimizer = optax.apply_if_finite(optax.adamw(learning_rate), 10)

    best_params = params

    if opt_state is None:
        opt_state = optimizer.init(params)

    # train val split
    key, subkey = jr.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)
    losses = {"train": [], "val": []}

    loop = tqdm.tqdm(range(max_epochs), disable=not show_progress)

    for i in loop:
        # Shuffle data
        key, *subkeys = jr.split(key, 3)
        train_data = [jr.permutation(subkeys[0], a) for a in train_data]
        val_data = [jr.permutation(subkeys[1], a) for a in val_data]

        key, subkey = jr.split(key)
        batches = get_batches(train_data, batch_size)
        batch_losses = []

        if True:
            for batch in zip(*batches, strict=True):
                if should_stop is not None and should_stop():
                    break
                key, subkey = jr.split(key)
                params, opt_state, batch_loss = step(
                    params,
                    static,
                    *batch,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    loss_fn=loss_fn,
                    key=subkey,
                )
                batch_losses.append(batch_loss)
        else:
            params, opt_state, batch_losses = _step_batch_loop(
                params,
                static,
                opt_state,
                optimizer,
                loss_fn,
                subkey,
                *batches,
            )

        if not batch_losses:
            # Stopped before the first batch of this epoch
            break
        losses["train"].append((sum(batch_losses) / len(batch_losses)).item())

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            loss_i = loss_fn(params, static, *batch, key=subkey)
            batch_losses.append(loss_i)

        loss = sum(batch_losses) / len(batch_losses)
        losses["val"].append(loss)

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

        elif stop_value is not None and loss < stop_value:
            loop.set_postfix_str(f"{loop.postfix} (Stop value reached)")
            break

        if should_stop is not None and should_stop():
            break

    params = best_params if return_best else params
    dist = eqx.combine(params, static)
    return dist, losses, opt_state


@eqx.filter_jit
def _fit_lbfgs(params, static, data, loss_fn, *, max_steps, rtol, atol):
    def objective(params, args):
        return loss_fn(params, static, *args)

    solver = optx.LBFGS(rtol=rtol, atol=atol)
    sol = optx.minimise(
        objective, solver, params, args=data, max_steps=max_steps, throw=False
    )
    return sol.value, objective(sol.value, data)


@eqx.filter_jit
def res_fn(params, args):
    loss_fn, *args = args
    return loss_fn.residuals(params, *args)


def gn_factor_fn(params, args, draw_data):
    """`lmopt.fit`'s `factor_fn` for `res_fn`: one draw's exact Gauss-Newton
    block factors, see `FisherLoss.gauss_newton_factors`."""
    loss_fn, static = args
    return loss_fn.gauss_newton_factors(params, static, *draw_data)


def _conditioner_coordinates(flow):
    """The coordinate (index into the flattened unconstrained draw) that each
    conditioner of a triangular flow transforms, one array per bucket; `None`
    for other flows."""
    try:
        sandwich = unwrap(flow).bijection.bijections[0].bijections[0]
        order = np.ravel(np.asarray(sandwich.outer.permutation))
        members = sandwich.inner.bucket_members
    except (AttributeError, IndexError, TypeError):
        return None
    return [order[np.asarray(m)] for m in members]


def _fit_lm(
    params,
    static,
    data,
    loss_fn,
    *,
    max_steps,
    rtol,
    atol,
    linear_steps,
    min_loss,
    probe_batch,
    probes,
    probe_groups,
    probe_rounds,
    fit_affine,
    patience,
    line_search,
    forcing,
    lam0,
    exact_blocks,
    max_exact_block_size,
    verbose,
    print_blocks,
    diagnose,
    should_stop,
):
    if not hasattr(loss_fn, "residuals"):
        raise ValueError(
            "method='lm' requires loss_fn to have a `residuals` method "
            "(e.g. FisherLoss with gamma=None)."
        )

    from nutpie.lmopt import fit

    theta, hist = fit(
        params,
        res_fn,
        (loss_fn, static),
        data=data,
        n_groups=probe_groups,
        rounds=probe_rounds,
        fit_affine=fit_affine,
        rtol=rtol,
        patience=patience,
        line_search=line_search,
        forcing=forcing,
        **({} if lam0 is None else {"lam0": lam0}),
        factor_fn=gn_factor_fn if exact_blocks else None,
        max_exact_block_size=max_exact_block_size,
        n_steps=max_steps,
        verbose=verbose,
        print_blocks=print_blocks,
        diagnose=diagnose,
        conditioner_labels=(
            _conditioner_coordinates(eqx.combine(params, static)) if diagnose else None
        ),
        should_stop=should_stop,
        min_loss=min_loss,
        cg_max=linear_steps,
        m=probes,
        batch=probe_batch,
        precondition=True,
    )

    n_accepted = sum(bool(info["accept"]) for info in hist)
    return (
        theta,
        hist[-1]["F_out"],
        float(hist[-1]["lam_out"]),
        len(hist),
        n_accepted,
    )


def _fit_lm_rust(
    params,
    static,
    data,
    loss_fn,
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
):
    """`_fit_lm` with exact blocks and the line search, in Rust (see
    `nutpie.triangular_lm.fit`). Fits the conditioners of the flow's
    `SparseTriangularMap`; everything else stays as it is."""
    from nutpie.lmopt import _conditioners
    from nutpie.triangular_lm import (
        fit,
        make_residuals,
        map_data,
        pack_params,
        unpack_params,
    )

    if not isinstance(loss_fn, FisherLoss) or loss_fn.gamma is not None:
        raise ValueError("method='lm-rust' needs a FisherLoss with gamma=None.")
    draws, grads, *_ = data
    tmap, y, g = map_data(eqx.combine(params, static), draws, grads)
    problem = make_residuals(
        tmap, y, g, fisher_regularization=loss_fn.fisher_regularization
    )
    theta, lam, hist = fit(
        problem,
        pack_params(tmap),
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
    )

    fitted = eqx.filter(
        unpack_params(tmap, jnp.asarray(theta)).conditioners, eqx.is_inexact_array
    )
    if jax.tree.structure(fitted) != jax.tree.structure(_conditioners(params)):
        raise RuntimeError("The fitted conditioners do not match the flow's.")
    params = eqx.tree_at(_conditioners, params, fitted)

    accepted = [info for info in hist if info["accept"]]
    if hist:
        loss = hist[-1]["F_out"]
    else:
        r = problem.residuals(theta, record=False)
        loss = float(r @ r)
    # The damping to carry over is the one after the last accepted step:
    # each rejection in a final streak (typical of a patience stop) only
    # multiplied `lam` by a growing `nu`.
    if accepted:
        lam = accepted[-1]["lam_out"]
    return params, loss, float(lam), len(hist), len(accepted)


@eqx.filter_jit
def _step_batch_loop(params, static, opt_state, optimizer, loss_fn, key, *batches):
    def scan_fn(carry, batch):
        params, opt_state, key = carry
        key, subkey = jr.split(key)
        params, opt_state, loss_i = step(
            params,
            static,
            *batch,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
            key=subkey,
        )
        return (params, opt_state, key), loss_i

    (params, opt_state, _), batch_losses = jax.lax.scan(
        scan_fn, (params, opt_state, key), batches
    )

    return params, opt_state, batch_losses


@eqx.filter_jit
def inverse_gradient_and_val(bijection, draw, grad, logp, *, naive=False):
    if naive:
        x = bijection.inverse(draw)
        (_, fwd_log_det), pull_grad_fn = jax.vjp(
            lambda x: bijection.transform_and_log_det(x), x
        )
        (x_grad,) = pull_grad_fn((grad, jnp.ones(())))
        return (x, x_grad, logp + fwd_log_det)
    if hasattr(bijection, "inverse_gradient_and_val"):
        return bijection.inverse_gradient_and_val(draw, grad, logp)
    if isinstance(bijection, bijections.Chain):
        for b in bijection.bijections[::-1]:
            draw, grad, logp = inverse_gradient_and_val(b, draw, grad, logp)
        return draw, grad, logp
    elif isinstance(bijection, bijections.Permute):
        return (
            draw[bijection.inverse_permutation],
            grad[bijection.inverse_permutation],
            logp,
        )
    elif isinstance(bijection, bijections.Affine):
        draw, logdet = bijection.inverse_and_log_det(draw)
        grad = grad * unwrap(bijection.scale)
        return (draw, grad, logp - logdet)
    elif isinstance(bijection, Householder):
        params = unwrap(bijection.params)
        params = params / jnp.linalg.norm(params)
        draw = draw - 2 * params * (draw @ params)
        grad = grad - 2 * params * (grad @ params)
        return (draw, grad, logp)
    elif isinstance(bijection, bijections.Vmap):

        def inner(bijection, y, y_grad, y_logp):
            return inverse_gradient_and_val(bijection, y, y_grad, y_logp)

        y, y_grad, log_det = eqx.filter_vmap(
            inner,
            in_axes=(bijection.in_axes[0], 0, 0, None),
            axis_size=bijection.axis_size,
        )(bijection.bijection, draw, grad, jnp.zeros(()))
        return y, y_grad, jnp.sum(log_det) + logp
    elif isinstance(bijection, Scan):
        return bijection.inverse_gradient_and_val(draw, grad, logp)
    elif isinstance(bijection, bijections.Sandwich):
        draw, grad, logp = inverse_gradient_and_val(
            bijections.Invert(bijection.outer), draw, grad, logp
        )
        draw, grad, logp = inverse_gradient_and_val(bijection.inner, draw, grad, logp)
        draw, grad, logp = inverse_gradient_and_val(bijection.outer, draw, grad, logp)
        return draw, grad, logp
    # Disabeling the Coupling case for now, it slows down compile time for some reason?
    elif False and isinstance(bijection, Coupling):  # noqa: SIM223
        y, y_grad, y_logp = draw, grad, logp
        y_cond, y_trans = (
            y[: bijection.untransformed_dim],
            y[bijection.untransformed_dim :],
        )
        x_cond = y_cond

        y_grad_cond, y_grad_trans = (
            y_grad[: bijection.untransformed_dim],
            y_grad[bijection.untransformed_dim :],
        )

        def conditioner(x_cond):
            return bijection.conditioner(x_cond)

        transformer_params, nn_pull = jax.vjp(conditioner, x_cond)

        def pull_transformer_grad(transformer_params):
            transformer = bijection._flat_params_to_transformer(transformer_params)

            x_trans, x_grad_trans, x_logp = inverse_gradient_and_val(
                transformer, y_trans, y_grad_trans, y_logp
            )

            return (x_logp, x_trans), x_grad_trans

        ((x_logp, x_trans), pull_pull_transformer_grad, x_grad_trans) = jax.vjp(
            pull_transformer_grad, transformer_params, has_aux=True
        )

        (co_transformer_params,) = pull_pull_transformer_grad((1.0, -x_grad_trans))
        (co_x_cond,) = nn_pull(co_transformer_params)

        x = jnp.hstack((x_cond, x_trans))
        x_grad = jnp.hstack((y_grad_cond + co_x_cond, x_grad_trans))
        return x, x_grad, x_logp

    elif isinstance(bijection, bijections.Invert):
        inner = bijection.bijection
        x, _ = inner.transform_and_log_det(draw)
        (_, fwd_log_det), pull_grad_fn = jax.vjp(
            lambda x: inner.inverse_and_log_det(x), x
        )
        (x_grad,) = pull_grad_fn((grad, jnp.ones(())))
        return (x, x_grad, logp + fwd_log_det)
    else:
        x, _ = bijection.inverse_and_log_det(draw)
        (_, fwd_log_det), pull_grad_fn = jax.vjp(
            lambda x: bijection.transform_and_log_det(x), x
        )
        (x_grad,) = pull_grad_fn((grad, jnp.ones(())))
        return (x, x_grad, logp + fwd_log_det)


def _huberise(residuals, delta):
    """Rescale each draw's residual vector so its squared norm is Huber's.

    The Fisher divergence is a sum of squares, so a draw whose whitened
    residual norm is ``s`` contributes ``s**2`` -- one draw at ``s = 1e6``
    outweighs a million draws at ``s = 1``, and the fit ends up describing the
    outliers rather than the posterior. This rescales each draw's residual
    vector by ``sqrt(2 * huber(s)) / s``, so its squared norm becomes exactly
    ``2 * huber(s)``: unchanged below ``delta``, and growing linearly in ``s``
    rather than quadratically above it.

    Differentiating through the rescaled vector gives the *exact* Huber
    gradient -- from ``||r_tilde||**2 == 2 huber(s)`` it follows that
    ``r_tilde . dr_tilde/ds == huber'(s)`` -- while the Gauss-Newton model
    built from the rescaled Jacobian is the usual IRLS approximation, which is
    what LM wants anyway.

    Robustness is per *draw*, not per coordinate, because that is the failure
    mode: a draw deep in a funnel neck has a large residual in many
    coordinates at once. Note the cost -- those draws are exactly the hard
    region of the posterior, so a ``delta`` set too low buys a well-behaved
    fit that ignores the part of the space the sampler most needs help with.
    """
    square_norm = jnp.sum(residuals**2, axis=-1, keepdims=True)
    # Clamped from below so the `otherwise` branch stays finite (and carries
    # zero gradient) wherever `where` discards it; an unclamped sqrt at
    # ``s = 0`` would put a NaN into the cotangent regardless of the branch.
    norm = jnp.sqrt(jnp.maximum(square_norm, delta**2))
    scale = jnp.where(
        square_norm <= delta**2,
        1.0,
        jnp.sqrt(2.0 * delta * norm - delta**2) / norm,
    )
    return residuals * scale


class FisherLoss(eqx.Module):
    """Fisher-divergence training loss.

    The returned value is always the raw Fisher divergence (previously
    ``log(fisher_divergence)``), so it is directly comparable across calls
    and usable for thresholds like ``stop_value``. Internally, gradients are
    computed against the raw value divided by ``target_norm`` (a
    straight-through estimator, via ``jax.lax.stop_gradient``), purely to
    keep gradient magnitudes well-scaled and avoid blowups; this does not
    change the reported loss value.

    ``target_norm`` is expected to hold an exponential moving average of
    Fisher divergence values from previous windows, updated externally (see
    ``TransformAdapter``). It is a genuine pytree leaf (not a plain Python
    attribute) so that updating it does not trigger recompilation of jitted
    training steps that close over this loss.

    ``residual_batch_size`` chunks `residuals` over draws: it is the memory
    knob of the LM path's residual evaluation, and it *multiplies* with
    `lmopt.build_blocks`' own probe batching, since each concurrent probe
    carries a full reverse pass through this function. ``None`` restores the
    unchunked `jax.vmap`, which is fastest and uses the most memory. It is a
    static field, so changing it triggers a recompile.

    ``huber_delta`` optionally robustifies `residuals` against draws that
    dominate the fit, see `_huberise`. ``__call__`` deliberately keeps
    reporting the *raw* divergence either way, so numbers stay comparable
    across windows and across the setting; only what LM minimises changes.

    ``fisher_regularization`` (``lambda``) adds, for a triangular flow, the
    residuals ``sqrt(lambda) d/dy_j log q(y_i | y_pa)`` for every draw,
    variable and parent (`SparseTriangularMap.parent_scores`, in the map's
    standardized coordinates): an empirical penalty on how fast each
    conditional changes with its parents, in the Fisher metric (option B of
    `notes/flow_fisher_regularizer.md`). Like the Huber rescaling it only
    changes what LM minimises, ``__call__`` still reports the divergence.
    """

    gamma: float | None = eqx.field(static=True, default=None)
    log_inside_batch: bool = eqx.field(static=True, default=False)
    target_norm: jax.Array = eqx.field(converter=jnp.asarray, default=1.0)
    residual_batch_size: int | None = eqx.field(static=True, default=256)
    huber_delta: float | None = eqx.field(static=True, default=None)
    # See `SparseTriangularMap.gauss_newton_factors`
    cholesky_jitter: float | None = eqx.field(static=True, default=None)
    fisher_regularization: float | None = eqx.field(static=True, default=None)

    @eqx.filter_jit
    def __call__(
        self,
        params,
        static,
        draws,
        grads,
        logps,
        condition=None,
        key=None,
        return_all_costs=False,
        return_elemwise_costs=False,
    ):
        flow = unwrap(eqx.combine(params, static, is_leaf=eqx.is_inexact_array))

        if return_elemwise_costs:

            def compute_loss(bijection, draw, grad, logp):
                draw, grad, logp = inverse_gradient_and_val(bijection, draw, grad, logp)
                cost = (draw + grad) ** 2
                return cost

            costs = jax.vmap(compute_loss, [None, 0, 0, 0])(
                flow.bijection,
                draws,
                grads,
                logps,
            )
            return costs.mean(0)

        if self.gamma is None:

            def compute_loss(bijection, draw, grad, logp):
                draw, grad, logp = inverse_gradient_and_val(bijection, draw, grad, logp)
                cost = ((draw + grad) ** 2).sum()
                return cost

            costs = jax.vmap(compute_loss, [None, 0, 0, 0])(
                flow.bijection,
                draws,
                grads,
                logps,
            )

            if return_all_costs:
                return costs

            if self.log_inside_batch:
                raw = costs.mean()
                normalized = (costs / self.target_norm).mean()
            else:
                raw = costs.mean()
                normalized = raw / self.target_norm

            # stick the landing
            if False:
                flow = unwrap(eqx.combine(params, static, is_leaf=eqx.is_inexact_array))

                def compute_residual(bijection, draw, grad, logp):
                    draw, grad, logp = inverse_gradient_and_val(
                        bijection, draw, grad, logp
                    )
                    return draw, grad

                draws, grads = jax.vmap(compute_residual, [None, 0, 0, 0])(
                    flow.bijection, draws, grads, logps
                )

                resid = jax.lax.stop_gradient(draws + grads)
                return (resid * draws).sum()

            return jnp.log(raw)

            return normalized + jax.lax.stop_gradient(raw - normalized)

        else:

            def transform(draw, grad, logp):
                return inverse_gradient_and_val(flow.bijection, draw, grad, logp)

            draws, grads, logps = jax.vmap(transform, [0, 0, 0], (0, 0, 0))(
                draws, grads, logps
            )
            fisher_loss = ((draws + grads) ** 2).sum(1).mean(0)
            normal_logps = -(draws * draws).sum(1) / 2
            var_loss = (logps - normal_logps).var()
            raw = fisher_loss + self.gamma * var_loss
            normalized = raw / self.target_norm
            return normalized + jax.lax.stop_gradient(raw - normalized)

    def residuals(self, params, static, draws, grads, logps, condition=None, key=None):
        """Per-draw, per-dimension residuals whose sum of squares (divided by
        ``target_norm``) equals the ``gamma=None`` loss from ``__call__``.

        Used by the Levenberg-Marquardt fitting method (``optimistix.least_squares``
        needs an actual residual vector, not just a scalar loss, to form its
        Gauss-Newton steps).
        """
        if self.gamma is not None:
            raise ValueError(
                "FisherLoss.residuals is only defined when gamma is None, since "
                "the variance term is not expressible as a sum of squared residuals."
            )

        flow = unwrap(eqx.combine(params, static, is_leaf=eqx.is_inexact_array))

        def compute_residual(draw_grad_logp):
            draw, grad, logp = draw_grad_logp
            if self.fisher_regularization is None:
                x, grad_x, _ = inverse_gradient_and_val(
                    flow.bijection, draw, grad, logp
                )
                return x + grad_x
            # One pass for both. The Fisher part stays in the map's order,
            # without the final permutation, which doesn't change its norm.
            tmap, y, grad_y = _triangular_map_input(flow, draw, grad, logp)
            x, grad_x, _, scores = tmap.inverse_gradient_and_val(
                y, grad_y, logp, parent_scores=True
            )
            scores = jnp.concatenate([s.ravel() for s in scores])
            return jnp.concatenate(
                [x + grad_x, jnp.sqrt(self.fisher_regularization) * scores]
            )

        # `jax.vjp(res_fn, theta)` in `lmopt.step` otherwise saves every draw's
        # intermediates, and `inverse_gradient_and_val` nests autodiff (an inner
        # `value_and_grad` w.r.t. the draw inside the outer one w.r.t. the
        # parameters), so the tape holds the inner forward *and* backward. That
        # is `O(n_draws * n_dim * c)` and -- unlike peak working memory -- it
        # does not shrink with `residual_batch_size`, because reverse-mode
        # through `lax.map` stacks residuals across all chunks. Remat trades it
        # for one extra forward per draw.
        #
        # Note the cost here is not the usual ~1.3x: `vjpf` is built once and
        # applied `m` times in `build_blocks` plus once per CG iteration, and
        # each application re-runs the checkpointed forward rather than reusing
        # a saved tape.
        compute = (
            jax.checkpoint(compute_residual)
            if CHECKPOINT_RESIDUAL
            else compute_residual
        )

        if self.residual_batch_size is None:
            residuals = jax.vmap(compute)((draws, grads, logps))
        else:
            residuals = jax.lax.map(
                compute,
                (draws, grads, logps),
                batch_size=self.residual_batch_size,
            )
        n_draws = draws.shape[0]
        return residuals / jnp.sqrt(n_draws)

    def gauss_newton_factors(self, params, static, draw, grad, logp):
        """One draw's factors of the exact Gauss-Newton blocks of `residuals`,
        per conditioner bucket of the flow's `SparseTriangularMap` (see its
        `gauss_newton_factors`). Unscaled: `residuals` divides by
        ``sqrt(n_draws)``, which the caller applies to the summed blocks.

        Only the triangular map's conditioners get blocks. Everything else in
        the flow must be frozen: the diagonal affine before the map just
        changes the data it sees, and the permutation after it is orthogonal,
        so neither changes the blocks. This mirrors `inverse_gradient_and_val`
        on the flow `make_flow(kind="triangular")` builds.
        """
        flow = unwrap(eqx.combine(params, static, is_leaf=eqx.is_inexact_array))
        tmap, draw, grad = _triangular_map_input(flow, draw, grad, logp)
        return tmap.gauss_newton_factors(
            draw,
            grad,
            cholesky_jitter=self.cholesky_jitter,
            fisher_regularization=self.fisher_regularization,
        )


def _triangular_map_input(flow, draw, grad, logp):
    """``(tmap, y, grad_y)``: the `SparseTriangularMap` of a flow
    `make_flow(kind="triangular")` builds, and a draw and its gradient as
    that map sees them (after the diagonal affine and the permutation)."""
    sandwich = flow.bijection.bijections[0].bijections[0]
    affine = flow.bijection.bijections[1]
    draw, grad, _ = inverse_gradient_and_val(affine, draw, grad, logp)
    draw, grad, _ = inverse_gradient_and_val(
        bijections.Invert(sandwich.outer), draw, grad, logp
    )
    return sandwich.inner, draw, grad


def _describe_flow(bijection):
    """Size and structure of a flow, for the verbose output."""
    from nutpie.triangular import SparseTriangularMap

    bijection = unwrap(bijection)
    arrays = jax.tree.leaves(eqx.filter(bijection, eqx.is_inexact_array))
    text = f"{bijection.shape[0]} dimensions, {sum(a.size for a in arrays)} parameters"
    maps = [
        leaf
        for leaf in jax.tree.leaves(
            bijection, is_leaf=lambda x: isinstance(x, SparseTriangularMap)
        )
        if isinstance(leaf, SparseTriangularMap)
    ]
    for tmap in maps:
        dim = tmap.shape[0]
        # Padded parent slots read the dummy index `dim`
        max_parents = max(
            (
                int((np.asarray(parents) < dim).sum(axis=1).max(initial=0))
                for parents in tmap.bucket_parent_indices
            ),
            default=0,
        )
        text += f", at most {max_parents} parents, {tmap.n_levels} levels"
    return text


def _format_log_f(value):
    value = float(value)
    return f"{value:+.2f}" if np.isfinite(value) else "  nan"


def fit_flow(key, bijection, loss_fn, draws, grads, logps, **kwargs):
    flow = flowjax.flows.Transformed(
        flowjax.distributions.StandardNormal(bijection.shape), bijection
    )

    key, train_key = jax.random.split(key)

    fit, losses, opt_state = fit_to_data(
        key=train_key,
        dist=flow,
        x=(draws, grads, logps),
        loss_fn=loss_fn,
        return_best=True,
        stop_value=_LOG_STOP_VALUE,
        **kwargs,
    )
    return fit.bijection, losses, opt_state


@eqx.filter_jit
def _init_from_transformed_position(logp_fn, bijection, transformed_position, clip):
    bijection = unwrap(bijection)
    (untransformed_position, logdet), pull_grad = jax.vjp(
        bijection.transform_and_log_det, transformed_position
    )
    logp, untransformed_gradient = jax.value_and_grad(lambda x: logp_fn(x)[0])(
        untransformed_position
    )

    if clip is not None:
        untransformed_gradient = clip * jnp.arcsinh(untransformed_gradient / clip)

    (transformed_gradient,) = pull_grad((untransformed_gradient, 1.0))
    return (
        logp,
        logdet,
        untransformed_position,
        untransformed_gradient,
        transformed_gradient,
    )


@eqx.filter_jit
def _init_from_transformed_position_part1(logp_fn, bijection, transformed_position):
    bijection = unwrap(bijection)
    (untransformed_position, logdet) = bijection.transform_and_log_det(
        transformed_position
    )

    return (logdet, untransformed_position)


@eqx.filter_jit
def _init_from_transformed_position_part2(
    bijection,
    part1,
    untransformed_gradient,
):
    logdet, _untransformed_position, transformed_position = part1
    bijection = unwrap(bijection)
    _, pull_grad = jax.vjp(bijection.transform_and_log_det, transformed_position)
    (transformed_gradient,) = pull_grad((untransformed_gradient, 1.0))
    return (
        logdet,
        transformed_gradient,
    )


@eqx.filter_jit
def _init_from_untransformed_position(logp_fn, bijection, untransformed_position, clip):
    logp, untransformed_gradient = jax.value_and_grad(lambda x: logp_fn(x)[0])(
        untransformed_position
    )
    if clip is not None:
        untransformed_gradient = clip * jnp.arcsinh(untransformed_gradient / clip)

    logdet, transformed_position, transformed_gradient = _inv_transform(
        bijection, untransformed_position, untransformed_gradient
    )
    return (
        logp,
        logdet,
        untransformed_gradient,
        transformed_position,
        transformed_gradient,
    )


@eqx.filter_jit
def _inv_transform(bijection, untransformed_position, untransformed_gradient):
    bijection = unwrap(bijection)
    transformed_position, transformed_gradient, logdet = inverse_gradient_and_val(
        bijection, untransformed_position, untransformed_gradient, 0.0
    )
    return logdet, transformed_position, transformed_gradient


class TransformAdapter:
    def __init__(
        self,
        seed,
        position,
        gradient,
        chain,
        *,
        logp_fn,
        make_flow_fn,
        verbose=False,
        window_size=2000,
        show_progress=False,
        num_diag_windows=10,
        learning_rate=1e-3,
        zero_init=True,
        untransformed_dim=None,
        batch_size=128,
        reuse_opt_state=True,
        max_patience=5,
        gamma=None,
        log_inside_batch=False,
        fisher_ema_alpha=0.1,
        initial_skip=500,
        extension_windows=None,
        extend_dct=False,
        extension_var_count=6,
        extension_var_trafo_count=4,
        debug_save_bijection=False,
        make_optimizer=None,
        num_layers=9,
        max_epochs=200,
        method="adam",
        solver_rtol=1e-3,
        solver_atol=1e-6,
        lm_linear_steps=300,
        lm_min_loss=math.exp(-3),
        lm_probe_batch=32,
        lm_probes=64,
        lm_residual_batch=256,
        lm_cholesky_jitter=None,
        lm_fisher_regularization=None,
        lm_probe_groups=None,
        lm_probe_rounds=1,
        lm_fit_affine=True,
        lm_patience=5,
        lm_line_search=False,
        lm_forcing="residual",
        lm_exact_blocks=False,
        lm_max_exact_block_size=256,
        lm_lmp_size=0,
        lm_lmp_tol=1e-8,
        native_flow=True,
        stop_event=None,
    ):
        self._logp_fn = logp_fn
        self._make_flow_fn = make_flow_fn
        self._chain = chain
        # 0: silent, 1: one line per window, 2: also the LM iterations
        self._verbose = int(verbose)
        self._printed_lm_blocks = False
        self._window_size = window_size
        self._initial_skip = initial_skip
        self._num_layers = num_layers
        if make_optimizer is None:
            self._make_optimizer = lambda: optax.apply_if_finite(
                optax.adamw(learning_rate), 10
            )
        else:
            self._make_optimizer = make_optimizer
        self._optimizer = self._make_optimizer()
        self._loss_fn = FisherLoss(
            gamma,
            log_inside_batch,
            residual_batch_size=lm_residual_batch,
            huber_delta=None,
            cholesky_jitter=lm_cholesky_jitter,
            fisher_regularization=lm_fisher_regularization,
        )
        self._fisher_ema = None
        self._fisher_ema_alpha = fisher_ema_alpha
        self._show_progress = show_progress
        self._num_diag_windows = num_diag_windows
        self._zero_init = zero_init
        self._untransformed_dim = untransformed_dim
        self._batch_size = batch_size
        self._reuse_opt_state = reuse_opt_state
        self._opt_state = None
        self._max_patience = max_patience
        self._count_trace = []
        self._last_extend_dct = True
        self._extend_dct = extend_dct
        self._extension_var_count = extension_var_count
        self._extension_var_trafo_count = extension_var_trafo_count
        self._debug_save_bijection = debug_save_bijection
        self._layers = 0
        self._max_epochs = max_epochs
        self._method = method
        self._solver_rtol = solver_rtol
        self._solver_atol = solver_atol
        self._lm_linear_steps = lm_linear_steps
        self._lm_min_loss = lm_min_loss
        self._lm_probe_batch = lm_probe_batch
        self._lm_probes = lm_probes
        self._lm_probe_groups = lm_probe_groups
        self._lm_probe_rounds = lm_probe_rounds
        self._lm_fit_affine = lm_fit_affine
        self._lm_patience = lm_patience
        self._lm_line_search = lm_line_search
        self._lm_forcing = lm_forcing
        self._lm_exact_blocks = lm_exact_blocks
        self._lm_max_exact_block_size = lm_max_exact_block_size
        self._lm_lmp_size = lm_lmp_size
        self._lm_lmp_tol = lm_lmp_tol
        # Whether the sampler may run the flow natively in its leapfrog steps,
        # see `flow_transform_layout`.
        self._native_flow = native_flow
        # Set by the main thread when sampling is aborted. The training runs
        # in a chain's thread, where Python never raises KeyboardInterrupt.
        self._stop_event = stop_event
        # Damping the previous LM fit ended with, to start the next one from:
        # consecutive windows fit nearly the same problem, and restarting from
        # `lam0` makes each fit rediscover the scale, typically overshooting on
        # its way down.
        self._lm_lam = None

        if extension_windows is None:
            self._extension_windows = []
        else:
            self._extension_windows = extension_windows

        try:
            self._bijection = make_flow_fn(seed, [position], [gradient], n_layers=0)
        except Exception as e:
            print("make_flow", e)
            print(traceback.format_exc())
            raise
        self.index = 0

    @property
    def transformation_id(self):
        return self.index

    def _sync_loss_target_norm(self):
        """Point ``self._loss_fn.target_norm`` at the EMA of past windows'
        Fisher divergence (or 1.0 before any window has been measured).

        This only affects the internal gradient scaling used while fitting
        the current window; ``self._loss_fn`` still always *reports* the raw
        Fisher divergence.
        """
        target_norm = 1.0 if self._fisher_ema is None else self._fisher_ema
        self._loss_fn = eqx.tree_at(
            lambda loss: loss.target_norm, self._loss_fn, jnp.asarray(target_norm)
        )

    def _record_fisher_divergence(self, raw_loss):
        """Fold a newly observed raw Fisher divergence into the EMA used to
        normalize gradients in later windows."""
        raw_loss = float(raw_loss)
        if not np.isfinite(raw_loss):
            return
        raw_loss = max(raw_loss, 1e-12)
        if self._fisher_ema is None:
            self._fisher_ema = raw_loss
        else:
            self._fisher_ema = (
                self._fisher_ema_alpha * raw_loss
                + (1 - self._fisher_ema_alpha) * self._fisher_ema
            )

    def _report(self, n_draws, message):
        """One line about the current window, for ``verbose >= 1``."""
        if self._verbose:
            print(
                f"flow chain {self._chain} window {self.index:3d} "
                f"({n_draws:5d} draws): {message}"
            )

    def _should_stop(self):
        return self._stop_event is not None and self._stop_event.is_set()

    def update(self, seed, positions, gradients, logps):
        self.index += 1
        n_draws = len(positions)
        if self._should_stop():
            return
        assert n_draws == len(positions)
        assert n_draws == len(gradients)
        assert n_draws == len(logps)
        self._count_trace.append(n_draws)
        if n_draws == 0:
            return
        try:
            if self.index <= self._num_diag_windows:
                positions = np.asarray(positions)
                gradients = np.asarray(gradients)
                logps = np.asarray(logps)

                # The newest `size // 5 + 3` distinct draws. Repeats (rejected
                # transitions) say nothing new about the score, and a fit to
                # a few distinct points matches them exactly.
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
                n_repeats = size - start - len(keep)
                repeats = f", {n_repeats} repeats skipped" if n_repeats else ""
                if len(keep) < _MIN_DIAG_DRAWS:
                    self._report(
                        n_draws,
                        f"keep diag, only {len(keep)} distinct draws{repeats}",
                    )
                    return

                keep = keep[::-1]
                positions = positions[keep]
                gradients = gradients[keep]
                logps = logps[keep]

                fit = self._make_flow_fn(seed, positions, gradients, n_layers=0)

                flow = flowjax.flows.Transformed(
                    flowjax.distributions.StandardNormal(fit.shape), fit
                )
                params, static = eqx.partition(flow, eqx.is_inexact_array)
                self._sync_loss_target_norm()
                new_loss = self._loss_fn(params, static, positions, gradients, logps)
                self._record_fisher_divergence(new_loss)

                self._report(
                    n_draws,
                    f"log F  diag {_format_log_f(new_loss)}  "
                    f"({len(keep)} draws{repeats})",
                )

                if np.isfinite(new_loss):
                    self._bijection = fit
                    self._opt_state = None

                return

            hist_positions = positions[self._initial_skip :]
            hist_gradients = gradients[self._initial_skip :]
            hist_logps = logps[self._initial_skip :]

            total_hist_len = len(hist_positions)
            if total_hist_len < 10:
                self._report(
                    n_draws,
                    f"keep flow, waiting for draws after the first {self._initial_skip}",
                )
                return

            # Number of draws that arrived since the previous update() call.
            # (The cadence itself is controlled on the Rust side via
            # transform_update_freq, not by window_size.)
            if len(self._count_trace) >= 2:
                stride = self._count_trace[-1] - self._count_trace[-2]
            else:
                stride = self._count_trace[-1]
            new_part_size = min(max(stride, 1), total_hist_len)

            window = self._window_size
            tail_len = min(total_hist_len, new_part_size + 3 * window)
            tail_positions = np.array(hist_positions[-tail_len:])
            tail_gradients = np.array(hist_gradients[-tail_len:])
            tail_logps = np.array(hist_logps[-tail_len:])

            new_positions = tail_positions[-new_part_size:]
            new_gradients = tail_gradients[-new_part_size:]
            new_logps = tail_logps[-new_part_size:]

            history_positions = tail_positions[:-new_part_size]
            history_gradients = tail_gradients[:-new_part_size]
            history_logps = tail_logps[:-new_part_size]
            history_len = len(history_positions)

            rng = np.random.default_rng(seed)

            if history_len == 0:
                pool_positions = new_positions
                pool_gradients = new_gradients
                pool_logps = new_logps
            else:
                # A random subset of the last three windows, equally sized
                # to the new draws.
                replace = history_len < new_part_size
                idx = rng.choice(history_len, size=new_part_size, replace=replace)
                pool_positions = np.concatenate([new_positions, history_positions[idx]])
                pool_gradients = np.concatenate([new_gradients, history_gradients[idx]])
                pool_logps = np.concatenate([new_logps, history_logps[idx]])

            # Final subsample (with replacement if the pool is smaller than
            # the window) down to the configured window size.
            pool_len = len(pool_positions)
            replace = pool_len < window
            final_idx = rng.choice(pool_len, size=window, replace=replace)
            positions = pool_positions[final_idx]
            gradients = pool_gradients[final_idx]
            logps = pool_logps[final_idx]

            if len(positions) < 10:
                return

            if self._verbose >= 2 and not np.isfinite(gradients).all():
                print(gradients)
                print(gradients.shape)
                print((~np.isfinite(gradients)).nonzero())

            assert np.isfinite(positions).all()
            assert np.isfinite(gradients).all()
            assert np.isfinite(logps).all()

            # TODO don't reuse seed
            key = jax.random.PRNGKey(seed % (2**63))

            if len(self._bijection.bijections) == 1:
                base = self._make_flow_fn(
                    seed,
                    positions,
                    gradients,
                    n_layers=self._num_layers,
                    untransformed_dim=self._untransformed_dim,
                    zero_init=self._zero_init,
                )
                flow = flowjax.flows.Transformed(
                    flowjax.distributions.StandardNormal(base.shape), base
                )
                params, static = eqx.partition(flow, eqx.is_inexact_array)
                self._report(n_draws, f"new flow: {_describe_flow(base)}")
                if self._verbose >= 2:
                    fresh_loss = self._loss_fn(
                        params,
                        static,
                        positions[-128:],
                        gradients[-128:],
                        logps[-128:],
                    )
                    self._report(
                        n_draws, f"log F  fresh flow {_format_log_f(fresh_loss)}"
                    )
            else:
                base = self._bijection

            if self.index in self._extension_windows:
                self._report(n_draws, "extending the flow")
                self._last_extend_dct = not self._last_extend_dct
                dct = self._last_extend_dct and self._extend_dct
                base = extend_flow(
                    key,
                    base,
                    self._loss_fn,
                    positions,
                    gradients,
                    logps,
                    self._layers,
                    dct=dct,
                    extension_var_count=self._extension_var_count,
                    extension_var_trafo_count=self._extension_var_trafo_count,
                    verbose=self._verbose,
                )
                self._optimizer = self._make_optimizer()
                self._opt_state = None
                self._layers += 1

            # make_flow might still onreturn a single trafo for 1d problems
            if len(base.bijections) == 1:
                self._bijection = base
                self._opt_state = None

                if self._debug_save_bijection:
                    _BIJECTION_TRACE.append(
                        (self.index, base, (positions, gradients, logps))
                    )
                return

            flow = flowjax.flows.Transformed(
                flowjax.distributions.StandardNormal(self._bijection.shape),
                self._bijection,
            )
            params, static = eqx.partition(flow, eqx.is_inexact_array)

            self._sync_loss_target_norm()
            old_loss = self._loss_fn(
                params, static, positions[-128:], gradients[-128:], logps[-128:]
            )
            self._record_fisher_divergence(old_loss)

            skip_training = old_loss < _LOG_SKIP_TRAINING_VALUE and self.index > 10
            if np.isfinite(old_loss) and skip_training:
                self._report(
                    n_draws,
                    f"log F  current {_format_log_f(old_loss)}  -> keep flow, "
                    f"already below {_format_log_f(_LOG_SKIP_TRAINING_VALUE)}",
                )
                return

            fit, fit_losses, opt_state = fit_flow(
                key,
                base,
                self._loss_fn,
                positions,
                gradients,
                logps,
                show_progress=self._show_progress,
                verbose=self._verbose >= 2,
                lm_print_blocks=self._verbose >= 2 and not self._printed_lm_blocks,
                optimizer=self._optimizer,
                batch_size=self._batch_size,
                opt_state=self._opt_state if self._reuse_opt_state else None,
                max_patience=self._max_patience,
                max_epochs=self._max_epochs,
                method=self._method,
                solver_rtol=self._solver_rtol,
                solver_atol=self._solver_atol,
                lm_linear_steps=self._lm_linear_steps,
                lm_min_loss=self._lm_min_loss,
                lm_probe_batch=self._lm_probe_batch,
                lm_probes=self._lm_probes,
                lm_probe_groups=self._lm_probe_groups,
                lm_probe_rounds=self._lm_probe_rounds,
                lm_fit_affine=self._lm_fit_affine,
                lm_patience=self._lm_patience,
                lm_line_search=self._lm_line_search,
                lm_forcing=self._lm_forcing,
                lm_lam0=self._lm_lam,
                lm_exact_blocks=self._lm_exact_blocks,
                lm_max_exact_block_size=self._lm_max_exact_block_size,
                lm_lmp_size=self._lm_lmp_size,
                lm_lmp_tol=self._lm_lmp_tol,
                lm_diagnose=self._verbose >= 3,
                should_stop=self._should_stop,
            )
            if self._should_stop():
                # Sampling was aborted, keep the flow as it is.
                return

            # Kept even if the fit is discarded below: the damping scale says
            # something about the problem, whether or not this fit won. But
            # not from a fit that made no progress at all, where every
            # rejected step only raised it.
            no_progress = fit_losses.get("lm_accepted") == 0
            if not no_progress:
                self._lm_lam = fit_losses.get("lm_lam", self._lm_lam)
            if self._method == "lm":
                self._printed_lm_blocks = True

            flow = flowjax.flows.Transformed(
                flowjax.distributions.StandardNormal(fit.shape), fit
            )
            params, static = eqx.partition(flow, eqx.is_inexact_array)

            new_loss = self._loss_fn(
                params, static, positions[-128:], gradients[-128:], logps[-128:]
            )

            def report(decision):
                steps = fit_losses.get("lm_steps")
                self._report(
                    n_draws,
                    f"log F  current {_format_log_f(old_loss)}  "
                    f"refit {_format_log_f(new_loss)}  -> {decision}"
                    + (
                        ""
                        if steps is None
                        else f"  ({steps} LM step{'' if steps == 1 else 's'})"
                    ),
                )

            if self._verbose >= 2 and not np.isfinite(old_loss):
                flow = flowjax.flows.Transformed(
                    flowjax.distributions.StandardNormal(self._bijection.shape),
                    self._bijection,
                )
                params, static = eqx.partition(flow, eqx.is_inexact_array)
                print(
                    self._loss_fn(
                        params,
                        static,
                        positions[-128:],
                        gradients[-128:],
                        logps[-128:],
                        return_all_costs=True,
                    )
                )

            if self._verbose >= 2 and not np.isfinite(new_loss):
                flow = flowjax.flows.Transformed(
                    flowjax.distributions.StandardNormal(fit.shape), fit
                )
                params, static = eqx.partition(flow, eqx.is_inexact_array)
                print(
                    self._loss_fn(
                        params,
                        static,
                        positions[-128:],
                        gradients[-128:],
                        logps[-128:],
                        return_all_costs=True,
                    )
                )

            if self._debug_save_bijection:
                _BIJECTION_TRACE.append(
                    (self.index, fit, (positions, gradients, logps))
                )

            def valid_new_logp():
                logdet, pos, grad = _inv_transform(
                    fit,
                    jnp.array(positions[-1]),
                    jnp.array(gradients[-1]),
                )
                return (
                    np.isfinite(logdet)
                    and np.isfinite(pos[0]).all()
                    and np.isfinite(grad[0]).all()
                )

            if no_progress:
                report("keep flow, no LM step accepted")
                return

            if (not np.isfinite(old_loss)) and (not np.isfinite(new_loss)):
                report("reset to a diagonal flow, both are invalid")
                self._bijection = self._make_flow_fn(
                    seed, positions, gradients, n_layers=0
                )
                self._opt_state = None
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
            self._bijection = fit
            self._opt_state = opt_state

        except Exception as e:
            print("update error:", e)
            print(traceback.format_exc())
            raise

    def init_from_transformed_position(self, transformed_position, clip):
        try:
            logp, logdet, *arrays = _init_from_transformed_position(
                self._logp_fn,
                self._bijection,
                jnp.array(transformed_position),
                clip,
            )
            return (
                float(logp),
                float(logdet),
                *[np.array(val, dtype="float64") for val in arrays],
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise

    def flow_transform_layout(self):
        """The current flow's layout for the sampler's native leapfrog
        transform, or `None` to keep using JAX (see
        `triangular_rust.flow_transform_layout`). The Rust side asks again
        after every `update`."""
        if not self._native_flow:
            return None
        from nutpie.triangular_rust import flow_transform_layout

        return flow_transform_layout(self._bijection)

    def init_from_transformed_position_part1(self, transformed_position):
        try:
            transformed_position = jnp.array(transformed_position)
            logdet, untransformed_position = _init_from_transformed_position_part1(
                self._logp_fn,
                self._bijection,
                transformed_position,
            )
            part1 = (logdet, untransformed_position, transformed_position)
            return np.array(untransformed_position, dtype="float64"), part1
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise

    def init_from_transformed_position_part2(
        self,
        part1,
        untransformed_gradient,
    ):
        try:
            # TODO We could extract the arrays from the pull_grad function
            # to reuse computation from part1
            logdet, *arrays = _init_from_transformed_position_part2(
                self._bijection,
                part1,
                untransformed_gradient,
            )
            return float(logdet), *[np.array(val, dtype="float64") for val in arrays]
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise

    def init_from_untransformed_position(self, untransformed_position, clip):
        try:
            logp, logdet, *arrays = _init_from_untransformed_position(
                self._logp_fn, self._bijection, jnp.array(untransformed_position), clip
            )
            arrays = [np.array(val, dtype="float64") for val in arrays]
            return float(logp), float(logdet), *arrays
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise

    def inv_transform(self, position, gradient):
        try:
            logdet, *arrays = _inv_transform(
                self._bijection, jnp.array(position), jnp.array(gradient)
            )
            return logdet, *[np.array(val, dtype="float64") for val in arrays]
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise


def make_transform_adapter(
    *,
    verbose=False,
    window_size=512,
    show_progress=False,
    nn_depth=None,
    nn_width=8,
    num_layers=8,
    num_diag_windows=9,
    learning_rate=5e-4,
    untransformed_dim=None,
    zero_init=True,
    batch_size=128,
    reuse_opt_state=False,
    max_patience=20,
    householder_layer=False,
    dct_layer=False,
    gamma=None,
    log_inside_batch=False,
    fisher_ema_alpha=0.1,
    initial_skip=120,
    extension_windows=None,
    extend_dct=False,
    extension_var_count=4,
    extension_var_trafo_count=2,
    debug_save_bijection=False,
    make_optimizer=None,
    coupling_type="triangular",
    mvscale_layer=False,
    num_project=None,
    num_embed=None,
    num_householder=8,
    twin_layers=False,
    activation=None,
    max_epochs=30,
    affine_transformer=False,
    contract_transformer=1,
    asymmetric_transformer=False,
    tangent_sas_transformer=0,
    tangent_sas_fix_b=False,
    reuse_embed=True,
    order=None,
    sparsity=None,
    location_skip=True,
    feature_degree=None,
    method="lm-rust",
    solver_rtol=5e-2,
    solver_atol=1e-6,
    lm_linear_steps=150,
    lm_min_loss=math.exp(-3),
    lm_probe_batch=32,
    lm_probes=1024,
    lm_residual_batch=128,
    lm_cholesky_jitter=None,
    lm_fisher_regularization=None,
    lm_probe_groups=None,
    lm_probe_rounds=1,
    lm_fit_affine=False,
    lm_patience=5,
    lm_line_search=True,
    lm_forcing="model",
    lm_exact_blocks=True,
    lm_max_exact_block_size=256,
    lm_lmp_size=8,
    lm_lmp_tol=1e-8,
    native_flow=True,
    stop_event=None,
):
    if extension_windows is None:
        extension_windows = []

    return partial(
        TransformAdapter,
        verbose=verbose,
        window_size=window_size,
        make_flow_fn=partial(
            make_flow,
            householder_layer=householder_layer,
            dct_layer=dct_layer,
            nn_depth=nn_depth,
            nn_width=nn_width,
            n_embed=num_project,
            n_deembed=num_embed,
            mvscale=mvscale_layer,
            kind=coupling_type,
            num_householder=num_householder,
            twin_layers=twin_layers,
            activation=activation,
            affine_transformer=affine_transformer,
            contract_transformer=contract_transformer,
            asymmetric_transformer=asymmetric_transformer,
            tangent_sas_transformer=tangent_sas_transformer,
            tangent_sas_fix_b=tangent_sas_fix_b,
            reuse_embed=reuse_embed,
            order=order,
            sparsity=sparsity,
            location_skip=location_skip,
            feature_degree=feature_degree,
        ),
        show_progress=show_progress,
        num_diag_windows=num_diag_windows,
        learning_rate=learning_rate,
        zero_init=zero_init,
        untransformed_dim=untransformed_dim,
        batch_size=batch_size,
        reuse_opt_state=reuse_opt_state,
        max_patience=max_patience,
        gamma=gamma,
        log_inside_batch=log_inside_batch,
        fisher_ema_alpha=fisher_ema_alpha,
        initial_skip=initial_skip,
        extension_windows=extension_windows,
        extend_dct=extend_dct,
        extension_var_count=extension_var_count,
        extension_var_trafo_count=extension_var_trafo_count,
        debug_save_bijection=debug_save_bijection,
        make_optimizer=make_optimizer,
        num_layers=num_layers,
        max_epochs=max_epochs,
        method=method,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        lm_linear_steps=lm_linear_steps,
        lm_min_loss=lm_min_loss,
        lm_probe_batch=lm_probe_batch,
        lm_probes=lm_probes,
        lm_residual_batch=lm_residual_batch,
        lm_cholesky_jitter=lm_cholesky_jitter,
        lm_fisher_regularization=lm_fisher_regularization,
        lm_probe_groups=lm_probe_groups,
        lm_probe_rounds=lm_probe_rounds,
        lm_fit_affine=lm_fit_affine,
        lm_patience=lm_patience,
        lm_line_search=lm_line_search,
        lm_forcing=lm_forcing,
        lm_exact_blocks=lm_exact_blocks,
        lm_max_exact_block_size=lm_max_exact_block_size,
        lm_lmp_size=lm_lmp_size,
        lm_lmp_tol=lm_lmp_tol,
        native_flow=native_flow,
        stop_event=stop_event,
    )
