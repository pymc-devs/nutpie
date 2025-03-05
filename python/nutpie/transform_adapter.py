from functools import partial
from importlib.util import find_spec
from typing import Callable
import time

if find_spec("flowjax") is None:
    raise ImportError(
        "The 'flowjax' package is required to use normalizing flow adaptation."
    )

from flowjax import bijections
from jaxtyping import ArrayLike, PyTree
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import traceback
import flowjax
import flowjax.flows
import flowjax.train
from flowjax.train.losses import MaximumLikelihoodLoss, PRNGKeyArray
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)
import optax
from paramax import unwrap, NonTrainable

from nutpie.normalizing_flow import Coupling, extend_flow, make_flow
import tqdm

_BIJECTION_TRACE = []


def fit_to_data(
    key: PRNGKeyArray,
    dist: PyTree,  # Custom losses may support broader types than AbstractDistribution
    x,
    *,
    condition: ArrayLike | None = None,
    loss_fn: Callable | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    return_best: bool = True,
    show_progress: bool = True,
    opt_state=None,
    verbose: bool = False,
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
        max_epochs: Maximum number of epochs. Defaults to 100.
        max_patience: Number of consecutive epochs with no validation loss improvement
            after which training is terminated. Defaults to 5.
        batch_size: Batch size. Defaults to 100.
        val_prop: Proportion of data to use in validation set. Defaults to 0.1.
        learning_rate: Adam learning rate. Defaults to 5e-4.
        optimizer: Optax optimizer. If provided, this overrides the default Adam
            optimizer, and the learning_rate is ignored. Defaults to None.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing the trained distribution and the losses.
    """
    if not isinstance(x, tuple):
        x = (x,)
    data = x if condition is None else (*x, condition)
    data = tuple(jnp.asarray(a) for a in data)

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    if loss_fn is None:
        loss_fn = MaximumLikelihoodLoss()

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )
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
        start = time.time()
        key, *subkeys = jr.split(key, 3)
        train_data = [jr.permutation(subkeys[0], a) for a in train_data]
        val_data = [jr.permutation(subkeys[1], a) for a in val_data]
        if verbose and i == 0:
            print("shuffle timing:", time.time() - start)

        start = time.time()

        key, subkey = jr.split(key)
        batches = get_batches(train_data, batch_size)
        batch_losses = []

        if verbose and i == 0:
            print("batch timing:", time.time() - start)

        start = time.time()

        if True:
            for batch in zip(*batches, strict=True):
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

        losses["train"].append((sum(batch_losses) / len(batch_losses)).item())

        if verbose and i == 0:
            print("step timing:", time.time() - start)

        start = time.time()
        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            loss_i = loss_fn(params, static, *batch, key=subkey)
            batch_losses.append(loss_i)
        losses["val"].append(sum(batch_losses) / len(batch_losses))

        if verbose and i == 0:
            print("val timing:", time.time() - start)

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    dist = eqx.combine(params, static)
    return dist, losses, opt_state


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
def inverse_gradient_and_val(bijection, draw, grad, logp):
    if False:
        x = bijection.inverse(draw)
        (_, fwd_log_det), pull_grad_fn = jax.vjp(
            lambda x: bijection.transform_and_log_det(x), x
        )
        (x_grad,) = pull_grad_fn((grad, jnp.ones(())))
        return (x, x_grad, logp + fwd_log_det)
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
        grad = grad * bijection.scale
        return (draw, grad, logp - logdet)
    elif isinstance(bijection, bijections.Vmap):

        def inner(bijection, y, y_grad, y_logp):
            return inverse_gradient_and_val(bijection, y, y_grad, y_logp)

        y, y_grad, log_det = eqx.filter_vmap(
            inner,
            in_axes=(bijection.in_axes[0], 0, 0, None),
            axis_size=bijection.axis_size,
        )(bijection.bijection, draw, grad, jnp.zeros(()))
        return y, y_grad, jnp.sum(log_det) + logp
    elif isinstance(bijection, bijections.Sandwich):
        draw, grad, logp = inverse_gradient_and_val(
            bijections.Invert(bijection.outer), draw, grad, logp
        )
        draw, grad, logp = inverse_gradient_and_val(bijection.inner, draw, grad, logp)
        draw, grad, logp = inverse_gradient_and_val(bijection.outer, draw, grad, logp)
        return draw, grad, logp
    # Disabeling the Coupling case for now, it slows down compile time for some reason?
    elif False and isinstance(bijection, Coupling):
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


class FisherLoss:
    def __init__(self, gamma=None, log_inside_batch=False):
        self._gamma = gamma
        self._log_inside_batch = log_inside_batch

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

        if self._gamma is None:

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

            if self._log_inside_batch:
                return jnp.log(costs).mean()
            else:
                return jnp.log(costs.mean())

        else:

            def transform(draw, grad, logp):
                return inverse_gradient_and_val(flow.bijection, draw, grad, logp)

            draws, grads, logps = jax.vmap(transform, [0, 0, 0], (0, 0, 0))(
                draws, grads, logps
            )
            fisher_loss = ((draws + grads) ** 2).sum(1).mean(0)
            normal_logps = -(draws * draws).sum(1) / 2
            var_loss = (logps - normal_logps).var()
            return jnp.log(fisher_loss + self._gamma * var_loss)


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
        max_epochs=500,
        return_best=True,
        **kwargs,
    )
    return fit.bijection, losses, opt_state


@eqx.filter_jit
def _init_from_transformed_position(logp_fn, bijection, transformed_position):
    bijection = unwrap(bijection)
    (untransformed_position, logdet), pull_grad = jax.vjp(
        bijection.transform_and_log_det, transformed_position
    )
    logp, untransformed_gradient = jax.value_and_grad(lambda x: logp_fn(x)[0])(
        untransformed_position
    )
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
    logdet, untransformed_position, transformed_position = part1
    bijection = unwrap(bijection)
    _, pull_grad = jax.vjp(bijection.transform_and_log_det, transformed_position)
    (transformed_gradient,) = pull_grad((untransformed_gradient, 1.0))
    return (
        logdet,
        transformed_gradient,
    )


@eqx.filter_jit
def _init_from_untransformed_position(logp_fn, bijection, untransformed_position):
    logp, untransformed_gradient = jax.value_and_grad(lambda x: logp_fn(x)[0])(
        untransformed_position
    )
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
        initial_skip=500,
        extension_windows=None,
        extend_dct=False,
        extension_var_count=6,
        extension_var_trafo_count=4,
        debug_save_bijection=False,
        make_optimizer=None,
        num_layers=9,
    ):
        self._logp_fn = logp_fn
        self._make_flow_fn = make_flow_fn
        self._chain = chain
        self._verbose = verbose
        self._window_size = window_size
        self._initial_skip = initial_skip
        self._num_layers = num_layers
        if make_optimizer is None:
            self._make_optimizer = lambda: optax.apply_if_finite(
                optax.adamw(learning_rate), 50
            )
        else:
            self._make_optimizer = make_optimizer
        self._optimizer = self._make_optimizer()
        self._loss_fn = FisherLoss(gamma, log_inside_batch)
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

    def update(self, seed, positions, gradients, logps):
        self.index += 1
        if self._verbose:
            print(
                f"Chain {self._chain}: Total available points: {len(positions)}, seed {seed}"
            )
        n_draws = len(positions)
        assert n_draws == len(positions)
        assert n_draws == len(gradients)
        assert n_draws == len(logps)
        self._count_trace.append(n_draws)
        if n_draws == 0:
            return
        try:
            if self.index <= self._num_diag_windows:
                size = len(positions)
                lower_idx = -size // 5 + 3
                positions_slice = positions[lower_idx:]
                gradients_slice = gradients[lower_idx:]
                logp_slice = logps[lower_idx:]

                if len(positions_slice) > 0:
                    positions = positions_slice
                    gradients = gradients_slice
                    logps = logp_slice

                positions = np.array(positions)
                gradients = np.array(gradients)
                logps = np.array(logps)

                fit = self._make_flow_fn(seed, positions, gradients, n_layers=0)

                flow = flowjax.flows.Transformed(
                    flowjax.distributions.StandardNormal(fit.shape), fit
                )
                params, static = eqx.partition(flow, eqx.is_inexact_array)
                new_loss = self._loss_fn(params, static, positions, gradients, logps)

                if self._verbose:
                    print("loss from diag:", new_loss)

                if np.isfinite(new_loss):
                    self._bijection = fit
                    self._opt_state = None

                return

            positions = np.array(positions[self._initial_skip :][-self._window_size :])
            gradients = np.array(gradients[self._initial_skip :][-self._window_size :])
            logps = np.array(logps[self._initial_skip :][-self._window_size :])

            if len(positions) < 10:
                return

            if self._verbose and not np.isfinite(gradients).all():
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
                if self._verbose:
                    print(
                        "loss before optimization: ",
                        self._loss_fn(
                            params,
                            static,
                            positions[-128:],
                            gradients[-128:],
                            logps[-128:],
                        ),
                    )
            else:
                base = self._bijection

            if self.index in self._extension_windows:
                if self._verbose:
                    print("Extending flow...")
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
                        (self.index, fit, (positions, gradients, logps))
                    )
                return

            flow = flowjax.flows.Transformed(
                flowjax.distributions.StandardNormal(self._bijection.shape),
                self._bijection,
            )
            params, static = eqx.partition(flow, eqx.is_inexact_array)

            start = time.time()
            old_loss = self._loss_fn(
                params, static, positions[-128:], gradients[-128:], logps[-128:]
            )
            if self._verbose:
                print("loss function time: ", time.time() - start)

            if np.isfinite(old_loss) and old_loss < -5 and self.index > 10:
                if self._verbose:
                    print(f"Loss is low ({old_loss}), skipping training")
                return

            fit, _, opt_state = fit_flow(
                key,
                base,
                self._loss_fn,
                positions,
                gradients,
                logps,
                show_progress=self._show_progress,
                verbose=self._verbose,
                optimizer=self._optimizer,
                batch_size=self._batch_size,
                opt_state=self._opt_state if self._reuse_opt_state else None,
                max_patience=self._max_patience,
            )

            flow = flowjax.flows.Transformed(
                flowjax.distributions.StandardNormal(fit.shape), fit
            )
            params, static = eqx.partition(flow, eqx.is_inexact_array)

            start = time.time()
            new_loss = self._loss_fn(
                params, static, positions[-128:], gradients[-128:], logps[-128:]
            )
            if self._verbose:
                print("new loss function time: ", time.time() - start)

            if self._verbose:
                print(f"Chain {self._chain}: New loss {new_loss}, old loss {old_loss}")

            if not np.isfinite(old_loss):
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

            if not np.isfinite(new_loss):
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

            if (not np.isfinite(old_loss)) and (not np.isfinite(new_loss)):
                self._bijection = self._make_flow_fn(
                    seed, positions, gradients, n_layers=0
                )
                self._opt_state = None
                return

            if not valid_new_logp():
                if self._verbose:
                    print("Invalid new logp. Skipping update.")
                return

            if not np.isfinite(new_loss):
                if self._verbose:
                    print("Invalid new loss. Skipping update.")
                return

            if new_loss > old_loss:
                return

            self._bijection = fit
            self._opt_state = opt_state

        except Exception as e:
            print("update error:", e)
            print(traceback.format_exc())
            raise

    def init_from_transformed_position(self, transformed_position):
        try:
            logp, logdet, *arrays = _init_from_transformed_position(
                self._logp_fn,
                self._bijection,
                jnp.array(transformed_position),
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

    def init_from_untransformed_position(self, untransformed_position):
        try:
            logp, logdet, *arrays = _init_from_untransformed_position(
                self._logp_fn,
                self._bijection,
                jnp.array(untransformed_position),
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
    window_size=600,
    show_progress=False,
    nn_depth=1,
    nn_width=None,
    num_layers=9,
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
    initial_skip=120,
    extension_windows=[],
    extend_dct=False,
    extension_var_count=4,
    extension_var_trafo_count=2,
    debug_save_bijection=False,
    make_optimizer=None,
):
    return partial(
        TransformAdapter,
        verbose=verbose,
        window_size=window_size,
        make_flow_fn=partial(
            make_flow,
            householder_layer=householder_layer,
            dct_layer=dct_layer,
            nn_width=nn_width,
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
        initial_skip=initial_skip,
        extension_windows=extension_windows,
        extend_dct=extend_dct,
        extension_var_count=extension_var_count,
        extension_var_trafo_count=extension_var_trafo_count,
        debug_save_bijection=debug_save_bijection,
        make_optimizer=make_optimizer,
    )
