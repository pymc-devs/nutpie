import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np

from nutpie import _lib  # type: ignore
from nutpie.sample import CompiledModel

SeedType = int


def make_transform_adapter(*, verbose=True, window_size=2000, show_progress=False):
    import jax
    import equinox as eqx
    import jax.numpy as jnp
    import flowjax
    import flowjax.train
    import flowjax.flows
    import optax
    import traceback

    class FisherLoss:
        @eqx.filter_jit
        def __call__(
            self,
            params,
            static,
            x,
            condition=None,
            key=None,
        ):
            flow = flowjax.train.losses.unwrap(
                eqx.combine(params, static, is_leaf=eqx.is_inexact_array)
            )

            def compute_loss(bijection, draw, grad):
                draw, grad, _ = bijection.inverse_gradient_and_val(
                    draw, grad, jnp.array(0.0)
                )
                return ((draw + grad) ** 2).sum()

            assert x.shape[1] == 2
            draws = x[:, 0, :]
            grads = x[:, 1, :]
            return jnp.log(
                jax.vmap(compute_loss, [None, 0, 0])(
                    flow.bijection, draws, grads
                ).mean()
            )

    def fit_flow(key, bijection, loss_fn, positions, gradients, **kwargs):
        flow = flowjax.flows.Transformed(
            flowjax.distributions.StandardNormal(bijection.shape), bijection
        )

        points = jnp.transpose(jnp.array([positions, gradients]), [1, 0, 2])

        key, train_key = jax.random.split(key)

        fit, losses, opt_state = flowjax.train.fit_to_data(
            key=train_key,
            dist=flow,
            x=points,
            loss_fn=loss_fn,
            **kwargs,
        )

        final_cost = losses["val"][-1]
        return fit, final_cost, opt_state

    def make_flow(seed, positions, gradients, *, n_layers):
        positions = np.array(positions)
        gradients = np.array(gradients)

        n_draws, n_dim = positions.shape

        if n_dim < 2:
            n_layers = 0

        assert positions.shape == gradients.shape
        assert n_draws > 0

        if n_draws == 0:
            raise ValueError("No draws")
        elif n_draws == 1:
            diag = 1 / jnp.abs(gradients[0])
            mean = jnp.zeros_like(diag)
        else:
            diag = jnp.sqrt(positions.std(0) / gradients.std(0))
            mean = positions.mean(0) + diag * gradients.mean(0)

        key = jax.random.PRNGKey(seed % (2**63))

        flows = [
            flowjax.flows.Affine(loc=mean, scale=diag),
        ]

        for layer in range(n_layers):
            key, key_couple, key_permute = jax.random.split(key, 3)

            scale = flowjax.wrappers.Parameterize(
                lambda x: jnp.exp(jnp.arcsinh(x)), jnp.array(0.0)
            )
            affine = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=flowjax.bijections.Affine(),
                replace=scale,
            )

            coupling = flowjax.bijections.coupling.Coupling(
                key_couple,
                transformer=affine,
                untransformed_dim=n_dim // 2,
                dim=n_dim,
                nn_activation=jax.nn.gelu,
                nn_width=n_dim // 2,
                nn_depth=1,
            )

            if layer == n_layers - 1:
                flow = coupling
            else:
                flow = flowjax.flows._add_default_permute(coupling, n_dim, key_permute)

            flows.append(flow)

        return flowjax.bijections.Chain(flows[::-1])

    @eqx.filter_jit
    def _init_from_transformed_position(logp_fn, bijection, transformed_position):
        bijection = flowjax.train.losses.unwrap(bijection)
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
        bijection = flowjax.train.losses.unwrap(bijection)
        transformed_position, transformed_gradient, logdet = (
            bijection.inverse_gradient_and_val(
                untransformed_position, untransformed_gradient, 0.0
            )
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
        ):
            self._logp_fn = logp_fn
            self._make_flow_fn = make_flow_fn
            self._chain = chain
            self._verbose = verbose
            self._window_size = window_size
            self._optimizer = optax.adabelief(1e-3)
            self._loss_fn = FisherLoss()
            self._show_progress = show_progress
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

        def update(self, seed, positions, gradients):
            self.index += 1
            if self._verbose:
                print(f"Chain {self._chain}: Total available points: {len(positions)}")
            n_draws = len(positions)
            if n_draws == 0:
                return
            try:
                if self.index <= 10:
                    self._bijection = self._make_flow_fn(
                        seed, positions[-10:], gradients[-10:], n_layers=0
                    )
                    return

                positions = np.array(positions[500:][-self._window_size :])
                gradients = np.array(gradients[500:][-self._window_size :])

                if len(positions) == 0:
                    return

                assert np.isfinite(positions).all()
                assert np.isfinite(gradients).all()

                if len(self._bijection.bijections) == 1:
                    self._bijection = self._make_flow_fn(
                        seed, positions, gradients, n_layers=8
                    )

                # make_flow might still only return a single trafo if the for 1d problems
                if len(self._bijection.bijections) == 1:
                    return

                # TODO don't reuse seed
                key = jax.random.PRNGKey(seed % (2**63))
                fit, final_cost, _ = fit_flow(
                    key,
                    self._bijection,
                    self._loss_fn,
                    positions,
                    gradients,
                    show_progress=self._show_progress,
                    optimizer=self._optimizer,
                    batch_size=128,
                )
                if self._verbose:
                    print(f"Chain {self._chain}: final cost {final_cost}")
                if np.isfinite(final_cost).all():
                    self._bijection = fit.bijection
                else:
                    self._bijection = self._make_flow_fn(
                        seed, positions, gradients, n_layers=0
                    )
            except Exception as e:
                print("update error:", e)
                print(traceback.format_exc())

        def init_from_transformed_position(self, transformed_position):
            try:
                logp, logdet, *arrays = _init_from_transformed_position(
                    self._logp_fn,
                    self._bijection,
                    jnp.array(transformed_position),
                )
                return float(logp), float(logdet), *[np.array(val) for val in arrays]
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
                return float(logp), float(logdet), *[np.array(val) for val in arrays]
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                raise

        def inv_transform(self, position, gradient):
            try:
                logdet, *arrays = _inv_transform(
                    self._bijection, jnp.array(position), jnp.array(gradient)
                )
                return logdet, *[np.array(val) for val in arrays]
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                raise

    return partial(
        TransformAdapter,
        verbose=verbose,
        window_size=window_size,
        make_flow_fn=make_flow,
        show_progress=show_progress,
    )


@dataclass(frozen=True)
class PyFuncModel(CompiledModel):
    _make_logp_func: Callable
    _make_expand_func: Callable
    _make_initial_points: Callable[[SeedType], np.ndarray] | None
    _shared_data: dict[str, Any]
    _n_dim: int
    _variables: list[_lib.PyVariable]
    _coords: dict[str, Any]
    _make_transform_adapter: Callable | None
    _raw_logp_fn: Callable | None

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {var.name: tuple(var.dtype.shape) for var in self._variables}

    @property
    def coords(self):
        return self._coords

    @property
    def n_dim(self):
        return self._n_dim

    def with_data(self, **updates):
        for name in updates:
            if name not in self._shared_data:
                raise ValueError(f"Unknown data variable: {name}")

        updated = self._shared_data.copy()
        updated.update(**updates)
        return dataclasses.replace(self, _shared_data=updated)

    def _make_sampler(self, settings, init_mean, cores, progress_type):
        model = self._make_model(init_mean)
        return _lib.PySampler.from_pyfunc(
            settings,
            cores,
            model,
            progress_type,
        )

    def _make_model(self, init_mean):
        def make_logp_func():
            logp_fn = self._make_logp_func()
            return partial(logp_fn, **self._shared_data)

        def make_expand_func(seed1, seed2, chain):
            expand_fn = self._make_expand_func(seed1, seed2, chain)
            return partial(expand_fn, **self._shared_data)

        if self._raw_logp_fn is not None:
            make_adapter = partial(make_transform_adapter(), logp_fn=self._raw_logp_fn)
        else:
            make_adapter = None

        return _lib.PyModel(
            make_logp_func,
            make_expand_func,
            self._variables,
            self.n_dim,
            self._make_initial_points,
            make_transform_adapter,
            make_adapter,
        )


def from_pyfunc(
    ndim: int,
    make_logp_fn: Callable,
    make_expand_fn: Callable,
    expanded_dtypes: list[np.dtype],
    expanded_shapes: list[tuple[int, ...]],
    expanded_names: list[str],
    *,
    coords: dict[str, Any] | None = None,
    dims: dict[str, tuple[str, ...]] | None = None,
    shared_data: dict[str, Any] | None = None,
    make_initial_point_fn: Callable[[SeedType], np.ndarray] | None,
    make_transform_adapter=None,
    raw_logp_fn=None,
):
    variables = []
    for name, shape, dtype in zip(
        expanded_names, expanded_shapes, expanded_dtypes, strict=True
    ):
        shape = _lib.TensorShape(list(shape))
        if dtype == np.float64:
            dtype = _lib.ExpandDtype.float64_array(shape)
        elif dtype == np.float32:
            dtype = _lib.ExpandDtype.float32_array(shape)
        elif dtype == np.int64:
            dtype = _lib.ExpandDtype.int64_array(shape)
        variables.append(_lib.PyVariable(name, dtype))

    if coords is None:
        coords = {}
    if dims is None:
        dims = {}
    if shared_data is None:
        shared_data = {}

    return PyFuncModel(
        _n_dim=ndim,
        dims=dims,
        _coords=coords,
        _make_logp_func=make_logp_fn,
        _make_expand_func=make_expand_fn,
        _make_initial_points=make_initial_point_fn,
        _variables=variables,
        _shared_data=shared_data,
        _make_transform_adapter=make_transform_adapter,
        _raw_logp_fn=raw_logp_fn,
    )
