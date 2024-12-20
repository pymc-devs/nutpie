from typing import cast

_BIJECTION_TRACE = []


def make_transform_adapter(
    *,
    verbose=False,
    window_size=600,
    show_progress=False,
    nn_depth=1,
    nn_width=16,
    num_layers=3,
    num_diag_windows=10,
    learning_rate=1e-3,
    untransformed_dim=[1, None, -1],
    zero_init=True,
    batch_size=128,
    reuse_opt_state=False,
    max_patience=60,
    householder_layer=True,
    dct_layer=False,
    gamma=None,
    log_inside_batch=False,
    initial_skip=120,
    extension_windows=[16, 20, 24],
    extend_dct=False,
    extension_var_count=4,
    extension_var_trafo_count=2,
    debug_save_bijection=False,
    make_optimizer=None,
):
    import traceback
    from functools import partial

    import equinox as eqx
    import flowjax
    import flowjax.flows
    import flowjax.train
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from paramax import Parameterize, unwrap

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
                    if True:
                        draw, grad, logp = bijection.inverse_gradient_and_val(
                            draw, grad, logp
                        )
                    else:
                        draw, grad, logp = (
                            flowjax.bijections.AbstractBijection.inverse_gradient_and_val(
                                bijection, draw, grad, logp
                            )
                        )
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
                    draw, grad, logp = bijection.inverse_gradient_and_val(
                        draw, grad, logp
                    )
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
                    return flow.bijection.inverse_gradient_and_val(draw, grad, logp)

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

        fit, losses = flowjax.train.fit_to_data(
            key=train_key,
            dist=flow,
            x=(draws, grads, logps),
            loss_fn=loss_fn,
            max_epochs=1000,
            return_best=True,
            **kwargs,
        )
        return fit.bijection, losses, losses["opt_state"]

    def make_flow(
        seed,
        positions,
        gradients,
        *,
        zero_init=False,
        householder_layer=False,
        dct_layer=False,
        untransformed_dim: int | list[int | None] | None = None,
        n_layers,
    ):
        from flowjax import bijections

        positions = np.array(positions)
        gradients = np.array(gradients)

        if len(positions) == 0:
            return

        n_draws, n_dim = positions.shape

        if n_dim < 2:
            n_layers = 0

        assert positions.shape == gradients.shape

        if n_draws == 0:
            raise ValueError("No draws")
        elif n_draws == 1:
            assert np.all(gradients != 0)
            diag = np.clip(1 / jnp.sqrt(jnp.abs(gradients[0])), 1e-5, 1e5)
            assert np.isfinite(diag).all()
            mean = jnp.zeros_like(diag)
        else:
            pos_std = np.clip(positions.std(0), 1e-8, 1e8)
            grad_std = np.clip(gradients.std(0), 1e-8, 1e8)
            diag = jnp.sqrt(pos_std / grad_std)
            mean = positions.mean(0) + gradients.mean(0) * diag * diag

        print("seed", seed)
        key = jax.random.PRNGKey(seed % (2**63))

        diag_param = Parameterize(
            lambda x: x + jnp.sqrt(1 + x**2),
            (diag**2 - 1) / (2 * diag),
        )

        diag_affine = eqx.tree_at(
            where=lambda aff: aff.scale,
            pytree=bijections.Affine(mean, diag),
            replace=diag_param,
        )

        flows = [
            #flowjax.flows.Affine(loc=mean, scale=diag),
            diag_affine,
        ]

        if n_layers == 0:
            return flowjax.flows.Chain(flows)

        scale = Parameterize(
            lambda x: x + jnp.sqrt(1 + x**2),
            jnp.zeros(n_dim),
        )
        affine = eqx.tree_at(
            where=lambda aff: aff.scale,
            pytree=flowjax.bijections.Affine(jnp.zeros(n_dim), jnp.ones(n_dim)),
            replace=scale,
        )
        params = jnp.ones(n_dim) * 1e-5
        params = params.at[-1].set(1.0)

        hh = flowjax.bijections.Householder(params)
        flows.append(
            bijections.Sandwich(
                bijections.Chain(
                    [
                        bijections.Vmap(bijections.SoftPlusX(), axis_size=n_dim),
                        hh,
                    ]
                ),
                affine,
            )
        )

        if untransformed_dim is None:
            untransformed_dim = n_dim // 2

        untransformed_dim = cast(list[int | None] | int, untransformed_dim)

        def make_layer(key, untransformed_dim: int | None):
            key, key_couple, key_permute = jax.random.split(key, 3)

            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.array(0.0),
            )
            affine = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=flowjax.bijections.Affine(),
                replace=scale,
            )

            if nn_width is None:
                width = n_dim // 2
            else:
                width = nn_width

            if untransformed_dim is None:
                untransformed_dim = n_dim // 2

            if untransformed_dim < 0:
                untransformed_dim = n_dim + untransformed_dim

            coupling = flowjax.bijections.coupling.Coupling(
                key_couple,
                transformer=affine,
                untransformed_dim=untransformed_dim,
                dim=n_dim,
                nn_activation=jax.nn.gelu,
                nn_width=width,
                nn_depth=nn_depth,
            )
            if zero_init:
                coupling = jax.tree_util.tree_map(
                    lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
                    coupling,
                )

            def add_default_permute(bijection, dim, key):
                if dim == 1:
                    return bijection
                if dim == 2:
                    outer = flowjax.bijections.Flip((dim,))
                else:
                    outer = flowjax.bijections.Permute(
                        jax.random.permutation(key, jnp.arange(dim))
                    )

                return flowjax.bijections.Sandwich(outer, bijection)

            flow = add_default_permute(coupling, n_dim, key_permute)

            if householder_layer:
                params = jnp.ones(n_dim) * 1e-5
                params = params.at[0].set(1.0)

                outer = flowjax.bijections.Householder(params)
                flow = flowjax.bijections.Sandwich(outer, flow)

            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.zeros(n_dim),
            )
            affine = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=flowjax.bijections.Affine(jnp.zeros(n_dim), jnp.ones(n_dim)),
                replace=scale,
            )
            params = jnp.ones(n_dim) * 1e-5
            params = params.at[-1].set(1.0)

            hh = flowjax.bijections.Householder(params)
            flow = bijections.Chain(
                [
                    bijections.Sandwich(
                        bijections.Chain(
                            [
                                bijections.Vmap(
                                    bijections.SoftPlusX(), axis_size=n_dim
                                ),
                                hh,
                            ]
                        ),
                        affine,
                    ),
                    flow,
                ]
            )
            return flow

        keys = jax.random.split(key, n_layers)

        if isinstance(untransformed_dim, int):
            make_layers = eqx.filter_vmap(make_layer)
            layers = make_layers(keys, untransformed_dim)
            bijection = flowjax.bijections.Scan(layers)
        else:
            layers = []
            for i, (key, num_untrafo) in enumerate(zip(keys, untransformed_dim)):
                if i % 2 == 0 or not dct_layer:
                    layers.append(make_layer(key, num_untrafo))
                else:
                    inner = make_layer(key, num_untrafo)
                    outer = flowjax.bijections.DCT(inner.shape)

                    layers.append(flowjax.bijections.Sandwich(outer, inner))

                    scale_val = jnp.ones(n_dim)
                    scale = Parameterize(
                        lambda x: x + jnp.sqrt(1 + x**2),
                        jnp.zeros(n_dim),
                    )
                    mean = jnp.zeros(n_dim)
                    inner = eqx.tree_at(
                        where=lambda aff: aff.scale,
                        pytree=flowjax.bijections.Affine(mean, scale_val),
                        replace=scale,
                    )
                    outer = flowjax.bijections.DCT(inner.shape)
                    layers.append(flowjax.bijections.Sandwich(outer, inner))

            bijection = flowjax.bijections.Chain(layers)

        return flowjax.bijections.Chain([bijection, *flows])

    def extend_flow(
        key,
        base,
        loss_fn,
        positions,
        gradients,
        logps,
        layer: int,
        *,
        extension_var_count=4,
        zero_init=False,
        householder_layer=False,
        untransformed_dim: int | list[int | None] | None = None,
        dct: bool = False,
        extension_var_trafo_count=2,
        verbose: bool = False,
    ):
        from flowjax import bijections

        n_draws, n_dim = positions.shape

        if n_dim < 2:
            return base

        if n_dim <= extension_var_count:
            extension_var_count = n_dim - 1
            extension_var_trafo_count = 1

        if dct:
            flow = flowjax.flows.Transformed(
                flowjax.distributions.StandardNormal(base.shape),
                bijections.Chain([bijections.DCT(shape=(n_dim,)), base]),
            )
        else:
            flow = flowjax.flows.Transformed(
                flowjax.distributions.StandardNormal(base.shape), base
            )

        params, static = eqx.partition(flow, eqx.is_inexact_array)
        costs = loss_fn(
            params,
            static,
            positions,
            gradients,
            logps,
            return_elemwise_costs=True,
        )

        if verbose:
            print(max(costs), costs)
            print("dct:", dct)
        idxs = np.argsort(costs)

        permute = bijections.Permute(idxs)

        if False:
            identity = bijections.Identity(shape=(n_dim - extension_var_count,))
            print(costs[idxs])

            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.array(0.0),
            )
            affine = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=flowjax.bijections.Affine(),
                replace=scale,
            )
            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.array(0.0),
            )
            affine2 = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=flowjax.bijections.Affine(),
                replace=scale,
            )

            pre = []
            if layer % 2 == 0:
                pre.append(bijections.Neg(shape=(n_dim,)))

            nonlin_affine = bijections.Chain(
                [
                    bijections.Sandwich(
                        bijections.Chain(
                            [
                                *pre,
                                bijections.Vmap(
                                    bijections.SoftPlusX(), axis_size=n_dim
                                ),
                            ]
                        ),
                        affine,
                    ),
                    affine2,
                ]
            )

            if nn_width is None:
                width = extension_var_count * 16
            else:
                width = nn_width * 16

            if untransformed_dim is None:
                untransformed_dim = extension_var_count // 2

            coupling = flowjax.bijections.coupling.Coupling(
                key,
                transformer=nonlin_affine,
                #transformer=affine,
                untransformed_dim=untransformed_dim,
                dim=extension_var_count,
                nn_activation=jax.nn.gelu,
                nn_width=width,
                nn_depth=nn_depth + 1,
            )
            if zero_init:
                coupling = jax.tree_util.tree_map(
                    lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
                    coupling,
                )

            params = jnp.ones(extension_var_count) * 1e-5
            params = params.at[-1].set(1.0)

            hh = flowjax.bijections.Householder(params)
            inner_permute = flowjax.bijections.Permute(
                jax.random.permutation(key, jnp.arange(extension_var_count))
            )
            coupling = flowjax.bijections.Sandwich(
                inner_permute,
                flowjax.bijections.Sandwich(hh, coupling),
            )

            inner = bijections.Concatenate([identity, coupling])
        else:
            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.array(0.0),
            )
            affine = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=flowjax.bijections.Affine(),
                replace=scale,
            )

            """
            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.array(0.0),
            )
            affine2 = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=flowjax.bijections.Affine(),
                replace=scale,
            )
            pre = []
            if layer % 2 == 0:
                pre.append(bijections.Neg(shape=()))

            nonlin_affine = bijections.Chain([
                bijections.Sandwich(
                    bijections.Chain([
                        *pre,
                        bijections.SoftPlusX(),
                    ]),
                    affine,
                ),
                affine2,
            ])
            """

            do_flip = layer % 2 == 0

            if nn_width is None:
                width = 16
            else:
                width = nn_width

            #if untransformed_dim is None:
            #    untransformed_dim = extension_var_count // 2

            if do_flip:
                coupling = flowjax.bijections.coupling.Coupling(
                    key,
                    transformer=affine,
                    untransformed_dim=n_dim - extension_var_trafo_count,
                    dim=n_dim,
                    nn_activation=jax.nn.gelu,
                    nn_width=width,
                    nn_depth=nn_depth,
                )

                inner_permute = flowjax.bijections.Permute(
                    jnp.concatenate(
                        [
                            jnp.arange(n_dim - extension_var_count),
                            jax.random.permutation(
                                key, jnp.arange(n_dim - extension_var_count, n_dim)
                            ),
                        ]
                    )
                )
            else:
                coupling = flowjax.bijections.coupling.Coupling(
                    key,
                    transformer=affine,
                    untransformed_dim=extension_var_trafo_count,
                    dim=n_dim,
                    nn_activation=jax.nn.gelu,
                    nn_width=width,
                    nn_depth=nn_depth,
                )

                inner_permute = flowjax.bijections.Permute(
                    jnp.concatenate(
                        [
                            jax.random.permutation(
                                key, jnp.arange(n_dim - extension_var_count, n_dim)
                            ),
                            jnp.arange(n_dim - extension_var_count),
                        ]
                    )
                )

            if zero_init:
                coupling = jax.tree_util.tree_map(
                    lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
                    coupling,
                )

            inner = bijections.Sandwich(inner_permute, coupling)

            if False:
                scale = Parameterize(
                    lambda x: x + jnp.sqrt(1 + x**2),
                    jnp.array(0.0),
                )
                affine = eqx.tree_at(
                    where=lambda aff: aff.scale,
                    pytree=flowjax.bijections.Affine(),
                    replace=scale,
                )

                if nn_width is None:
                    width = 16
                else:
                    width = nn_width

                #if untransformed_dim is None:
                #    untransformed_dim = extension_var_count // 2

                coupling = flowjax.bijections.coupling.Coupling(
                    key,
                    transformer=affine,
                    untransformed_dim=extension_var_trafo_count,
                    dim=n_dim,
                    nn_activation=jax.nn.gelu,
                    nn_width=width,
                    nn_depth=nn_depth,
                )

                if zero_init:
                    coupling = jax.tree_util.tree_map(
                        lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
                        coupling,
                    )

                if verbose:
                    print(costs[permute.permutation][inner.outer.permutation])

                """
                params = jnp.ones(n_dim) * 1e-5
                params = params.at[-1].set(0.)

                hh = flowjax.bijections.Householder(params)

                coupling = bijections.Sandwich(hh, coupling)
                """

                inner = bijections.Sandwich(
                    inner.outer,
                    bijections.Chain(
                        [
                            bijections.Sandwich(
                                bijections.Flip(shape=(n_dim,)), coupling
                            ),
                            inner.inner,
                        ]
                    ),
                )

        if dct:
            new_layer = bijections.Sandwich(
                bijections.DCT(shape=(n_dim,)),
                bijections.Sandwich(permute, inner),
            )
        else:
            new_layer = bijections.Sandwich(permute, inner)

        scale = Parameterize(
            lambda x: x + jnp.sqrt(1 + x**2),
            jnp.zeros(n_dim),
        )
        affine = eqx.tree_at(
            where=lambda aff: aff.scale,
            pytree=flowjax.bijections.Affine(jnp.zeros(n_dim), jnp.ones(n_dim)),
            replace=scale,
        )

        pre = []
        if layer % 2 == 0:
            pre.append(bijections.Neg(shape=(n_dim,)))

        nonlin_layer = bijections.Sandwich(
            bijections.Chain(
                [
                    *pre,
                    bijections.Vmap(bijections.SoftPlusX(), axis_size=n_dim),
                ]
            ),
            affine,
        )
        scale = Parameterize(
            lambda x: x + jnp.sqrt(1 + x**2),
            jnp.zeros(n_dim),
        )
        affine = eqx.tree_at(
            where=lambda aff: aff.scale,
            pytree=flowjax.bijections.Affine(jnp.zeros(n_dim), jnp.ones(n_dim)),
            replace=scale,
        )
        return bijections.Chain([new_layer, nonlin_layer, affine, base])

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
        ):
            self._logp_fn = logp_fn
            self._make_flow_fn = make_flow_fn
            self._chain = chain
            self._verbose = verbose
            self._window_size = window_size
            self._initial_skip = initial_skip
            if make_optimizer is None:
                self._make_optimizer = lambda: optax.apply_if_finite(
                    optax.adabelief(learning_rate), 50
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
                print(f"Chain {self._chain}: Total available points: {len(positions)}")
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
                    new_loss = self._loss_fn(
                        params, static, positions, gradients, logps
                    )

                    if self._verbose:
                        print("loss from diag:", new_loss)

                    if np.isfinite(new_loss):
                        self._bijection = fit
                        self._opt_state = None

                    return

                positions = np.array(
                    positions[self._initial_skip :][-self._window_size :]
                )
                gradients = np.array(
                    gradients[self._initial_skip :][-self._window_size :]
                )
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
                        n_layers=num_layers,
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
                                positions[-100:],
                                gradients[-100:],
                                logps[-100:],
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
                    return

                flow = flowjax.flows.Transformed(
                    flowjax.distributions.StandardNormal(self._bijection.shape),
                    self._bijection,
                )
                params, static = eqx.partition(flow, eqx.is_inexact_array)
                old_loss = self._loss_fn(
                    params, static, positions[-128:], gradients[-128:], logps[-128:]
                )

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
                    optimizer=self._optimizer,
                    batch_size=self._batch_size,
                    opt_state=self._opt_state if self._reuse_opt_state else None,
                    max_patience=self._max_patience,
                )

                flow = flowjax.flows.Transformed(
                    flowjax.distributions.StandardNormal(fit.shape), fit
                )
                params, static = eqx.partition(flow, eqx.is_inexact_array)
                new_loss = self._loss_fn(
                    params, static, positions[-128:], gradients[-128:], logps[-128:]
                )

                if self._verbose:
                    print(
                        f"Chain {self._chain}: New loss {new_loss}, old loss {old_loss}"
                    )

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

                if (not np.isfinite(old_loss)) and (not np.isfinite(new_loss)):
                    self._bijection = self._make_flow_fn(
                        seed, positions, gradients, n_layers=0
                    )
                    self._opt_state = None
                    return

                if not np.isfinite(new_loss):
                    return

                if new_loss > old_loss:
                    return

                self._bijection = fit
                self._opt_state = opt_state

                if self._debug_save_bijection:
                    _BIJECTION_TRACE.append(
                        (self.index, fit, (positions, gradients, logps))
                    )

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
                return float(logdet), *[
                    np.array(val, dtype="float64") for val in arrays
                ]
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

    return partial(
        TransformAdapter,
        verbose=verbose,
        window_size=window_size,
        make_flow_fn=partial(
            make_flow,
            householder_layer=householder_layer,
            dct_layer=dct_layer,
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
