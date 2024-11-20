def make_transform_adapter(
    *,
    verbose=False,
    window_size=2000,
    show_progress=False,
    nn_depth=1,
    nn_width=None,
    num_layers=8,
    num_diag_windows=10,
    learning_rate=1e-3,
    untransformed_dim=None,
    zero_init=True,
    batch_size=128,
    reuse_opt_state=True,
    max_patience=5,
    householder_layer=True,
    dct_layer=True,
    gamma=None,
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
        def __init__(self, gamma=None):
            self._gamma = gamma

        @eqx.filter_jit
        def __call__(
            self,
            params,
            static,
            x,
            condition=None,
            key=None,
            return_all_costs=False,
        ):
            flow = unwrap(eqx.combine(params, static, is_leaf=eqx.is_inexact_array))

            def compute_loss(bijection, draw, grad):
                draw, grad, logp = bijection.inverse_gradient_and_val(
                    draw, grad, jnp.array(0.0)
                )
                cost = ((draw + grad) ** 2).sum()
                if self._gamma is not None:
                    normal_logp = -draw @ draw / 2
                    cost = cost + self._gamma * (logp - normal_logp).sum()

                return cost

            assert x.shape[1] == 2
            draws = x[:, 0, :]
            grads = x[:, 1, :]

            if return_all_costs:
                return jax.vmap(compute_loss, [None, 0, 0])(
                    flow.bijection, draws, grads
                )

            return jnp.log(
                jax.vmap(compute_loss, [None, 0, 0])(
                    flow.bijection, draws, grads
                ).mean()
            )

    def fit_flow(key, bijection, loss_fn, points, **kwargs):
        flow = flowjax.flows.Transformed(
            flowjax.distributions.StandardNormal(bijection.shape), bijection
        )

        key, train_key = jax.random.split(key)

        fit, losses = flowjax.train.fit_to_data(
            key=train_key,
            dist=flow,
            x=points,
            loss_fn=loss_fn,
            max_epochs=500,
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
        untransformed_dim: int | list[int] | None = None,
        n_layers,
    ):
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

        key = jax.random.PRNGKey(seed % (2**63))

        flows = [
            flowjax.flows.Affine(loc=mean, scale=diag),
        ]

        if n_layers == 0:
            return flowjax.flows.Chain(flows)

        if untransformed_dim is None:
            untransformed_dim = n_dim // 2

        untransformed_dim = cast(list[int] | int, untransformed_dim)

        def make_layer(key, untransformed_dim, is_last=False):
            key, key_couple, key_permute = jax.random.split(key, 3)

            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.array(0.0),
                # lambda x: jnp.exp(jnp.arcsinh(x)), jnp.array(0.0),
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

            if is_last:
                flow = flowjax.bijections.Chain([coupling])
            else:
                flow = flowjax.flows._add_default_permute(coupling, n_dim, key_permute)

            if householder_layer:
                bijections = list(flow.bijections)
                params = jnp.ones(n_dim) * 1e-5
                params = params.at[0].set(0.)
                bijections.append(flowjax.bijections.Householder(params))
                flow = flowjax.bijections.Chain(bijections)

            return flow

        if n_layers == 1:
            num_untrafo = (
                untransformed_dim
                if isinstance(untransformed_dim, int)
                else untransformed_dim[-1]
            )
            bijection = make_layer(
                key,
                untransformed_dim=num_untrafo,
                is_last=True,
            )
        else:
            keys = jax.random.split(key, n_layers - 1)

            if isinstance(untransformed_dim, int):
                make_layers = eqx.filter_vmap(make_layer)
                layers = make_layers(keys, untransformed_dim)
                bijection = flowjax.bijections.Scan(layers)

                key, key_layer = jax.random.split(key)
                last = make_layer(key_layer, untransformed_dim, is_last=True)

                bijection = flowjax.bijections.Chain([bijection, last])
            else:
                layers = []
                for i, (key, num_untrafo) in enumerate(zip(keys[:-1], untransformed_dim[:1])):

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

                layers.append(make_layer(keys[-1], untransformed_dim[-1], is_last=True))

                bijection = flowjax.bijections.Chain(layers)

        return flowjax.bijections.Chain([bijection, *flows])

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
        ):
            self._logp_fn = logp_fn
            self._make_flow_fn = make_flow_fn
            self._chain = chain
            self._verbose = verbose
            self._window_size = window_size
            self._optimizer = optax.apply_if_finite(optax.adabelief(learning_rate), 50)
            self._loss_fn = FisherLoss(gamma)
            self._show_progress = show_progress
            self._num_diag_windows = num_diag_windows
            self._zero_init = zero_init
            self._untransformed_dim = untransformed_dim
            self._batch_size = batch_size
            self._reuse_opt_state = reuse_opt_state
            self._opt_state = None
            self._max_patience = max_patience
            self._count_trace = []
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
            self._count_trace.append(n_draws)
            if n_draws == 0:
                return
            try:
                if self.index <= self._num_diag_windows:
                    size = len(positions)
                    lower_idx = -size // 5 + 3
                    positions_slice = positions[-size // 5 + 3 :]
                    gradients_slice = gradients[-size // 5 + 3 :]

                    if len(positions_slice) > 0:
                        positions = positions_slice
                        gradients = gradients_slice

                    fit = self._make_flow_fn(seed, positions, gradients, n_layers=0)
                    points = jnp.transpose(jnp.array([positions, gradients]), [1, 0, 2])

                    flow = flowjax.flows.Transformed(
                        flowjax.distributions.StandardNormal(fit.shape), fit
                    )
                    params, static = eqx.partition(flow, eqx.is_inexact_array)
                    new_loss = self._loss_fn(params, static, points)

                    if self._verbose:
                        print("loss from diag:", new_loss)

                    if np.isfinite(new_loss):
                        self._bijection = fit
                        self._opt_state = None

                    return

                positions = np.array(positions[500:][-self._window_size :])
                gradients = np.array(gradients[500:][-self._window_size :])

                if len(positions) == 0:
                    return

                if not np.isfinite(gradients).all():
                    print(gradients)
                    print(gradients.shape)
                    print((~np.isfinite(gradients)).nonzero())

                assert np.isfinite(positions).all()
                assert np.isfinite(gradients).all()

                # TODO don't reuse seed
                key = jax.random.PRNGKey(seed % (2**63))
                points = jnp.transpose(jnp.array([positions, gradients]), [1, 0, 2])

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
                            self._loss_fn(params, static, points[-500:]),
                        )
                else:
                    base = self._bijection

                # make_flow might still only return a single trafo for 1d problems
                if len(base.bijections) == 1:
                    self._bijection = base
                    self._opt_state = None
                    return

                fit, _, opt_state = fit_flow(
                    key,
                    base,
                    self._loss_fn,
                    points,
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
                new_loss = self._loss_fn(params, static, points[-500:])

                flow = flowjax.flows.Transformed(
                    flowjax.distributions.StandardNormal(self._bijection.shape),
                    self._bijection,
                )
                params, static = eqx.partition(flow, eqx.is_inexact_array)
                old_loss = self._loss_fn(params, static, points[-500:])

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
                            params, static, points[-500:], return_all_costs=True
                        )
                    )

                if not np.isfinite(new_loss):
                    flow = flowjax.flows.Transformed(
                        flowjax.distributions.StandardNormal(fit.shape), fit
                    )
                    params, static = eqx.partition(flow, eqx.is_inexact_array)
                    print(
                        self._loss_fn(
                            params, static, points[-500:], return_all_costs=True
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
    )
