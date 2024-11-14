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
    scale_layer=False,
    untransformed_dim=None,
    zero_init=True,
):
    import jax
    import equinox as eqx
    import jax.numpy as jnp
    import flowjax
    import flowjax.train
    import flowjax.flows
    import optax
    import traceback
    from paramax import Parameterize, unwrap
    from functools import partial

    import numpy as np

    class FisherLoss:
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
                draw, grad, _ = bijection.inverse_gradient_and_val(
                    draw, grad, jnp.array(0.0)
                )
                return ((draw + grad) ** 2).sum()

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
            **kwargs,
        )
        return fit.bijection, losses

    def make_flow(
        seed,
        positions,
        gradients,
        *,
        zero_init=False,
        scale_layer=False,
        untransformed_dim=None,
        n_layers,
    ):
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

        def make_layer(key, is_last=False):
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

            if untransformed_dim is None:
                untrans_dim = n_dim // 2
            else:
                untrans_dim = untransformed_dim

            coupling = flowjax.bijections.coupling.Coupling(
                key_couple,
                transformer=affine,
                untransformed_dim=untrans_dim,
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

            if scale_layer:
                from flowjax.bijections import mvscale

                bijections = list(flow.bijections)
                bijections.append(mvscale.MvScale4(jnp.ones(n_dim) * 1e-5))
                # bijections.append(mvscale.MvScale3(jnp.ones(n_dim) * 1e-5))
                flow = flowjax.bijections.Chain(bijections)

            return flow

        if n_layers == 1:
            bijection = make_layer(key, is_last=True)
        else:
            keys = jax.random.split(key, n_layers - 1)
            layers = eqx.filter_vmap(make_layer)(keys)
            bijection = flowjax.bijections.Scan(layers)

            key, key_layer = jax.random.split(key)
            last = make_layer(key_layer, is_last=True)

            bijection = flowjax.bijections.Chain([bijection, last])

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
        ):
            self._logp_fn = logp_fn
            self._make_flow_fn = make_flow_fn
            self._chain = chain
            self._verbose = verbose
            self._window_size = window_size
            self._optimizer = optax.apply_if_finite(optax.adabelief(learning_rate), 50)
            self._loss_fn = FisherLoss()
            self._show_progress = show_progress
            self._num_diag_windows = num_diag_windows
            self._zero_init = zero_init
            self._untransformed_dim = untransformed_dim
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
                if self.index <= self._num_diag_windows:
                    size = len(positions)
                    positions = positions[-size // 5 + 3 :]
                    gradients = gradients[-size // 5 + 3 :]

                    fit = self._make_flow_fn(seed, positions, gradients, n_layers=0)
                    points = jnp.transpose(jnp.array([positions, gradients]), [1, 0, 2])

                    flow = flowjax.flows.Transformed(
                        flowjax.distributions.StandardNormal(fit.shape), fit
                    )
                    params, static = eqx.partition(flow, eqx.is_inexact_array)
                    new_loss = self._loss_fn(params, static, points)

                    print("loss from diag:", new_loss)

                    if np.isfinite(new_loss):
                        self._bijection = fit

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

                # make_flow might still only return a single trafo if the for 1d problems
                if len(base.bijections) == 1:
                    self._bijection = base
                    return

                fit, _ = fit_flow(
                    key,
                    base,
                    self._loss_fn,
                    points,
                    show_progress=self._show_progress,
                    optimizer=self._optimizer,
                    batch_size=128,
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
                    return

                if not np.isfinite(new_loss):
                    return

                if new_loss > old_loss:
                    return

                self._bijection = fit

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
        make_flow_fn=partial(make_flow, scale_layer=scale_layer),
        show_progress=show_progress,
        num_diag_windows=num_diag_windows,
        learning_rate=learning_rate,
        zero_init=zero_init,
        untransformed_dim=untransformed_dim,
    )
