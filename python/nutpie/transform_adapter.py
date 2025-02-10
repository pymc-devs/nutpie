from typing import Callable, Literal, Union, cast
import itertools
import math
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp

_BIJECTION_TRACE = []

class Linear(eqx.Module, strict=True):
    """Performs a linear transformation."""

    weight: jax.Array
    bias: jax.Array | None
    in_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        *,
        key,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        #dtype = default_floating_dtype() if dtype is None else dtype
        dtype = np.float32 if dtype is None else dtype
        wkey, bkey = jax.random.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        if in_features_ == 0:
            lim = 1.0
        else:
            lim = 1 / math.sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = eqx.nn._misc.default_init(wkey, wshape, dtype, lim)
        bshape = (out_features_,)
        self.bias = eqx.nn._misc.default_init(bkey, bshape, dtype, lim) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    @jax.named_scope("eqx.nn.Linear")
    def __call__(self, x: jax.Array, *, key=None) -> jax.Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x

class FactoredMLP(eqx.Module, strict=True):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: tuple[tuple[Linear, Linear], ...]
    activation: tuple[Callable, ...]
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_size: tuple[int, ...] = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int | tuple[int | tuple[int, int], ...],
        depth: int,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        #dtype = default_floating_dtype() if dtype is None else dtype
        keys = jax.random.split(key, depth + 1)
        layers = []
        if isinstance(width_size, int):
            width_size = (width_size,) * depth

        assert len(width_size) == depth
        activations: list[Callable] = []

        if depth == 0:
            layers.append(
                Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[0])
            )
        else:
            if isinstance(width_size[0], tuple):
                n, k = width_size[0]
                key1, key2 = jax.random.split(keys[0])
                U = Linear(in_size, n, use_bias=False, dtype=dtype, key=key1)
                K = Linear(n, k, use_bias=True, dtype=dtype, key=key2)
                layers.append((U, K))
            else:
                k = width_size[0]
                layers.append(
                    Linear(in_size, k, use_bias, dtype=dtype, key=keys[0])
                )
            activations.append(eqx.filter_vmap(lambda: activation, axis_size=k)())

            for i in range(depth - 1):
                if isinstance(width_size[i + 1], tuple):
                    n, k_new = width_size[i + 1]
                    key1, key2 = jax.random.split(keys[i + 1])
                    U = Linear(k, n, use_bias=False, dtype=dtype, key=key1)
                    K = Linear(n, k_new, use_bias=True, dtype=dtype, key=key2)
                    layers.append((U, K))
                    k = k_new
                else:
                    layers.append(
                        Linear(
                            k, width_size[i + 1], use_bias, dtype=dtype, key=keys[i + 1]
                        )
                    )
                    k = width_size[i + 1]
                activations.append(eqx.filter_vmap(lambda: activation, axis_size=k)())

            if isinstance(out_size, tuple):
                n, k_new = out_size
                key1, key2 = jax.random.split(keys[-1])
                U = Linear(k, n, use_bias=False, dtype=dtype, key=key1)
                K = Linear(n, k_new, use_bias=True, dtype=dtype, key=key2)
                k = k_new
                layers.append((U, K))
            else:
                layers.append(
                    Linear(k, out_size, use_final_bias, dtype=dtype, key=keys[-1])
                )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = tuple(activations)
        #self.activation = eqx.filter_vmap(
        #    eqx.filter_vmap(lambda: activation), axis_size=depth
        #)()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(self, x: jax.Array, *, key = None) -> jax.Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for i, (layer, act) in enumerate(zip(self.layers[:-1], self.activation)):
            if isinstance(layer, tuple):
                U, K = layer
                x = U(x)
                x = K(x)
            else:
                x = layer(x)
            layer_activation = jax.tree.map(
                lambda x: x[i] if eqx.is_array(x) else x, act
            )
            x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, x)

        if isinstance(self.layers[-1], tuple):
            U, K = self.layers[-1]
            x = U(x)
            x = K(x)
        else:
            x = self.layers[-1](x)

        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x


class MLP(eqx.Module, strict=True):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: tuple[Linear, ...]
    activation: tuple[Callable, ...]
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_size: tuple[int, ...] = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int | tuple[int, ...],
        depth: int,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        #dtype = default_floating_dtype() if dtype is None else dtype
        keys = jax.random.split(key, depth + 1)
        layers = []
        if isinstance(width_size, int):
            width_size = (width_size,) * depth

        assert len(width_size) == depth
        activations: list[Callable] = []

        if depth == 0:
            layers.append(
                Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[0])
            )
        else:
            layers.append(
                Linear(in_size, width_size[0], use_bias, dtype=dtype, key=keys[0])
            )
            activations.append(eqx.filter_vmap(lambda: activation, axis_size=width_size[0])())
            for i in range(depth - 1):
                layers.append(
                    Linear(
                        width_size[i], width_size[i + 1], use_bias, dtype=dtype, key=keys[i + 1]
                    )
                )
                activations.append(eqx.filter_vmap(lambda: activation, axis_size=width_size[i])())
            layers.append(
                eqx.nn.Linear(width_size[-1], out_size, use_final_bias, dtype=dtype, key=keys[-1])
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = tuple(activations)
        #self.activation = eqx.filter_vmap(
        #    eqx.filter_vmap(lambda: activation), axis_size=depth
        #)()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(self, x: jax.Array, *, key = None) -> jax.Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for i, (layer, act) in enumerate(zip(self.layers[:-1], self.activation)):
            x = layer(x)
            layer_activation = jax.tree.map(
                lambda x: x[i] if eqx.is_array(x) else x, act
            )
            x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.layers[-1](x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x

def generate_sequences(k, r_vals):
    """
    Generate all binary sequences of length k with exactly r 1's.
    The sequences are stored in a preallocated boolean NumPy array of shape (N, k),
    where N = comb(k, r). A True value represents a '1' and False represents a '0'.

    Parameters:
        k (int): The length of each sequence.
        r (int): The exact number of ones in each sequence.

    Returns:
        A NumPy boolean array of shape (comb(k, r), k) containing all sequences.
    """
    all_sequences = []
    for r in r_vals:
        N = math.comb(k, r)  # number of sequences
        sequences = np.zeros((N, k), dtype=bool)
        # Use enumerate on all combinations where ones appear.
        for i, ones_positions in enumerate(itertools.combinations(range(k), r)):
            sequences[i, list(ones_positions)] = True
        all_sequences.append(sequences)
    return np.concatenate(all_sequences, axis=0)

def max_run_length(seq):
    """
    Given a 1D boolean NumPy array 'seq', compute the maximum run length of consecutive
    identical values (either True or False).

    Parameters:
        seq (np.array): A 1D boolean array.

    Returns:
        The length (int) of the longest run.
    """
    # If the sequence is empty, return 0.
    if seq.size == 0:
        return 0

    # Convert boolean to int (0 or 1) so we can use np.diff.
    arr = seq.astype(int)
    # Compute differences between consecutive elements.
    diffs = np.diff(arr)
    # Positions where the value changes:
    change_indices = np.nonzero(diffs)[0]

    if change_indices.size == 0:
        # No changes at all, so the entire sequence is one run.
        return seq.size

    # To compute the run lengths, add the "start" index (-1) and the last index.
    # For example, if change_indices = [i1, i2, ..., in],
    # then the runs are: (i1 - (-1)), (i2 - i1), ..., (seq.size-1 - in).
    boundaries = np.concatenate(([-1], change_indices, [seq.size - 1]))
    run_lengths = np.diff(boundaries)
    return int(run_lengths.max())

def filter_sequences(sequences, m):
    """
    Filter a 2D NumPy boolean array 'sequences' (each row a binary sequence) so that
    only sequences with maximum run length (of 0's or 1's) at most m are kept.

    Parameters:
        sequences (np.array): A 2D boolean array of shape (N, k).
        m (int): Maximum allowed run length.

    Returns:
        A NumPy array containing only the rows (sequences) that pass the filter.
    """
    filtered = []
    for seq in sequences:
        if max_run_length(seq) <= m:
            filtered.append(seq)
    return np.array(filtered)


def generate_permutations(rng, n_dim, n_layers, max_run=3):
    if n_layers == 1:
        r = [0, 1]
    elif n_layers == 2:
        r = [1]
    else:
        if n_layers % 2 == 0:
            half = n_layers // 2
            r = [half - 1, half, half + 1]
        else:
            half = n_layers // 2
            r = [half, half + 1]

    all_sequences = generate_sequences(n_layers, r)
    valid_sequences = filter_sequences(all_sequences, max_run)

    valid_sequences = np.repeat(valid_sequences, n_dim // len(valid_sequences) + 1, axis=0)
    rng.shuffle(valid_sequences, axis=0)
    is_in_first = valid_sequences[:n_dim]
    rng = np.random.default_rng(42)
    permutations = (~is_in_first).argsort(axis=0, kind="stable")
    return permutations.T, is_in_first.sum(0)

def make_transform_adapter(
    *,
    verbose=False,
    window_size=600,
    show_progress=False,
    nn_depth=1,
    nn_width=16,
    num_layers=9,
    num_diag_windows=10,
    learning_rate=5e-4,
    untransformed_dim=None,
    zero_init=True,
    batch_size=128,
    reuse_opt_state=False,
    max_patience=20,
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
    from flowjax import bijections
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
                        draw, grad, logp = bijection.inverse_gradient_and_val_(
                            draw, grad, logp
                        )
                    else:
                        draw, grad, logp = (
                            flowjax.bijections.AbstractBijection.inverse_gradient_and_val_(
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
                    draw, grad, logp = bijection.inverse_gradient_and_val_(
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
                    return flow.bijection.inverse_gradient_and_val_(draw, grad, logp)

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

    def make_mvscale(key, n_dim, size, randomize_base=False):
        def make_single_hh(key, idx):
            key1, key2 = jax.random.split(key)
            params = jax.random.normal(key1, (n_dim,))
            params = params / jnp.linalg.norm(params)
            mvscale = bijections.MvScale(params)
            return mvscale

        keys = jax.random.split(key, size)

        if randomize_base:
            key, key_base = jax.random.split(key)
            indices = jax.random.randint(key_base, (size,), 0, n_dim)
        else:
            indices = [val % n_dim for val in range(size)]

        return bijections.Chain([make_single_hh(key, idx) for key, idx in zip(keys, indices)])

    def make_hh(key, n_dim, size, randomize_base=False):
        def make_single_hh(key, idx):
            key1, key2 = jax.random.split(key)
            params = jax.random.normal(key1, (n_dim,)) * 1e-2
            return bijections.Householder(params, base_index=idx)

        keys = jax.random.split(key, size)

        if randomize_base:
            key, key_base = jax.random.split(key)
            indices = jax.random.randint(key_base, (size,), 0, n_dim)
        else:
            indices = [val % n_dim for val in range(size)]

        return bijections.Chain([make_single_hh(key, idx) for key, idx in zip(keys, indices)])

    def make_elemwise_trafo(key, n_dim, *, count=1):
        def make_elemwise(key, loc):
            key1, key2 = jax.random.split(key)
            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.zeros(())
            )
            theta = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.zeros(())
            )

            affine = bijections.AsymmetricAffine(
                loc,
                jnp.ones(()),
                jnp.ones(()),
            )

            affine = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=affine,
                replace=scale,
            )
            affine = eqx.tree_at(
                where=lambda aff: aff.theta,
                pytree=affine,
                replace=theta,
            )

            return affine

        def make(key):
            keys = jax.random.split(key, count + 1)
            key, keys = keys[0], keys[1:]
            loc = jax.random.normal(key=key, shape=(count,)) * 2
            loc = loc - loc.mean()
            return bijections.Chain([make_elemwise(key, mu) for key, mu in zip(keys, loc)])

        keys = jax.random.split(key, n_dim)
        make_affine = eqx.filter_vmap(make, axis_size=n_dim)(keys)
        return bijections.Vmap(make_affine, in_axes=eqx.if_array(0))

    def make_elemwise_trafo_(key, n_dim, *, count=1):
        def make_elemwise(key):
            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jax.random.normal(key=key) / 5,
            )
            theta = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jax.random.normal(key=key) / 5,
            )

            affine = bijections.AsymmetricAffine(
                jax.random.normal(key=key) * 2,
                jnp.ones(()),
                jnp.ones(()),
            )

            affine = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=affine,
                replace=scale,
            )
            affine = eqx.tree_at(
                where=lambda aff: aff.theta,
                pytree=affine,
                replace=theta,
            )

            return affine

        def make(key):
            keys = jax.random.split(key, count)
            return bijections.Scan(eqx.filter_vmap(make_elemwise)(keys))

        keys = jax.random.split(key, n_dim)
        make_affine = eqx.filter_vmap(make)(keys)
        return bijections.Vmap(make_affine())

    def make_coupling(key, dim, n_untransformed, **kwargs):
        n_transformed = dim - n_untransformed

        mvscale = make_mvscale(key, n_transformed, 1, randomize_base=True)

        transformer = bijections.Chain([
            make_elemwise_trafo(key, n_transformed, count=3),
            mvscale,
        ])
        mlp1 = lambda out_size: FactoredMLP(
            n_untransformed,
            (128, out_size),
            kwargs["nn_width"],
            depth=len(kwargs["nn_width"]),
            key=key,
            dtype=jnp.float32,
            activation=jax.nn.gelu,
        )
        return bijections.Coupling(key, transformer=transformer, untransformed_dim=n_untransformed, dim=dim, conditioner=mlp1, **kwargs)

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
        diag_affine = bijections.Affine(mean, diag)
        diag_affine = eqx.tree_at(
            where=lambda aff: aff.scale,
            pytree=diag_affine,
            replace=diag_param,
        )

        flows = [
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

        def make_layer(key, untransformed_dim: int | None, permutation=None):
            key, key_couple, key_permute, key_hh = jax.random.split(key, 4)

            if nn_width is None:
                width = n_dim // 2
            else:
                width = nn_width

            if untransformed_dim is None:
                untransformed_dim = n_dim // 2

            if untransformed_dim < 0:
                untransformed_dim = n_dim + untransformed_dim

            if isinstance(width, int) and width > 16 * untransformed_dim:
                width = 16 * untransformed_dim

            coupling = make_coupling(
                key_couple,
                n_dim,
                untransformed_dim,
                nn_width=width,
                nn_depth=nn_depth,
                nn_activation=jax.nn.gelu,
            )

            if zero_init:
                coupling = jax.tree_util.tree_map(
                    lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
                    coupling,
                )

            flow = coupling

            if householder_layer:
                hh = make_hh(key_hh, n_dim, 1, randomize_base=False)
                flow = bijections.Sandwich(hh, flow)

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

            if permutation is None:
                flow = add_default_permute(flow, n_dim, key_permute)
            else:
                flow = bijections.Sandwich(bijections.Permute(permutation), flow)

            mvscale = make_mvscale(key, n_dim, 1, randomize_base=True)

            flow = bijections.Chain(
                [
                    #make_elemwise_trafo(key, n_dim, count=1),
                    mvscale,
                    flow,
                ]
            )

            return flow

        key, key_permute = jax.random.split(key)
        keys = jax.random.split(key, n_layers)

        if untransformed_dim is None:
            # TODO better rng?
            rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))
            permutation, lengths = generate_permutations(rng, n_dim, n_layers)
            layers = []
            for i, (key, p, length) in enumerate(zip(keys, permutation, lengths)):
                layers.append(make_layer(key, int(length), p))
            bijection = flowjax.bijections.Chain(layers)
        elif isinstance(untransformed_dim, int):
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

        if True:
            scale = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.array(0.0),
            )
            theta = Parameterize(
                lambda x: x + jnp.sqrt(1 + x**2),
                jnp.array(0.0),
            )

            affine = flowjax.bijections.AsymmetricAffine(jnp.zeros(()), jnp.ones(()), jnp.ones(()))

            affine = eqx.tree_at(
                where=lambda aff: aff.scale,
                pytree=affine,
                replace=scale,
            )
            affine = eqx.tree_at(
                where=lambda aff: aff.theta,
                pytree=affine,
                replace=theta,
            )

            do_flip = layer % 2 == 0

            if nn_width is None:
                width = 16
            else:
                width = nn_width

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
            bijection.inverse_gradient_and_val_(
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
                    #optax.adamw(learning_rate), 50
                    optax.adabelief(learning_rate), 50
                    #optax.adam(learning_rate), 50
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
                    return np.isfinite(logdet) and np.isfinite(pos[0]).all() and np.isfinite(grad[0]).all()

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
