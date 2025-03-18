from typing import Any, ClassVar, Union, Literal, Callable
import math
import itertools

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.coupling import get_ravelled_pytree_constructor
from flowjax.utils import arraylike_to_array
import jax
import jax.numpy as jnp
import equinox as eqx
from flowjax import bijections
import flowjax.distributions
import flowjax.flows
from jaxtyping import Array, ArrayLike, PyTree
import numpy as np
from paramax import NonTrainable, Parameterize, unwrap
from equinox.nn import Linear
from paramax.wrappers import AbstractUnwrappable


_NN_ACTIVATION = jax.nn.gelu


def _generate_sequences(k, r_vals):
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
    if k > 30:
        raise ValueError("Too many sequences to enumerate.")
    all_sequences = []
    for r in r_vals:
        N = math.comb(k, r)  # number of sequences
        sequences = np.zeros((N, k), dtype=bool)
        # Use enumerate on all combinations where ones appear.
        for i, ones_positions in enumerate(itertools.combinations(range(k), r)):
            sequences[i, list(ones_positions)] = True
        all_sequences.append(sequences)
    return np.concatenate(all_sequences, axis=0)


def _max_run_length(seq):
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


def _filter_sequences(sequences, m):
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
        if _max_run_length(seq) <= m:
            filtered.append(seq)
    return np.array(filtered)


def _generate_permutations(rng, n_dim, n_layers, max_run=3):
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

    all_sequences = _generate_sequences(n_layers, r)
    valid_sequences = _filter_sequences(all_sequences, max_run)

    valid_sequences = np.repeat(
        valid_sequences, n_dim // len(valid_sequences) + 1, axis=0
    )
    rng.shuffle(valid_sequences, axis=0)
    is_in_first = valid_sequences[:n_dim]
    rng = np.random.default_rng(42)
    permutations = (~is_in_first).argsort(axis=0, kind="stable")
    return permutations.T, is_in_first.sum(0)


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
                layers.append(Linear(in_size, k, use_bias, dtype=dtype, key=keys[0]))
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
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(self, x: jax.Array, *, key=None) -> jax.Array:
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


class AsymmetricAffine(bijections.AbstractBijection):
    """An asymmetric bijection that applies different scaling factors for
    positive and negative inputs.

    This bijection implements a continuous, differentiable transformation that
    scales positive and negative inputs differently while maintaining smoothness
    at zero. It's particularly useful for modeling data with different variances
    in positive and negative regions.

    The forward transformation is defined as:
        y = σ θ x     for x ≥ 0
        y = σ x/θ     for x < 0
    where:
        - σ (scale) controls the overall scaling
        - θ (theta) controls the asymmetry between positive and negative regions
        - μ (loc) controls the location shift

    The transformation uses a smooth transition between the two regions to
    maintain differentiability.

    For θ = 0, this is exactly an affine function with the specified location
    and scale.

    Attributes:
        shape: The shape of the transformation parameters
        cond_shape: Shape of conditional inputs (None as this bijection is
            unconditional)
        loc: Location parameter μ for shifting the distribution
        scale: Scale parameter σ (positive)
        theta: Asymmetry parameter θ (positive)
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None
    loc: Array
    scale: Array | AbstractUnwrappable[Array]
    theta: Array | AbstractUnwrappable[Array]

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
        theta: ArrayLike = 1,
    ):
        self.loc, scale, theta = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (loc, scale, theta)),
        )
        self.shape = scale.shape
        assert self.shape == ()
        self.scale = Parameterize(lambda x: x + jnp.sqrt(1 + x**2), jnp.zeros(()))
        self.theta = Parameterize(lambda x: x + jnp.sqrt(1 + x**2), jnp.zeros(()))

    def _log_derivative_f(self, x, mu, sigma, theta):
        abs_x = jnp.abs(x)
        theta = jnp.log(theta)

        sinh_theta = jnp.sinh(theta)
        # sinh_theta = (theta - 1 / theta) / 2
        cosh_theta = jnp.cosh(theta)
        # cosh_theta = (theta + 1 / theta) / 2
        numerator = sinh_theta * x * (abs_x + 2.0)
        denominator = (abs_x + 1.0) ** 2
        term = numerator / denominator
        dy_dx = sigma * (cosh_theta + term)
        return jnp.log(dy_dx)

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        def transform(x, mu, sigma, theta):
            weight = (jax.nn.soft_sign(x) + 1) / 2
            z = x * sigma
            y_pos = z * theta
            y_neg = z / theta
            y = weight * y_pos + (1.0 - weight) * y_neg + mu
            return y

        mu, sigma, theta = self.loc, self.scale, self.theta

        y = transform(x, mu, sigma, theta)
        logjac = self._log_derivative_f(x, mu, sigma, theta)
        return y, logjac.sum()
        # y, jac = jax.value_and_grad(transform, argnums=0)(x, mu, sigma, theta)
        # return y, jnp.log(jac)

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        def inverse(y, mu, sigma, theta):
            delta = y - mu
            inv_theta = 1 / theta

            # Case 1: y >= mu (delta >= 0)
            a = sigma * (theta + inv_theta)
            discriminant_pos = (
                jnp.square(a - 2.0 * delta) + 16.0 * sigma * theta * delta
            )
            discriminant_pos = jnp.where(discriminant_pos < 0, 1.0, discriminant_pos)
            sqrt_pos = jnp.sqrt(discriminant_pos)
            numerator_pos = 2.0 * delta - a + sqrt_pos
            denominator_pos = 4.0 * sigma * theta
            x_pos = numerator_pos / denominator_pos

            # Case 2: y < mu (delta < 0)
            sigma_part = sigma * (1.0 + theta * theta)
            term2 = 2.0 * delta * theta
            inside_sqrt_neg = (
                jnp.square(sigma_part + term2) - 16.0 * sigma * delta * theta
            )
            inside_sqrt_neg = jnp.where(inside_sqrt_neg < 0, 1.0, inside_sqrt_neg)
            sqrt_neg = jnp.sqrt(inside_sqrt_neg)
            numerator_neg = sigma_part + term2 - sqrt_neg
            denominator_neg = 4.0 * sigma
            x_neg = numerator_neg / denominator_neg

            # Combine cases based on delta
            x = jnp.where(delta >= 0.0, x_pos, x_neg)
            return x

        mu, sigma, theta = self.loc, self.scale, self.theta

        x = inverse(y, mu, sigma, theta)
        logjac = self._log_derivative_f(x, mu, sigma, theta)
        return x, -logjac.sum()
        # x, jac = jax.value_and_grad(inverse, argnums=0)(y, mu, sigma, theta)
        # return x, jnp.log(jac)


class MvScale(bijections.AbstractBijection):
    shape: tuple[int, ...]
    params: Array
    cond_shape = None
    base_index: int

    def __init__(self, params: Array, base_index: int = 0):
        self.shape = (params.shape[-1],)
        self.params = params
        self.base_index = base_index

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        scale = jnp.linalg.norm(self.params)
        v = self.params / scale
        y = x + ((v @ x) * (scale - 1)) * v
        return y, jnp.log(scale)

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        scale = jnp.linalg.norm(self.params)
        v = self.params / scale
        x = y + ((v @ y) * (1 / scale - 1)) * v
        return x, -jnp.log(scale)


class MaskedVmap(AbstractBijection):
    bijection: AbstractBijection
    in_axes: tuple
    axis_size: int
    cond_shape: tuple[int, ...] | None
    mask: Array

    def __init__(
        self,
        bijection: AbstractBijection,
        mask: Array,
        *,
        in_axes: PyTree | None | int | Callable = None,
        axis_size: int | None = None,
        in_axes_condition: int | None = None,
    ):
        if in_axes is not None and axis_size is not None:
            raise ValueError("Cannot specify both in_axes and axis_size.")

        if axis_size is None:
            if in_axes is None:
                raise ValueError("Either axis_size or in_axes must be provided.")
            # _check_no_unwrappables(in_axes)
            from flowjax.bijections.jax_transforms import _infer_axis_size_from_params

            axis_size = _infer_axis_size_from_params(unwrap(bijection), in_axes)

        self.in_axes = (0, in_axes, 0, in_axes_condition)
        self.bijection = bijection
        self.axis_size = axis_size
        self.cond_shape = self.get_cond_shape(in_axes_condition)
        self.mask = mask

    def vmap(self, f: Callable):
        return eqx.filter_vmap(f, in_axes=self.in_axes, axis_size=self.axis_size)

    def transform_and_log_det(self, x, condition=None):
        def _transform_and_log_det(mask, bijection, x, condition):
            y, det = bijection.transform_and_log_det(x, condition)
            return jnp.where(mask, y, x), jnp.where(mask, det, jnp.zeros(()))

        y, log_det = self.vmap(_transform_and_log_det)(
            self.mask, self.bijection, x, condition
        )
        return y, jnp.sum(log_det)

    def inverse_and_log_det(self, y, condition=None):
        def _inverse_and_log_det(mask, bijection, y, condition):
            x, det = bijection.inverse_and_log_det(y, condition)
            return jnp.where(mask, x, y), jnp.where(mask, det, jnp.zeros(()))

        x, log_det = self.vmap(_inverse_and_log_det)(
            self.mask, self.bijection, y, condition
        )
        return x, jnp.sum(log_det)

    @property
    def shape(self):
        return (self.axis_size, *self.bijection.shape)

    def get_cond_shape(self, cond_ax):
        if self.bijection.cond_shape is None or cond_ax is None:
            return self.bijection.cond_shape
        return (
            *self.bijection.cond_shape[:cond_ax],
            self.axis_size,
            *self.bijection.cond_shape[cond_ax:],
        )


class Mask(eqx.Module):
    mask: Array

    def __init__(self, mask: Array):
        assert mask.dtype == jnp.bool_
        self.mask = mask

    def __call__(self, x: Array, *, key=None) -> Array:
        return x * self.mask


class Scan(AbstractBijection):
    """Repeatedly apply the same bijection with different parameter values.

    Internally, uses `jax.lax.scan` to reduce compilation time. Often it is convenient
    to construct these using ``equinox.filter_vmap``.

    Args:
        bijection: A bijection, in which the arrays leaves have an additional leading
            axis to scan over. It is often can convenient to create compatible
            bijections with ``equinox.filter_vmap``.

    Example:
        Below is equivilent to ``Chain([Affine(p) for p in params])``.

        .. doctest::

            >>> from flowjax.bijections import Scan, Affine
            >>> import jax.numpy as jnp
            >>> import equinox as eqx
            >>> params = jnp.ones((3, 2))
            >>> affine = eqx.filter_vmap(Affine)(params)
            >>> affine = Scan(affine)
    """

    bijection: AbstractBijection
    filter_spec: Any = None

    def transform_and_log_det(self, x, condition=None):
        def step(carry, bijection):
            x, log_det = carry
            y, log_det_i = bijection.transform_and_log_det(x, condition)
            return ((y, log_det + log_det_i.sum()), None)

        (y, log_det), _ = _filter_scan(
            step, (x, 0), self.bijection, filter_spec=self.filter_spec
        )
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        def step(carry, bijection):
            y, log_det = carry
            x, log_det_i = bijection.inverse_and_log_det(y, condition)
            return ((x, log_det + log_det_i.sum()), None)

        (y, log_det), _ = _filter_scan(
            step, (y, 0), self.bijection, reverse=True, filter_spec=self.filter_spec
        )
        return y, log_det

    def inverse_gradient_and_val(
        self,
        y: Array,
        y_grad: Array,
        y_logp: Array,
        condition: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        def step(carry, bijection):
            from nutpie.transform_adapter import inverse_gradient_and_val

            carry = inverse_gradient_and_val(bijection, *carry)
            return (carry, None)

        (y, y_grad, y_logp), _ = _filter_scan(
            step,
            (y, y_grad, y_logp),
            self.bijection,
            reverse=True,
            filter_spec=self.filter_spec,
        )
        return y, y_grad, y_logp

    @property
    def shape(self):
        return self.bijection.shape

    @property
    def cond_shape(self):
        return self.bijection.cond_shape


def _filter_scan(f, init, xs, *, reverse=False, filter_spec=None):
    if filter_spec is None:
        filter_spec = eqx.is_array
    params, static = eqx.partition(xs, filter_spec=filter_spec)

    def _scan_fn(carry, x):
        module = eqx.combine(x, static)
        carry, y = f(carry, module)
        return carry, y

    return jax.lax.scan(_scan_fn, init, params, reverse=reverse)


class Coupling(bijections.AbstractBijection):
    """Coupling layer implementation (https://arxiv.org/abs/1605.08803).

    Args:
        key: Jax key
        transformer: Unconditional bijection with shape () to be parameterised by the
            conditioner neural netork. Parameters wrapped with ``NonTrainable``
            are excluded from being parameterized.
        untransformed_dim: Number of untransformed conditioning variables (e.g. dim//2).
        dim: Total dimension.
        cond_dim: Dimension of additional conditioning variables. Defaults to None.
        nn_width: Neural network hidden layer width.
        nn_depth: Neural network hidden layer size.
        nn_activation: Neural network activation function. Defaults to jnn.relu.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    untransformed_dim: int
    dim: int
    transformer_constructor: Callable
    requires_vmap: bool
    conditioner: eqx.nn.MLP | eqx.Module

    def __init__(
        self,
        key,
        *,
        transformer: bijections.AbstractBijection,
        untransformed_dim: int,
        dim: int,
        cond_dim: int | None = None,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jax.nn.relu,
        conditioner: eqx.Module | None = None,
    ):
        if transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers are supported.",
            )
        n_transformed = dim - untransformed_dim
        if n_transformed < 0:
            raise ValueError(
                "The number of untransformed variables must be less than the total "
                "dimension.",
            )
        if transformer.shape != () and transformer.shape != (n_transformed,):
            raise ValueError(
                "The transformer must have shape () or (n_transformed,), "
                f"got {transformer.shape}.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )

        if transformer.shape == ():
            self.requires_vmap = True
            conditioner_output_size = num_params * n_transformed
        else:
            self.requires_vmap = False
            conditioner_output_size = num_params

        self.transformer_constructor = constructor
        self.untransformed_dim = untransformed_dim
        self.dim = dim
        self.shape = (dim,)
        self.cond_shape = (cond_dim,) if cond_dim is not None else None

        if conditioner is None:
            conditioner = eqx.nn.MLP(
                in_size=(
                    untransformed_dim
                    if cond_dim is None
                    else untransformed_dim + cond_dim
                ),
                out_size=conditioner_output_size,
                width_size=nn_width,
                depth=nn_depth,
                activation=nn_activation,
                key=key,
            )
        self.conditioner = conditioner(conditioner_output_size)

    def transform_and_log_det(self, x, condition=None):
        x_cond, x_trans = x[: self.untransformed_dim], x[self.untransformed_dim :]
        nn_input = x_cond if condition is None else jnp.hstack((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        y_trans, log_det = transformer.transform_and_log_det(x_trans)
        y = jnp.hstack((x_cond, y_trans))
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        x_cond, y_trans = y[: self.untransformed_dim], y[self.untransformed_dim :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        x_trans, log_det = transformer.inverse_and_log_det(y_trans)
        x = jnp.hstack((x_cond, x_trans))
        return x, log_det

    def _flat_params_to_transformer(self, params: Array):
        """Reshape to dim X params_per_dim, then vmap."""
        if self.requires_vmap:
            dim = self.dim - self.untransformed_dim
            transformer_params = jnp.reshape(params, (dim, -1))
            transformer = eqx.filter_vmap(self.transformer_constructor)(
                transformer_params
            )
            return bijections.Vmap(transformer, in_axes=eqx.if_array(0))
        else:
            transformer = self.transformer_constructor(params)
            return transformer


class MaskedCoupling(bijections.AbstractBijection):
    """Coupling layer implementation (https://arxiv.org/abs/1605.08803).

    Args:
        key: Jax key
        transformer: Unconditional bijection with shape () to be parameterised by the
            conditioner neural netork. Parameters wrapped with ``NonTrainable``
            are excluded from being parameterized.
        untransformed_dim: Number of untransformed conditioning variables (e.g. dim//2).
        dim: Total dimension.
        cond_dim: Dimension of additional conditioning variables. Defaults to None.
        nn_width: Neural network hidden layer width.
        nn_depth: Neural network hidden layer size.
        nn_activation: Neural network activation function. Defaults to jnn.relu.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    untransformed_mask: Array
    dim: int
    transformer_constructor: Callable
    requires_vmap: bool
    conditioner: eqx.nn.MLP | eqx.Module

    @classmethod
    def conditioner_output_size(cls, dim, transformer):
        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )
        return num_params * dim

    def __init__(
        self,
        key,
        *,
        transformer: bijections.AbstractBijection,
        untransformed_mask: Array,
        dim: int,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jax.nn.relu,
        conditioner: eqx.Module | None = None,
    ):
        if transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers are supported.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )

        assert transformer.shape == ()
        self.requires_vmap = True
        conditioner_output_size = num_params * dim

        self.transformer_constructor = constructor
        self.dim = dim
        self.shape = (dim,)
        self.cond_shape = None
        self.untransformed_mask = untransformed_mask

        if conditioner is None:
            self.conditioner = eqx.nn.Sequential(
                [
                    Mask(untransformed_mask),
                    eqx.nn.MLP(
                        in_size=dim,
                        out_size=conditioner_output_size,
                        width_size=nn_width,
                        depth=nn_depth,
                        activation=nn_activation,
                        key=key,
                    ),
                ]
            )
        else:
            self.conditioner = eqx.nn.Sequential(
                [
                    Mask(untransformed_mask),
                    conditioner,
                ]
            )

    def transform_and_log_det(self, x, condition=None):
        transformer_params = self.conditioner(x.astype(jnp.float32))
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.transform_and_log_det(x)

    def inverse_and_log_det(self, y, condition=None):
        transformer_params = self.conditioner(y.astype(jnp.float32))
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.inverse_and_log_det(y)

    def _flat_params_to_transformer(self, params: Array):
        """Reshape to dim X params_per_dim, then vmap."""
        assert self.requires_vmap

        transformer_params = jnp.reshape(params, (self.dim, -1))
        transformer = eqx.filter_vmap(self.transformer_constructor)(transformer_params)
        return MaskedVmap(
            transformer, ~self.untransformed_mask, in_axes=eqx.if_array(0)
        )


def make_mvscale(key, n_dim, size, randomize_base=False):
    def make_single_hh(key, idx):
        key1, key2 = jax.random.split(key)
        params = jax.random.normal(key1, (n_dim,))
        params = params / jnp.linalg.norm(params)
        mvscale = MvScale(params)
        return mvscale

    keys = jax.random.split(key, size)

    if randomize_base:
        key, key_base = jax.random.split(key)
        indices = jax.random.randint(key_base, (size,), 0, n_dim)
    else:
        indices = [val % n_dim for val in range(size)]

    return bijections.Chain(
        [make_single_hh(key, idx) for key, idx in zip(keys, indices)]
    )


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

    return bijections.Chain(
        [make_single_hh(key, idx) for key, idx in zip(keys, indices)]
    )


def make_elemwise_trafo(key, n_dim, *, count=1, vmap=True):
    def make_elemwise(key, loc):
        key1, key2 = jax.random.split(key)
        scale = Parameterize(lambda x: x + jnp.sqrt(1 + x**2), jnp.zeros(()))
        theta = Parameterize(lambda x: x + jnp.sqrt(1 + x**2), jnp.zeros(()))

        affine = AsymmetricAffine(
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

        return bijections.Invert(affine)

    def make(key):
        keys = jax.random.split(key, count + 1)
        key, keys = keys[0], keys[1:]
        loc = jax.random.normal(key=key, shape=(count,)) * 2
        loc = loc - loc.mean()
        if count == 1:
            return make_elemwise(key, loc[0])
        return bijections.Chain([make_elemwise(key, mu) for key, mu in zip(keys, loc)])

    if vmap:
        keys = jax.random.split(key, n_dim)
        make_affine = eqx.filter_vmap(make, axis_size=n_dim)(keys)
        return bijections.Vmap(make_affine, in_axes=eqx.if_array(0))
    else:
        return make(key)


def make_coupling(key, dim, n_untransformed, *, inner_mvscale=False, **kwargs):
    n_transformed = dim - n_untransformed

    nn_width = kwargs.get("nn_width", None)
    nn_depth = kwargs.get("nn_depth", None)

    if nn_width is None:
        if dim > 128:
            nn_width = (64, 2 * dim)
        else:
            nn_width = 2 * dim

    if nn_depth is None:
        if isinstance(nn_width, int):
            nn_depth = 1
        else:
            nn_depth = len(nn_width)

    transformer = make_elemwise_trafo(key, n_transformed, count=3)

    if inner_mvscale:
        mvscale = make_mvscale(key, n_transformed, 1, randomize_base=True)
        transformer = bijections.Chain([transformer, mvscale])

    def make_mlp(out_size):
        if isinstance(nn_width, tuple):
            out = (nn_width[0], out_size)
        else:
            out = out_size

        return FactoredMLP(
            n_untransformed,
            out,
            nn_width,
            depth=nn_depth,
            key=key,
            dtype=jnp.float32,
            activation=_NN_ACTIVATION,
        )

    return Coupling(
        key,
        transformer=transformer,
        untransformed_dim=n_untransformed,
        dim=dim,
        conditioner=make_mlp,
        **kwargs,
    )


class Add(eqx.Module):
    bias: Array

    def __init__(self, bias):
        self.bias = bias

    def __call__(self, x: Array, *, key=None) -> Array:
        return x + self.bias


def make_flow_scan(
    key,
    n_dim,
    *,
    zero_init=False,
    n_layers,
    nn_width=None,
    nn_depth=None,
    n_embed=None,
    n_deembed=None,
    mvscale=False,
):
    dim = n_dim

    if nn_width is None:
        nn_width = 32
    if n_embed is None:
        n_embed = 2 * nn_width
    if n_deembed is None:
        n_deembed = 2 * nn_width
    if nn_depth is None:
        nn_depth = 1

    def make_transformer():
        elemwises = []
        # loc = bijections.Loc(jnp.zeros(()))
        # elemwises.append(loc)

        for loc in [0.0]:
            scale = Parameterize(lambda x: x + jnp.sqrt(1 + x**2), jnp.zeros(()))
            theta = Parameterize(lambda x: x + jnp.sqrt(1 + x**2), jnp.zeros(()))

            affine = AsymmetricAffine(
                jnp.zeros(()) + loc,
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
            elemwises.append(bijections.Invert(affine))

        if len(elemwises) == 1:
            return elemwises[0]
        return bijections.Chain(elemwises)

    # Just to get at the size
    transformer = make_transformer()
    size = MaskedCoupling.conditioner_output_size(dim, transformer)

    key, key1 = jax.random.split(key)
    embed = eqx.nn.Sequential(
        [
            eqx.nn.Linear(dim, n_embed, key=key1, dtype=jnp.float32, use_bias=True),
            # Activation(_NN_ACTIVATION),
            # eqx.nn.LayerNorm(shape=(n_embed,), dtype=jnp.float32),
        ]
    )
    key, key1 = jax.random.split(key)
    embed_back = eqx.nn.Linear(
        n_deembed, size, key=key1, dtype=jnp.float32, use_bias=True
    )
    embed_back = jax.tree_util.tree_map(
        lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
        embed_back,
    )

    rng = np.random.default_rng(42)  # TODO
    order, counts = _generate_permutations(rng, dim, n_layers)
    mask = order == 0
    mask[...] = False
    for i in range(len(mask)):
        mask[i, order[i, : counts[i]]] = True

    def make_mvscale(key, n_dim):
        params = jax.random.normal(key, (n_dim,))
        params = params / jnp.linalg.norm(params)
        return MvScale(params)

    def make_layer(key, mask, embed, embed_back):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        transformer = make_transformer()
        bias = Add(jax.random.normal(key5, (size,)) * 0.001)
        inner = eqx.nn.MLP(
            n_embed,
            n_deembed,
            width_size=nn_width,
            depth=nn_depth,
            key=key2,
            dtype=jnp.float32,
            activation=_NN_ACTIVATION,
        )
        inner = jax.tree_util.tree_map(
            lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
            inner,
        )

        conditioner = eqx.nn.Sequential(
            [
                embed,
                inner,
                eqx.nn.Sequential(
                    [
                        embed_back,
                        bias,
                    ]
                ),
            ]
        )

        coupling = MaskedCoupling(
            key=key3,
            transformer=transformer,
            untransformed_mask=mask,
            dim=dim,
            conditioner=conditioner,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

        if mvscale:
            scale = make_mvscale(key4, dim)
            return bijections.Chain([coupling, scale])
        else:
            return bijections.Chain([coupling])

    keys = jax.random.split(key, n_layers)

    base = make_layer(key, mask[0], embed, embed_back)
    out_axes = eqx.tree_at(
        lambda tree: tree.bijections[0].conditioner.layers[1].layers[0],
        pytree=base,
        replace=None,
    )
    out_axes = eqx.tree_at(
        lambda tree: tree.bijections[0].conditioner.layers[1].layers[-1].layers[0],
        pytree=out_axes,
        replace=None,
    )
    out_axes = jax.tree.map(lambda leaf: eqx.if_array(0)(leaf), out_axes)

    vectorized = eqx.filter_vmap(
        make_layer, in_axes=(0, 0, None, None), out_axes=out_axes
    )

    vectorize = jax.tree.map(lambda leaf: eqx.is_array(leaf), base)
    vectorize = eqx.tree_at(
        lambda tree: tree.bijections[0].conditioner.layers[1].layers[0],
        pytree=vectorize,
        replace=False,
    )
    vectorize = eqx.tree_at(
        lambda tree: tree.bijections[0].conditioner.layers[1].layers[-1].layers[0],
        pytree=vectorize,
        replace=False,
    )

    return Scan(
        vectorized(keys, mask, embed, embed_back),
        filter_spec=vectorize,
    )


def make_flow_loop(
    key,
    n_dim,
    *,
    zero_init=False,
    householder_layer=False,
    dct_layer=False,
    untransformed_dim: int | list[int | None] | None = None,
    n_layers,
    nn_width=None,
    nn_depth=None,
):
    def make_layer(key, untransformed_dim: int | None, permutation=None):
        key, key_couple, key_permute, key_hh = jax.random.split(key, 4)

        if untransformed_dim is None:
            untransformed_dim = n_dim // 2

        if untransformed_dim < 0:
            untransformed_dim = n_dim + untransformed_dim

        coupling = make_coupling(
            key_couple,
            n_dim,
            untransformed_dim,
            nn_activation=_NN_ACTIVATION,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

        if zero_init:
            coupling = jax.tree_util.tree_map(
                lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
                coupling,
            )

        flow = coupling

        if householder_layer:
            hh = make_hh(key_hh, n_dim, 1, randomize_base=False)
            flow = bijections.Sandwich(flow, hh)

        def add_default_permute(bijection, dim, key):
            if dim == 1:
                return bijection
            if dim == 2:
                outer = bijections.Flip((dim,))
            else:
                outer = bijections.Permute(jax.random.permutation(key, jnp.arange(dim)))

            return bijections.Sandwich(bijection, outer)

        if permutation is None:
            flow = add_default_permute(flow, n_dim, key_permute)
        else:
            flow = bijections.Sandwich(flow, bijections.Permute(permutation))

        mvscale = make_mvscale(key, n_dim, 1, randomize_base=True)

        flow = bijections.Chain(
            [
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
        permutation, lengths = _generate_permutations(rng, n_dim, n_layers)
        layers = []
        for i, (key, p, length) in enumerate(zip(keys, permutation, lengths)):
            layers.append(make_layer(key, int(length), p))
        bijection = bijections.Chain(layers)
    elif isinstance(untransformed_dim, int):
        make_layers = eqx.filter_vmap(make_layer)
        layers = make_layers(keys, untransformed_dim)
        bijection = bijections.Scan(layers)
    else:
        layers = []
        for i, (key, num_untrafo) in enumerate(zip(keys, untransformed_dim)):
            if i % 2 == 0 or not dct_layer:
                layers.append(make_layer(key, num_untrafo))
            else:
                inner = make_layer(key, num_untrafo)
                outer = bijections.DCT(inner.shape)

                layers.append(bijections.Sandwich(inner, outer))

        bijection = bijections.Chain(layers)

    return bijection


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
    nn_width=None,
    nn_depth=None,
    n_embed=None,
    n_deembed=None,
    kind="subset",
    mvscale=False,
):
    positions = np.array(positions)
    gradients = np.array(gradients)

    if len(positions) == 0:
        return

    n_draws, n_dim = positions.shape
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
        return bijections.Chain(flows)

    if kind == "subset":
        inner = make_flow_loop(
            key,
            n_dim,
            zero_init=zero_init,
            householder_layer=householder_layer,
            dct_layer=dct_layer,
            untransformed_dim=untransformed_dim,
            n_layers=n_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
    elif kind == "masked":
        inner = make_flow_scan(
            key,
            n_dim,
            zero_init=zero_init,
            n_layers=n_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            n_embed=n_embed,
            n_deembed=n_deembed,
            mvscale=mvscale,
        )
    else:
        raise ValueError(f"Unknown flow kind: {kind}")
    return bijections.Chain([inner, *flows])


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
    nn_width=None,
    nn_depth=None,
):
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

        affine = bijections.AsymmetricAffine(jnp.zeros(()), jnp.ones(()), jnp.ones(()))

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
            coupling = bijections.coupling.Coupling(
                key,
                transformer=affine,
                untransformed_dim=n_dim - extension_var_trafo_count,
                dim=n_dim,
                nn_activation=_NN_ACTIVATION,
                nn_width=width,
                nn_depth=nn_depth,
            )

            inner_permute = bijections.Permute(
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
            coupling = bijections.coupling.Coupling(
                key,
                transformer=affine,
                untransformed_dim=extension_var_trafo_count,
                dim=n_dim,
                nn_activation=_NN_ACTIVATION,
                nn_width=width,
                nn_depth=nn_depth,
            )

            inner_permute = bijections.Permute(
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

        inner = bijections.Sandwich(coupling, inner_permute)

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
                nn_activation=_NN_ACTIVATION,
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
                bijections.Chain(
                    [
                        bijections.Sandwich(coupling, bijections.Flip(shape=(n_dim,))),
                        inner.inner,
                    ]
                ),
                inner.outer,
            )

    if dct:
        new_layer = bijections.Sandwich(
            bijections.Sandwich(inner, permute),
            bijections.DCT(shape=(n_dim,)),
        )
    else:
        new_layer = bijections.Sandwich(inner, permute)

    scale = Parameterize(
        lambda x: x + jnp.sqrt(1 + x**2),
        jnp.zeros(n_dim),
    )
    affine = eqx.tree_at(
        where=lambda aff: aff.scale,
        pytree=bijections.Affine(jnp.zeros(n_dim), jnp.ones(n_dim)),
        replace=scale,
    )

    pre = []
    if layer % 2 == 0:
        pre.append(bijections.Neg(shape=(n_dim,)))

    nonlin_layer = bijections.Sandwich(
        affine,
        bijections.Chain(
            [
                *pre,
                bijections.Vmap(bijections.SoftPlusX(), axis_size=n_dim),
            ]
        ),
    )
    scale = Parameterize(
        lambda x: x + jnp.sqrt(1 + x**2),
        jnp.zeros(n_dim),
    )
    affine = eqx.tree_at(
        where=lambda aff: aff.scale,
        pytree=bijections.Affine(jnp.zeros(n_dim), jnp.ones(n_dim)),
        replace=scale,
    )
    return bijections.Chain([new_layer, nonlin_layer, affine, base])
