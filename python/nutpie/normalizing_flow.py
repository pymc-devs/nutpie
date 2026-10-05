import itertools
import math
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import equinox as eqx
import flowjax.distributions
import flowjax.flows
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
from equinox.nn import Linear
from flowjax import bijections
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.coupling import get_ravelled_pytree_constructor
from flowjax.utils import arraylike_to_array
from jaxtyping import Array, ArrayLike, PyTree
from paramax import NonTrainable, Parameterize, unwrap
from paramax.wrappers import AbstractUnwrappable

from nutpie.triangular import SparseTriangularMap


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
    in_size: int | Literal["scalar"] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    width_size: tuple[int, ...] = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
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
                lambda x: x[i] if eqx.is_array(x) else x,  # noqa: B023
                act,
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


def _scale_last_layer(mlp, scale):
    """Return a copy of an MLP-like conditioner (``FactoredMLP`` or
    ``eqx.nn.MLP``) with only its output layer scaled down."""
    last = jax.tree_util.tree_map(
        lambda x: x * scale if eqx.is_inexact_array(x) else x, mlp.layers[-1]
    )
    return eqx.tree_at(lambda m: m.layers[-1], mlp, last)


def zero_init_conditioners(bijection, scale=1e-3):
    """Shrink a freshly-initialized bijection towards the identity by
    scaling down only the *output* layer of each conditioner MLP it
    contains, leaving hidden layers at their normal initialization scale.

    This replaces naively scaling every parameter in the bijection by
    ``scale``: doing that shrinks every layer of a conditioner's MLP, so the
    signal (and gradient) passing through an ``nn_depth``-layer network gets
    attenuated roughly like ``scale ** nn_depth``, which can leave training
    with essentially no usable gradient to start from. Scaling only the
    final layer keeps the network at (near-)identity output while hidden
    layers, and thus the gradients flowing back through them, stay at their
    normal scale.
    """
    is_mlp = lambda x: isinstance(x, (FactoredMLP, eqx.nn.MLP))
    return jax.tree_util.tree_map(
        lambda leaf: _scale_last_layer(leaf, scale) if is_mlp(leaf) else leaf,
        bijection,
        is_leaf=is_mlp,
    )


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


class Householder(AbstractBijection):
    """A Householder reflection.

    A linear transformation reflecting vectors across a hyperplane defined by a normal
    vector (params). The transformation is its own inverse and volume-preserving
    (determinant = -1). Given a unit vector :math:`v`, the transformation is
    :math:`y = x - 2(x^T v)v`.

    It is often desirable to stack multiple such transforms (e.g. up to the
    dimensionality of the data):

    .. doctest::

        >>> from flowjax.bijections import Householder, Scan
        >>> import jax.random as jr
        >>> import equinox as eqx
        >>> import jax.numpy as jnp

        >>> dim = 5
        >>> keys = jr.split(jr.key(0), dim)
        >>> householder_stack = Scan(
        ...    eqx.filter_vmap(lambda key: Householder(jr.normal(key, dim)))(keys)
        ... )

    Args:
        params: Normal vector defining the reflection hyperplane. The vector is
            normalized in the transformation, so scaling params will have no effect
            on the bijection.
    """

    shape: tuple[int, ...]
    params: Array
    cond_shape = None

    def __init__(self, params: ArrayLike):
        params = arraylike_to_array(params)
        if params.ndim != 1:
            raise ValueError("params must be a vector.")
        self.shape = params.shape
        self.params = params

    def _householder(self, x: Array) -> Array:
        unit_vec = self.params / jnp.linalg.norm(self.params)
        return x - 2 * unit_vec * (x @ unit_vec)

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        return self._householder(x), jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        return self._householder(y), jnp.zeros(())


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
        _constructor, num_params = get_ravelled_pytree_constructor(
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
        transformer_params = self.conditioner(x.astype(jnp.float32)).astype(jnp.float64)
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.transform_and_log_det(x)

    def inverse_and_log_det(self, y, condition=None):
        transformer_params = self.conditioner(y.astype(jnp.float32)).astype(jnp.float64)
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
        params = jax.random.normal(key, (n_dim,))
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
        params = jax.random.normal(key, (n_dim,)) * 1e-3
        params = params.at[idx].set(1.0)
        return Householder(params)

    keys = jax.random.split(key, size)

    if randomize_base:
        key, key_base = jax.random.split(key)
        indices = jax.random.randint(key_base, (size,), 0, n_dim)
    else:
        indices = [val % n_dim for val in range(size)]

    if size == 1:
        return make_single_hh(keys[0], indices[0])

    make_single_hh_vec = eqx.filter_vmap(make_single_hh, axis_size=size)(keys, indices)
    return bijections.Scan(make_single_hh_vec)

    chain = bijections.Chain(
        [make_single_hh(key, idx) for key, idx in zip(keys, indices)]
    )
    if len(chain.bijections) == 1:
        return chain.bijections[0]
    return chain


def make_elemwise_trafo(key, n_dim, *, count=1, vmap=True):
    def make_elemwise(key, loc):
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


def make_coupling(
    key, dim, n_untransformed, *, activation, inner_mvscale=False, **kwargs
):
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
            activation=activation,
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


class UnconstrainedAffine(bijections.AbstractBijection):
    loc: Array
    unconstrained_scale: Array
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None = None

    def __init__(self, loc, unconstrained_scale):
        self.loc = loc
        self.unconstrained_scale = unconstrained_scale
        self.shape = loc.shape

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        scale = self.unconstrained_scale + jnp.sqrt(1 + self.unconstrained_scale**2)
        y = self.loc + scale * x
        return y, jnp.sum(jnp.log(scale))

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        scale = self.unconstrained_scale + jnp.sqrt(1 + self.unconstrained_scale**2)
        x = (y - self.loc) / scale
        return x, -jnp.sum(jnp.log(scale))


def pairwise_rotation(x, thetas):
    """
    Applies a rotation to each consecutive pair in x.

    Parameters:
      x (jnp.ndarray): 1D array containing the values to be rotated.
      thetas (jnp.ndarray): 1D array of angles (in radians) for each pair.
                            Length must equal x.shape[0] // 2.

    Returns:
      jnp.ndarray: The rotated vector where each pair (x[2*i], x[2*i+1])
                   is rotated by the corresponding angle thetas[i]. If x has
                   an odd length, the last element is unchanged.
    """
    n = x.shape[0]
    num_pairs = n // 2

    # Reshape the first 2*num_pairs elements into pairs
    x_pairs = x[: num_pairs * 2].reshape(num_pairs, 2)

    # Compute cosine and sine of each rotation angle for the pairs
    cos_thetas = jnp.cos(thetas)
    sin_thetas = jnp.sin(thetas)

    # Apply the rotation to each pair without forming a 2x2 matrix:
    # rotated_x = x * cos(theta) - y * sin(theta)
    # rotated_y = x * sin(theta) + y * cos(theta)
    rotated_x = x_pairs[:, 0] * cos_thetas - x_pairs[:, 1] * sin_thetas
    rotated_y = x_pairs[:, 0] * sin_thetas + x_pairs[:, 1] * cos_thetas

    # Stack the rotated coordinates and flatten back into a 1D array
    rotated_pairs = jnp.stack([rotated_x, rotated_y], axis=1)
    y_rotated = rotated_pairs.reshape(-1)

    # If x has an odd length, append the last unchanged element.
    if n % 2 == 1:
        y = jnp.concatenate([y_rotated, x[num_pairs * 2 :]])
    else:
        y = y_rotated

    return y


class Rotations(bijections.AbstractBijection):
    theta: Array
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None = None

    def __init__(self, key, ndim):
        n_rotations = ndim // 2
        self.theta = jax.random.normal(key, (n_rotations,)) / 10
        self.shape = (ndim,)

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        return pairwise_rotation(x, self.theta), jnp.zeros(())

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        return pairwise_rotation(y, -self.theta), jnp.zeros(())


class Orthogonal(bijections.AbstractBijection):
    theta: Array
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None = None

    def __init__(self, key, ndim, k):
        self.theta = jax.random.normal(key, (ndim, k))
        self.shape = (ndim,)

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        q, _ = jnp.linalg.qr(self.theta, mode="reduced")
        q = q.T
        qx = q @ x
        return x - 2 * q.T @ qx, jnp.zeros(())

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        q, _ = jnp.linalg.qr(self.theta, mode="reduced")
        q = q.T
        qy = q @ y
        return y - 2 * q.T @ qy, jnp.zeros(())


class Planar(bijections.AbstractBijection):
    u: Array

    # One dimensional transformation (assumed to operate on shape (..., 1))
    inner: bijections.AbstractBijection

    # The full input shape (e.g. (ndim,))
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None = None

    def __init__(self, key, inner, ndim: int):
        self.inner = inner
        self.shape = (ndim,)
        # Initialize u as a random vector of shape (ndim,)
        self.u = jax.random.normal(key, (ndim,))

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        # Normalize u to have unit norm.
        u = self.u / jnp.linalg.norm(self.u)
        # Compute the scalar projection d = <x, u>.
        d = x @ u
        # Apply the one-dimensional bijection to d.
        f_d, logdet_inner = self.inner.transform_and_log_det(d, condition)
        # Lift the 1D transformation to the full space:
        # y = x + (f(d) - d) * u
        y = x + (f_d - d) * u
        return y, logdet_inner

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        # Normalize u.
        u = self.u / jnp.linalg.norm(self.u)
        # Compute the projected coordinate from y.
        d_y = y @ u
        # Invert the inner bijection to get d.
        d, logdet_inner = self.inner.inverse_and_log_det(d_y, condition)
        # Invert the full transformation:
        # x = y + (d - f(d)) * u, but note that f(d)=d_y.
        x = y + (d - d_y) * u
        # The log–determinant of the inverse is the negative of the forward.
        return x, -logdet_inner


def _log_cosh(v):
    """log(cosh(v)), stable for large |v|."""
    a = jnp.abs(v)
    return a + jnp.log1p(jnp.exp(-2.0 * a)) - jnp.log(2.0)


def _bounded_log_gamma(unbounded, low, high):
    """Squash ``unbounded`` into ``(low, high)``, fixing 0 -> 0.

    A shifted, scaled logistic, chosen so that ``f(0) = 0`` and ``f'(0) = 1``:
    the map is unchanged to first order around the identity, so zero-init and
    anything calibrated against the unbounded parameterization still see the
    same local scaling. It
    is ``C^inf`` and strictly monotone, and handles asymmetric bounds --
    ``low < 0 < high`` is required, since ``log gamma = 0`` is the identity
    and has to stay reachable in the interior.
    """
    width = high - low
    at_zero = -low / width  # sigmoid(offset), so that f(0) == 0
    offset = jnp.log(at_zero / (1.0 - at_zero))
    slope = width / (-low * high)  # so that f'(0) == 1
    return low + width * jax.nn.sigmoid(slope * unbounded + offset)


class Contract2(bijections.AbstractBijection):
    shape: tuple[int, ...]
    alpha: Array | None
    beta: Array
    sigma: Array
    mu: Array
    nu: Array
    cond_shape: tuple[int, ...] | None = None
    log_gamma_bounds: tuple[float, float] | None = eqx.field(static=True, default=None)

    def __init__(self, alpha, beta, sigma, mu, nu, log_gamma_bounds=None):
        """
        Args:
            log_gamma_bounds: Optional ``(low, high)`` on ``log gamma``, with
                ``low < 0 < high``. ``alpha`` itself stays unconstrained; it is
                ``log gamma = asinh(alpha)`` that is squashed. Bounds are in
                logs because ``gamma`` is a tail *exponent* -- ``T(x) ~
                x**gamma`` -- so it composes multiplicatively along a chain and
                additively in logs, and a symmetric bound ``(-b, b)`` means
                "at most ``e**b`` times heavier or lighter tails". Unbounded,
                a ``gamma`` fitted on one window's draws maps the next
                window's slightly wider draws to astronomical values. Defaults
                to unbounded, reproducing the previous behaviour exactly.
        """
        if log_gamma_bounds is not None:
            low, high = log_gamma_bounds
            if not low < 0.0 < high:
                raise ValueError(
                    "log_gamma_bounds must satisfy low < 0 < high, got "
                    f"{log_gamma_bounds}."
                )
            log_gamma_bounds = (float(low), float(high))
        self.log_gamma_bounds = log_gamma_bounds
        if alpha is not None:
            self.alpha = jnp.array(alpha)
        else:
            self.alpha = None
        if beta is not None:
            self.beta = jnp.array(beta)
        else:
            self.beta = None
        if sigma is not None:
            self.sigma = jnp.array(sigma)
        else:
            self.sigma = None
        if mu is not None:
            self.mu = jnp.array(mu)
        else:
            self.mu = None
        if nu is not None:
            self.nu = jnp.array(nu)
        else:
            self.nu = None
        self.shape = ()

    def _log_params(self):
        """log gamma, log delta, log sigma_mod.

        The original computes gamma = alpha + sqrt(1 + alpha**2), which is
        exp(asinh(alpha)); taking the log directly is exact everywhere.

        Both `transform_and_log_det` and `inverse_and_log_det` route through
        here, so `log_gamma_bounds` constrains the two directions consistently.
        """
        if self.alpha is not None:
            log_gamma = jnp.arcsinh(self.alpha)
            if self.log_gamma_bounds is not None:
                log_gamma = _bounded_log_gamma(log_gamma, *self.log_gamma_bounds)
        else:
            log_gamma = jnp.zeros(())
        if self.beta is not None:
            log_beta = jnp.arcsinh(self.beta)
        else:
            log_beta = jnp.zeros(())
        if self.sigma is not None:
            log_sigma = jnp.arcsinh(self.sigma)
        else:
            log_sigma = jnp.zeros(())

        return log_gamma, log_beta, log_sigma

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Forward transformation:

            T(x) = (2*sigma_mod/gamma) * sinh(gamma*u + 2*log(delta)) + mu,
            u    = asinh((x - nu)/2),

        which is the original expression with z = exp(u) and
        delta**2 * z**gamma - delta**(-2) * z**(-gamma) = 2*sinh(gamma*u + 2*log(delta)).

            log T'(x) = log(sigma_mod) + logcosh(gamma*u + 2*log(delta)) - logcosh(u)
        """
        log_gamma, log_delta, log_sigma = self._log_params()
        gamma = jnp.exp(log_gamma)

        if self.nu is not None:
            centred = x - self.nu
        else:
            centred = x
        u = jnp.arcsinh(centred / 2)
        arg = gamma * u + 2.0 * log_delta

        y = 2.0 * jnp.exp(log_sigma - log_gamma) * jnp.sinh(arg)

        if self.mu is not None:
            y = y + self.mu
        log_det = log_sigma + _log_cosh(arg) - _log_cosh(u)
        return y, log_det

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Inverse transformation.

        With A = gamma*(y - mu)/sigma_mod, the forward relation is
        sinh(gamma*u + 2*log(delta)) = A/2, so

            gamma*u + 2*log(delta) = asinh(A/2),
            x = 2*sinh(u) + nu.

        asinh(A/2) replaces (A + sqrt(A**2 + 4))/2, which cancels for A << 0.
        """
        log_gamma, log_delta, log_sigma = self._log_params()
        gamma = jnp.exp(log_gamma)

        if self.nu is None:
            nu = jnp.zeros(())
        else:
            nu = self.nu

        if self.mu is None:
            mu = jnp.zeros(())
        else:
            mu = self.mu

        half_a = jnp.exp(log_gamma - log_sigma) * (y - mu) / 2.0
        arg = jnp.arcsinh(half_a)  # == gamma*u + 2*log(delta)
        u = (arg - 2.0 * log_delta) / gamma

        x = 2.0 * jnp.sinh(u) + nu
        # logcosh(asinh(half_a)) = 0.5*log1p(half_a**2)
        log_det = _log_cosh(u) - log_sigma - 0.5 * jnp.log1p(half_a * half_a)
        return x, log_det


def _optional(value):
    return None if value is None else jnp.array(value)


def _positive(x):
    """``x + sqrt(1 + x**2)``: positive, and 1 at zero."""
    return x + jnp.sqrt(1 + x**2)


class TangentSAS(bijections.AbstractBijection):
    """Tangent sinh-arcsinh: one sinh-arcsinh layer pinned to the identity to
    first order at its centre ``nu``,

        S(u) = [sinh(c asinh(b u) / b + eps) - sinh(eps)] / (c cosh(eps)),
        y    = nu + S(x - nu),

    so ``S(0) = 0`` and ``S'(0) = 1``. ``b`` sets where the tails start,
    ``r = c / b`` is the tail exponent (``S(u) ~ u**r``), and ``eps`` the
    skew. With ``c = r b`` the argument is ``r asinh(b u) + eps``.

    The fields are unconstrained: ``b = sigmoid(b_raw)`` in ``(0, 1)`` and
    ``r = r_raw + sqrt(1 + r_raw**2)`` positive. A field that is `None` stays
    at its value for zero, and all zeros give the identity.
    """

    shape: tuple[int, ...]
    nu: Array | None
    eps: Array | None
    b: Array | None
    r: Array | None
    cond_shape: tuple[int, ...] | None = None

    def __init__(self, nu, eps, b, r):
        self.nu = _optional(nu)
        self.eps = _optional(eps)
        self.b = _optional(b)
        self.r = _optional(r)
        self.shape = ()

    def _params(self):
        """nu, eps, b, r."""
        zero = jnp.zeros(())
        nu = zero if self.nu is None else self.nu
        eps = zero if self.eps is None else self.eps
        b = jax.nn.sigmoid(zero if self.b is None else self.b)
        r = jnp.ones(()) if self.r is None else _positive(self.r)
        return nu, eps, b, r

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """``log S'(u) = logcosh(arg) - logcosh(eps) - 0.5 log1p((b u)**2)``."""
        nu, eps, b, r = self._params()
        w = b * (x - nu)
        arg = r * jnp.arcsinh(w) + eps
        y = nu + (jnp.sinh(arg) - jnp.sinh(eps)) / (r * b * jnp.cosh(eps))
        log_det = _log_cosh(arg) - _log_cosh(eps) - 0.5 * jnp.log1p(w * w)
        return y, log_det

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """With ``q = sinh(arg) = r b cosh(eps) (y - nu) + sinh(eps)``:
        ``a = asinh(b u) = (asinh(q) - eps) / r`` and ``x = nu + sinh(a) / b``.
        In the log det, ``logcosh(asinh(q)) = 0.5 log1p(q**2)`` and
        ``0.5 log1p((b u)**2) = logcosh(a)``."""
        nu, eps, b, r = self._params()
        q = r * b * jnp.cosh(eps) * (y - nu) + jnp.sinh(eps)
        a = (jnp.arcsinh(q) - eps) / r
        x = nu + jnp.sinh(a) / b
        log_det = _log_cosh(a) + _log_cosh(eps) - 0.5 * jnp.log1p(q * q)
        return x, log_det


class PositiveAffine(bijections.AbstractBijection):
    """``y = loc + scale_mod * x`` with ``scale_mod = scale + sqrt(1 +
    scale**2)``, so ``scale`` is unconstrained and zero is the identity."""

    shape: tuple[int, ...]
    loc: Array
    scale: Array
    cond_shape: tuple[int, ...] | None = None

    def __init__(self, loc, scale):
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)
        self.shape = ()

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        # log(scale_mod) == asinh(scale)
        return self.loc + _positive(self.scale) * x, jnp.arcsinh(self.scale)

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        return (y - self.loc) / _positive(self.scale), -jnp.arcsinh(self.scale)


class DipBij(bijections.AbstractBijection):
    b: jnp.ndarray  # raw parameter (scalar)
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None = None

    def __init__(self):
        # Store the parameter b (a scalar). Then a = sigmoid(b) ∈ (0,1).
        self.b = jnp.zeros(())
        self.shape = self.b.shape
        # We expect b to be scalar.
        assert self.shape == (), "Parameter b must be a scalar."

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        # Compute a = sigmoid(b)
        a = jnp.tanh(self.b)

        # Define the forward transformation.
        def f(x):
            return x - (a * x) / (1 + x**2)

        # Use jax.jvp to compute the derivative of f at x.
        y, tangent = jax.jvp(f, (x,), (jnp.ones_like(x),))
        # For a 1d transformation the log-det is just the log of the absolute derivative.
        logdet = jnp.sum(jnp.log(jnp.abs(tangent)))
        return y, logdet

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        # Compute a = sigmoid(b)
        a = jnp.tanh(self.b)

        # The forward map is: f(x) = x - (a*x)/(1+x**2).
        # Its inverse is given by solving for x in
        #   x - (a*x)/(1+x^2) = y,
        # which can be rearranged to:
        #   x^3 - y*x^2 + (1 - a)*x - y = 0.
        # We solve this cubic by first shifting: let x = z + y/3.
        m = y / 3
        # The depressed cubic is: z^3 + P*z + Q = 0, with
        P = (1 - a) - y**2 / 3
        Q = -2 * y**3 / 27 - (a + 2) * y / 3
        # Compute the discriminant:
        delta = (Q / 2) ** 2 + (P / 3) ** 3
        # Cardano's formula for the real solution:
        z = jnp.cbrt(-Q / 2 + jnp.sqrt(delta)) + jnp.cbrt(-Q / 2 - jnp.sqrt(delta))
        x = z + m

        # To compute the log-det for the inverse, note that it is the negative of
        # the forward log-det. We compute f'(x) via jax.jvp.
        def f(x):
            return x - (a * x) / (1 + x**2)

        _, fprime = jax.jvp(f, (x,), (jnp.ones_like(x),))
        logdet = -jnp.sum(jnp.log(jnp.abs(fprime)))
        return x, logdet


class Contract(bijections.AbstractBijection):
    alpha: Array
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None = None

    def __init__(self, alpha):
        self.alpha = jnp.array(alpha)
        self.shape = alpha.shape
        assert self.shape == ()

    def transform_and_log_det(
        self, x: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        beta = jax.scipy.special.expit(self.alpha)

        def trafo(x):
            z = 2 * jnp.asinh(x / 2)
            return jnp.sinh(beta * z) / beta

        y, det = jax.jvp(trafo, [x], [jnp.ones(())])
        return y, jnp.sum(jnp.log(det))

    def inverse_and_log_det(
        self, y: ArrayLike, condition: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        beta = jax.scipy.special.expit(self.alpha)

        def trafo(y):
            z = jnp.asinh(beta * y) / beta / 2
            return 2 * jnp.sinh(z)

        x, det = jax.jvp(trafo, [y], [jnp.ones(())])
        return x, jnp.sum(jnp.log(det))


class Activation(eqx.Module):
    fn: Callable

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.fn(*args)


class ElementwiseTransformer(bijections.AbstractBijection):
    """A `make_transformer` chain that knows which parameter is its location.

    The location is a parameter entering the forward map purely additively,
    ``transform(x) = T0(x) + location``, so in the density direction it is the
    shift ``y - location`` applied directly to the model-space value.
    Conditioners use it to route a linear function of the parents straight to
    the conditional mean (see `SparseTriangularMap`).

    `location_field` is ``(layer, field)``: the location is attribute `field`
    of ``chain.bijections[layer]``. `None` if the chain has no such parameter.
    """

    chain: bijections.Chain
    location_field: tuple[int, str] | None = eqx.field(static=True)
    shape: tuple[int, ...] = ()
    cond_shape: tuple[int, ...] | None = None

    def location(self):
        if self.location_field is None:
            return None
        layer, field = self.location_field
        return getattr(self.chain.bijections[layer], field)

    def transform_and_log_det(self, x, condition=None):
        return self.chain.transform_and_log_det(x, condition)

    def inverse_and_log_det(self, y, condition=None):
        return self.chain.inverse_and_log_det(y, condition)


def make_transformer(
    affine_transformer=False,
    contract_transformer=True,
    asymmetric_transformer=True,
    log_gamma_bounds=None,
    tangent_sas_transformer=0,
):
    """Elementwise transformer as a chain of the requested layers.

    Each argument is a *count*; ``True``/``False`` still work as 1/0. Layers
    are chained in the order affine -> asymmetric -> contract -> tangent SAS,
    and tangent SAS layers are followed by one `PositiveAffine`.

    Parameters the chain makes redundant are disabled here rather than left
    for the optimizer to find. Two additive shifts applied back to back
    compose, so only their difference is identified: `Contract2` ends with
    ``+ mu`` and starts with ``- nu``, so in a stack of ``n`` every interior
    ``mu`` is exactly redundant with the following ``nu``. Keeping ``nu``
    throughout and dropping all but the last ``mu`` gives ``4n + 1``
    parameters at full rank.

    The alternative of dropping ``nu`` everywhere is also full rank at ``4n``,
    but measurably weaker: on 1-d targets it cost ~566x on Student-t5 and
    ~48x on Student-t3 against the ``4n + 1`` form, which itself matches an
    unrestricted ``5n`` chain to two significant figures (and at ``n = 3``
    beats it -- removing exactly-flat directions helps the optimizer, the
    same effect `lmopt.MARQUARDT_FLOOR` guards against).

    A leading `bijections.Affine` also ends in a shift, so it makes the first
    `Contract2`'s ``nu`` redundant in the same way and that one is dropped
    too. The inverted `AsymmetricAffine` does *not*: inverting turns its
    output shift into an input shift, leaving a following ``nu`` identified.

    Args:
        affine_transformer: Number of leading affine layers. More than one is
            pointless -- consecutive affines compose to a single affine.
        contract_transformer: Number of `Contract2` (sinh-arcsinh) layers.
            ``alpha``, ``beta`` and ``sigma`` stay free on every layer: each
            layer needs its own shape parameters, and concentrating them in
            one layer forfeits most of what stacking buys (measured ~32x on
            Student-t3, and a collapse from 3.5e-2 to 2.8 on a bimodal
            target). ``alpha`` in particular sets the tail exponent
            ``gamma``, which multiplies along the chain -- pin it to 1 and
            the whole stack is asymptotically linear however deep it is.
        asymmetric_transformer: Number of `AsymmetricAffine` layers, applied
            inverted, with initial locations spread evenly over ``[-2, 2]``
            (so a count of 3 reproduces the previous ``[-2, 0, 2]``).
        log_gamma_bounds: Optional ``(low, high)`` bound on each `Contract2`
            layer's ``log gamma``, see that class. Note the bound is per
            layer and log gamma adds along a chain, so ``n`` layers bounded at
            ``high`` reach ``n * high`` overall.
        tangent_sas_transformer: Number of `TangentSAS` layers. If nonzero,
            they are followed by a `PositiveAffine`, which holds the location.

    Returns:
        An `ElementwiseTransformer`, whose location is the last layer's output
        shift: the trailing `PositiveAffine`'s ``loc``, the last `Contract2`'s
        ``mu``, or the `Affine`'s ``loc`` when that is the only layer kind.
        None if the chain ends in an inverted `AsymmetricAffine`.
    """
    n_affine = int(affine_transformer)
    n_contract = int(contract_transformer)
    n_asymmetric = int(asymmetric_transformer)
    n_tangent_sas = int(tangent_sas_transformer)

    if min(n_affine, n_contract, n_asymmetric, n_tangent_sas) < 0:
        raise ValueError("transformer counts must be non-negative.")
    if n_affine + n_contract + n_asymmetric + n_tangent_sas == 0:
        raise ValueError("make_transformer needs at least one layer.")

    elemwises = []

    for _ in range(n_affine):
        affine = bijections.Affine(jnp.zeros(()), jnp.ones(()))
        scale = Parameterize(lambda x: x + jnp.sqrt(1 + x**2), jnp.zeros(()))
        affine = eqx.tree_at(
            where=lambda aff: aff.scale,
            pytree=affine,
            replace=scale,
        )
        elemwises.append(affine)

    if n_asymmetric:
        locs = (
            [0.0] if n_asymmetric == 1 else list(np.linspace(-2.0, 2.0, n_asymmetric))
        )
        for loc in locs:
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

    for index in range(n_contract):
        is_last = index == n_contract - 1
        # A preceding affine already ends in a free shift, so this layer's own
        # input shift would only ever enter as the difference of the two.
        shift_already_available = index == 0 and n_affine > 0
        elemwises.append(
            Contract2(
                alpha=jnp.zeros(()),
                beta=jnp.zeros(()),
                sigma=jnp.zeros(()),
                mu=jnp.zeros(()) if is_last else None,
                nu=None if shift_already_available else jnp.zeros(()),
                log_gamma_bounds=log_gamma_bounds,
            )
        )

    if n_tangent_sas:
        for _ in range(n_tangent_sas):
            elemwises.append(
                TangentSAS(
                    nu=jnp.zeros(()),
                    eps=jnp.zeros(()),
                    b=jnp.zeros(()),
                    r=jnp.zeros(()),
                )
            )
        elemwises.append(PositiveAffine(jnp.zeros(()), jnp.zeros(())))

    # The location is the last layer's output shift, which nothing after it
    # undoes. `Contract2` ends in ``+ mu`` and the affines in ``+ loc``; the
    # inverted `AsymmetricAffine` has only an *input* shift, so a chain ending
    # in one has no location.
    last = elemwises[-1]
    if isinstance(last, PositiveAffine):
        location_field = (len(elemwises) - 1, "loc")
    elif isinstance(last, Contract2):
        location_field = (len(elemwises) - 1, "mu")
    elif isinstance(last, bijections.Affine):
        location_field = (len(elemwises) - 1, "loc")
    else:
        location_field = None
    return ElementwiseTransformer(bijections.Chain(elemwises), location_field)


def make_twin_flow_scan(
    key,
    n_dim,
    *,
    zero_init=False,
    n_layers,
    nn_width=None,
    nn_depth=None,
    num_householder=1,
    affine_transformer=False,
    contract_transformer=True,
    asymmetric_transformer=True,
    activation,
):
    if nn_width is None:
        nn_width = 32
    if nn_depth is None:
        nn_depth = 1

    def make_layer(key):
        keys = jax.random.split(key, 4)

        def make_coupling(key):
            transformer = make_transformer(
                affine_transformer=affine_transformer,
                contract_transformer=contract_transformer,
                asymmetric_transformer=asymmetric_transformer,
            )

            coupling = bijections.Coupling(
                key=key,
                transformer=transformer,
                untransformed_dim=n_dim // 2,
                dim=n_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=activation,
            )

            if zero_init:
                coupling = zero_init_conditioners(coupling)
            return coupling

        layers = []

        if num_householder > 0:
            layers.append(make_hh(keys[0], n_dim, num_householder, randomize_base=True))

        # layers.append(Rotations(keys[0], n_dim))

        # layers.append(Orthogonal(keys[0], n_dim, 8))

        inner = Contract2(
            jnp.zeros(()),
            jnp.zeros(()),
            jnp.zeros(()),
            jnp.zeros(()),
            jnp.zeros(()),
        )
        layers.append(Planar(keys[0], inner, n_dim))

        layers.append(make_coupling(keys[1]))

        # layers.append(bijections.Flip((n_dim,)))
        # layers.append(make_coupling(keys[2]))

        permutation = jax.random.permutation(keys[3], n_dim)
        layers.append(bijections.Permute(permutation))

        return bijections.Chain(layers)

    keys = jax.random.split(key, n_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    return bijections.Scan(layers)


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
    num_householder=1,
    twin_layers=False,
    affine_transformer=False,
    contract_transformer=True,
    asymmetric_transformer=True,
    sandwich_householder=False,
    activation,
    reuse_embed=True,
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

    # Just to get at the size
    transformer = make_transformer(
        affine_transformer=affine_transformer,
        contract_transformer=contract_transformer,
        asymmetric_transformer=asymmetric_transformer,
    )
    size = MaskedCoupling.conditioner_output_size(dim, transformer)

    key, key1 = jax.random.split(key)
    embed = eqx.nn.Sequential(
        [
            eqx.nn.Linear(dim, n_embed, key=key1, dtype=jnp.float32, use_bias=True),
            eqx.nn.LayerNorm(shape=(n_embed,), dtype=jnp.float32),
        ]
    )
    key, key1 = jax.random.split(key)
    embed_back = eqx.nn.Linear(
        n_deembed, size, key=key1, dtype=jnp.float32, use_bias=False
    )
    embed_back = jax.tree_util.tree_map(
        lambda x: x * 1e-3 if eqx.is_inexact_array(x) else x,
        embed_back,
    )

    key, key1 = jax.random.split(key)
    seeds = jax.random.randint(key1, (4,), 0, 1 << 31 - 1)
    rng = np.random.default_rng([int(seed) for seed in seeds])
    order, counts = _generate_permutations(rng, dim, n_layers)
    mask = order == 0
    mask[...] = False
    for i in range(len(mask)):
        mask[i, order[i, : counts[i]]] = True

    if False:
        if n_layers >= 12 and dim > 2:
            mask[n_layers // 2, :] = True
            mask[n_layers // 2, 0] = False
            mask[n_layers // 2 + 1, :] = False
            mask[n_layers // 2 + 1, 0] = True
            mask[n_layers // 2 + 2, :] = True
            mask[n_layers // 2 + 2, -1] = False
            mask[n_layers // 2 + 3, :] = False
            mask[n_layers // 2 + 3, -1] = True

    if twin_layers:
        interleaved = np.empty((mask.shape[0] * 2, mask.shape[1]), dtype=mask.dtype)
        interleaved[0::2] = mask
        interleaved[1::2] = ~mask
        mask = interleaved
        n_layers = len(mask)

    def make_mvscale(key, n_dim):
        params = jax.random.normal(key, (n_dim,))
        params = params / jnp.linalg.norm(params)
        return MvScale(params)

    def make_layer(key, mask, embed, embed_back):
        _key1, key2, key3, key4, _key5 = jax.random.split(key, 5)
        transformer = make_transformer(
            affine_transformer=affine_transformer,
            contract_transformer=contract_transformer,
            asymmetric_transformer=asymmetric_transformer,
        )
        inner = eqx.nn.MLP(
            n_embed,
            n_deembed,
            width_size=nn_width,
            depth=nn_depth,
            key=key2,
            dtype=jnp.float32,
            activation=activation,
        )
        inner = jax.tree_util.tree_map(
            lambda x: x * 1e-2 if eqx.is_inexact_array(x) else x,
            inner,
        )

        conditioner = eqx.nn.Sequential(
            [
                embed,
                inner,
                eqx.nn.Sequential(
                    [
                        embed_back,
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

        if num_householder == 0:
            return bijections.Chain([coupling])
        if sandwich_householder:
            hh = make_hh(key4, dim, num_householder, randomize_base=True)
            return bijections.Sandwich(coupling, hh)
        else:
            hh = make_hh(key4, dim, num_householder, randomize_base=True)
            inner = Contract2(
                jnp.zeros(()),
                jnp.zeros(()),
                jnp.zeros(()),
                jnp.zeros(()),
                jnp.zeros(()),
            )
            inner = bijections.Chain([inner, DipBij()])
            planar = Planar(key4, inner, n_dim)
            return bijections.Chain([coupling, hh, planar])

    keys = jax.random.split(key, n_layers)

    base = make_layer(key, mask[0], embed, embed_back)

    if sandwich_householder:

        def select_coupling(tree):
            return tree.inner
    else:

        def select_coupling(tree):
            return tree.bijections[0]

    if reuse_embed:
        out_axes = eqx.tree_at(
            lambda tree: select_coupling(tree).conditioner.layers[1].layers[0],
            pytree=base,
            replace=None,
        )

        out_axes = eqx.tree_at(
            lambda tree: (
                select_coupling(tree).conditioner.layers[1].layers[-1].layers[0]
            ),
            pytree=out_axes,
            replace=None,
        )
        out_axes = jax.tree.map(eqx.if_array(0), out_axes)
    else:
        out_axes = jax.tree.map(eqx.if_array(0), base)

    vectorized = eqx.filter_vmap(
        make_layer, in_axes=(0, 0, None, None), out_axes=out_axes
    )

    vectorize = jax.tree.map(eqx.is_array, base)

    if reuse_embed:
        vectorize = eqx.tree_at(
            lambda tree: select_coupling(tree).conditioner.layers[1].layers[0],
            pytree=vectorize,
            replace=False,
        )
        vectorize = eqx.tree_at(
            lambda tree: (
                select_coupling(tree).conditioner.layers[1].layers[-1].layers[0]
            ),
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
    activation,
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
            nn_activation=activation,
            nn_width=nn_width,
            nn_depth=nn_depth,
            activation=activation,
        )

        if zero_init:
            coupling = zero_init_conditioners(coupling)

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

    key, _key_permute = jax.random.split(key)
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


def make_sparse_triangular_map(
    key,
    n_dim,
    *,
    order: ArrayLike,
    sparsity: ArrayLike,
    zero_init=True,
    n_buckets=8,
    n_level_segments=8,
    nn_width=None,
    nn_depth=None,
    activation,
    location_skip=True,
    affine_transformer=False,
    contract_transformer=False,
    asymmetric_transformer=False,
    tangent_sas_transformer=0,
    feature_degree=None,
):
    """Build a `SparseTriangularMap` bijection for the given ordering.

    Unlike coupling layers, a single triangular map already gives every
    variable an arbitrarily flexible, invertible conditional transform given
    its parents, so (unlike the other flows in this module) there is no
    benefit to stacking several of these with the same `order` and
    `sparsity`: doing so would not add any expressivity, only cost. `order`
    is applied by sandwiching the (order-less) `SparseTriangularMap` between
    a `bijections.Permute` and its inverse, rather than baking the ordering
    into the triangular map itself.

    Args:
        order: Permutation of ``range(n_dim)`` giving the variable ordering
            used for the triangular structure. ``order[k]`` is the index (in
            the original, unpermuted variable space) of the variable
            transformed at position ``k``; a variable can only depend on
            variables earlier in this order.

            If `sparsity` comes from a symbolic Cholesky factorization, this
            must be the **reverse** of the elimination order that factorization
            used (``order = p[::-1]`` for a CHOLMOD/AMD permutation ``p``),
            because the map is lower triangular and so factorizes the
            precision as ``Lambda = C^T C`` rather than ``L L^T``. See
            `SparseTriangularMap` for the full argument. Reversing costs no
            fill: the fill count is the one the elimination order achieved.
        sparsity: ``(n_dim, n_dim)`` dense or scipy sparse matrix
            convertible to boolean, the Markov-blanket adjacency matrix, see
            `SparseTriangularMap`.
            ``sparsity[i, j]`` being truthy means ``j`` may be used to
            parameterize the transform of ``i``, provided ``j`` precedes
            ``i`` in ``order``.
        location_skip: Linear map from each variable's parents straight to
            its transformer location, see `SparseTriangularMap`.
        affine_transformer, contract_transformer, asymmetric_transformer,
        tangent_sas_transformer:
            Layer counts of the elementwise transformer, see
            `make_transformer`. If all are zero, `SparseTriangularMap`'s own
            default transformer is used.
        feature_degree: Degree of the `HermiteFeatures` the conditioners see
            instead of the raw parents, see `SparseTriangularMap`. None for
            the raw parents.
    """
    if nn_width is None:
        nn_width = 16
    if nn_depth is None:
        nn_depth = 1

    order = np.asarray(order)
    if scipy.sparse.issparse(sparsity):
        # The triangular map still works with a dense pattern for now
        sparsity = sparsity.toarray()
    sparsity = np.asarray(sparsity, dtype=bool)

    if order.shape != (n_dim,):
        raise ValueError(f"order must have shape ({n_dim},), got {order.shape}.")
    if not np.array_equal(np.sort(order), np.arange(n_dim)):
        raise ValueError("order must be a permutation of range(n_dim).")
    if sparsity.shape != (n_dim, n_dim):
        raise ValueError(
            f"sparsity must have shape {(n_dim, n_dim)}, got {sparsity.shape}."
        )

    # Reindex the sparsity pattern into the "sorted" position space given by
    # `order`, so that `SparseTriangularMap` (which assumes variable `i`
    # precedes variable `j` whenever `i < j`) can be used unchanged.
    sparsity_sorted = sparsity[np.ix_(order, order)]

    transformer = None
    if (
        affine_transformer
        or contract_transformer
        or asymmetric_transformer
        or tangent_sas_transformer
    ):
        transformer = make_transformer(
            affine_transformer=affine_transformer,
            contract_transformer=contract_transformer,
            asymmetric_transformer=asymmetric_transformer,
            tangent_sas_transformer=tangent_sas_transformer,
        )

    layer = SparseTriangularMap(
        key,
        blanket=sparsity_sorted,
        transformer=transformer,
        n_buckets=n_buckets,
        n_level_segments=n_level_segments,
        nn_width=nn_width,
        nn_depth=nn_depth,
        nn_activation=activation,
        location_skip=location_skip,
        feature_degree=feature_degree,
    )
    if zero_init:
        layer = zero_init_conditioners(layer)

    layer = bijections.Sandwich(layer, bijections.Permute(jnp.asarray(order)))

    """
    contract = eqx.filter_vmap(
        lambda: Contract2(jnp.zeros(()), jnp.zeros(()), jnp.zeros(()), jnp.zeros(()), jnp.zeros(())),
        #lambda: Contract2(None, None, None, None, jnp.zeros(())),
        axis_size=len(order),
    )()
    contract = bijections.Vmap(contract, in_axes=eqx.if_array(0))
    #contract = bijections.Invert(contract)
    """
    return bijections.Chain([layer])


def _pattern_lower_indices(pattern):
    """Row/column indices of the lower triangle (incl. diagonal) of `pattern`."""
    pattern = np.asarray(pattern, dtype=bool)
    dim = pattern.shape[0]
    mask = np.tril(pattern | pattern.T | np.eye(dim, dtype=bool))
    return np.nonzero(mask)


def fisher_optimal_precision(
    draws,
    grads,
    pattern,
    *,
    maxiter: int = 200,
    tol: float = 1e-10,
):
    """Precision matrix of the Fisher-optimal linear map with a given sparsity.

    A linear triangular map ``w = C s`` has whitened Fisher divergence

    .. code-block:: text

        E ||C s + C^-T g||^2 = tr(M Sigma) + tr(M^-1 G) - 2 dim,   M = C^T C

    (the cross term is constant because ``C^T C^-T = I`` and
    ``E[g s^T] = -I``). So the loss sees ``C`` only through ``M = C^T C``,
    and because `pattern` is fill-completed for the map's order, ``{C^T C : C
    lower triangular with this pattern}`` is exactly ``{M positive definite
    with this pattern}``. The problem is therefore *convex* in ``M``:
    ``tr(M Sigma)`` is linear, ``tr(M^-1 G)`` is convex, and the objective
    diverges as ``M`` approaches singularity, so it is self-barriering.

    Note what this is *not*. The tempting cheap alternative -- regressing
    ``-g`` on ``s`` row by row, i.e. minimizing ``E||M s + g||^2`` -- expands
    to ``tr(M Sigma M) - 2 tr(M) + tr(G)``, whose minimizer is ``M =
    Sigma^-1``: the score covariance drops out entirely and the result is the
    draw covariance in disguise. The stationarity condition here is instead
    ``M Sigma M = G``, whose unconstrained solution is the matrix geometric
    mean of ``Sigma^-1`` and ``G``.

    Nothing dense is ever formed. The gradient is
    ``P[Sigma - M^-1 G M^-1]``, which in sample form is a difference of two
    empirical second moments evaluated only on the pattern,

    .. code-block:: text

        grad_ij = mean_k [ s_ki s_kj - y_ki y_kj ],    M y_k = g_k

    so an iteration costs one sparse solve per draw plus ``O(n_draws * nnz)``
    to accumulate the moments. With an empty pattern this reduces to
    ``M_ii = sqrt(G_ii / Sigma_ii)``, the diagonal geometric mean `make_flow`
    already uses.

    The map has to be affine rather than merely linear, because the whitened
    residual of a Gaussian with mean ``mu`` is ``C (s - mu)``: without an
    intercept it is off by a constant whenever the coordinates the map sees
    are not exactly centered, which is the normal case (`make_flow`'s
    preceding affine layer only centers approximately). That costs nothing
    to handle. Writing the map as ``w = C (s - c)`` and minimizing over
    ``c`` gives ``c = mean(s) + M^-1 mean(g)``; substituting it back
    collapses every ``c``-dependent term into ``-mean(g)^T M^-1 mean(g)``,
    leaving exactly the objective above with **both** moments centered. So
    the intercept is profiled out rather than iterated on: use covariances
    instead of second moments, solve once, then read ``c`` off the result.

    Args:
        draws: ``(n_draws, dim)`` draws, in the order the map uses.
        grads: ``(n_draws, dim)`` gradients of the target log density at
            `draws`, in the same order.
        pattern: ``(dim, dim)`` boolean adjacency matrix, fill-completed for
            the map's order (see `make_sparse_triangular_map`).
        maxiter: Maximum number of L-BFGS iterations.
        tol: Gradient tolerance for the L-BFGS convergence check.

    Returns:
        ``(M, center)``, with ``M`` a ``scipy.sparse`` CSC matrix and
        ``center`` the offset the map subtracts, i.e. the optimal linear map
        is ``w = C (s - center)``.
    """
    import scipy.sparse as sp
    from scipy.optimize import minimize
    from scipy.sparse.linalg import splu

    draws = np.asarray(draws, dtype=np.float64)
    grads = np.asarray(grads, dtype=np.float64)
    n_draws, dim = draws.shape

    rows, cols = _pattern_lower_indices(pattern)
    is_diag = rows == cols
    # Off-diagonal entries appear twice in the symmetric matrix, so their
    # directional derivative picks up a factor of two.
    grad_weight = np.where(is_diag, 1.0, 2.0)

    # Centered, because the intercept is profiled out (see above).
    draw_mean = draws.mean(0)
    grad_mean = grads.mean(0)
    centered_draws = draws - draw_mean
    centered_grads = grads - grad_mean

    # Empirical covariance, evaluated only on the pattern.
    sigma_vals = (centered_draws[:, rows] * centered_draws[:, cols]).sum(0) / n_draws

    def to_matrix(theta):
        lower = sp.coo_matrix((theta, (rows, cols)), shape=(dim, dim))
        strict = sp.coo_matrix(
            (theta[~is_diag], (cols[~is_diag], rows[~is_diag])), shape=(dim, dim)
        )
        return (lower + strict).tocsc()

    def objective(theta):
        matrix = to_matrix(theta)
        try:
            # `diag_pivot_thresh=0` turns SuperLU into a Cholesky-like
            # factorization for symmetric positive definite input; a
            # non-positive pivot then means we left the feasible set.
            factor = splu(matrix, diag_pivot_thresh=0, permc_spec="MMD_AT_PLUS_A")
        except RuntimeError:
            return np.inf, np.zeros_like(theta)
        if not (factor.U.diagonal() > 0).all():
            return np.inf, np.zeros_like(theta)

        y = factor.solve(centered_grads.T).T
        value = (centered_draws * (centered_draws @ matrix)).sum() / n_draws
        value = value + (centered_grads * y).sum() / n_draws

        y_vals = (y[:, rows] * y[:, cols]).sum(0) / n_draws
        return value, (sigma_vals - y_vals) * grad_weight

    # Start from the diagonal geometric mean, which is the exact solution
    # when the pattern is empty and a feasible (positive definite) point
    # otherwise.
    diag0 = np.sqrt(centered_grads.var(0) / np.maximum(centered_draws.var(0), 1e-300))
    theta0 = np.where(is_diag, diag0[rows], 0.0)

    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "gtol": tol, "ftol": 1e-15},
    )
    matrix = to_matrix(result.x)

    # c = mean(s) + M^-1 mean(g), the offset that makes the whitened residual
    # mean-free.
    factor = splu(matrix, diag_pivot_thresh=0, permc_spec="MMD_AT_PLUS_A")
    center = draw_mean + factor.solve(grad_mean)
    return matrix, center


def reverse_cholesky(matrix):
    """Lower-triangular ``C`` with ``C.T @ C == matrix``.

    This is the factorization a `SparseTriangularMap` needs (see that class's
    note on the triangle convention), as opposed to the usual ``L @ L.T``.
    It is computed as an ordinary Cholesky of the reversed matrix: reversing
    a lower-triangular factor gives an upper-triangular one, and
    ``(X[J][:, J]).T == X.T[J][:, J]``.
    """
    import scipy.sparse as sp
    from sksparse.cholmod import cholesky

    dim = matrix.shape[0]
    rev = np.arange(dim)[::-1]
    reversed_matrix = sp.csc_matrix(matrix)[rev][:, rev]
    # `order="natural"` is essential: any fill-reducing permutation here
    # would destroy the ordering the triangular map is built around.
    factor, perm = cholesky(sp.csc_matrix(reversed_matrix), order="natural", lower=True)
    assert np.array_equal(perm, np.arange(dim))
    return sp.csc_matrix(factor).T[rev][:, rev]


def make_flow(
    seed,
    positions,
    gradients,
    *,
    zero_init=True,
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
    num_householder=1,
    twin_layers=False,
    affine_transformer=False,
    contract_transformer=False,
    asymmetric_transformer=False,
    tangent_sas_transformer=0,
    sandwich_householder=False,
    activation=None,
    reuse_embed=False,
    order: ArrayLike | None = None,
    sparsity: ArrayLike | None = None,
    location_skip: bool = True,
    feature_degree: int | None = None,
    n_buckets: int = 8,
):
    if activation is None:
        activation = jax.nn.leaky_relu
    if activation == "gelu":
        activation = jax.nn.gelu
    if activation == "relu":
        activation = jax.nn.relu
    if activation == "leaky_relu":
        activation = jax.nn.leaky_relu
    if activation == "tanh":
        activation = jnp.tanh
    if activation == "sigmoid":
        activation = jax.nn.sigmoid

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
        diag = np.clip(1 / jnp.sqrt(jnp.abs(gradients[0])), 1e-8, 1e8)
        assert np.isfinite(diag).all()
        mean = jnp.zeros_like(diag)
    else:
        pos_std = np.clip(positions.std(0), 1e-8, 1e8)
        grad_std = np.clip(gradients.std(0), 1e-8, 1e8)
        diag = jnp.sqrt(pos_std / grad_std)
        mean = positions.mean(0) + gradients.mean(0) * diag * diag

    key = jax.random.key(seed % (2**63), impl="threefry2x32")

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
            activation=activation,
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
            num_householder=num_householder,
            twin_layers=twin_layers,
            activation=activation,
            affine_transformer=affine_transformer,
            contract_transformer=contract_transformer,
            asymmetric_transformer=asymmetric_transformer,
            reuse_embed=reuse_embed,
            sandwich_householder=sandwich_householder,
        )
    elif kind == "flowjax_coupling":
        base_dist = flowjax.distributions.StandardNormal((n_dim,))
        if nn_width is None:
            nn_width = 32
        if nn_depth is None:
            nn_depth = 1

        if contract_transformer:
            transformer = Contract2(
                jnp.zeros(()),
                jnp.zeros(()),
                jnp.zeros(()),
                jnp.zeros(()),
                jnp.zeros(()),
            )
        else:
            transformer = None

        inner = flowjax.flows.coupling_flow(
            key,
            base_dist=base_dist,
            flow_layers=n_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            transformer=transformer,
            nn_activation=activation,
        )
        inner = inner.bijection
    elif kind == "twin":
        inner = make_twin_flow_scan(
            key,
            n_dim,
            zero_init=zero_init,
            n_layers=n_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            num_householder=num_householder,
            affine_transformer=affine_transformer,
            contract_transformer=contract_transformer,
            activation=activation,
        )
    elif kind == "triangular":
        if sparsity is None:
            raise ValueError(
                "kind='triangular' requires a `sparsity` argument "
                "(a boolean Markov-blanket adjacency matrix of shape "
                "(n_dim, n_dim))."
            )
        if order is None:
            order = np.arange(n_dim)
        inner = make_sparse_triangular_map(
            key,
            n_dim,
            order=order,
            sparsity=sparsity,
            zero_init=zero_init,
            nn_width=nn_width,
            nn_depth=nn_depth,
            activation=activation,
            location_skip=location_skip,
            affine_transformer=affine_transformer,
            contract_transformer=contract_transformer,
            asymmetric_transformer=asymmetric_transformer,
            tangent_sas_transformer=tangent_sas_transformer,
            feature_degree=feature_degree,
            n_buckets=n_buckets,
        )
        if feature_degree is not None:
            # The marginal maps of the features, fitted on the draws as the
            # triangular map sees them: standardized, then in its order.
            from nutpie.triangular import fit_marginal_maps

            standardized = ((positions - np.asarray(mean)) / np.asarray(diag))[
                :, np.asarray(order)
            ]
            where = lambda chain: chain.bijections[0].inner
            inner = eqx.tree_at(
                where,
                inner,
                where(inner).with_marginal_maps(fit_marginal_maps(standardized)),
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
    activation,
):
    _n_draws, n_dim = positions.shape

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
                nn_activation=activation,
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
                nn_activation=activation,
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
            coupling = zero_init_conditioners(coupling)

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
                nn_activation=activation,
                nn_width=width,
                nn_depth=nn_depth,
            )

            if zero_init:
                coupling = zero_init_conditioners(coupling)

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
