"""How strongly a fitted triangular flow lets parameters influence each other.

For fixed parents, the flow gives parameter ``i`` a one-dimensional
conditional distribution ``p_theta(y_i)``, where ``theta = net(y_parents)``
are the parameters of its transformer ``y = T_theta(x)`` with standard normal
``x``. A change of parent ``j`` moves ``theta`` along
``g_j = d theta / d y_j``. With the Fisher information ``I(theta)`` of the
one-dimensional family,

    speed(i <- j) = sqrt(g_j^T I(theta) g_j)

is how fast the conditional distribution of ``i`` changes in the Fisher-Rao
metric when parent ``j`` moves by one unit. It does not depend on how the
transformer is parameterized.

It is computed in the latent space: the latent velocity of the change is
``V(x) = sum_a g_ja d_a T(x) / T'(x)``, the corresponding score is
``x V(x) - V'(x)`` (a Stein operator, which maps Hermite polynomials
``He_n`` to ``He_{n+1}``), and the speed is its root mean square, with
Gauss-Hermite quadrature over ``x``. This only needs the forward map.

Splitting ``V`` into its even and odd part in ``x`` splits the squared speed
exactly, since the scores of the two parts are odd and even, and so
uncorrelated under the normal distribution:

- even velocities move all latent points the same way or asymmetrically:
  a change of *location and skew*,
- odd velocities stretch symmetrically: a change of *scale and tails*.

The speeds are averaged (as a root mean square) over posterior draws, taken
in the input space of the triangular map, after the flow's diagonal affine
layer. That layer standardizes the parameters roughly, so a speed of 1 means
that moving the parent by about one posterior standard deviation changes the
conditional distribution of the child by about as much as shifting a normal
distribution by one standard deviation.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

KINDS = {
    "total": "Fisher speed",
    "location_skew": "location and skew",
    "scale_tails": "scale and tails",
}


@dataclass(frozen=True, repr=False)
class FlowInfluence:
    """Influence of each parent on the conditional distribution of each
    parameter under a fitted triangular flow, see `fisher_influence`.

    Attributes
    ----------
    total:
        Sparse ``(n_dim, n_dim)`` matrix, ``total[i, j]`` is the RMS Fisher
        speed of the conditional distribution of ``i`` per unit of parent
        ``j``. Only parents of the flow have entries.
    location_skew, scale_tails:
        The parts of the speed from changes of location and skew, and of
        scale and tails. Their squares add up to the square of `total`.
    order:
        The flow order: ``order[k]`` is the parameter at position ``k``.
    unconstrained_parameters:
        Names of the unconstrained parameters.
    num_draws:
        Number of draws the speeds are averaged over.
    """

    total: sp.csr_array
    location_skew: sp.csr_array
    scale_tails: sp.csr_array
    order: np.ndarray
    unconstrained_parameters: list[str]
    num_draws: int

    @property
    def n_dim(self) -> int:
        return len(self.order)

    def plot(
        self,
        kinds=("total", "location_skew", "scale_tails"),
        *,
        variables=None,
        max_variables=10,
        axes=None,
    ):
        """Spy-like plots of the influences, in the flow order.

        Row ``k`` shows the influence of the parents of the parameter at
        position ``k``, so only the lower triangle has entries.

        Parameters
        ----------
        kinds:
            Which influences to show, one panel each: ``"total"``,
            ``"location_skew"`` and ``"scale_tails"``.
        variables:
            The model variable of each unconstrained parameter, e.g.
            ``compiled.factorization.variables``. If given, coloured strips
            along the axes show them.
        max_variables:
            Number of model variables that get their own colour.
        axes:
            Matplotlib axes, one per kind. A new figure is created if None.

        Returns
        -------
        The matplotlib axes.
        """
        import matplotlib.pyplot as plt

        from nutpie.sparsity import _draw_cells, _variable_strips

        if axes is None:
            _, axes = plt.subplots(
                1, len(kinds), figsize=(5.5 * len(kinds), 5), squeeze=False
            )
            axes = axes[0]

        n = self.n_dim
        position = np.empty_like(self.order)
        position[self.order] = np.arange(n)
        handles = []
        for ax, kind in zip(axes, kinds, strict=True):
            coo = getattr(self, kind).tocoo()
            cells = _draw_cells(
                ax, position[coo.col], position[coo.row], array=coo.data
            )
            cells.set_clim(0, max(float(coo.data.max(initial=0)), 1e-12))
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_title(KINDS[kind])
            ax.figure.colorbar(cells, ax=ax, fraction=0.046, pad=0.04)
            if variables is not None:
                handles = _variable_strips(ax, variables, self.order, max_variables)
        if handles:
            # Below the middle panel, under the tick labels and strips
            axes[len(axes) // 2].legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncols=min(len(handles), 6),
            )
        return axes

    def __repr__(self):
        return (
            f"FlowInfluence(n_dim={self.n_dim}, "
            f"edges={self.total.nnz}, draws={self.num_draws})"
        )


def _draws(trace):
    """Unconstrained draws ``(n_draws, n_dim)`` and their names."""
    stats = trace["sample_stats"]
    if "unconstrained_draw" not in stats:
        raise ValueError(
            "The trace has no unconstrained draws. "
            "Sample with `store_unconstrained=True`."
        )
    draws = stats["unconstrained_draw"]
    values = np.asarray(draws.values, dtype=np.float64)
    names = [str(name) for name in draws.coords[draws.dims[-1]].values]
    return values.reshape(-1, values.shape[-1]), names


def _quadrature(num_nodes, dtype):
    """Gauss-Hermite nodes and normalized weights for a standard normal. The
    nodes are symmetric: reversing them negates them."""
    import jax.numpy as jnp

    nodes, weights = np.polynomial.hermite_e.hermegauss(num_nodes)
    return jnp.asarray(nodes, dtype=dtype), jnp.asarray(
        weights / weights.sum(), dtype=dtype
    )


def latent_velocities(constructor, theta, nodes):
    """Latent velocities ``V_a(x) = d_a T(x) / T'(x)`` of the transformer
    parameters and their ``x``-derivatives at `nodes`, each ``(nodes, q)``."""
    import jax

    def T(theta, x):
        return constructor(theta).transform_and_log_det(x)[0]

    def velocity(x):
        return jax.grad(T, argnums=0)(theta, x) / jax.grad(T, argnums=1)(theta, x)

    return jax.vmap(velocity)(nodes), jax.vmap(jax.jacfwd(velocity))(nodes)


def fisher_information(constructor, theta, num_nodes=16):
    """Fisher information of the one-dimensional family ``y = T_theta(x)``,
    ``x`` standard normal: ``E[s_a s_b]`` with the scores
    ``s_a = x V_a(x) - V_a'(x)``, see `latent_velocities`."""
    import jax.numpy as jnp

    nodes, weights = _quadrature(num_nodes, theta.dtype)
    V, dV = latent_velocities(constructor, theta, nodes)
    scores = nodes[:, None] * V - dV
    return jnp.einsum("k,ka,kb->ab", weights, scores, scores)


def fisher_influence(bijection, trace, *, num_nodes=16, batch_size=256):
    """Influence of each parent on each parameter under a fitted flow.

    Parameters
    ----------
    bijection:
        The full flow bijection of a triangular flow, as stored in
        `nutpie.transform_adapter._BIJECTION_TRACE`: diagonal affine layer,
        permutation and `SparseTriangularMap`.
    trace:
        A trace sampled with ``store_unconstrained=True``. Only the
        unconstrained draws in ``sample_stats`` are used.
    num_nodes:
        Number of Gauss-Hermite nodes for each one-dimensional conditional
        distribution.
    batch_size:
        Draws evaluated at once, the memory knob.

    Returns
    -------
    A `FlowInfluence`, see there and the module docstring.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from flowjax import bijections
    from paramax import unwrap

    draws, names = _draws(trace)
    bijection = unwrap(bijection)
    sandwich = bijection.bijections[0].bijections[0]
    affine = bijection.bijections[1]
    tmap = sandwich.inner
    # flowjax stores the permutation as an index tuple ``(array,)``
    order = np.ravel(np.asarray(sandwich.outer.permutation))
    dim = tmap.shape[0]
    if draws.shape[1] != dim:
        raise ValueError(f"The draws have {draws.shape[1]} dimensions, the flow {dim}.")

    # Into the input space of the triangular map, as in `gauss_newton_factors`
    to_map = jax.jit(
        jax.vmap(lambda d: bijections.Invert(sandwich.outer).inverse(affine.inverse(d)))
    )
    constructor = tmap.transformer_constructor

    def speeds(net, parents):
        """Squared speeds per parent slot, for one conditioner and draw:
        total, even (location and skew) and odd (scale and tails) part."""
        nodes, weights = _quadrature(num_nodes, parents.dtype)
        theta = net(parents)
        G = jax.jacfwd(net)(parents)  # (num_params, num_parents)
        V, dV = latent_velocities(constructor, theta, nodes)
        V, dV = V @ G, dV @ G  # per parent slot, (nodes, num_parents)
        # The nodes are symmetric, so reversing them gives V(-x).
        even = (V + V[::-1]) / 2
        odd = (V - V[::-1]) / 2
        d_even = (dV - dV[::-1]) / 2
        d_odd = (dV + dV[::-1]) / 2
        x = nodes[:, None]
        even_part = weights @ (x * even - d_even) ** 2
        odd_part = weights @ (x * odd - d_odd) ** 2
        return even_part + odd_part, even_part, odd_part

    @eqx.filter_jit
    def bucket_sums(conditioner, parent_values):
        # vmap over the draws, then over the conditioners of the bucket
        per_draw = jax.vmap(
            lambda values: eqx.filter_vmap(speeds)(conditioner, values)
        )(parent_values)
        return [part.sum(0) for part in per_draw]

    rows, cols, values = [], [], {kind: [] for kind in KINDS}
    for bucket, conditioner in enumerate(tmap.conditioners):
        members = np.asarray(tmap.bucket_members[bucket])
        parent_indices = np.asarray(tmap.bucket_parent_indices[bucket])
        if parent_indices.shape[1] == 0:
            continue

        sums = None
        for start in range(0, len(draws), batch_size):
            y = to_map(jnp.asarray(draws[start : start + batch_size]))
            y_padded = jnp.concatenate([y, jnp.zeros((len(y), 1), y.dtype)], axis=1)
            batch = bucket_sums(conditioner, y_padded[:, parent_indices])
            sums = batch if sums is None else [s + b for s, b in zip(sums, batch)]

        # Padded parent slots read a constant and have no influence.
        real = parent_indices < dim
        child = np.broadcast_to(members[:, None], parent_indices.shape)
        rows.append(order[child[real]])
        cols.append(order[parent_indices[real]])
        for kind, total in zip(KINDS, sums, strict=True):
            values[kind].append(np.sqrt(np.asarray(total)[real] / len(draws)))

    def matrix(kind):
        data = np.concatenate(values[kind]) if values[kind] else np.zeros(0)
        row = np.concatenate(rows) if rows else np.zeros(0, int)
        col = np.concatenate(cols) if cols else np.zeros(0, int)
        return sp.csr_array((data, (row, col)), shape=(dim, dim))

    return FlowInfluence(
        total=matrix("total"),
        location_skew=matrix("location_skew"),
        scale_tails=matrix("scale_tails"),
        order=order,
        unconstrained_parameters=names,
        num_draws=len(draws),
    )
