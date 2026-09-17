"""How strongly a fitted triangular flow lets parameters influence each other.

For fixed parents, the flow gives parameter ``i`` a one-dimensional
conditional distribution ``p_theta(y_i)``, where ``theta = net(y_parents)``
are the parameters of its transformer. A change of parent ``j`` moves
``theta`` along ``g_j = d theta / d y_j``. With the Fisher information
``I(theta)`` of the one-dimensional family,

    speed(i <- j) = sqrt(g_j^T I(theta) g_j)

is how fast the conditional distribution of ``i`` changes in the Fisher-Rao
metric when parent ``j`` moves by one unit. It does not depend on how the
transformer is parameterized, so location, scale and shape parameters are
combined consistently. ``I(theta)`` is computed exactly enough with
Gauss-Hermite quadrature in the latent space.

The speed is averaged (as a root mean square) over posterior draws, taken in
the input space of the triangular map, after the flow's diagonal affine
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
    "location": "location only",
    "scale_shape": "scale and shape",
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
    location:
        The same, with only the change of the location, or None if the
        transformer has no location parameter.
    scale_shape:
        The same, with the change of everything but the location.
    order:
        The flow order: ``order[k]`` is the parameter at position ``k``.
    unconstrained_parameters:
        Names of the unconstrained parameters.
    num_draws:
        Number of draws the speeds are averaged over.
    """

    total: sp.csr_array
    location: sp.csr_array | None
    scale_shape: sp.csr_array
    order: np.ndarray
    unconstrained_parameters: list[str]
    num_draws: int

    @property
    def n_dim(self) -> int:
        return len(self.order)

    def plot(
        self,
        kinds=("total", "location", "scale_shape"),
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
            ``"location"`` and ``"scale_shape"``.
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

        kinds = [kind for kind in kinds if getattr(self, kind) is not None]
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


def _find_location_index(conditioner, constructor, num_params):
    """Index of the transformer parameter that is its location, or None."""
    from nutpie.triangular import LocationSkipMlp, _location_index

    if isinstance(conditioner, LocationSkipMlp):
        return conditioner.location_index
    import jax.numpy as jnp

    try:
        transformer = constructor(jnp.zeros((num_params,)))
        return _location_index(transformer, constructor, num_params)
    except (AttributeError, ValueError):
        return None


def fisher_information(constructor, theta, num_nodes=16):
    """Fisher information of the one-dimensional family ``p_theta(y)``,
    where ``x = constructor(theta).inverse(y)`` is standard normal.

    ``log p_theta(y) = log phi(x) + log |dx/dy|``. The expectation over
    ``y ~ p_theta`` is taken with Gauss-Hermite quadrature on ``x``.
    """
    import jax
    import jax.numpy as jnp

    nodes, weights = np.polynomial.hermite_e.hermegauss(num_nodes)
    nodes = jnp.asarray(nodes, dtype=theta.dtype)
    weights = jnp.asarray(weights / weights.sum(), dtype=theta.dtype)

    def log_p(theta, y):
        x, log_det = constructor(theta).inverse_and_log_det(y)
        return -0.5 * x**2 + log_det

    ys = jax.vmap(lambda x: constructor(theta).transform_and_log_det(x)[0])(nodes)
    scores = jax.vmap(jax.grad(log_p), in_axes=(None, 0))(
        theta, jax.lax.stop_gradient(ys)
    )
    return jnp.einsum("k,ki,kj->ij", weights, scores, scores)


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
        Number of Gauss-Hermite nodes for the Fisher information of each
        one-dimensional conditional distribution.
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

    def speeds(net, parents, location_index):
        """Squared speeds per parent slot, for one conditioner and draw."""
        theta = net(parents)
        G = jax.jacfwd(net)(parents)  # (num_params, num_parents)
        info = fisher_information(constructor, theta, num_nodes)
        total = jnp.einsum("pk,pq,qk->k", G, info, G)
        if location_index is None:
            return total, jnp.zeros_like(total), total
        location = info[location_index, location_index] * G[location_index] ** 2
        G_rest = G.at[location_index].set(0.0)
        rest = jnp.einsum("pk,pq,qk->k", G_rest, info, G_rest)
        return total, location, rest

    @eqx.filter_jit
    def bucket_sums(conditioner, parent_values, location_index):
        # vmap over the draws, then over the conditioners of the bucket
        per_draw = jax.vmap(
            lambda values: eqx.filter_vmap(
                lambda net, parents: speeds(net, parents, location_index)
            )(conditioner, values)
        )(parent_values)
        return [part.sum(0) for part in per_draw]

    rows, cols, values = [], [], {kind: [] for kind in KINDS}
    has_location = True
    for bucket, conditioner in enumerate(tmap.conditioners):
        members = np.asarray(tmap.bucket_members[bucket])
        parent_indices = np.asarray(tmap.bucket_parent_indices[bucket])
        if parent_indices.shape[1] == 0:
            continue
        one = jax.tree.map(
            lambda leaf: leaf[0] if eqx.is_array(leaf) else leaf, conditioner
        )
        num_params = int(one(jnp.zeros(parent_indices.shape[1])).shape[0])
        location_index = _find_location_index(conditioner, constructor, num_params)
        has_location &= location_index is not None

        sums = None
        for start in range(0, len(draws), batch_size):
            y = to_map(jnp.asarray(draws[start : start + batch_size]))
            y_padded = jnp.concatenate([y, jnp.zeros((len(y), 1), y.dtype)], axis=1)
            batch = bucket_sums(
                conditioner, y_padded[:, parent_indices], location_index
            )
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
        location=matrix("location") if has_location else None,
        scale_shape=matrix("scale_shape"),
        order=order,
        unconstrained_parameters=names,
        num_draws=len(draws),
    )
