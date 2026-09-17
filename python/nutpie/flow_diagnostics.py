"""Diagnostics for fitted triangular flows.

- `fisher_influence`: how strongly the flow lets parameters influence each
  other.
- `conditioner_capacity`: how much of each conditioner's hidden layer the
  flow actually uses, to tell whether ``nn_width`` is too small or larger
  than necessary.

Neither depends on how the transformer is parameterized. The influence is
measured in the Fisher-Rao metric, which splits exactly into the
conditionals; the capacity in the Fisher divergence the flow is fitted to,
so it is in the units of the ``log F`` the fit reports.

Influence
---------

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

Capacity
--------

Each conditioner is ``theta = W h(y_parents) + b``, plus a linear skip to the
location, where ``h`` is its last hidden layer. Gating the hidden units,
``theta = W diag(1 + g) h + b``, makes removing or merging units a change of
``theta`` that is linear in ``g``, with the constant part absorbed by ``b``.
Its effect on the Fisher divergence ``F`` of the fit is ``g^T G g`` to
second order, with ``G`` the Gauss-Newton curvature of ``F`` in the gates,
from the exact Gauss-Newton factors of the LM fit. The eigenvalues of ``G``,
after projecting out what the linear skip already does, are how much ``F``
rises when an independent direction of the hidden layer is dropped: small
ones are width the flow does not use.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
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
        Names of the unconstrained parameters, None if the draws were given
        as an array.
    num_draws:
        Number of draws the speeds are averaged over.
    """

    total: sp.csr_array
    location_skew: sp.csr_array
    scale_tails: sp.csr_array
    order: np.ndarray
    unconstrained_parameters: list[str] | None
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
        max_labels=40,
        axes=None,
    ):
        """Spy-like plots of the influences, in the flow order.

        Row ``k`` shows the influence of the parents of the parameter at
        position ``k``, so only the lower triangle has entries. The axes are
        labelled with the names of the unconstrained parameters if there are
        at most `max_labels` of them.

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
        max_labels:
            Largest number of parameters whose names label the axes.
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
        labels = _labels(self.unconstrained_parameters, n)[self.order]
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
            if self.unconstrained_parameters is not None and n <= max_labels:
                ax.set_xticks(np.arange(n), labels, rotation=90, fontsize="small")
                ax.set_yticks(np.arange(n), labels, fontsize="small")
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


def _labels(names, n_dim) -> pd.Index:
    """Index of the unconstrained parameters: their names, or positions if
    the draws had none."""
    if names is None:
        return pd.RangeIndex(n_dim, name="unconstrained_parameter")
    return pd.Index(names, name="unconstrained_parameter")


def _draws(trace):
    """Unconstrained draws ``(n_draws, n_dim)`` and their names.

    `trace` may also be an array of draws, which have no names.
    """
    if hasattr(trace, "shape"):
        values = np.asarray(trace, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(
                f"Expected draws of shape (n_draws, n_dim), got {values.shape}."
            )
        return values, None
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


def _triangular_map(bijection, draws):
    """The `SparseTriangularMap` of a triangular flow, the flow order
    (``order[k]`` is the parameter at position ``k``), and a jitted map of
    draws into the input space of the triangular map."""
    import jax
    from flowjax import bijections
    from paramax import unwrap

    bijection = unwrap(bijection)
    sandwich = bijection.bijections[0].bijections[0]
    affine = bijection.bijections[1]
    tmap = sandwich.inner
    # flowjax stores the permutation as an index tuple ``(array,)``
    order = np.ravel(np.asarray(sandwich.outer.permutation))
    dim = tmap.shape[0]
    if draws.shape[1] != dim:
        raise ValueError(f"The draws have {draws.shape[1]} dimensions, the flow {dim}.")

    # As in `gauss_newton_factors`
    to_map = jax.jit(
        jax.vmap(lambda d: bijections.Invert(sandwich.outer).inverse(affine.inverse(d)))
    )
    return tmap, order, to_map


def _parent_batches(tmap, bucket, draws, to_map, batch_size):
    """Parent values ``(draws, conditioners, parent slots)`` of one bucket,
    `batch_size` draws at a time. Padded parent slots read 0."""
    import jax.numpy as jnp

    parent_indices = np.asarray(tmap.bucket_parent_indices[bucket])
    for start in range(0, len(draws), batch_size):
        y = to_map(jnp.asarray(draws[start : start + batch_size]))
        y_padded = jnp.concatenate([y, jnp.zeros((len(y), 1), y.dtype)], axis=1)
        yield y_padded[:, parent_indices]


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
        unconstrained draws in ``sample_stats`` are used. Or an array
        ``(n_draws, n_dim)`` of unconstrained draws, e.g. a window's
        positions from `_BIJECTION_TRACE`.
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

    draws, names = _draws(trace)
    tmap, order, to_map = _triangular_map(bijection, draws)
    dim = tmap.shape[0]
    constructor = tmap.transformer_constructor

    def speeds(net, parents, parent_indices):
        """Squared speeds per parent slot, for one conditioner and draw:
        total, even (location and skew) and odd (scale and tails) part."""
        nodes, weights = _quadrature(num_nodes, parents.dtype)
        condition = lambda p: tmap._condition(net, p, parent_indices)
        theta = condition(parents)
        G = jax.jacfwd(condition)(parents)  # (num_params, num_parents)
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
    def bucket_sums(conditioner, parent_values, parent_indices):
        # vmap over the draws, then over the conditioners of the bucket
        per_draw = jax.vmap(
            lambda values: eqx.filter_vmap(speeds)(
                conditioner, values, parent_indices
            )
        )(parent_values)
        return [part.sum(0) for part in per_draw]

    rows, cols, values = [], [], {kind: [] for kind in KINDS}
    for bucket, conditioner in enumerate(tmap.conditioners):
        members = np.asarray(tmap.bucket_members[bucket])
        parent_indices = np.asarray(tmap.bucket_parent_indices[bucket])
        if parent_indices.shape[1] == 0:
            continue

        sums = None
        for parents in _parent_batches(tmap, bucket, draws, to_map, batch_size):
            batch = bucket_sums(conditioner, parents, parent_indices)
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


@dataclass(frozen=True, repr=False)
class ConditionerCapacity:
    """How much of each conditioner's last hidden layer a fitted triangular
    flow uses, see `conditioner_capacity`.

    Attributes
    ----------
    eigenvalues:
        DataFrame with a row per unconstrained parameter (its conditioner)
        and a column per ``hidden_direction``, in decreasing order.
        Eigenvalue ``mu`` is an independent direction of the hidden layer,
        beyond the linear location skip, whose removal raises the Fisher
        divergence ``F`` of the fit by about ``mu``.
    num_parents:
        Series, number of parents of each unconstrained parameter.
    order:
        The flow order: ``order[k]`` is the parameter at position ``k``.
    num_draws:
        Number of draws the Gram matrices are averaged over.

    The index is ``unconstrained_parameter``, with the names from the trace,
    or positions if the draws were given as an array.
    """

    eigenvalues: pd.DataFrame
    num_parents: pd.Series
    order: np.ndarray
    num_draws: int

    @property
    def n_dim(self) -> int:
        return len(self.order)

    @property
    def width(self) -> int:
        return self.eigenvalues.shape[1]

    def effective_width(self, threshold=1e-3) -> pd.Series:
        """Number of hidden directions per conditioner whose removal raises
        the Fisher divergence ``F`` of the fit by more than `threshold`.
        Compare it with the ``F`` the fit reaches, e.g. ``lm_min_loss``
        (``exp(-3) ~ 0.05`` by default).

        About equal to `width` for many conditioners: the capacity is fully
        used, and the width may be too small. Well below it for most: the
        width is larger than necessary.
        """
        return (self.eigenvalues > threshold).sum(axis=1).rename("effective_width")

    def plot(self, threshold=1e-3, *, highlight=5, axes=None):
        """Eigenvalue spectrum of every conditioner, and a histogram of the
        effective widths at `threshold`.

        The `highlight` conditioners that use the most of their width are
        drawn in colour and named in the legend.

        Returns
        -------
        The two matplotlib axes.
        """
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(11, 4))
        spectra, counts = axes
        rank = self.eigenvalues.columns
        floor = threshold * 1e-4
        effective = self.effective_width(threshold)
        # Most used first: effective width, then the total of the spectrum
        busiest = (
            pd.DataFrame({"width": effective, "total": self.eigenvalues.sum(axis=1)})
            .sort_values(["width", "total"], ascending=False)
            .index[:highlight]
        )
        for name, eigenvalues in self.eigenvalues.iterrows():
            if name not in busiest:
                spectra.plot(
                    rank, np.maximum(eigenvalues, floor), color="0.6", alpha=0.3
                )
        for i, name in enumerate(busiest):
            spectra.plot(
                rank,
                np.maximum(self.eigenvalues.loc[name], floor),
                color=f"C{i + 1}",
                label=f"{name} ({self.num_parents[name]} parents)",
            )
        spectra.axhline(threshold, color="C0", linestyle="--", label="threshold")
        spectra.set_yscale("log")
        spectra.set_xlabel("hidden direction")
        spectra.set_ylabel("rise of F when dropped")
        spectra.set_title("conditioner spectra")
        spectra.legend(fontsize="small")

        counts.hist(effective, bins=np.arange(self.width + 2) - 0.5)
        counts.set_xlabel(f"effective width (of {self.width})")
        counts.set_ylabel("conditioners")
        counts.set_title("effective width")
        return axes

    def __repr__(self):
        effective = self.effective_width()
        return (
            f"ConditionerCapacity(n_dim={self.n_dim}, width={self.width}, "
            f"draws={self.num_draws}, effective width median "
            f"{effective.median():g}, max {effective.max()})"
        )


def _unit_features(tmap, net):
    """Features of one conditioner of `tmap` at its parent values and
    indices: its last hidden layer, then the parent values that feed the
    location skip."""
    import equinox as eqx
    import jax.numpy as jnp

    from nutpie.triangular import LocationSkipMlp

    mlp = net.mlp if isinstance(net, LocationSkipMlp) else net
    if mlp.depth == 0:
        raise ValueError("The conditioners have no hidden layer (nn_depth=0).")
    hidden_fn = eqx.tree_at(lambda m: m.layers[-1], mlp, eqx.nn.Identity())

    def features(parents, parent_indices):
        inputs = tmap.mlp_inputs(parents, parent_indices)
        hidden = hidden_fn(parents if inputs is None else inputs)
        if isinstance(net, LocationSkipMlp):
            return jnp.concatenate([hidden, parents])
        return hidden

    return features


def _gate_directions(net, mean):
    """``(n_params, units)``: the change of the conditioner's weights, in
    `ravel_pytree` order as in `gauss_newton_factors`, when each unit's
    output is scaled by ``1 + g``. The bias absorbs the unit's mean output,
    so the directions only change how the output varies with the parents."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from nutpie.triangular import LocationSkipMlp

    arrays, _ = eqx.partition(net, eqx.is_inexact_array)
    skip = isinstance(net, LocationSkipMlp)
    mlp = (lambda n: n.mlp) if skip else (lambda n: n)
    last = mlp(net).layers[-1]
    width = last.weight.shape[1]

    def tangent(gates):
        hidden, skipped = gates[:width], gates[width:]
        shift = last.weight @ (hidden * mean[:width])
        zero = jax.tree.map(jnp.zeros_like, arrays)
        change = eqx.tree_at(
            lambda n: mlp(n).layers[-1].weight, zero, last.weight * hidden[None, :]
        )
        if skip:
            change = eqx.tree_at(lambda n: n.skip, change, net.skip * skipped)
            skip_mean = net.skip @ (skipped * mean[width:])
            shift = shift.at[net.location_index].add(skip_mean)
        change = eqx.tree_at(lambda n: mlp(n).layers[-1].bias, change, -shift)
        return ravel_pytree(change)[0]

    return jax.jacfwd(tangent)(jnp.zeros_like(mean))


def _gradients(trace, gradients, n_draws):
    """Log density gradients at the draws, ``(n_draws, n_dim)``."""
    if gradients is not None:
        values = np.asarray(gradients, dtype=np.float64)
        return values.reshape(n_draws, -1)
    if hasattr(trace, "shape"):
        raise ValueError(
            "Pass the log density gradients at the draws as `gradients`."
        )
    stats = trace["sample_stats"]
    if "gradient" not in stats:
        raise ValueError(
            "The trace has no gradients. Sample with `store_gradient=True`, "
            "or pass them as `gradients`."
        )
    values = np.asarray(stats["gradient"].values, dtype=np.float64)
    return values.reshape(n_draws, -1)


def conditioner_capacity(
    bijection, trace, *, gradients=None, batch_size=32, cholesky_jitter=None
):
    """How much of each conditioner's last hidden layer a fitted triangular
    flow uses, measured in the Fisher divergence the flow is fitted to,
    without refitting it.

    Gating the hidden units of a conditioner, ``theta = W diag(1 + g) h + b``
    with the bias absorbing their mean output, changes the fit's residuals
    by ``J g`` to first order. ``J`` comes from the exact Gauss-Newton
    factors (`SparseTriangularMap.gauss_newton_factors`), so it includes how
    the change moves the residuals of the parents. Then

        G = E_draws[J^T J]

    is the Gauss-Newton curvature of the Fisher divergence ``F`` in the
    gates: at a converged fit, a change ``g`` raises ``F`` by about
    ``g^T G g``, in the units of the ``log F`` the fit reports. The parents
    feeding the linear location skip enter as extra units and are projected
    out (Schur complement), so the eigenvalues of ``G`` only count what the
    hidden layer adds beyond the skip.

    The conditioners are treated one at a time: the curvature between
    conditioners (a child whose residual moves with a parent's conditioner)
    is left out, as in the LM preconditioner.

    Parameters
    ----------
    bijection:
        The full flow bijection of a triangular flow, as stored in
        `nutpie.transform_adapter._BIJECTION_TRACE`.
    trace:
        A trace sampled with ``store_unconstrained=True`` and
        ``store_gradient=True``, or an array ``(n_draws, n_dim)`` of
        unconstrained draws, e.g. the window's positions stored with the
        bijection.
    gradients:
        The log density gradients at the draws, ``(n_draws, n_dim)``.
        Required if `trace` is an array, e.g. the window's gradients.
    batch_size:
        Draws evaluated at once, the memory knob.
    cholesky_jitter:
        Passed to `gauss_newton_factors`.

    Returns
    -------
    A `ConditionerCapacity`, see there.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from flowjax import bijections
    from paramax import unwrap

    from nutpie.transform_adapter import inverse_gradient_and_val

    draws, names = _draws(trace)
    grads = _gradients(trace, gradients, len(draws))
    tmap, order, to_map = _triangular_map(bijection, draws)
    dim = tmap.shape[0]

    flow = unwrap(bijection)
    sandwich = flow.bijections[0].bijections[0]
    affine = flow.bijections[1]

    @jax.jit
    @jax.vmap
    def into_map(draw, grad):
        # As in `gauss_newton_factors` of the transform adapter
        logp = jnp.zeros((), draw.dtype)
        draw, grad, _ = inverse_gradient_and_val(affine, draw, grad, logp)
        draw, grad, _ = inverse_gradient_and_val(
            bijections.Invert(sandwich.outer), draw, grad, logp
        )
        return draw, grad

    @eqx.filter_jit
    def feature_sums(conditioner, parent_values, parent_indices):
        # vmap over the draws, then over the conditioners of the bucket
        per_draw = jax.vmap(
            lambda values: eqx.filter_vmap(
                lambda net, parents, indices: _unit_features(tmap, net)(
                    parents, indices
                )
            )(conditioner, values, parent_indices)
        )(parent_values)
        return per_draw.sum(0)

    @eqx.filter_jit
    def gram_sums(tmap, ys, gs, directions):
        def one(y, g):
            factors = tmap.gauss_newton_factors(y, g, cholesky_jitter=cholesky_jitter)
            grams = []
            for V, D in zip(factors, directions, strict=True):
                J = jnp.einsum("mkp,mpu->mku", V, D)
                grams.append(jnp.einsum("mku,mkv->muv", J, J))
            return grams

        return [gram.sum(0) for gram in jax.vmap(one)(ys, gs)]

    # First pass: the mean output of each unit, absorbed by the bias
    directions = []
    for bucket, conditioner in enumerate(tmap.conditioners):
        parent_indices = tmap.bucket_parent_indices[bucket]
        sums = sum(
            feature_sums(conditioner, parents, parent_indices)
            for parents in _parent_batches(tmap, bucket, draws, to_map, batch_size)
        )
        directions.append(
            eqx.filter_vmap(_gate_directions)(conditioner, sums / len(draws))
        )

    # Second pass: the Gauss-Newton curvature in the gates
    grams = None
    for start in range(0, len(draws), batch_size):
        ys, gs = into_map(
            jnp.asarray(draws[start : start + batch_size]),
            jnp.asarray(grads[start : start + batch_size]),
        )
        batch = gram_sums(tmap, ys, gs, directions)
        grams = batch if grams is None else [a + b for a, b in zip(grams, batch)]

    eigenvalues = None
    num_parents = np.zeros(dim, dtype=int)
    for bucket, conditioner in enumerate(tmap.conditioners):
        members = np.asarray(tmap.bucket_members[bucket])
        parent_indices = np.asarray(tmap.bucket_parent_indices[bucket])
        num_parents[order[members]] = (parent_indices < dim).sum(axis=1)
        width = _hidden_width(conditioner)
        if eigenvalues is None:
            eigenvalues = np.zeros((dim, width))
        bucket_grams = np.asarray(grams[bucket], dtype=np.float64) / len(draws)
        for member, gram in zip(members, bucket_grams, strict=True):
            hidden, skip = gram[:width, :width], gram[:width, width:]
            if skip.size:
                hidden = hidden - skip @ np.linalg.pinv(
                    gram[width:, width:], rcond=1e-10, hermitian=True
                ) @ skip.T
            values = np.linalg.eigvalsh(hidden)[::-1]
            eigenvalues[order[member]] = np.maximum(values, 0.0)

    labels = _labels(names, dim)
    directions = pd.RangeIndex(1, eigenvalues.shape[1] + 1, name="hidden_direction")
    return ConditionerCapacity(
        eigenvalues=pd.DataFrame(eigenvalues, index=labels, columns=directions),
        num_parents=pd.Series(num_parents, index=labels, name="num_parents"),
        order=order,
        num_draws=len(draws),
    )


def _hidden_width(conditioner):
    """Width of the last hidden layer of a bucket's conditioners."""
    from nutpie.triangular import LocationSkipMlp

    mlp = conditioner.mlp if isinstance(conditioner, LocationSkipMlp) else conditioner
    return mlp.layers[-1].weight.shape[-1]
