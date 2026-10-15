"""Diagnostics for fitted triangular flows (`nutpie.triangular_flow`).

- `fisher_influence`: how strongly the flow lets parameters influence each
  other.
- `conditioner_capacity`: how much of each conditioner's hidden layer the
  flow actually uses, to tell whether ``nn_width`` is too small or larger
  than necessary.

Both are computed in Rust (`src/triangular/lm/diagnostics.rs`). Neither
depends on how the transformer is parameterized. The influence is measured
in the Fisher-Rao metric, which splits exactly into the conditionals; the
capacity in the Fisher divergence the flow is fitted to, so it is in the
units of the ``log F`` the fit reports.

Influence
---------

For fixed parents, the flow gives parameter ``i`` a one-dimensional
conditional distribution ``p_theta(y_i)``, where ``theta`` are the
coefficients of its transformer ``y = T_theta(x)`` with standard normal
``x``, the sum of its conditioners' outputs at the parents. A change of
parent ``j`` moves ``theta`` along ``g_j = d theta / d y_j``. With the
Fisher information ``I(theta)`` of the one-dimensional family,

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

A variable's coefficients are a constant, a linear conditioner in the
parents and MLPs ``W h(y_parents)`` without output bias, with ``h`` their
hidden units. Gating the hidden units, ``W diag(1 + g) h``, with the constant
absorbing their mean output, makes removing or merging units a change of the
weights that is linear in ``g``. Its effect on the Fisher divergence ``F`` of
the fit is ``g^T G g`` to second order, with ``G`` the Gauss-Newton curvature
of ``F`` in the gates, from the exact Gauss-Newton blocks of the LM fit. The
eigenvalues of ``G``, after projecting out what the linear conditioner
already does, are how much ``F`` rises when an independent direction of the
hidden layer is dropped: small ones are width the flow does not use.
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


def _quadrature(num_nodes):
    """Gauss-Hermite nodes and normalized weights for a standard normal. The
    nodes are symmetric: reversing them negates them."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(num_nodes)
    return nodes, weights / weights.sum()


def _trim(flow, *arrays):
    """`arrays` (draws first) without their oldest draws, down to a multiple
    of the Rust kernels' SIMD width, and C-contiguous."""
    from nutpie._lib import FisherResiduals

    draws = arrays[0]
    if draws.shape[1] != flow.n_var:
        raise ValueError(
            f"The draws have {draws.shape[1]} dimensions, the flow {flow.n_var}."
        )
    keep = len(draws) - len(draws) % FisherResiduals.simd_width()
    if keep == 0:
        raise ValueError(
            f"Need at least {FisherResiduals.simd_width()} draws, got {len(draws)}."
        )
    return [np.ascontiguousarray(array[len(array) - keep :]) for array in arrays]


def fisher_influence(flow, trace, *, num_nodes=16):
    """Influence of each parent on each parameter under a fitted flow.

    Parameters
    ----------
    flow:
        A fitted `TriangularFlow`, e.g. as stored in
        `nutpie.triangular_adapter._BIJECTION_TRACE`.
    trace:
        A trace sampled with ``store_unconstrained=True``. Only the
        unconstrained draws in ``sample_stats`` are used. Or an array
        ``(n_draws, n_dim)`` of unconstrained draws, e.g. a window's
        positions from `_BIJECTION_TRACE`. The oldest draws are dropped down
        to a multiple of the SIMD width (`FisherResiduals.simd_width()`).
    num_nodes:
        Number of Gauss-Hermite nodes for each one-dimensional conditional
        distribution.

    Returns
    -------
    A `FlowInfluence`, see there and the module docstring.
    """
    draws, names = _draws(trace)
    (draws,) = _trim(flow, draws)
    nodes, weights = _quadrature(num_nodes)
    child, parent, location_skew, scale_tails = flow.fisher_speeds(
        draws, nodes, weights
    )
    order = np.asarray(flow.permutation)
    dim = len(order)

    def matrix(squares):
        return sp.csr_array(
            (np.sqrt(squares), (order[child], order[parent])), shape=(dim, dim)
        )

    return FlowInfluence(
        total=matrix(location_skew + scale_tails),
        location_skew=matrix(location_skew),
        scale_tails=matrix(scale_tails),
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


def _gradients(trace, gradients, n_draws):
    """Log density gradients at the draws, ``(n_draws, n_dim)``."""
    if gradients is not None:
        values = np.asarray(gradients, dtype=np.float64)
        return values.reshape(n_draws, -1)
    if hasattr(trace, "shape"):
        raise ValueError("Pass the log density gradients at the draws as `gradients`.")
    stats = trace["sample_stats"]
    if "gradient" not in stats:
        raise ValueError(
            "The trace has no gradients. Sample with `store_gradient=True`, "
            "or pass them as `gradients`."
        )
    values = np.asarray(stats["gradient"].values, dtype=np.float64)
    return values.reshape(n_draws, -1)


def conditioner_capacity(flow, trace, *, gradients=None):
    """How much of each conditioner's hidden layer a fitted triangular flow
    uses, measured in the Fisher divergence the flow is fitted to, without
    refitting it.

    Gating the hidden units of a variable's MLPs, ``W diag(1 + g) h`` with
    the constant absorbing their mean output, changes the fit's residuals by
    ``J g`` to first order. ``J`` comes from the exact Gauss-Newton blocks
    of the LM fit, so it includes how the change moves the residuals of the
    parents. Then

        G = E_draws[J^T J]

    is the Gauss-Newton curvature of the Fisher divergence ``F`` in the
    gates: at a converged fit, a change ``g`` raises ``F`` by about
    ``g^T G g``, in the units of the ``log F`` the fit reports. The linear
    conditioner's weights enter as extra gates and are projected out (Schur
    complement), so the eigenvalues of ``G`` only count what the hidden
    layer adds beyond them.

    The variables are treated one at a time: the curvature between
    variables (a child whose residual moves with a parent's conditioner) is
    left out, as in the LM preconditioner. With several MLPs, their units
    are gated together.

    Parameters
    ----------
    flow:
        A fitted `TriangularFlow` with an MLP conditioner, e.g. as stored in
        `nutpie.triangular_adapter._BIJECTION_TRACE`.
    trace:
        A trace sampled with ``store_unconstrained=True`` and
        ``store_gradient=True``, or an array ``(n_draws, n_dim)`` of
        unconstrained draws, e.g. the window's positions stored with the
        flow. The oldest draws are dropped down to a multiple of the SIMD
        width (`FisherResiduals.simd_width()`).
    gradients:
        The log density gradients at the draws, ``(n_draws, n_dim)``.
        Required if `trace` is an array, e.g. the window's gradients.

    Returns
    -------
    A `ConditionerCapacity`, see there.
    """
    draws, names = _draws(trace)
    grads = _gradients(trace, gradients, len(draws))
    draws, grads = _trim(flow, draws, grads)
    order = np.asarray(flow.permutation)
    dim = len(order)

    # Rows in the flow order, by parameter
    values = flow.gate_eigenvalues(draws, grads)
    eigenvalues = np.empty_like(values)
    eigenvalues[order] = values
    num_parents = np.empty(dim, dtype=int)
    num_parents[order] = flow.n_parents

    labels = _labels(names, dim)
    directions = pd.RangeIndex(1, eigenvalues.shape[1] + 1, name="hidden_direction")
    return ConditionerCapacity(
        eigenvalues=pd.DataFrame(eigenvalues, index=labels, columns=directions),
        num_parents=pd.Series(num_parents, index=labels, name="num_parents"),
        order=order,
        num_draws=len(draws),
    )
