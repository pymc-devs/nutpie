"""Hessian sparsity patterns and symbolic factorizations.

The Hessian sparsity pattern of the log density on the unconstrained space
is the conditional-dependency graph of the posterior. A symbolic
factorization adds an elimination order and the fill-in of a Cholesky
factorization with that order. The triangular flow uses it to decide which
earlier variables each variable may depend on.

Patterns are stored as boolean `scipy.sparse.csr_array` with sorted indices
and without duplicates.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from math import prod

import numpy as np
import pandas as pd
import scipy.sparse as sp

from nutpie import _lib

ORDER_METHODS = ("metis", "amd", "natural")


def _pattern(rows, cols, n) -> sp.csr_array:
    """Boolean CSR pattern with entries at `(rows, cols)`."""
    pattern = sp.csr_array((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
    pattern.sum_duplicates()
    pattern.sort_indices()
    return pattern


def check_hessian_sparsity(sparsity, n_dim) -> sp.csr_array:
    """Validate a Hessian sparsity pattern, given as a dense or sparse
    matrix, and return it as a symmetric CSR pattern with a true diagonal."""
    if sp.issparse(sparsity):
        pattern = sp.coo_array(sparsity)
        pattern.eliminate_zeros()
    else:
        pattern = sp.coo_array(np.asarray(sparsity, dtype=bool))
    if pattern.shape != (n_dim, n_dim):
        raise ValueError(
            f"Hessian sparsity must have shape {(n_dim, n_dim)}, got {pattern.shape}."
        )
    diag = np.arange(n_dim)
    return _pattern(
        np.concatenate([pattern.row, pattern.col, diag]),
        np.concatenate([pattern.col, pattern.row, diag]),
        n_dim,
    )


def _off_diagonal(pattern) -> sp.csr_array:
    coo = pattern.tocoo()
    keep = coo.row != coo.col
    return _pattern(coo.row[keep], coo.col[keep], pattern.shape[0])


def _adjacency(graph):
    """CSR adjacency (indptr, indices) of a graph without the diagonal."""
    return graph.indptr.astype(np.int64), graph.indices.astype(np.int64)


def _metis_order(graph):
    try:
        import pymetis
    except ImportError as err:
        raise ImportError(
            "The METIS order requires pymetis. Please install it with something "
            "like 'pip install pymetis', or use `order='amd'`."
        ) from err
    indptr, indices = _adjacency(graph)
    # The first array is the elimination order: separators come last.
    elim, _ = pymetis.nested_dissection(adjacency=pymetis.CSRAdjacency(indptr, indices))
    return np.asarray(elim, dtype=np.int64)


def _elimination_order(graph, method):
    """Fill-reducing elimination order of a graph without the diagonal."""
    n = graph.shape[0]
    if method == "natural":
        # Natural flow order, which is the reversed elimination order
        return np.arange(n)[::-1]
    if n == 0:
        return np.arange(n)
    if method == "metis":
        return _metis_order(graph)
    if method == "amd":
        return np.asarray(_lib.amd_order(*_adjacency(graph)))
    raise ValueError(f"Unknown order {method!r}. Expected one of {ORDER_METHODS}.")


def _canonicalize(hessian_sparsity, order, fixed=()):
    """Use the natural order among variables that are interchangeable.

    Twins, variables with the same neighbours, can be swapped without
    changing the fill or the levels, so we sort them to keep the order
    readable and stable. Twins are either adjacent with the same closed
    neighbourhood (row of the pattern with diagonal), or not adjacent with
    the same open neighbourhood, like the leaves of a star. A variable can't
    have both kinds of twins. Variables in `fixed` are not moved.
    """
    indptr, indices = hessian_sparsity.indptr, hessian_sparsity.indices
    fixed = {int(i) for i in fixed}
    classes = {}
    for i in range(hessian_sparsity.shape[0]):
        if i not in fixed:
            closed = indices[indptr[i] : indptr[i + 1]]
            classes.setdefault(b"closed" + closed.tobytes(), []).append(i)
            open_ = closed[closed != i]
            classes.setdefault(b"open" + open_.tobytes(), []).append(i)
    order = np.asarray(order).copy()
    position = np.empty_like(order)
    position[order] = np.arange(len(order))
    for members in classes.values():
        if len(members) < 2:
            continue
        order[np.sort(position[members])] = np.sort(members)
    return order


def _fill(graph, order) -> sp.csr_array:
    """Symmetric pattern of the Cholesky factor without the diagonal, for
    the flow order `order`.

    The flow order is the reverse of the elimination order, because the
    triangular map factorizes the precision as ``C^T C``.
    """
    n = graph.shape[0]
    if n == 0:
        return _pattern([], [], 0)
    elim = np.ascontiguousarray(np.asarray(order, dtype=np.int64)[::-1])
    rows, cols = _lib.symbolic_fill(*_adjacency(graph), elim)
    return _pattern(np.concatenate([rows, cols]), np.concatenate([cols, rows]), n)


def _draw_cells(ax, x, y, **kwargs):
    """Draw unit squares centred at `(x, y)` in data coordinates, so that
    neighbouring cells meet without gaps at any size or resolution. Returns
    the `PolyCollection`, `kwargs` are passed on to it."""
    from matplotlib.collections import PolyCollection

    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    centers = np.stack([x, y], axis=-1).astype(np.float64)
    cells = PolyCollection(
        centers[:, None, :] + corners[None, :, :],
        edgecolors="none",
        antialiased=False,
        rasterized=True,
        **kwargs,
    )
    ax.add_collection(cells)
    return cells


def _variable_strips(ax, variables, perm, max_variables):
    """Coloured strips left of and below `ax` that show the model variable
    of each unconstrained parameter, in the order `perm`. The largest
    `max_variables` variables get a colour, the others are grey. Returns the
    legend handles, in the order in which the variables first appear."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n = len(perm)
    codes, names = pd.factorize(pd.Series(variables))
    sizes = np.bincount(codes, minlength=len(names))
    coloured = np.argsort(-sizes, kind="stable")[:max_variables]
    colors = ["0.85"] * len(names)
    for i, code in enumerate(sorted(coloured)):
        colors[code] = f"C{i % 10}"
    cmap = ListedColormap(colors)
    strip = codes[perm]

    # Outside of the tick labels, and not on top where the title goes
    divider = make_axes_locatable(ax)
    for side, values in [("left", strip[:, None]), ("bottom", strip[None, :])]:
        strip_ax = divider.append_axes(side, size="3%", pad=0.4)
        strip_ax.imshow(
            values,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(names) - 0.5,
            aspect="auto",
            interpolation="nearest",
        )
        strip_ax.set_axis_off()

    first = np.full(len(names), n)
    np.minimum.at(first, strip, np.arange(n))
    handles = [
        Patch(color=colors[code], label=names[code])
        for code in sorted(coloured, key=lambda code: first[code])
    ]
    if len(names) > len(coloured):
        handles.append(Patch(color="0.85", label="other"))
    return handles


@dataclass(frozen=True, repr=False)
class Factorization:
    """Symbolic factorization of the Hessian sparsity pattern.

    Attributes
    ----------
    order:
        The flow order: ``order[k]`` is the variable at position ``k``. A
        variable may only depend on variables earlier in this order. This is
        the reverse of the elimination order of the Cholesky factorization.
    filled:
        Symmetric boolean sparse ``(n_dim, n_dim)`` pattern of the Cholesky
        factor without the diagonal: the Hessian sparsity plus fill-in.
        ``filled[i, j]`` for ``j`` earlier than ``i`` means that ``j`` is a
        parent of ``i``.
    hessian_sparsity:
        The Hessian sparsity pattern the factorization was computed from.
    method:
        How the order was chosen.
    unconstrained_parameters:
        Names of the unconstrained parameters, as in the
        ``unconstrained_parameter`` coordinate of the trace.
    variables:
        The model variable each unconstrained parameter belongs to.
    """

    order: np.ndarray
    filled: sp.csr_array
    hessian_sparsity: sp.csr_array
    method: str
    unconstrained_parameters: list[str]
    variables: list[str]

    @property
    def n_dim(self) -> int:
        return len(self.order)

    @cached_property
    def position(self) -> np.ndarray:
        """Position of each variable in the flow order."""
        position = np.empty_like(self.order)
        position[self.order] = np.arange(self.n_dim)
        return position

    @cached_property
    def _parent_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """`(child, parent)` index arrays of all parents."""
        coo = self.filled.tocoo()
        earlier = self.position[coo.col] < self.position[coo.row]
        return coo.row[earlier], coo.col[earlier]

    @cached_property
    def parents(self) -> sp.csr_array:
        """Pattern with the parents of each variable in its row."""
        return _pattern(*self._parent_pairs, self.n_dim)

    @cached_property
    def num_parents(self) -> np.ndarray:
        """Number of parents of each variable."""
        return np.diff(self.parents.indptr)

    @cached_property
    def num_fill_parents(self) -> np.ndarray:
        """Number of parents of each variable that are not Hessian neighbours."""
        children, parents = self._parent_pairs
        hessian = self.hessian_sparsity.tocoo()
        n = np.int64(self.n_dim)
        is_neighbour = np.isin(
            children.astype(np.int64) * n + parents,
            hessian.row.astype(np.int64) * n + hessian.col,
        )
        return np.bincount(children[~is_neighbour], minlength=self.n_dim)

    @cached_property
    def levels(self) -> np.ndarray:
        """Level of each variable: the length of the longest chain of parents.

        Variables of the same level can be transformed in parallel.
        """
        indptr, indices = self.parents.indptr, self.parents.indices
        levels = np.zeros(self.n_dim, dtype=np.int64)
        for var in self.order:
            parents = indices[indptr[var] : indptr[var + 1]]
            if len(parents):
                levels[var] = levels[parents].max() + 1
        return levels

    @property
    def num_levels(self) -> int:
        return int(self.levels.max(initial=-1)) + 1

    @property
    def max_parents(self) -> int:
        return int(self.num_parents.max(initial=0))

    @property
    def num_edges(self) -> int:
        """Number of nonzero pairs in the Hessian sparsity, without the diagonal."""
        return (self.hessian_sparsity.nnz - self.n_dim) // 2

    @property
    def num_fill(self) -> int:
        """Number of pairs added by the factorization."""
        return int(self.num_fill_parents.sum())

    def summary(self) -> pd.DataFrame:
        """Per-variable summary, in flow order."""
        neighbours = np.diff(self.hessian_sparsity.indptr) - 1
        frame = pd.DataFrame(
            {
                "variable": self.variables,
                "position": self.position,
                "level": self.levels,
                "neighbours": neighbours,
                "parents": self.num_parents,
                "fill_parents": self.num_fill_parents,
            },
            index=pd.Index(
                self.unconstrained_parameters, name="unconstrained_parameter"
            ),
        )
        return frame.iloc[self.order]

    def by_variable(self) -> pd.DataFrame:
        """Summary of the dependencies between model variables.

        One row for each pair of model variables that interact, with the
        number of nonzero Hessian entries between them, the number of
        entries after fill-in, and the number of possible entries.
        """
        codes, names = pd.factorize(pd.Series(self.variables))
        sizes = np.bincount(codes, minlength=len(names))

        def count_pairs(pattern):
            coo = pattern.tocoo()
            upper = coo.row < coo.col
            first = codes[coo.row[upper]]
            second = codes[coo.col[upper]]
            pairs = pd.DataFrame(
                {
                    "first": np.minimum(first, second),
                    "second": np.maximum(first, second),
                }
            )
            return pairs.value_counts()

        counts = pd.DataFrame(
            {
                "hessian": count_pairs(self.hessian_sparsity),
                "filled": count_pairs(self.filled),
            }
        )
        counts = counts.fillna(0).astype(np.int64).sort_index()
        first = counts.index.get_level_values("first").to_numpy()
        second = counts.index.get_level_values("second").to_numpy()
        possible = np.where(
            first == second,
            sizes[first] * (sizes[first] - 1) // 2,
            sizes[first] * sizes[second],
        )
        return pd.DataFrame(
            {
                "hessian": counts["hessian"].to_numpy(),
                "filled": counts["filled"].to_numpy(),
                "possible": possible,
            },
            index=pd.MultiIndex.from_arrays(
                [names[first], names[second]], names=["variable", "other"]
            ),
        )

    def plot_spy(self, order="flow", *, ax=None, max_variables=10):
        """Spy plot of the Hessian sparsity and the fill-in.

        Entries of the Hessian sparsity and entries added by the fill-in are
        drawn in two shades of grey. Coloured strips along the axes show which
        model variable each unconstrained parameter belongs to.

        Parameters
        ----------
        order:
            ``"flow"`` shows the lower triangle in the flow order, so that row
            ``k`` shows the parents of the parameter at position ``k``.
            ``"original"`` shows the symmetric pattern in the original order of
            the unconstrained parameters.
        ax:
            Matplotlib axes to draw into. A new figure is created if None.
        max_variables:
            Number of model variables that get their own colour, the largest
            ones first. The others are drawn in grey.

        Returns
        -------
        The matplotlib axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        n = self.n_dim
        if order == "flow":
            perm = self.order
        elif order == "original":
            perm = np.arange(n)
        else:
            raise ValueError(f"order must be 'flow' or 'original', got {order!r}.")
        position = np.empty_like(perm)
        position[perm] = np.arange(n)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        def entries(pattern, exclude=None):
            coo = pattern.tocoo()
            rows, cols = position[coo.row], position[coo.col]
            keep = np.ones(len(rows), dtype=bool)
            if order == "flow":
                keep &= cols <= rows
            if exclude is not None:
                other = exclude.tocoo()
                keep &= ~np.isin(
                    coo.row.astype(np.int64) * n + coo.col,
                    other.row.astype(np.int64) * n + other.col,
                )
            return cols[keep], rows[keep]

        _draw_cells(ax, *entries(self.hessian_sparsity), facecolors="0.2")
        _draw_cells(
            ax, *entries(self.filled, exclude=self.hessian_sparsity), facecolors="0.7"
        )
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")

        handles = [
            Line2D([], [], marker="s", linestyle="", color="0.2", label="Hessian"),
            Line2D([], [], marker="s", linestyle="", color="0.7", label="fill-in"),
        ]
        handles += _variable_strips(ax, self.variables, perm, max_variables)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1))
        return ax

    def __repr__(self):
        return (
            f"Factorization({self.method}, n_dim={self.n_dim}, "
            f"edges={self.num_edges}, fill={self.num_fill}, "
            f"max_parents={self.max_parents}, levels={self.num_levels})"
        )


def factorize(
    hessian_sparsity,
    *,
    order: str | Sequence[int] = "metis",
    front: Sequence[int] = (),
    unconstrained_parameters: list[str] | None = None,
    variables: list[str] | None = None,
) -> Factorization:
    """Compute a symbolic factorization of a Hessian sparsity pattern.

    Parameters
    ----------
    hessian_sparsity:
        Boolean ``(n_dim, n_dim)`` Hessian sparsity pattern, dense or sparse.
    order:
        ``"metis"`` (nested dissection), ``"amd"`` (approximate minimum
        degree), ``"natural"``, or an explicit flow order.
    front:
        Variables to put at the front of the flow order, in this order,
        before the automatically ordered remaining variables. Only allowed
        with an automatic order.
    unconstrained_parameters, variables:
        Names of the unconstrained parameters and of the model variables
        they belong to. Default to the parameter indices.
    """
    n = hessian_sparsity.shape[0]
    hessian_sparsity = check_hessian_sparsity(hessian_sparsity, n)
    graph = _off_diagonal(hessian_sparsity)

    if unconstrained_parameters is None:
        unconstrained_parameters = [str(i) for i in range(n)]
    if variables is None:
        variables = list(unconstrained_parameters)

    front = np.asarray(front, dtype=np.int64)
    if len(np.unique(front)) != len(front):
        raise ValueError("Variables in `front` must be unique.")
    if len(front) and (front.min() < 0 or front.max() >= n):
        raise ValueError("Variables in `front` are out of range.")

    if isinstance(order, str):
        # Variables eliminated last don't change the fill between the
        # other variables, so we can order the rest without them.
        rest = np.setdiff1d(np.arange(n), front)
        rest_graph = sp.csr_array(graph[rest][:, rest])
        rest_graph.sort_indices()
        elim_rest = rest[_elimination_order(rest_graph, order)]
        flow_order = _canonicalize(
            hessian_sparsity, np.concatenate([front, elim_rest[::-1]]), fixed=front
        )
        method = order if not len(front) else f"{order} (with front)"
    else:
        if len(front):
            raise ValueError("`front` can only be used with an automatic order.")
        flow_order = np.asarray(order, dtype=np.int64)
        if not np.array_equal(np.sort(flow_order), np.arange(n)):
            raise ValueError(f"order must be a permutation of range({n}).")
        method = "custom"

    return Factorization(
        order=flow_order,
        filled=_fill(graph, flow_order),
        hessian_sparsity=hessian_sparsity,
        method=method,
        unconstrained_parameters=list(unconstrained_parameters),
        variables=list(variables),
    )


def resolve_variables(
    items, unconstrained_parameters: list[str], variables: list[str]
) -> list[int]:
    """Indices of unconstrained parameters, given as indices, names of
    unconstrained parameters, or names of model variables (all their
    parameters, in index order)."""
    indices = []
    for item in items:
        if isinstance(item, (int, np.integer)):
            indices.append(int(item))
        elif item in variables:
            indices.extend(i for i, var in enumerate(variables) if var == item)
        elif item in unconstrained_parameters:
            indices.append(unconstrained_parameters.index(item))
        else:
            raise KeyError(f"Unknown variable {item!r}.")
    return indices


def variables_from_layout(names, shapes) -> list[str]:
    """The variable of each unconstrained parameter, for variables with
    the given names and shapes that make up the unconstrained point."""
    return [name for name, shape in zip(names, shapes) for _ in range(prod(shape))]
