from importlib.util import find_spec

import numpy as np
import pytest
import scipy.sparse

from nutpie.sparsity import _canonicalize, check_hessian_sparsity, factorize

ORDERS = ["amd", "natural"] + (["metis"] if find_spec("pymetis") else [])


def star(n):
    pattern = np.eye(n, dtype=bool)
    pattern[0, :] = pattern[:, 0] = True
    return pattern


def path(n):
    pattern = np.eye(n, dtype=bool)
    for i in range(n - 1):
        pattern[i, i + 1] = pattern[i + 1, i] = True
    return pattern


def random_pattern(n, density, seed):
    rng = np.random.default_rng(seed)
    pattern = rng.random((n, n)) < density
    return pattern | pattern.T | np.eye(n, dtype=bool)


def brute_force_fill(pattern, order):
    """Fill by eliminating the variables in reverse flow order."""
    adjacency = pattern.copy()
    np.fill_diagonal(adjacency, False)
    filled = np.zeros_like(adjacency)
    done = np.zeros(len(pattern), dtype=bool)
    for var in np.asarray(order)[::-1]:
        later = np.flatnonzero(adjacency[var] & ~done)
        filled[var, later] = filled[later, var] = True
        adjacency[np.ix_(later, later)] = True
        done[var] = True
    np.fill_diagonal(filled, False)
    return filled


def test_check_hessian_sparsity():
    pattern = np.zeros((3, 3), dtype=bool)
    pattern[0, 2] = True
    for value in [pattern, scipy.sparse.coo_array(pattern)]:
        checked = check_hessian_sparsity(value, 3)
        assert isinstance(checked, scipy.sparse.csr_array)
        expected = np.eye(3, dtype=bool)
        expected[0, 2] = expected[2, 0] = True
        np.testing.assert_array_equal(checked.toarray(), expected)
    with pytest.raises(ValueError, match="shape"):
        check_hessian_sparsity(pattern, 4)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("seed", range(5))
def test_fill(order, seed):
    pattern = random_pattern(40, 0.08, seed)
    factorization = factorize(pattern, order=order)
    np.testing.assert_array_equal(
        factorization.filled.toarray(),
        brute_force_fill(pattern, factorization.order),
    )
    # Dense and sparse input give the same result
    sparse = factorize(scipy.sparse.csr_array(pattern), order=order)
    np.testing.assert_array_equal(sparse.order, factorization.order)


@pytest.mark.parametrize("order", [o for o in ORDERS if o != "natural"])
def test_star(order):
    factorization = factorize(star(10), order=order)
    # The hub comes first, then the leaves in natural order
    np.testing.assert_array_equal(factorization.order, np.arange(10))
    assert factorization.num_fill == 0
    assert factorization.max_parents == 1
    assert factorization.num_levels == 2


def test_hub_last_fills_everything():
    factorization = factorize(star(6), order=np.arange(6)[::-1])
    assert factorization.method == "custom"
    assert factorization.num_fill == 10
    assert factorization.max_parents == 5


@pytest.mark.skipif(find_spec("pymetis") is None, reason="needs pymetis")
def test_metis_path_levels():
    # Nested dissection trades fill for fewer levels
    metis = factorize(path(31), order="metis")
    natural = factorize(path(31), order="natural")
    assert natural.num_fill == 0
    assert natural.num_levels == 31
    assert metis.num_levels < 10


def test_canonicalize_twins_keeps_fill():
    # Hubs with leaves, and a clique: open and closed twins
    pattern = np.eye(12, dtype=bool)
    for hub, leaves in [(0, [3, 1, 5]), (2, [4, 6])]:
        pattern[hub, leaves] = pattern[leaves, hub] = True
    clique = [7, 9, 8, 11]
    pattern[np.ix_(clique, clique)] = True
    pattern[0, 2] = pattern[2, 0] = True
    rng = np.random.default_rng(0)
    for _ in range(10):
        order = rng.permutation(12)
        canonical = _canonicalize(check_hessian_sparsity(pattern, 12), order)
        assert sorted(canonical) == list(range(12))
        assert (
            brute_force_fill(pattern, canonical).sum()
            == brute_force_fill(pattern, order).sum()
        )


def test_front():
    factorization = factorize(path(9), order="amd", front=[4, 7])
    np.testing.assert_array_equal(factorization.order[:2], [4, 7])
    assert factorization.method == "amd (with front)"
    np.testing.assert_array_equal(
        factorization.filled.toarray(), brute_force_fill(path(9), factorization.order)
    )
    with pytest.raises(ValueError, match="automatic order"):
        factorize(path(9), order=np.arange(9), front=[1])


def test_summaries():
    variables = ["a", "b", "b", "b", "c", "c"]
    factorization = factorize(star(6), order="amd", variables=variables)
    summary = factorization.summary()
    assert summary.index.name == "unconstrained_parameter"
    assert list(summary["parents"]) == [0, 1, 1, 1, 1, 1]
    by_variable = factorization.by_variable()
    assert by_variable.loc[("a", "b")].tolist() == [3, 3, 3]
    assert by_variable.loc[("a", "c")].tolist() == [2, 2, 2]
    assert ("b", "b") not in by_variable.index


@pytest.mark.skipif(find_spec("matplotlib") is None, reason="needs matplotlib")
def test_plot_spy():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = ["a", "b", "b", "b", "c", "c"]
    factorization = factorize(star(6), order="amd", variables=variables)
    for order in ["flow", "original"]:
        ax = factorization.plot_spy(order=order, max_variables=2)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["Hessian", "fill-in", "b", "c", "other"]
        plt.close(ax.figure)
    with pytest.raises(ValueError, match="order"):
        factorization.plot_spy(order="elimination")

    # The legend follows the first appearance in the plot, the colours stay
    variables = ["a", "a", "b", "c"]
    factorization = factorize(
        np.eye(4, dtype=bool), order=[3, 2, 0, 1], variables=variables
    )
    legends = {}
    for order in ["flow", "original"]:
        ax = factorization.plot_spy(order=order)
        legend = ax.get_legend()
        legends[order] = {
            text.get_text(): tuple(handle.get_facecolor())
            for text, handle in zip(legend.get_texts(), legend.legend_handles)
            if text.get_text() in variables
        }
        plt.close(ax.figure)
    assert list(legends["flow"]) == ["c", "b", "a"]
    assert list(legends["original"]) == ["a", "b", "c"]
    assert legends["flow"] == legends["original"]
