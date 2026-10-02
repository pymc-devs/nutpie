//! Orderings and symbolic Cholesky factorizations of sparsity patterns.
//!
//! Graphs are given as symmetric CSR adjacency (`indptr`, `indices`)
//! without the diagonal.

use anyhow::{bail, Result};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::amd;
use faer::sparse::SymbolicSparseColMatRef;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn check_graph(indptr: &[usize], indices: &[usize]) -> Result<usize> {
    let Some(&nnz) = indptr.last() else {
        bail!("indptr must not be empty");
    };
    let n = indptr.len() - 1;
    if nnz != indices.len() || indptr.windows(2).any(|w| w[0] > w[1]) {
        bail!("Invalid indptr");
    }
    if indices.iter().any(|&j| j >= n) {
        bail!("Graph index out of range");
    }
    Ok(n)
}

/// Approximate minimum degree elimination order of a graph.
pub fn amd_order(indptr: &[usize], indices: &[usize]) -> Result<Vec<usize>> {
    let n = check_graph(indptr, indices)?;
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    if n == 0 {
        return Ok(perm);
    }
    // The pattern is symmetric, so CSR and CSC are the same.
    let graph = SymbolicSparseColMatRef::new_checked(n, n, indptr, None, indices);
    let mut buffer = MemBuffer::new(amd::order_maybe_unsorted_scratch::<usize>(n, indices.len()));
    amd::order_maybe_unsorted(
        &mut perm,
        &mut perm_inv,
        graph,
        amd::Control::default(),
        MemStack::new(&mut buffer),
    )?;
    Ok(perm)
}

/// Off-diagonal pattern of the Cholesky factor, when the variables are
/// eliminated in the order `elim`.
///
/// Returns the pairs `(i, j)` in the original indices, where `i` is
/// eliminated after `j` and `L[i, j]` is nonzero. Uses the elimination tree
/// and row subtrees (see Davis, "Direct Methods for Sparse Linear Systems"),
/// so the cost is proportional to the number of nonzeros of `L`.
pub fn symbolic_fill(
    indptr: &[usize],
    indices: &[usize],
    elim: &[usize],
) -> Result<Vec<(usize, usize)>> {
    const NONE: usize = usize::MAX;
    let n = check_graph(indptr, indices)?;
    if elim.len() != n {
        bail!("Order has length {} (expected {n})", elim.len());
    }
    let mut position = vec![NONE; n];
    for (k, &var) in elim.iter().enumerate() {
        if var >= n || position[var] != NONE {
            bail!("Order is not a permutation");
        }
        position[var] = k;
    }
    let neighbours = |k: usize| {
        let var = elim[k];
        indices[indptr[var]..indptr[var + 1]]
            .iter()
            .map(|&j| position[j])
            .filter(move |&p| p < k)
    };

    // Elimination tree with path compression
    let mut parent = vec![NONE; n];
    let mut ancestor = vec![NONE; n];
    for k in 0..n {
        for p in neighbours(k) {
            let mut i = p;
            while i != NONE && i < k {
                let next = ancestor[i];
                ancestor[i] = k;
                if next == NONE {
                    parent[i] = k;
                }
                i = next;
            }
        }
    }

    // Row k of L: the union of the paths from its neighbours up to k
    let mut mark = vec![NONE; n];
    let mut pairs = Vec::new();
    for k in 0..n {
        mark[k] = k;
        for p in neighbours(k) {
            let mut i = p;
            while mark[i] != k {
                mark[i] = k;
                pairs.push((elim[k], elim[i]));
                i = parent[i];
            }
        }
    }
    Ok(pairs)
}

fn to_usize(values: PyReadonlyArray1<'_, i64>) -> Result<Vec<usize>> {
    values
        .as_slice()?
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| anyhow::anyhow!("Negative index")))
        .collect()
}

fn to_numpy(py: Python<'_>, values: impl IntoIterator<Item = usize>) -> Bound<'_, PyArray1<i64>> {
    PyArray1::from_vec(py, values.into_iter().map(|x| x as i64).collect())
}

/// AMD elimination order of the graph with CSR adjacency `indptr`, `indices`.
#[pyfunction(name = "amd_order")]
pub fn py_amd_order<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
) -> Result<Bound<'py, PyArray1<i64>>> {
    let order = amd_order(&to_usize(indptr)?, &to_usize(indices)?)?;
    Ok(to_numpy(py, order))
}

/// Off-diagonal pattern of the Cholesky factor with elimination order
/// `elim`, as arrays `(rows, cols)` in the original indices.
#[pyfunction(name = "symbolic_fill")]
pub fn py_symbolic_fill<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    elim: PyReadonlyArray1<'py, i64>,
) -> Result<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let pairs = symbolic_fill(&to_usize(indptr)?, &to_usize(indices)?, &to_usize(elim)?)?;
    let rows = to_numpy(py, pairs.iter().map(|&(i, _)| i));
    let cols = to_numpy(py, pairs.iter().map(|&(_, j)| j));
    Ok((rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(n: usize, edges: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
        let mut rows = vec![vec![]; n];
        for &(i, j) in edges {
            rows[i].push(j);
            rows[j].push(i);
        }
        let mut indptr = vec![0];
        let mut indices = vec![];
        for mut row in rows {
            row.sort_unstable();
            indices.extend(row);
            indptr.push(indices.len());
        }
        (indptr, indices)
    }

    /// Fill by explicit elimination of the graph.
    fn brute_force_fill(n: usize, edges: &[(usize, usize)], elim: &[usize]) -> Vec<(usize, usize)> {
        let mut adj = vec![vec![false; n]; n];
        for &(i, j) in edges {
            adj[i][j] = true;
            adj[j][i] = true;
        }
        let mut done = vec![false; n];
        let mut pairs = vec![];
        for &v in elim {
            let later: Vec<usize> = (0..n).filter(|&u| adj[v][u] && !done[u]).collect();
            for &u in later.iter() {
                pairs.push((u, v));
            }
            for &a in later.iter() {
                for &b in later.iter() {
                    if a != b {
                        adj[a][b] = true;
                    }
                }
            }
            done[v] = true;
        }
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn fill_matches_elimination() {
        let n = 8;
        let edges = [
            (0, 1),
            (0, 5),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (2, 6),
        ];
        let (indptr, indices) = csr(n, &edges);
        for elim in [
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![7, 6, 5, 4, 3, 2, 1, 0],
            vec![3, 0, 7, 5, 1, 6, 2, 4],
        ] {
            let mut pairs = symbolic_fill(&indptr, &indices, &elim).unwrap();
            pairs.sort_unstable();
            assert_eq!(pairs, brute_force_fill(n, &edges, &elim));
        }
    }

    #[test]
    fn star_center_first_fills_everything() {
        let n = 5;
        let edges: Vec<_> = (1..n).map(|j| (0, j)).collect();
        let (indptr, indices) = csr(n, &edges);
        let pairs = symbolic_fill(&indptr, &indices, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(pairs.len(), 4 + 6);
        let pairs = symbolic_fill(&indptr, &indices, &[1, 2, 3, 4, 0]).unwrap();
        assert_eq!(pairs.len(), 4);
    }

    #[test]
    fn amd_is_permutation_and_avoids_star_fill() {
        let n = 20;
        let edges: Vec<_> = (1..n).map(|j| (0, j)).collect();
        let (indptr, indices) = csr(n, &edges);
        let order = amd_order(&indptr, &indices).unwrap();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
        let pairs = symbolic_fill(&indptr, &indices, &order).unwrap();
        assert_eq!(pairs.len(), n - 1);
    }
}
