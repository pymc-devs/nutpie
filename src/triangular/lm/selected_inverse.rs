//! Where the selected inverse `Sigma = (J^T J)^{-1}` is kept.

use anyhow::{bail, Result};

use crate::triangular::pattern::{Offsets, Pattern, Ragged};

/// The slots of `Sigma` in a store: the diagonal, then one slot per edge,
/// `Sigma[i, p]` for each parent `p` of `i`.
///
/// Takahashi's recurrence and the exact blocks read `Sigma[p, q]` for pairs
/// of parents of one variable, so the parents must be closed under
/// elimination: if `p` and `q < p` are both parents of `i`, then `q` is a
/// parent of `p`. The parents from a symbolic factorization are; [`Self::new`]
/// only checks it.
#[derive(Debug, Clone)]
pub(crate) struct SelectedInverseLayout {
    n_var: usize,
    n_edge: usize,
    /// Per variable `i`, the slots of `Sigma[P(i), P(i)]`, `|P(i)| x |P(i)|`
    /// row-major with the parents in edge order.
    parent_pairs: Ragged<u32>,
}

impl SelectedInverseLayout {
    pub(crate) fn new(graph: &Pattern) -> Result<Self> {
        let (n_var, n_edge) = (graph.n_var(), graph.n_edge());
        if u32::try_from(n_var + n_edge).is_err() {
            bail!("the selected inverse has too many entries");
        }
        let offsets = Offsets::from_lens((0..n_var).map(|i| graph.n_parent(i).pow(2)));
        let mut parent_pairs = Ragged::filled(offsets, 0u32);

        // `edge_of[q]` is the edge from `q` into `owner[q]`, for the row
        // loaded last.
        let mut edge_of = vec![0; n_var];
        let mut owner = vec![usize::MAX; n_var];
        for i in 0..n_var {
            let parents = graph.parents(i);
            let n_p = parents.len();
            let pairs = parent_pairs.row_mut(i);
            for (a, &p) in parents.iter().enumerate() {
                for (e, q) in graph.entries(p) {
                    edge_of[q] = e;
                    owner[q] = p;
                }
                // `Sigma[p, q]` for `q <= p` is on row `p`; `q > p` is filled
                // from row `q`, by symmetry.
                for (b, &q) in parents.iter().enumerate() {
                    let slot = if q == p {
                        p
                    } else if q < p {
                        if owner[q] != p {
                            bail!(
                                "the parents are not closed under elimination: {q} and \
                                 {p} are parents of {i}, but {q} is not a parent of {p}. \
                                 Use the `filled` pattern of a symbolic factorization."
                            );
                        }
                        n_var + edge_of[q]
                    } else {
                        continue;
                    };
                    pairs[a * n_p + b] = slot as u32;
                    pairs[b * n_p + a] = slot as u32;
                }
            }
        }
        Ok(Self {
            n_var,
            n_edge,
            parent_pairs,
        })
    }

    /// The number of slots.
    pub(crate) fn len(&self) -> usize {
        self.n_var + self.n_edge
    }

    /// The slot of `Sigma[i, i]`.
    #[inline(always)]
    pub(crate) fn diagonal(&self, i: usize) -> usize {
        i
    }

    /// The slot of `Sigma[i, p]` for the edge `e` from `p` into `i`.
    #[inline(always)]
    pub(crate) fn edge(&self, e: usize) -> usize {
        self.n_var + e
    }

    /// The slots of `Sigma[P(i), P(i)]`, see [`Self::parent_pairs`].
    #[inline(always)]
    pub(crate) fn parent_pairs(&self, i: usize) -> &[u32] {
        self.parent_pairs.row(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(rows: &[&[usize]]) -> Result<SelectedInverseLayout> {
        let rows = Ragged::from_rows(rows.iter().map(|r| r.iter().copied()));
        SelectedInverseLayout::new(&Pattern::new(rows).unwrap())
    }

    #[test]
    fn requires_closed_parents() {
        // 2 has parents 0 and 1, so 1 needs parent 0.
        assert!(layout(&[&[], &[], &[1, 0]]).is_err());
        let l = layout(&[&[], &[0], &[1, 0]]).unwrap();
        // Edges: 1 <- 0 is edge 0, 2 <- 1 is edge 1, 2 <- 0 is edge 2.
        assert_eq!(l.len(), 6);
        // Sigma[{1, 0}, {1, 0}] for variable 2: [[1, 3 + 0], [3 + 0, 0]].
        assert_eq!(l.parent_pairs(2), [1, 3, 3, 0]);
    }
}
