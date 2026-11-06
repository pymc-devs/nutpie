//! Sparsity patterns and ragged arrays.
//!
//! The map's structure is a strictly lower triangular [`Pattern`]: row `i`
//! lists the parents of variable `i`, in conditioner input order. Its entries
//! are numbered row by row, and an entry's number is the id of that edge,
//! which indexes every per-edge array (Jacobian entries, skip weights,
//! cotangents). [`Children`] is its transpose, for sweeps that push work from
//! a variable to the variables that read it.

use std::ops::Range;

use anyhow::{anyhow, bail, Result};

/// Where `n` consecutive rows of a flat array start and end: row `i` is
/// [`Self::range`]`(i)`.
#[derive(Debug, Clone)]
pub(crate) struct Offsets(Vec<usize>);

impl Offsets {
    /// Checks that `offsets` starts at zero and does not decrease.
    pub(crate) fn new(offsets: Vec<usize>) -> Result<Self> {
        if offsets.first() != Some(&0) || offsets.windows(2).any(|w| w[0] > w[1]) {
            bail!("offsets must start at zero and must not decrease");
        }
        Ok(Self(offsets))
    }

    /// From Python's `int64` offsets.
    pub(crate) fn from_i64(offsets: &[i64]) -> Result<Self> {
        Self::new(to_usize(offsets)?)
    }

    pub(crate) fn from_lens(lens: impl IntoIterator<Item = usize>) -> Self {
        let mut offsets = vec![0];
        let mut total = 0;
        for len in lens {
            total += len;
            offsets.push(total);
        }
        Self(offsets)
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.0.len() - 1
    }

    /// The length of the flat array.
    pub(crate) fn total(&self) -> usize {
        self.0[self.n_rows()]
    }

    #[inline(always)]
    pub(crate) fn start(&self, i: usize) -> usize {
        self.0[i]
    }

    #[inline(always)]
    pub(crate) fn range(&self, i: usize) -> Range<usize> {
        self.0[i]..self.0[i + 1]
    }

    #[inline(always)]
    pub(crate) fn row_len(&self, i: usize) -> usize {
        self.0[i + 1] - self.0[i]
    }

    pub(crate) fn max_len(&self) -> usize {
        self.0.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
    }

    /// The row holding flat position `position`.
    pub(crate) fn row_of(&self, position: usize) -> usize {
        self.0.partition_point(|&o| o <= position) - 1
    }

    pub(crate) fn as_slice(&self) -> &[usize] {
        &self.0
    }
}

/// Rows of varying length, stored flat.
#[derive(Debug, Clone)]
pub(crate) struct Ragged<T> {
    offsets: Offsets,
    values: Vec<T>,
}

impl<T> Ragged<T> {
    pub(crate) fn new(offsets: Offsets, values: Vec<T>) -> Result<Self> {
        if offsets.total() != values.len() {
            bail!(
                "offsets cover {} values, but there are {}",
                offsets.total(),
                values.len()
            );
        }
        Ok(Self { offsets, values })
    }

    /// Rows of the lengths `offsets` gives, every value `value`.
    pub(crate) fn filled(offsets: Offsets, value: T) -> Self
    where
        T: Clone,
    {
        let values = vec![value; offsets.total()];
        Self { offsets, values }
    }

    #[cfg(test)]
    pub(crate) fn from_rows<R: IntoIterator<Item = T>>(rows: impl IntoIterator<Item = R>) -> Self {
        let mut offsets = vec![0];
        let mut values = Vec::new();
        for row in rows {
            values.extend(row);
            offsets.push(values.len());
        }
        Self {
            offsets: Offsets(offsets),
            values,
        }
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.offsets.n_rows()
    }

    #[inline(always)]
    pub(crate) fn row(&self, i: usize) -> &[T] {
        &self.values[self.offsets.range(i)]
    }

    #[inline(always)]
    pub(crate) fn row_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.values[self.offsets.range(i)]
    }

    #[inline(always)]
    pub(crate) fn range(&self, i: usize) -> Range<usize> {
        self.offsets.range(i)
    }

    pub(crate) fn offsets(&self) -> &Offsets {
        &self.offsets
    }

    /// All rows, concatenated.
    pub(crate) fn values(&self) -> &[T] {
        &self.values
    }
}

/// A variable index as a [`Pattern`] stores it: `usize`, or `u32` to halve
/// the index traffic of the transform's sweeps.
pub(crate) trait VarIndex: Copy + Eq + std::fmt::Debug {
    fn from_usize(value: usize) -> Option<Self>;
    fn index(self) -> usize;
}

impl VarIndex for usize {
    #[inline(always)]
    fn from_usize(value: usize) -> Option<Self> {
        Some(value)
    }

    #[inline(always)]
    fn index(self) -> usize {
        self
    }
}

impl VarIndex for u32 {
    #[inline(always)]
    fn from_usize(value: usize) -> Option<Self> {
        u32::try_from(value).ok()
    }

    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

/// The parents of each variable: a strictly lower triangular pattern, with
/// edges numbered row by row.
#[derive(Debug, Clone)]
pub(crate) struct Pattern<I = usize> {
    rows: Ragged<I>,
}

impl<I: VarIndex> Pattern<I> {
    /// Checks that every variable lists distinct parents that precede it, and
    /// that edge ids fit `I`.
    pub(crate) fn new(rows: Ragged<I>) -> Result<Self> {
        if I::from_usize(rows.values().len()).is_none() {
            bail!("too many edges for the index type");
        }
        let n_var = rows.n_rows();
        // `seen[p] == i` once row `i` has listed `p`.
        let mut seen = vec![usize::MAX; n_var];
        for i in 0..n_var {
            for &p in rows.row(i) {
                let p = p.index();
                if p >= i {
                    bail!("variable {i} has parent {p}, which does not precede it");
                }
                if seen[p] == i {
                    bail!("variable {i} lists parent {p} twice");
                }
                seen[p] = i;
            }
        }
        Ok(Self { rows })
    }

    /// From Python's `parent_indptr` and `parent_index`.
    pub(crate) fn from_i64(indptr: &[i64], index: &[i64]) -> Result<Self> {
        let index = index
            .iter()
            .map(|&v| {
                usize::try_from(v)
                    .ok()
                    .and_then(I::from_usize)
                    .ok_or_else(|| anyhow!("parent index {v} is out of range"))
            })
            .collect::<Result<Vec<I>>>()?;
        Self::new(Ragged::new(Offsets::from_i64(indptr)?, index)?)
    }

    pub(crate) fn n_var(&self) -> usize {
        self.rows.n_rows()
    }

    pub(crate) fn n_edge(&self) -> usize {
        self.rows.values().len()
    }

    /// The ids of the edges into variable `i`.
    #[inline(always)]
    pub(crate) fn edges(&self, i: usize) -> Range<usize> {
        self.rows.range(i)
    }

    #[inline(always)]
    pub(crate) fn parents(&self, i: usize) -> &[I] {
        self.rows.row(i)
    }

    #[inline(always)]
    pub(crate) fn n_parent(&self, i: usize) -> usize {
        self.rows.offsets().row_len(i)
    }

    pub(crate) fn max_parent(&self) -> usize {
        self.rows.offsets().max_len()
    }

    /// `(edge, parent)` for each parent of variable `i`.
    #[inline(always)]
    pub(crate) fn entries(&self, i: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.edges(i).zip(self.parents(i).iter().map(|p| p.index()))
    }

    /// The transpose, by a counting sort over the parents.
    pub(crate) fn children(&self) -> Children<I> {
        let n_var = self.n_var();
        let mut lens = vec![0usize; n_var];
        for &p in self.rows.values() {
            lens[p.index()] += 1;
        }
        let offsets = Offsets::from_lens(lens);
        let mut cursor = offsets.as_slice()[..n_var].to_vec();
        let mut child = vec![I::from_usize(0).expect("zero fits"); self.n_edge()];
        let mut edge = child.clone();
        for i in 0..n_var {
            for (e, p) in self.entries(i) {
                let slot = &mut cursor[p];
                child[*slot] = I::from_usize(i).expect("variables fit, as edges do");
                edge[*slot] = I::from_usize(e).expect("checked in `new`");
                *slot += 1;
            }
        }
        Children {
            rows: Ragged {
                offsets,
                values: child,
            },
            edge,
        }
    }
}

/// The transpose of a [`Pattern`]: each variable's children, each with the
/// id of the edge it reads the variable through.
#[derive(Debug, Clone)]
pub(crate) struct Children<I = usize> {
    rows: Ragged<I>,
    edge: Vec<I>,
}

impl<I: VarIndex> Children<I> {
    #[inline(always)]
    pub(crate) fn children(&self, p: usize) -> &[I] {
        self.rows.row(p)
    }

    /// `(child, edge)` for each child of variable `p`.
    #[inline(always)]
    pub(crate) fn entries(&self, p: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let range = self.rows.range(p);
        self.rows.values()[range.clone()]
            .iter()
            .zip(&self.edge[range])
            .map(|(c, e)| (c.index(), e.index()))
    }
}

/// Python's `int64` indices as `usize`.
pub(crate) fn to_usize(values: &[i64]) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| anyhow!("negative index {v}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(rows: &[&[usize]]) -> Result<Pattern> {
        Pattern::new(Ragged::from_rows(rows.iter().map(|r| r.iter().copied())))
    }

    #[test]
    fn rejects_bad_parents() {
        assert!(pattern(&[&[], &[0], &[1, 0]]).is_ok());
        assert!(pattern(&[&[], &[1]]).is_err());
        assert!(pattern(&[&[], &[0], &[0, 0]]).is_err());
        assert!(Pattern::<usize>::from_i64(&[0, 1], &[0, 0]).is_err());
        assert!(Pattern::<usize>::from_i64(&[0, 0, 2], &[0]).is_err());
    }

    #[test]
    fn children_transpose_the_edges() {
        let p = pattern(&[&[], &[0], &[1, 0], &[2]]).unwrap();
        let c = p.children();
        assert_eq!(c.entries(0).collect::<Vec<_>>(), [(1, 0), (2, 2)]);
        assert_eq!(c.entries(1).collect::<Vec<_>>(), [(2, 1)]);
        assert_eq!(c.entries(2).collect::<Vec<_>>(), [(3, 3)]);
        assert!(c.children(3).is_empty());
    }
}
