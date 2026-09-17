//! Detect the sparsity pattern of a Hessian from Hessian-vector products.
//!
//! The pattern is found in two stages:
//!
//! 1. Bloom filter probing (Hovland, "Jacobian sparsity detection using Bloom
//!    filters", 2024). Every column gets `num_hashes` random bits out of
//!    `bloom_size`, and we compute one HVP per bit, with the columns of that
//!    bit set to random weights. Entry `(i, j)` is a candidate if row `i` is
//!    nonzero for all bits of column `j`. This never misses a nonzero, but can
//!    produce false positives. Since the Hessian is symmetric we only keep
//!    `(i, j)` if `(j, i)` is a candidate as well.
//! 2. Exact verification. We colour the columns so that no row has two
//!    candidates of the same colour, and compute one HVP per colour. Each
//!    candidate then shows up on its own in the result, and the false
//!    positives are exactly zero. Columns with many candidates get a colour
//!    of their own and are recovered by symmetry, so that a single dense
//!    row (like a scale parameter of a hierarchical model) doesn't force
//!    all columns into different colours.
//!
//! The random weights in stage 1 matter: with 0/1 probe vectors entries
//! like `+1` and `-1` in a row cancel exactly, which is common in
//! statistical models.
//!
//! Autodiff Hessian-vector products give exact zeros for entries that do
//! not depend on each other, so the result is exact at the evaluated points.
//! Dependencies in branches of the model that are not taken at any point
//! can't be detected. NaN values are treated as nonzero.

use anyhow::{bail, Result};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Something that can compute products of its Hessian with vectors.
pub trait HessianVectorProduct {
    fn dim(&self) -> usize;

    /// Store the product of the Hessian at `point` with `vector` in `out`.
    fn hvp(&mut self, point: &[f64], vector: &[f64], out: &mut [f64]) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct SparsityOptions {
    /// Number of HVPs per point in the Bloom filter stage. Defaults to
    /// `32 * ceil(log2(n))`. If this is not much smaller than the dimension,
    /// we use unit vectors instead.
    pub bloom_size: Option<usize>,
    /// Number of bits per column in the Bloom filter.
    pub num_hashes: usize,
    pub seed: u64,
}

impl Default for SparsityOptions {
    fn default() -> Self {
        Self {
            bloom_size: None,
            num_hashes: 3,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SparsityPattern {
    /// Sorted column indices of the nonzeros of each row. The pattern is
    /// symmetric and contains the diagonal.
    pub rows: Vec<Vec<usize>>,
    /// Total number of Hessian-vector products that were computed.
    pub num_hvps: usize,
    /// Number of colours of the verification stage (the dimension, if unit
    /// vectors were used).
    pub num_colors: usize,
}

impl SparsityPattern {
    /// The pattern in CSR format: `(indptr, indices)`.
    pub fn to_csr(&self) -> (Vec<i64>, Vec<i64>) {
        let mut indptr = Vec::with_capacity(self.rows.len() + 1);
        let mut indices = Vec::new();
        indptr.push(0);
        for row in self.rows.iter() {
            indices.extend(row.iter().map(|&j| j as i64));
            indptr.push(indices.len() as i64);
        }
        (indptr, indices)
    }

    /// The pattern as a dense row-major boolean matrix.
    #[cfg(test)]
    pub fn to_dense(&self) -> Vec<bool> {
        let n = self.rows.len();
        let mut dense = vec![false; n * n];
        for (i, row) in self.rows.iter().enumerate() {
            for &j in row {
                dense[i * n + j] = true;
            }
        }
        dense
    }
}

fn default_bloom_size(n: usize) -> usize {
    let log2 = (usize::BITS - n.max(2).saturating_sub(1).leading_zeros()) as usize;
    32 * log2
}

/// Detect the sparsity pattern of the Hessian, as the union of the patterns
/// at `points`.
pub fn hessian_sparsity<H: HessianVectorProduct + ?Sized>(
    hessian: &mut H,
    points: &[Vec<f64>],
    options: &SparsityOptions,
) -> Result<SparsityPattern> {
    let n = hessian.dim();
    if points.is_empty() {
        bail!("Need at least one point to detect the Hessian sparsity");
    }
    if let Some(point) = points.iter().find(|point| point.len() != n) {
        bail!("Point has length {} (expected {n})", point.len());
    }
    if options.num_hashes == 0 {
        bail!("num_hashes must be positive");
    }

    let bloom_size = options.bloom_size.unwrap_or_else(|| default_bloom_size(n));
    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    let mut num_hvps = 0;

    // With a Bloom filter about as large as the dimension, the
    // verification stage alone with unit vectors is cheaper.
    let use_bloom = bloom_size >= options.num_hashes && 2 * bloom_size < n;
    let candidates = if use_bloom {
        let (candidates, hvps) =
            bloom_candidates(hessian, points, bloom_size, options.num_hashes, &mut rng)?;
        num_hvps += hvps;
        Some(candidates)
    } else {
        None
    };

    let (nonzeros, num_colors, hvps) = match candidates {
        Some(candidates) => verify_candidates(hessian, points, &candidates)?,
        None => unit_vector_pattern(hessian, points)?,
    };
    num_hvps += hvps;

    // Symmetrize and add the diagonal
    let mut rows: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, row) in nonzeros.iter().enumerate() {
        rows[i].push(i);
        for &j in row {
            rows[i].push(j);
            rows[j].push(i);
        }
    }
    for row in rows.iter_mut() {
        row.sort_unstable();
        row.dedup();
    }

    Ok(SparsityPattern {
        rows,
        num_hvps,
        num_colors,
    })
}

/// Stage 1: Candidates for the nonzeros of each row, as sorted column
/// indices. Returns the candidates and the number of HVPs.
fn bloom_candidates<H: HessianVectorProduct + ?Sized>(
    hessian: &mut H,
    points: &[Vec<f64>],
    bloom_size: usize,
    num_hashes: usize,
    rng: &mut ChaCha8Rng,
) -> Result<(Vec<Vec<usize>>, usize)> {
    let n = hessian.dim();

    // Distinct random bits for each column. We use the same bits at all
    // points, so that false positives at different points coincide.
    let mut bit_columns: Vec<Vec<usize>> = vec![vec![]; bloom_size];
    let mut bits = Vec::with_capacity(num_hashes);
    for j in 0..n {
        bits.clear();
        while bits.len() < num_hashes {
            let bit = rng.random_range(0..bloom_size);
            if !bits.contains(&bit) {
                bits.push(bit);
            }
        }
        for &bit in bits.iter() {
            bit_columns[bit].push(j);
        }
    }

    let words = bloom_size.div_ceil(64);
    let mut row_bits = vec![0u64; n * words];
    let mut vector = vec![0f64; n];
    let mut out = vec![0f64; n];
    let mut hit_count = vec![0usize; n];
    let mut touched = Vec::new();
    let mut candidates: Vec<Vec<usize>> = vec![vec![]; n];
    let mut num_hvps = 0;

    for point in points {
        // Which bits are nonzero in each row
        row_bits.fill(0);
        for (bit, columns) in bit_columns.iter().enumerate() {
            vector.fill(0.);
            for &j in columns {
                vector[j] = rng.random_range(1.0..2.0);
            }
            hessian.hvp(point, &vector, &mut out)?;
            num_hvps += 1;
            for (i, &val) in out.iter().enumerate() {
                if val != 0.0 {
                    row_bits[i * words + bit / 64] |= 1 << (bit % 64);
                }
            }
        }

        // Row i has candidate j if all bits of column j are set in row i.
        let mut point_candidates: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, row_candidates) in point_candidates.iter_mut().enumerate() {
            let row = &row_bits[i * words..(i + 1) * words];
            for (word_idx, &word) in row.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let bit = word_idx * 64 + word.trailing_zeros() as usize;
                    word &= word - 1;
                    for &j in bit_columns[bit].iter() {
                        if hit_count[j] == 0 {
                            touched.push(j);
                        }
                        hit_count[j] += 1;
                    }
                }
            }
            for &j in touched.iter() {
                if hit_count[j] == num_hashes {
                    row_candidates.push(j);
                }
                hit_count[j] = 0;
            }
            touched.clear();
            row_candidates.sort_unstable();
        }

        // The Hessian is symmetric, so both (i, j) and (j, i) must pass.
        for (i, row) in point_candidates.iter().enumerate() {
            for &j in row {
                if point_candidates[j].binary_search(&i).is_ok() {
                    candidates[i].push(j);
                }
            }
        }
    }

    for row in candidates.iter_mut() {
        row.sort_unstable();
        row.dedup();
    }
    Ok((candidates, num_hvps))
}

/// Choose columns that get a colour of their own ("dense" columns).
///
/// The HVP for a column on its own gives the whole column, and by symmetry
/// the whole row. So the rows of dense columns don't constrain the colouring
/// of the other columns. This matters for columns with many candidates, like
/// a scale parameter in a hierarchical model, which would otherwise force
/// all columns into different colours.
///
/// We greedily take columns with the most remaining candidates, and use the
/// prefix of that sequence that minimizes `num_dense + longest remaining
/// row`, which estimates the number of colours. Taking only one of several
/// dense columns often doesn't help, so we don't stop at the first step
/// without improvement.
fn choose_dense_columns(candidates: &[Vec<usize>]) -> Vec<bool> {
    let n = candidates.len();
    let mut dense = vec![false; n];
    // Number of candidates of each row in non-dense columns
    let mut remaining: Vec<usize> = candidates.iter().map(|row| row.len()).collect();

    let longest = |remaining: &[usize], dense: &[bool]| {
        remaining
            .iter()
            .zip(dense)
            .filter(|(_, &dense)| !dense)
            .map(|(&len, _)| len)
            .max()
            .unwrap_or(0)
    };

    let mut sequence = vec![];
    let mut best_cost = longest(&remaining, &dense);
    let mut best_len = 0;
    // With `best_cost` dense columns we can't improve anymore
    while sequence.len() + 1 < best_cost {
        let Some(column) = (0..n).filter(|&j| !dense[j]).max_by_key(|&j| remaining[j]) else {
            break;
        };
        dense[column] = true;
        for &i in candidates[column].iter() {
            remaining[i] -= 1;
        }
        sequence.push(column);
        let cost = sequence.len() + longest(&remaining, &dense);
        if cost < best_cost {
            best_cost = cost;
            best_len = sequence.len();
        }
    }

    for &column in sequence[best_len..].iter() {
        dense[column] = false;
    }
    dense
}

/// Colour the columns for the verification stage.
///
/// Dense columns (see `choose_dense_columns`) get a colour of their own. The
/// other columns are coloured greedily, such that no non-dense row has
/// candidates in two non-dense columns of the same colour. Returns the
/// colours of the columns, the number of colours and which columns are
/// dense.
fn color_columns(candidates: &[Vec<usize>]) -> (Vec<usize>, usize, Vec<bool>) {
    const UNCOLORED: usize = usize::MAX;
    let n = candidates.len();
    let dense = choose_dense_columns(candidates);

    // The candidates are symmetric, so the rows with a candidate in column
    // j are `candidates[j]`. Colour columns with many candidates first.
    let mut order: Vec<usize> = (0..n).filter(|&j| !dense[j]).collect();
    order.sort_by_key(|&j| std::cmp::Reverse(candidates[j].len()));

    let mut colors = vec![UNCOLORED; n];
    // `forbidden[c] == j` marks colour c as taken by a neighbour of column j
    let mut forbidden: Vec<usize> = vec![];
    let mut num_colors = 0;
    for &j in order.iter() {
        for &i in candidates[j].iter().filter(|&&i| !dense[i]) {
            for &k in candidates[i].iter() {
                let color = colors[k];
                if color != UNCOLORED && !dense[k] {
                    forbidden[color] = j;
                }
            }
        }
        let color = (0..num_colors)
            .find(|&color| forbidden[color] != j)
            .unwrap_or_else(|| {
                num_colors += 1;
                forbidden.push(UNCOLORED);
                num_colors - 1
            });
        colors[j] = color;
    }

    for j in (0..n).filter(|&j| dense[j]) {
        colors[j] = num_colors;
        num_colors += 1;
    }
    (colors, num_colors, dense)
}

/// Stage 2: Check which candidates are actually nonzero. Returns the
/// nonzeros of each row, the number of colours and the number of HVPs.
fn verify_candidates<H: HessianVectorProduct + ?Sized>(
    hessian: &mut H,
    points: &[Vec<f64>],
    candidates: &[Vec<usize>],
) -> Result<(Vec<Vec<usize>>, usize, usize)> {
    let n = hessian.dim();
    let (colors, num_colors, dense) = color_columns(candidates);
    let mut color_columns: Vec<Vec<usize>> = vec![vec![]; num_colors];
    for (j, &color) in colors.iter().enumerate() {
        color_columns[color].push(j);
    }

    let mut vector = vec![0f64; n];
    let mut out = vec![0f64; n];
    let mut nonzeros: Vec<Vec<usize>> = vec![vec![]; n];
    let mut num_hvps = 0;

    for point in points {
        for columns in color_columns.iter() {
            vector.fill(0.);
            for &j in columns {
                vector[j] = 1.;
            }
            hessian.hvp(point, &vector, &mut out)?;
            num_hvps += 1;
            // A dense column is alone in its colour, so `out` is the whole
            // column. Otherwise a non-dense row i has at most one candidate
            // in this colour, so out[i] is exactly the entry (i, j) of that
            // candidate. Entries in dense rows are known from the dense
            // column by symmetry.
            for &j in columns {
                for &i in candidates[j].iter() {
                    if (dense[j] || !dense[i]) && out[i] != 0.0 {
                        nonzeros[i].push(j);
                    }
                }
            }
        }
    }
    Ok((nonzeros, num_colors, num_hvps))
}

/// Compute the pattern column by column with unit vectors. Returns the
/// nonzeros of each row, the number of colours and the number of HVPs.
fn unit_vector_pattern<H: HessianVectorProduct + ?Sized>(
    hessian: &mut H,
    points: &[Vec<f64>],
) -> Result<(Vec<Vec<usize>>, usize, usize)> {
    let n = hessian.dim();
    let mut vector = vec![0f64; n];
    let mut out = vec![0f64; n];
    let mut nonzeros: Vec<Vec<usize>> = vec![vec![]; n];
    let mut num_hvps = 0;
    for point in points {
        for j in 0..n {
            vector.fill(0.);
            vector[j] = 1.;
            hessian.hvp(point, &vector, &mut out)?;
            num_hvps += 1;
            for (i, &val) in out.iter().enumerate() {
                if val != 0.0 {
                    nonzeros[i].push(j);
                }
            }
        }
    }
    Ok((nonzeros, n, num_hvps))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense symmetric matrix that ignores the point.
    struct DenseHessian {
        n: usize,
        values: Vec<f64>,
    }

    impl DenseHessian {
        fn new(n: usize, entries: &[(usize, usize, f64)]) -> Self {
            let mut values = vec![0.; n * n];
            for &(i, j, val) in entries {
                values[i * n + j] = val;
                values[j * n + i] = val;
            }
            Self { n, values }
        }

        fn pattern(&self) -> Vec<bool> {
            let mut pattern: Vec<bool> = self.values.iter().map(|&val| val != 0.).collect();
            for i in 0..self.n {
                pattern[i * self.n + i] = true;
            }
            pattern
        }
    }

    impl HessianVectorProduct for DenseHessian {
        fn dim(&self) -> usize {
            self.n
        }

        fn hvp(&mut self, _point: &[f64], vector: &[f64], out: &mut [f64]) -> Result<()> {
            for (i, out) in out.iter_mut().enumerate() {
                *out = (0..self.n)
                    .map(|j| self.values[i * self.n + j] * vector[j])
                    .sum();
            }
            Ok(())
        }
    }

    fn random_sparse(n: usize, per_row: usize, seed: u64) -> DenseHessian {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut entries = vec![];
        for i in 0..n {
            entries.push((i, i, -1.));
            for _ in 0..per_row / 2 {
                let j = rng.random_range(0..n);
                entries.push((i, j, rng.random_range(-1.0..1.0)));
            }
        }
        DenseHessian::new(n, &entries)
    }

    fn check(hessian: &mut DenseHessian, options: &SparsityOptions) -> SparsityPattern {
        let points = vec![vec![0.; hessian.n]];
        let pattern = hessian_sparsity(hessian, &points, options).unwrap();
        assert_eq!(pattern.to_dense(), hessian.pattern());
        pattern
    }

    #[test]
    fn unit_vectors_for_small_problems() {
        let mut hessian = DenseHessian::new(5, &[(0, 1, 2.), (1, 3, -1.), (2, 2, 1.)]);
        let pattern = check(&mut hessian, &SparsityOptions::default());
        assert_eq!(pattern.num_hvps, 5);
    }

    #[test]
    fn bloom_random_sparse() {
        let mut hessian = random_sparse(1000, 10, 1);
        let pattern = check(&mut hessian, &SparsityOptions::default());
        assert!(pattern.num_hvps < 1000, "{} HVPs", pattern.num_hvps);
    }

    #[test]
    fn bloom_small_filter() {
        // Many false positives in stage 1, which stage 2 must remove
        let mut hessian = random_sparse(500, 10, 2);
        let options = SparsityOptions {
            bloom_size: Some(40),
            ..Default::default()
        };
        check(&mut hessian, &options);
    }

    #[test]
    fn bloom_cancellation() {
        // A chain x[i] ~ normal(x[i - 1], 1): the entries of each row sum
        // to zero, which 0/1 probes would miss.
        let n = 1000;
        let mut entries = vec![];
        for i in 0..n {
            entries.push((i, i, if i == 0 || i == n - 1 { -1. } else { -2. }));
            if i > 0 {
                entries.push((i, i - 1, 1.));
            }
        }
        let mut hessian = DenseHessian::new(n, &entries);
        let pattern = check(&mut hessian, &SparsityOptions::default());
        assert!(pattern.num_hvps < n, "{} HVPs", pattern.num_hvps);
    }

    #[test]
    fn bloom_dense_row() {
        // Arrowhead: one parameter interacts with all others
        let n = 2000;
        let mut entries: Vec<_> = (0..n).map(|i| (i, i, -1.)).collect();
        entries.extend((1..n).map(|j| (0, j, 0.5)));
        let mut hessian = DenseHessian::new(n, &entries);
        let pattern = check(&mut hessian, &SparsityOptions::default());
        assert!(pattern.num_colors <= 3, "{} colours", pattern.num_colors);
    }

    #[test]
    fn bloom_hierarchical() {
        // Two hyperparameters (0, 1) interact with all group effects, each
        // group effect with its own observations-level parameter.
        let groups = 500;
        let n = 2 + 2 * groups;
        let mut entries: Vec<_> = (0..n).map(|i| (i, i, -1.)).collect();
        entries.push((0, 1, 0.3));
        for g in 0..groups {
            let effect = 2 + g;
            let local = 2 + groups + g;
            entries.push((0, effect, 0.5));
            entries.push((1, effect, -0.5));
            entries.push((effect, local, 0.2));
        }
        let mut hessian = DenseHessian::new(n, &entries);
        let pattern = check(&mut hessian, &SparsityOptions::default());
        assert!(pattern.num_colors <= 6, "{} colours", pattern.num_colors);
    }

    #[test]
    fn union_over_points() {
        // Entries that are only nonzero at some of the points
        struct Switching;
        impl HessianVectorProduct for Switching {
            fn dim(&self) -> usize {
                3
            }
            fn hvp(&mut self, point: &[f64], vector: &[f64], out: &mut [f64]) -> Result<()> {
                out.copy_from_slice(vector);
                if point[0] > 0. {
                    out[1] += vector[2];
                    out[2] += vector[1];
                }
                Ok(())
            }
        }
        let points = vec![vec![-1., 0., 0.], vec![1., 0., 0.]];
        let pattern =
            hessian_sparsity(&mut Switching, &points, &SparsityOptions::default()).unwrap();
        assert_eq!(pattern.rows, vec![vec![0], vec![1, 2], vec![1, 2]]);
    }
}
