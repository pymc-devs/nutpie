//! Diagnostics of a fitted map, for `nutpie.flow_diagnostics`: how strongly
//! each parent moves the conditional distribution of its child
//! ([`FisherResiduals::fisher_speeds`]), and how much of the MLPs' hidden
//! layers the fit uses ([`FisherResiduals::gate_eigenvalues`]). The math is
//! in the module docstring of `nutpie.flow_diagnostics`.

use anyhow::{bail, Result};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par, Side};
use fearless_simd::{dispatch, prelude::*};
use rayon::prelude::*;

use super::local::LocalState;
use super::tiles::{Data, Linearization};
use super::FisherResiduals;
use crate::triangular::conditioner::Conditioner;
use crate::triangular::transformer::Transformer;

/// Eigenvalues of the skip block below this fraction of the largest are
/// dropped from its pseudo-inverse, as `numpy.linalg.pinv`'s `rcond`.
const SKIP_RCOND: f64 = 1e-10;

/// Per edge, in the graph's edge order, the squared Fisher speeds of the
/// child's conditional per unit of the parent, averaged over the draws:
/// from changes of location and skew (the odd part of the score), and of
/// scale and tails (the even part).
pub(crate) struct FisherSpeeds {
    pub(crate) location_skew: Vec<f64>,
    pub(crate) scale_tails: Vec<f64>,
}

impl FisherResiduals {
    /// The squared Fisher speeds of every edge at `theta`, over the data's
    /// draws (their gradients are not used).
    ///
    /// For one draw, variable `i` has the conditional `y_i = T(x; pi)`, `x`
    /// standard normal, with `pi` its conditioners' output. With the latent
    /// velocities `V_a = (dT/dpi_a) / T'` and the scores `s_a = x V_a -
    /// V_a'`, parent `j` moves the conditional with the score `s . C[:, j]`;
    /// its weighted mean square at the Gauss-Hermite `nodes` (symmetric
    /// about zero) and `weights` (summing to one) is the squared speed.
    pub(crate) fn fisher_speeds(
        &self,
        theta: &[f64],
        nodes: &[f64],
        weights: &[f64],
    ) -> Result<FisherSpeeds> {
        self.check_params(theta, "theta")?;
        let Some(data) = &self.data else {
            bail!("no data: call `set_data` first");
        };
        if nodes.is_empty() || nodes.len() != weights.len() {
            bail!("nodes and weights must be nonempty and of the same length");
        }
        let n_k = nodes.len();
        if (0..n_k).any(|k| (nodes[k] + nodes[n_k - 1 - k]).abs() > 1e-12) {
            bail!("the nodes must be ascending and symmetric about zero");
        }
        let per_var = dispatch!(self.level, simd => self.fisher_speeds_with(simd, theta, data, nodes, weights));
        let (location_skew, scale_tails) = per_var.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        Ok(FisherSpeeds {
            location_skew: location_skew.concat(),
            scale_tails: scale_tails.concat(),
        })
    }

    fn fisher_speeds_with<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        data: &Data,
        nodes: &[f64],
        weights: &[f64],
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        (0..self.n_var())
            .into_par_iter()
            .map_init(
                || self.local_state(simd),
                |st, i| self.variable_speeds(simd, theta, data, nodes, weights, i, st),
            )
            .collect()
    }

    /// Variable `i`'s squared speeds per parent, `(location_skew,
    /// scale_tails)`. The draws are summed in order, so the result does not
    /// depend on the threads.
    #[allow(clippy::too_many_arguments)]
    fn variable_speeds<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        data: &Data,
        nodes: &[f64],
        weights: &[f64],
        i: usize,
        st: &mut LocalState<S>,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_p = self.graph.n_parent(i);
        let (mut odd_sum, mut even_sum) = (vec![0.0; n_p], vec![0.0; n_p]);
        if n_p == 0 {
            return (odd_sum, even_sum);
        }
        let transformer = &self.conditional.transformer;
        let theta_i = self.params_of(theta, i);
        let (n_c, n_k) = (self.conditional.n_coefs(), nodes.len());
        let zero = S::f64s::splat(simd, 0.0);
        // `C`, column-major `(n_c, n_p)`, for the tile and for one lane.
        let mut jac = vec![zero; n_c * n_p];
        let mut jac_lane = vec![0.0; n_c * n_p];
        let mut pi = vec![0.0; n_c];
        let (mut dy, mut dl) = (vec![0.0; n_c], vec![0.0; n_c]);
        let mut score = vec![0.0; n_c];
        // Per node, the score of each parent.
        let mut speeds = vec![0.0; n_k * n_p];
        let mut tape = transformer.new_scratch();

        for tile in 0..data.n_tile() {
            st.y_parents.clear();
            st.y_parents.extend(
                self.graph
                    .parents(i)
                    .iter()
                    .map(|&p| data.y.value(simd, tile, p)),
            );
            self.conditional.evaluate(simd, theta_i, st);
            jac.fill(zero);
            self.conditional.conditioners.add_parent_jacobian_tile(
                simd,
                theta_i,
                &st.y_parents,
                &mut st.work,
                &mut jac,
            );
            for lane in 0..self.width {
                for (out, value) in pi.iter_mut().zip(&st.pi) {
                    *out = value[lane];
                }
                for (out, value) in jac_lane.iter_mut().zip(&jac) {
                    *out = value[lane];
                }
                for (k, &x) in nodes.iter().enumerate() {
                    // `log_det = log T'`, so `V_a' = dlog_det/dpi_a - V_a
                    // dlog_det/dx`.
                    let (_, _, dy_dx, dl_dx) =
                        transformer.forward_draw_with_grads(&mut tape, &pi, x, &mut dy, &mut dl);
                    for ((s, &dy), &dl) in score.iter_mut().zip(&dy).zip(&dl) {
                        let v = dy / dy_dx;
                        *s = x * v - (dl - v * dl_dx);
                    }
                    let row = &mut speeds[k * n_p..(k + 1) * n_p];
                    for (out, column) in row.iter_mut().zip(jac_lane.chunks_exact(n_c)) {
                        *out = score.iter().zip(column).map(|(s, c)| s * c).sum();
                    }
                }
                // Node `n_k - 1 - k` is `-x_k`. An even velocity has an odd
                // score: location and skew.
                for (k, &w) in weights.iter().enumerate() {
                    let r = n_k - 1 - k;
                    for j in 0..n_p {
                        let (a, b) = (speeds[k * n_p + j], speeds[r * n_p + j]);
                        odd_sum[j] += w * (0.5 * (a - b)).powi(2);
                        even_sum[j] += w * (0.5 * (a + b)).powi(2);
                    }
                }
            }
        }
        let inv_n = 1.0 / data.n_draw as f64;
        for value in odd_sum.iter_mut().chain(&mut even_sum) {
            *value *= inv_n;
        }
        (odd_sum, even_sum)
    }

    /// The hidden units of all MLPs together, the columns of
    /// [`Self::gate_eigenvalues`].
    pub(crate) fn hidden_width(&self) -> usize {
        self.conditional
            .conditioners
            .mlps
            .iter()
            .map(|mlp| mlp.n_unit())
            .sum()
    }

    /// How much of its MLPs' hidden units each variable's fit uses, at the
    /// linearization: row-major `(n_var, hidden_width)`, each row in
    /// decreasing order.
    ///
    /// Gating variable `i`'s units, `W2 diag(1 + g) h`, and its linear
    /// weights, `w_kj (1 + g_kj)`, with the constant absorbing their mean
    /// output over the draws (where it feeds the same coefficient), changes
    /// its weights by `D g` ([`Self::gate_directions`]). With `B` its exact
    /// Gauss-Newton block, `G = D^T B D` is the curvature of the Fisher
    /// divergence in the gates. The eigenvalues are those of `G`'s hidden
    /// block after projecting out the linear gates (a Schur complement), so
    /// they only count what the hidden layer adds beyond the linear
    /// conditioner: eigenvalue `mu` is a direction whose removal raises the
    /// Fisher divergence by about `mu`. Cross-variable curvature is left
    /// out.
    pub(crate) fn gate_eigenvalues(&self, lin: &Linearization) -> Result<Vec<f64>> {
        let width = self.hidden_width();
        if width == 0 {
            bail!("the flow has no MLP conditioner");
        }
        let means = dispatch!(self.level, simd => self.feature_means(simd, lin));
        let max_block = self.param_offset.max_len().max(1);
        let mut blocks: Vec<Option<Mat<f64>>> = (0..self.n_var()).map(|_| None).collect();
        for block in self.gauss_newton_blocks(lin, max_block)? {
            blocks[self.param_offset.row_of(block.start)] = Some(block.matrix);
        }
        let rows: Vec<Vec<f64>> = blocks
            .into_par_iter()
            .zip(means)
            .enumerate()
            .map(|(i, (block, mean))| match block {
                Some(block) => {
                    let d = self.gate_directions(i, &lin.theta, &mean);
                    hidden_eigenvalues(&gate_gram(&block, &d), width)
                }
                None => vec![0.0; width],
            })
            .collect();
        Ok(rows.concat())
    }

    /// Per variable, the mean over the draws of its gates' features at
    /// `lin`'s `theta`: the MLPs' activations, then its parents. Zero
    /// activations for a variable without parents.
    fn feature_means<S: Simd>(&self, simd: S, lin: &Linearization) -> Vec<Vec<f64>> {
        let data = &lin.data;
        let width = self.hidden_width();
        let mlps = &self.conditional.conditioners.mlps;
        (0..self.n_var())
            .into_par_iter()
            .map_init(
                || self.local_state(simd),
                |st, i| {
                    let n_p = self.graph.n_parent(i);
                    let mut sums = vec![0.0; width + n_p];
                    if n_p == 0 {
                        return sums;
                    }
                    let theta_i = self.params_of(&lin.theta, i);
                    for tile in 0..data.n_tile() {
                        st.y_parents.clear();
                        st.y_parents.extend(
                            self.graph
                                .parents(i)
                                .iter()
                                .map(|&p| data.y.value(simd, tile, p)),
                        );
                        self.conditional.evaluate(simd, theta_i, st);
                        let hidden = (0..mlps.len()).flat_map(|k| st.work.mlp_hidden(k));
                        for (sum, value) in sums.iter_mut().zip(hidden.chain(&st.y_parents)) {
                            *sum += (0..self.width).map(|lane| value[lane]).sum::<f64>();
                        }
                    }
                    let inv_n = 1.0 / data.n_draw as f64;
                    sums.iter_mut().for_each(|sum| *sum *= inv_n);
                    sums
                },
            )
            .collect()
    }

    /// `D`, `(n_theta, n_gate)`: the change of variable `i`'s weights per
    /// gate, see [`Self::gate_eigenvalues`]. The gates are the MLPs' units in
    /// order, then the linear conditioner's weights in their layout. `mean`
    /// is the variable's [`Self::feature_means`].
    pub(super) fn gate_directions(&self, i: usize, theta: &[f64], mean: &[f64]) -> Mat<f64> {
        let conditioners = &self.conditional.conditioners;
        let n_p = self.graph.n_parent(i);
        let theta_i = self.params_of(theta, i);
        let width = self.hidden_width();
        let n_linear = conditioners
            .linear
            .as_ref()
            .map_or(0, |c| c.outputs().len() * n_p);
        let mut d = Mat::<f64>::zeros(theta_i.len(), width + n_linear);

        let mut ranges = conditioners.weight_ranges(n_p).into_iter();
        let constant = conditioners
            .constant
            .as_ref()
            .map(|c| (c.outputs(), ranges.next().expect("a range per conditioner")));
        let linear = conditioners
            .linear
            .as_ref()
            .map(|c| (c.outputs(), ranges.next().expect("a range per conditioner")));
        // The constant takes the mean of a gate's output on coefficient `k`.
        let absorb = |d: &mut Mat<f64>, col: usize, k: usize, value: f64| {
            if let Some((outputs, range)) = &constant {
                if let Ok(j) = outputs.coefs().binary_search(&k) {
                    d[(range.start + j, col)] -= value;
                }
            }
        };

        let mut col = 0;
        for mlp in &conditioners.mlps {
            let range = ranges.next().expect("a range per conditioner");
            if n_p == 0 {
                col += mlp.n_unit();
                continue;
            }
            for u in 0..mlp.n_unit() {
                let (_, _, w2) = mlp.unit(&theta_i[range.clone()], n_p, u);
                let start = range.start + u * mlp.stride(n_p) + n_p + 1;
                for (j, (&w, &k)) in w2.iter().zip(mlp.outputs().coefs()).enumerate() {
                    d[(start + j, col)] = w;
                    absorb(&mut d, col, k, w * mean[col]);
                }
                col += 1;
            }
        }
        if let Some((outputs, range)) = linear {
            for (j, &k) in outputs.coefs().iter().enumerate() {
                for p in 0..n_p {
                    let index = range.start + j * n_p + p;
                    let w = theta_i[index];
                    d[(index, col)] = w;
                    absorb(&mut d, col, k, w * mean[width + p]);
                    col += 1;
                }
            }
        }
        d
    }
}

/// `D^T B D`, symmetrized.
pub(super) fn gate_gram(b: &Mat<f64>, d: &Mat<f64>) -> Mat<f64> {
    let mut bd = Mat::<f64>::zeros(b.nrows(), d.ncols());
    matmul(
        bd.as_mut(),
        Accum::Replace,
        b.as_ref(),
        d.as_ref(),
        1.0,
        Par::Seq,
    );
    let mut g = Mat::<f64>::zeros(d.ncols(), d.ncols());
    matmul(
        g.as_mut(),
        Accum::Replace,
        d.transpose(),
        bd.as_ref(),
        1.0,
        Par::Seq,
    );
    let n = g.nrows();
    for r in 0..n {
        for c in 0..r {
            let value = 0.5 * (g[(r, c)] + g[(c, r)]);
            g[(r, c)] = value;
            g[(c, r)] = value;
        }
    }
    g
}

/// The eigenvalues of the Schur complement of `gram`'s trailing block in
/// its leading `width` rows and columns, decreasing and clipped at zero. The
/// trailing block is pseudo-inverted, see [`SKIP_RCOND`].
fn hidden_eigenvalues(gram: &Mat<f64>, width: usize) -> Vec<f64> {
    let n_skip = gram.nrows() - width;
    let mut hidden = Mat::<f64>::from_fn(width, width, |r, c| gram[(r, c)]);
    if n_skip > 0 {
        let skip = Mat::<f64>::from_fn(n_skip, n_skip, |r, c| gram[(width + r, width + c)]);
        let eigen = skip
            .self_adjoint_eigen(Side::Lower)
            .expect("the eigendecomposition of a symmetric matrix");
        let (values, vectors) = (eigen.S().column_vector(), eigen.U());
        let largest = (0..n_skip).map(|l| values[l].abs()).fold(0.0, f64::max);
        for l in 0..n_skip {
            if values[l].abs() <= SKIP_RCOND * largest {
                continue;
            }
            // `K u_l`, with `K` the cross block.
            let ku: Vec<f64> = (0..width)
                .map(|r| {
                    (0..n_skip)
                        .map(|c| gram[(r, width + c)] * vectors[(c, l)])
                        .sum()
                })
                .collect();
            for r in 0..width {
                for c in 0..width {
                    hidden[(r, c)] -= ku[r] * ku[c] / values[l];
                }
            }
        }
    }
    let eigen = hidden
        .self_adjoint_eigen(Side::Lower)
        .expect("the eigendecomposition of a symmetric matrix");
    let values = eigen.S().column_vector();
    (0..width).rev().map(|l| values[l].max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use fearless_simd::dispatch;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::super::tests::{normal, problem_with};
    use super::gate_gram;
    use crate::triangular::transformer::Transformer;

    fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tol * expected.abs().max(1.0),
            "{what}: {actual} != {expected}"
        );
    }

    /// `D^T B D` from the exact blocks equals `(J D)^T (J D)` from the
    /// pushforward, for every variable.
    #[test]
    fn gate_gram_matches_pushforward() {
        let problem = problem_with(16);
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let theta = normal(&mut rng, problem.n_params(), 0.3);
        let (_, lin) = problem.residuals(&theta).unwrap();
        let means = dispatch!(problem.level, simd => problem.feature_means(simd, &lin));
        let blocks = problem
            .gauss_newton_blocks(&lin, problem.param_offset.max_len())
            .unwrap();
        for block in blocks {
            let i = problem.param_offset.row_of(block.start);
            let d = problem.gate_directions(i, &theta, &means[i]);
            let gram = gate_gram(&block.matrix, &d);
            let jd: Vec<Vec<f64>> = (0..d.ncols())
                .map(|c| {
                    let mut v = vec![0.0; problem.n_params()];
                    for (r, out) in v[problem.params(i)].iter_mut().enumerate() {
                        *out = d[(r, c)];
                    }
                    problem.pushforward(&lin, &v).unwrap()
                })
                .collect();
            for a in 0..d.ncols() {
                for b in 0..d.ncols() {
                    let expected = jd[a].iter().zip(&jd[b]).map(|(x, y)| x * y).sum();
                    assert_close(
                        gram[(a, b)],
                        expected,
                        1e-10,
                        &format!("var {i} ({a}, {b})"),
                    );
                }
            }
        }
    }

    /// Every gate leaves the mean coefficients over the draws unchanged: the
    /// constant (on every coefficient here) absorbs the mean output.
    #[test]
    fn gates_keep_the_mean_coefficients() {
        let problem = problem_with(16);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let theta = normal(&mut rng, problem.n_params(), 0.3);
        let (_, lin) = problem.residuals(&theta).unwrap();
        let means = dispatch!(problem.level, simd => problem.feature_means(simd, &lin));
        let data = &lin.data;
        let n_c = problem.conditional.n_coefs();
        dispatch!(problem.level, simd => {
            let mut st = problem.local_state(simd);
            let mut mean_pi = |i: usize, theta_i: &[f64]| {
                let mut sums = vec![0.0; n_c];
                for tile in 0..data.n_tile() {
                    st.y_parents.clear();
                    st.y_parents.extend(
                        problem.graph.parents(i).iter().map(|&p| data.y.value(simd, tile, p)),
                    );
                    problem.conditional.evaluate(simd, theta_i, &mut st);
                    for (sum, pi) in sums.iter_mut().zip(&st.pi) {
                        *sum += (0..problem.width).map(|lane| pi[lane]).sum::<f64>();
                    }
                }
                sums
            };
            for i in 0..problem.n_var() {
                let d = problem.gate_directions(i, &theta, &means[i]);
                let theta_i = problem.params_of(&theta, i).to_vec();
                let base = mean_pi(i, &theta_i);
                for c in 0..d.ncols() {
                    let moved: Vec<f64> =
                        theta_i.iter().enumerate().map(|(r, t)| t + d[(r, c)]).collect();
                    for (a, b) in mean_pi(i, &moved).iter().zip(&base) {
                        assert_close(*a, *b, 1e-10, &format!("var {i} gate {c}"));
                    }
                }
            }
        });
    }

    /// The speeds are the weighted mean squares of the score `d/dy_j log
    /// p(y_i | parents)` at `y_i = T(x_k)`, from finite differences of the
    /// transformer's inverse.
    #[test]
    fn fisher_speeds_match_finite_differences() {
        let problem = problem_with(16);
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let theta = normal(&mut rng, problem.n_params(), 0.3);
        let data = problem.data.as_ref().unwrap();
        let y = data.y.to_rows();
        let (n_var, n_draw) = (problem.n_var(), data.n_draw);
        let conditioners = &problem.conditional.conditioners;
        let transformer = &problem.conditional.transformer;
        let n_c = problem.conditional.n_coefs();

        for (nodes, weights) in [
            (vec![-1.5, -0.4, 0.4, 1.5], vec![0.1, 0.4, 0.4, 0.1]),
            (
                vec![-2.0, -0.7, 0.0, 0.7, 2.0],
                vec![0.05, 0.2, 0.5, 0.2, 0.05],
            ),
        ] {
            let speeds = problem.fisher_speeds(&theta, &nodes, &weights).unwrap();
            let mut edge = 0;
            for i in 0..n_var {
                let n_p = problem.graph.n_parent(i);
                let theta_i = problem.params_of(&theta, i);
                let packed = conditioners.pack(n_p, theta_i);
                let mut work = conditioners.draw_workspace(problem.graph.max_parent());
                let mut pi_of = |parents: &[f64]| {
                    let mut pi = vec![0.0; n_c];
                    conditioners.forward_draw(&packed, parents, &mut work, false, &mut pi);
                    pi
                };
                let (mut odd, mut even) = (vec![0.0; n_p], vec![0.0; n_p]);
                for draw in 0..n_draw {
                    let row = &y[draw * n_var..(draw + 1) * n_var];
                    let parents: Vec<f64> =
                        problem.graph.parents(i).iter().map(|&p| row[p]).collect();
                    let pi = pi_of(&parents);
                    let scores: Vec<Vec<f64>> = nodes
                        .iter()
                        .map(|&x| {
                            let (y_i, _) = transformer.forward_draw(&pi, x);
                            (0..n_p)
                                .map(|j| {
                                    let h = 1e-5;
                                    let mut log_p = |sign: f64| {
                                        let mut moved = parents.clone();
                                        moved[j] += sign * h;
                                        let (x, log_det) =
                                            transformer.inverse_draw(&pi_of(&moved), y_i);
                                        -0.5 * x * x + log_det
                                    };
                                    (log_p(1.0) - log_p(-1.0)) / (2.0 * h)
                                })
                                .collect()
                        })
                        .collect();
                    let n_k = nodes.len();
                    for (k, w) in weights.iter().enumerate() {
                        for j in 0..n_p {
                            let (a, b) = (scores[k][j], scores[n_k - 1 - k][j]);
                            odd[j] += w * (0.5 * (a - b)).powi(2) / n_draw as f64;
                            even[j] += w * (0.5 * (a + b)).powi(2) / n_draw as f64;
                        }
                    }
                }
                for j in 0..n_p {
                    let what = format!("var {i} parent {j}");
                    assert_close(speeds.location_skew[edge], odd[j], 1e-6, &what);
                    assert_close(speeds.scale_tails[edge], even[j], 1e-6, &what);
                    edge += 1;
                }
            }
            assert_eq!(edge, speeds.location_skew.len());
        }
    }
}
