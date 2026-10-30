//! The sweep's model for a flow built in Rust ([`crate::triangular::flow`]):
//! every variable's [`Conditional`], its conditioners through their per-draw
//! methods and its transformer in the generative direction.

use super::{Cell, Tape, VariableModel};
use crate::triangular::conditional::Conditional;
use crate::triangular::conditioner::{DrawWorkspace, PackedVariable, Stack, StackMut};
use crate::triangular::transformer::{AnyTransformer, Transformer};

pub struct ConditionalModel {
    conditional: Conditional,
    /// Each variable's conditioners' weights, packed for the per-draw
    /// kernels.
    packed: Vec<PackedVariable>,
    max_parent: usize,
}

impl ConditionalModel {
    /// The model of `conditional` with the weights of variable `i` in
    /// `weights(i)`, its parents' count in `n_parent(i)`.
    pub(crate) fn new<'a>(
        conditional: Conditional,
        n_var: usize,
        n_parent: impl Fn(usize) -> usize,
        weights: impl Fn(usize) -> &'a [f64],
    ) -> Self {
        let packed = (0..n_var)
            .map(|i| conditional.conditioners.pack(n_parent(i), weights(i)))
            .collect();
        let max_parent = (0..n_var).map(&n_parent).max().unwrap_or(0);
        Self {
            conditional,
            packed,
            max_parent,
        }
    }
}

/// Per-task working buffers for one variable's evaluation.
pub struct ConditionalPad {
    parents: Vec<f64>,
    coefs: Vec<f64>,
    /// `dy/dcoefs`, then `dlog_det/dcoefs`.
    coefs_bar: Vec<f64>,
    /// `d y / d parents`, then `d log_det / d parents`.
    edges: Vec<f64>,
    draw: DrawWorkspace,
    transformer: <AnyTransformer as Transformer>::Scratch,
}

impl VariableModel for ConditionalModel {
    type Pad = ConditionalPad;

    fn new_pad(&self) -> ConditionalPad {
        let n_coefs = self.conditional.n_coefs();
        ConditionalPad {
            parents: Vec::with_capacity(self.max_parent),
            coefs: vec![0.0; n_coefs],
            coefs_bar: vec![0.0; 2 * n_coefs],
            edges: Vec::with_capacity(2 * self.max_parent),
            draw: self
                .conditional
                .conditioners
                .draw_workspace(self.max_parent),
            transformer: self.conditional.transformer.new_scratch(),
        }
    }

    /// With `TAPE`, the backward pass through the conditioners runs here
    /// rather than in the reverse sweep: their weights are still in cache
    /// from the forward pass.
    #[inline]
    fn eval_variable<const TAPE: bool>(
        &self,
        variable: usize,
        parents: &[u32],
        edges: std::ops::Range<usize>,
        y: &[Cell],
        x: f64,
        pad: &mut ConditionalPad,
        tape: &Tape,
    ) -> (f64, f64) {
        let Conditional {
            conditioners,
            transformer,
        } = &self.conditional;
        let packed = &self.packed[variable];
        pad.parents.clear();
        pad.parents
            .extend(parents.iter().map(|&p| y[p as usize].get()));
        conditioners.forward_draw(packed, &pad.parents, &mut pad.draw, TAPE, &mut pad.coefs);
        if !TAPE {
            return transformer.forward_draw(&pad.coefs, x);
        }

        let (dy_dcoefs, dld_dcoefs) = pad.coefs_bar.split_at_mut(pad.coefs.len());
        let (value, log_det, dy_dx, dld_dx) = transformer.forward_draw_with_grads(
            &mut pad.transformer,
            &pad.coefs,
            x,
            dy_dcoefs,
            dld_dcoefs,
        );
        tape.diag[variable].set(dy_dx);
        tape.ld_dx[variable].set(dld_dx);

        let n_parent = parents.len();
        if n_parent > 0 {
            pad.edges.clear();
            pad.edges.resize(2 * n_parent, 0.0);
            conditioners.pullback_parents_draw(
                packed,
                &mut pad.draw,
                Stack::new(&pad.coefs_bar, 2),
                StackMut::new(&mut pad.edges, 2),
            );
            let (edge_y, edge_l) = pad.edges.split_at(n_parent);
            for (k, edge) in edges.enumerate() {
                tape.edges[2 * edge].set(edge_y[k]);
                tape.edges[2 * edge + 1].set(edge_l[k]);
            }
        }
        (value, log_det)
    }
}
