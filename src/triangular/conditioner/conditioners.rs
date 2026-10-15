//! A variable's conditioners together, see [`Conditioners`].

use anyhow::{bail, Result};
use fearless_simd::prelude::*;

use super::config::{roles, ConditionerConfig, Outputs, Role};
use super::{At, Conditioner, Constant, Linear, Mlp, SparseCotangents, Stack, StackMut};
use crate::triangular::transformer::Transformer;

/// A variable's conditioners, stored by kind, which fixes the summation
/// order and the layout of the variable's weights: the constant, the linear
/// conditioner, then the MLPs in list order. Every variable has the same
/// conditioners; a variable's weights depend on its number of parents.
#[derive(Debug, Clone)]
pub(crate) struct Conditioners {
    pub(crate) constant: Option<Constant>,
    pub(crate) linear: Option<Linear>,
    pub(crate) mlps: Vec<Mlp>,
    n_coefs: usize,
}

/// One variable's per-thread workspace for every conditioner: its tile
/// state and scratch.
pub(crate) struct Workspace<S: Simd> {
    constant: (
        <Constant as Conditioner>::TileState<S>,
        <Constant as Conditioner>::Scratch<S>,
    ),
    linear: (
        <Linear as Conditioner>::TileState<S>,
        <Linear as Conditioner>::Scratch<S>,
    ),
    mlps: Vec<(
        <Mlp as Conditioner>::TileState<S>,
        <Mlp as Conditioner>::Scratch<S>,
    )>,
}

impl<S: Simd> Workspace<S> {
    /// MLP `k`'s activations after a `forward_tile`, see [`MlpTile::hidden`].
    ///
    /// [`MlpTile::hidden`]: super::mlp::MlpTile::hidden
    pub(crate) fn mlp_hidden(&self, k: usize) -> &[S::f64s] {
        self.mlps[k].0.hidden()
    }
}

/// A variable's conditioners' weights in the per-draw kernels' layout, see
/// [`Conditioner::pack`]. The constant's has no parent dependence, so the
/// parent pullback skips it.
pub(crate) struct PackedVariable {
    constant: Option<<Constant as Conditioner>::Packed>,
    linear: Option<<Linear as Conditioner>::Packed>,
    mlps: Vec<<Mlp as Conditioner>::Packed>,
}

/// The per-draw states of a variable's conditioners; the constant and the
/// linear conditioner have none.
pub(crate) struct DrawWorkspace {
    mlps: Vec<<Mlp as Conditioner>::DrawState>,
}

/// Which of a flow's conditioners another's carry their weights over from,
/// see [`Conditioners::carry_plan`].
pub(crate) struct CarryPlan {
    constant: bool,
    linear: bool,
    /// Per MLP, the matching MLP of the other flow.
    pub(super) mlps: Vec<Option<usize>>,
}

/// The conditioners' weight tangent along `v`, from
/// [`Conditioners::pushforward_weights_tile`], for
/// [`Conditioners::pullback_parents_tangent_tile`]: it holds the workspace
/// with what the first left for the second, so nothing else can run between
/// them, and it carries `v`, so the second runs along the same direction.
#[must_use]
pub(crate) struct WeightTangent<'a, S: Simd> {
    v: &'a [f64],
    work: &'a mut Workspace<S>,
}

/// `parents_bar_bar` pushed through the conditioners, from
/// [`Conditioners::pushforward_parents_tile`], for
/// [`Conditioners::pullback_weights_tile`], as [`WeightTangent`].
#[must_use]
pub(crate) struct ParentsPushed<'a, S: Simd> {
    parents_bar_bar: Stack<'a, S::f64s>,
    work: &'a mut Workspace<S>,
}

/// Runs `$body` for each conditioner in the summation order, with `$c` the
/// conditioner, `$w` its range of the variable's weights and `$s` its
/// `(state, scratch)` in the [`Workspace`] `$work`.
macro_rules! each {
    ($self:expr, $n_parent:expr, $work:expr, |$c:ident, $w:ident, $s:ident| $body:block) => {{
        let mut offset = 0;
        if let Some($c) = &$self.constant {
            let $w = offset..offset + $c.n_weights($n_parent);
            offset = $w.end;
            let $s = &mut $work.constant;
            $body
        }
        if let Some($c) = &$self.linear {
            let $w = offset..offset + $c.n_weights($n_parent);
            offset = $w.end;
            let $s = &mut $work.linear;
            $body
        }
        for ($c, $s) in $self.mlps.iter().zip($work.mlps.iter_mut()) {
            let $w = offset..offset + $c.n_weights($n_parent);
            offset = $w.end;
            $body
        }
        let _ = offset;
    }};
}

/// [`each`] without a workspace.
macro_rules! each_range {
    ($self:expr, $n_parent:expr, |$c:ident, $w:ident| $body:block) => {{
        let mut offset = 0;
        if let Some($c) = &$self.constant {
            let $w = offset..offset + $c.n_weights($n_parent);
            offset = $w.end;
            $body
        }
        if let Some($c) = &$self.linear {
            let $w = offset..offset + $c.n_weights($n_parent);
            offset = $w.end;
            $body
        }
        for $c in &$self.mlps {
            let $w = offset..offset + $c.n_weights($n_parent);
            offset = $w.end;
            $body
        }
        let _ = offset;
    }};
}

/// The [`At`] of one conditioner: `$s` is its `(state, scratch)`.
macro_rules! at {
    ($weights:expr, $parents:expr, $s:expr) => {
        At {
            weights: $weights,
            parents: $parents,
            state: &$s.0,
        }
    };
}

impl Conditioners {
    pub(crate) fn new(
        configs: &[ConditionerConfig],
        transformer: &impl Transformer,
    ) -> Result<Self> {
        let roles = roles(transformer);
        let mut out = Self {
            constant: None,
            linear: None,
            mlps: Vec::new(),
            n_coefs: transformer.n_coefs(),
        };
        for config in configs {
            match config {
                ConditionerConfig::Constant { target, ridge } => {
                    if out.constant.is_some() {
                        bail!("at most one constant conditioner: two are one on the union of their targets");
                    }
                    out.constant = Some(Constant::new(Outputs::new(
                        target,
                        &ridge.groups()?,
                        &roles,
                    )?));
                }
                ConditionerConfig::Linear { target, ridge } => {
                    if out.linear.is_some() {
                        bail!("at most one linear conditioner: two are one on the union of their targets");
                    }
                    out.linear = Some(Linear::new(Outputs::new(target, &ridge.groups()?, &roles)?));
                }
                ConditionerConfig::Mlp {
                    width,
                    target,
                    activation,
                    input_squash,
                    init_output_scale,
                    ridge,
                } => {
                    let groups = ridge.groups()?;
                    out.mlps.push(Mlp::new(
                        Outputs::new(target, &groups, &roles)?,
                        *width,
                        *activation,
                        *input_squash,
                        *init_output_scale,
                        groups.hidden,
                    )?);
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn n_coefs(&self) -> usize {
        self.n_coefs
    }

    pub(crate) fn n_weights(&self, n_parent: usize) -> usize {
        let mut n = 0;
        each_range!(self, n_parent, |_c, w| {
            n = w.end;
        });
        n
    }

    /// Each conditioner's range of a variable's weights, in the order of
    /// [`Self`].
    pub(crate) fn weight_ranges(&self, n_parent: usize) -> Vec<std::ops::Range<usize>> {
        let mut out = Vec::new();
        each_range!(self, n_parent, |_c, w| {
            out.push(w);
        });
        out
    }

    /// Appends one ridge strength per weight of a variable, see
    /// [`Conditioner::ridge_strengths`].
    pub(crate) fn ridge_strengths(&self, n_parent: usize, loc_scale: f64, out: &mut Vec<f64>) {
        each_range!(self, n_parent, |c, _w| {
            c.ridge_strengths(n_parent, loc_scale, out);
        });
    }

    /// Appends, per weight of a variable, whether it makes its conditional
    /// nonlinear or non-Gaussian: the MLPs', the constant's into shape
    /// coefficients and the linear conditioner's into anything but the
    /// location.
    pub(crate) fn nonlinear_mask(&self, n_parent: usize, out: &mut Vec<bool>) {
        if let Some(c) = &self.constant {
            let o = c.outputs();
            out.extend((0..o.len()).map(|j| o.role(j) == Role::Shape));
        }
        if let Some(c) = &self.linear {
            let o = c.outputs();
            for j in 0..o.len() {
                let nonlinear = o.role(j) != Role::Location;
                out.extend(std::iter::repeat_n(nonlinear, n_parent));
            }
        }
        for c in &self.mlps {
            out.extend(std::iter::repeat_n(true, c.n_weights(n_parent)));
        }
    }

    /// A new variable's weights.
    pub(crate) fn init_weights<R: rand::Rng + ?Sized>(
        &self,
        n_parent: usize,
        rng: &mut R,
        out: &mut [f64],
    ) {
        each_range!(self, n_parent, |c, w| {
            c.init_weights(n_parent, rng, &mut out[w]);
        });
    }

    /// Which of `old`'s conditioners each of these carries its weights over
    /// from: the constant from the constant, the linear conditioner from the
    /// linear one, and an MLP from the one MLP of `old` with the same
    /// structure whose targets overlap with its own. A new MLP that two of
    /// `old`'s match, or two new MLPs matching the same one, are an error.
    pub(crate) fn carry_plan(&self, old: &Conditioners) -> Result<CarryPlan> {
        let mut mlps = Vec::with_capacity(self.mlps.len());
        let mut taken = vec![false; old.mlps.len()];
        for (k, mlp) in self.mlps.iter().enumerate() {
            let mut matches = old.mlps.iter().enumerate().filter(|(_, from)| {
                mlp.same_structure(from) && mlp.outputs().shared(from.outputs()).next().is_some()
            });
            let found = matches.next().map(|(m, _)| m);
            if matches.next().is_some() {
                bail!("MLP {k} matches several of the current flow's MLPs");
            }
            if let Some(m) = found {
                if std::mem::replace(&mut taken[m], true) {
                    bail!("two MLPs match the current flow's MLP {m}");
                }
            }
            mlps.push(found);
        }
        Ok(CarryPlan {
            constant: self.constant.is_some() && old.constant.is_some(),
            linear: self.linear.is_some() && old.linear.is_some(),
            mlps,
        })
    }

    /// A variable's weights `out`, already initialized, with what `plan`
    /// carries over from its weights `from` under `old`.
    pub(crate) fn carry_weights(
        &self,
        old: &Conditioners,
        plan: &CarryPlan,
        n_parent: usize,
        from: &[f64],
        out: &mut [f64],
    ) {
        let (ranges, old_ranges) = (self.weight_ranges(n_parent), old.weight_ranges(n_parent));
        // Both lists hold the constant first and the linear conditioner
        // second, each only if there is one.
        let (mut next, mut old_next) = (0, 0);
        if let Some(c) = &self.constant {
            if plan.constant {
                let from_c = old.constant.as_ref().expect("planned");
                c.carry_weights(
                    from_c,
                    n_parent,
                    &from[old_ranges[old_next].clone()],
                    &mut out[ranges[next].clone()],
                );
            }
            next += 1;
        }
        if old.constant.is_some() {
            old_next += 1;
        }
        if let Some(c) = &self.linear {
            if plan.linear {
                let from_c = old.linear.as_ref().expect("planned");
                c.carry_weights(
                    from_c,
                    n_parent,
                    &from[old_ranges[old_next].clone()],
                    &mut out[ranges[next].clone()],
                );
            }
            next += 1;
        }
        if old.linear.is_some() {
            old_next += 1;
        }
        for (c, found) in self.mlps.iter().zip(&plan.mlps) {
            if let Some(m) = *found {
                c.carry_weights(
                    &old.mlps[m],
                    n_parent,
                    &from[old_ranges[old_next + m].clone()],
                    &mut out[ranges[next].clone()],
                );
            }
            next += 1;
        }
    }

    /// A variable's weights after the coordinates change: its own value `v`
    /// becomes `a v + b` for `own = (a, b)`, and each parent's likewise with
    /// `parents`. Moves the location's constant and linear weights so that
    /// the location follows exactly (given a constant on it), and the
    /// scale's constant so that the scale does where nothing else feeds it.
    /// The MLPs, and the linear weights into anything but the location,
    /// keep their values.
    pub(crate) fn change_coordinates(
        &self,
        transformer: &impl Transformer,
        own: (f64, f64),
        parents: &[(f64, f64)],
        weights: &mut [f64],
    ) {
        let n_parent = parents.len();
        let ranges = self.weight_ranges(n_parent);
        let (a, b) = own;
        let mut next = 0;
        let constant = self.constant.as_ref().map(|c| {
            next += 1;
            (c, ranges[next - 1].clone())
        });
        let linear = self.linear.as_ref().map(|c| {
            next += 1;
            (c, ranges[next - 1].clone())
        });

        // A parent's `w_j v_j = (a w_j / a_j) v_j' - a w_j b_j / a_j`, in
        // units of the new location.
        let mut shift = 0.0;
        if let (Some((c, range)), Some(location)) = (&linear, transformer.location_coef()) {
            if let Some(j) = c.outputs().coefs().iter().position(|&k| k == location) {
                let row = &mut weights[range.clone()][j * n_parent..(j + 1) * n_parent];
                for (w, &(a_p, b_p)) in row.iter_mut().zip(parents) {
                    shift += a * *w * b_p / a_p;
                    *w *= a / a_p;
                }
            }
        }
        if let Some((c, range)) = &constant {
            let values = &mut weights[range.clone()];
            for (value, &k) in values.iter_mut().zip(c.outputs().coefs()) {
                if Some(k) == transformer.location_coef() {
                    *value = transformer.shifted_location_coef(*value, a, b) - shift;
                } else if Some(k) == transformer.scale_coef() {
                    *value = transformer.rescaled_scale_coef(*value, a);
                }
            }
        }
    }

    /// A variable's weights in the per-draw kernels' layout.
    pub(crate) fn pack(&self, n_parent: usize, weights: &[f64]) -> PackedVariable {
        let ranges = self.weight_ranges(n_parent);
        let mut ranges = ranges.into_iter();
        let mut next = |len| {
            let range = ranges.next().expect("a range per conditioner");
            debug_assert_eq!(range.len(), len);
            &weights[range]
        };
        PackedVariable {
            constant: self
                .constant
                .as_ref()
                .map(|c| c.pack(n_parent, next(c.n_weights(n_parent)))),
            linear: self
                .linear
                .as_ref()
                .map(|c| c.pack(n_parent, next(c.n_weights(n_parent)))),
            mlps: self
                .mlps
                .iter()
                .map(|c| c.pack(n_parent, next(c.n_weights(n_parent))))
                .collect(),
        }
    }

    pub(crate) fn draw_workspace(&self, max_parent: usize) -> DrawWorkspace {
        DrawWorkspace {
            mlps: self
                .mlps
                .iter()
                .map(|c| c.new_draw_state(max_parent))
                .collect(),
        }
    }

    /// `coefs = sum_c f_c(parents; weights_c)` at one draw; with `record`,
    /// keeping what [`Self::pullback_parents_draw`] needs.
    #[inline(always)]
    pub(crate) fn forward_draw(
        &self,
        packed: &PackedVariable,
        parents: &[f64],
        work: &mut DrawWorkspace,
        record: bool,
        coefs: &mut [f64],
    ) {
        coefs.fill(0.0);
        if let (Some(c), Some(packed)) = (&self.constant, &packed.constant) {
            c.forward_draw(packed, parents, &mut (), record, coefs);
        }
        if let (Some(c), Some(packed)) = (&self.linear, &packed.linear) {
            c.forward_draw(packed, parents, &mut (), record, coefs);
        }
        for ((c, packed), state) in self.mlps.iter().zip(&packed.mlps).zip(&mut work.mlps) {
            c.forward_draw(packed, parents, state, record, coefs);
        }
    }

    /// `parents_bar[i] += C^T coefs_bar[i]` at one draw, for the stack of
    /// `dy/dcoefs` and `dlog_det/dcoefs`, after a recording
    /// [`Self::forward_draw`].
    #[inline(always)]
    pub(crate) fn pullback_parents_draw(
        &self,
        packed: &PackedVariable,
        work: &mut DrawWorkspace,
        coefs_bar: Stack<f64>,
        mut parents_bar: StackMut<f64>,
    ) {
        let n_parent = parents_bar.get(0).len();
        if let (Some(c), Some(packed)) = (&self.linear, &packed.linear) {
            c.pullback_parents_draw(packed, &mut (), coefs_bar, parents_bar.sub(0..n_parent));
        }
        for ((c, packed), state) in self.mlps.iter().zip(&packed.mlps).zip(&mut work.mlps) {
            c.pullback_parents_draw(packed, state, coefs_bar, parents_bar.sub(0..n_parent));
        }
    }

    pub(crate) fn workspace<S: Simd>(&self, simd: S, max_parent: usize) -> Workspace<S> {
        Workspace {
            constant: ((), ()),
            linear: ((), ()),
            mlps: self
                .mlps
                .iter()
                .map(|c| {
                    (
                        c.new_tile_state(simd, max_parent),
                        c.new_scratch(simd, max_parent),
                    )
                })
                .collect(),
        }
    }

    /// `coefs = sum_c f_c(parents; weights_c)`.
    #[inline(always)]
    pub(crate) fn forward_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        work: &mut Workspace<S>,
        coefs: &mut [S::f64s],
    ) {
        coefs.fill(S::f64s::splat(simd, 0.0));
        each!(self, parents.len(), work, |c, w, s| {
            c.forward_tile(simd, &weights[w], parents, &mut s.0, coefs);
        });
    }

    /// See [`Conditioner::pullback_parents_tile`].
    #[inline(always)]
    pub(crate) fn pullback_parents_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        work: &mut Workspace<S>,
        coefs_bar: Stack<S::f64s>,
        mut parents_bar: StackMut<S::f64s>,
    ) {
        each!(self, parents.len(), work, |c, w, s| {
            let at = at!(&weights[w], parents, s);
            c.pullback_parents_tile(
                simd,
                &at,
                coefs_bar,
                parents_bar.sub(0..parents.len()),
                &mut s.1,
            );
        });
    }

    /// See [`Conditioner::pushforward_weights_tile`]. The returned
    /// [`WeightTangent`] is what [`Self::pullback_parents_tangent_tile`]
    /// needs: the same `v` and the work this left in `work`.
    #[inline(always)]
    pub(crate) fn pushforward_weights_tile<'a, S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        work: &'a mut Workspace<S>,
        v: &'a [f64],
        coefs_dot: &mut [S::f64s],
    ) -> WeightTangent<'a, S> {
        each!(self, parents.len(), work, |c, w, s| {
            let at = at!(&weights[w.clone()], parents, s);
            c.pushforward_weights_tile(simd, &at, &v[w], coefs_dot, &mut s.1);
        });
        WeightTangent { v, work }
    }

    /// See [`Conditioner::pullback_parents_tangent_tile`], along the
    /// `tangent`'s `v`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn pullback_parents_tangent_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        tangent: WeightTangent<S>,
        coefs_bar: Stack<S::f64s>,
        coefs_bar_dot: Stack<S::f64s>,
        mut parents_bar_dot: StackMut<S::f64s>,
    ) {
        let WeightTangent { v, work } = tangent;
        each!(self, parents.len(), work, |c, w, s| {
            let at = at!(&weights[w.clone()], parents, s);
            c.pullback_parents_tangent_tile(
                simd,
                &at,
                &v[w],
                coefs_bar,
                coefs_bar_dot,
                parents_bar_dot.sub(0..parents.len()),
                &mut s.1,
            );
        });
    }

    /// See [`Conditioner::pushforward_parents_tile`]. The returned
    /// [`ParentsPushed`] is what [`Self::pullback_weights_tile`] needs, with
    /// `parents_dot` as its `parents_bar_bar`.
    #[inline(always)]
    pub(crate) fn pushforward_parents_tile<'a, S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        work: &'a mut Workspace<S>,
        parents_dot: Stack<'a, S::f64s>,
        mut coefs_dot: StackMut<S::f64s>,
    ) -> ParentsPushed<'a, S> {
        let n_coefs = self.n_coefs;
        each!(self, parents.len(), work, |c, w, s| {
            let at = at!(&weights[w], parents, s);
            c.pushforward_parents_tile(simd, &at, parents_dot, coefs_dot.sub(0..n_coefs), &mut s.1);
        });
        ParentsPushed {
            parents_bar_bar: parents_dot,
            work,
        }
    }

    /// See [`Conditioner::pullback_weights_tile`], with the `pushed`
    /// `parents_bar_bar`; `weights_bar` is the variable's.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn pullback_weights_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        pushed: ParentsPushed<S>,
        coefs_bar: &[S::f64s],
        pulled_coefs_bar: Stack<S::f64s>,
        weights_bar: &mut [S::f64s],
    ) {
        let ParentsPushed {
            parents_bar_bar,
            work,
        } = pushed;
        each!(self, parents.len(), work, |c, w, s| {
            let at = at!(&weights[w.clone()], parents, s);
            c.pullback_weights_tile(
                simd,
                &at,
                coefs_bar,
                pulled_coefs_bar,
                parents_bar_bar,
                &mut weights_bar[w],
                &mut s.1,
            );
        });
    }

    /// `c += C`, column-major `(n_coefs, n_parent)`.
    #[inline(always)]
    pub(crate) fn add_parent_jacobian_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        work: &mut Workspace<S>,
        c_out: &mut [S::f64s],
    ) {
        each!(self, parents.len(), work, |c, w, s| {
            let at = at!(&weights[w], parents, s);
            c.add_parent_jacobian_tile(simd, &at, c_out, &mut s.1);
        });
    }

    /// See [`Conditioner::pullback_weights_stack_tile`]; `weights_bar` holds
    /// one vector of the variable's weights per cotangent.
    #[inline(always)]
    pub(crate) fn pullback_weights_stack_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        work: &mut Workspace<S>,
        cotangents: &SparseCotangents<S>,
        mut weights_bar: StackMut<S::f64s>,
    ) {
        each!(self, parents.len(), work, |c, w, s| {
            let at = at!(&weights[w.clone()], parents, s);
            c.pullback_weights_stack_tile(simd, &at, cotangents, weights_bar.sub(w), &mut s.1);
        });
    }
}
