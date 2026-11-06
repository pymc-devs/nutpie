//! The sampler's transform: the flow's generative direction
//! (`SparseTriangularMap::transform_and_log_det`) and the pullback of a
//! gradient through it, for the leapfrog step.
//!
//! Ancestral sampling through the sparse triangular flow is the one genuinely
//! sequential piece of a leapfrog step: a conditioner cannot run until the
//! variables it reads are resolved. The JAX implementation in
//! `nutpie/triangular.py` expresses that as a `lax.scan` over rectangular,
//! padded elimination levels, which is the only shape XLA can work with -- and
//! which degenerates to `dim` scan steps of pure dispatch overhead whenever the
//! blanket is banded or dense.
//!
//! Here the structure is kept ragged. `nutpie.triangular_layout` flattens the
//! map into per-variable parent lists and per-variable weight slices at their
//! true widths, and this module walks them under one of three schedules (see
//! [`Schedule`]).
//!
//! Only the value and log determinant are produced. Differentiation stays in
//! JAX, where the sparse solve in `inverse_gradient_and_val` lives.
//!
//! Note on BLAS: a level is a batch of *distinct* small matrices (every
//! variable has its own conditioner), not one shared matrix applied to many
//! vectors, so there is no GEMM here for faer or any other BLAS to accelerate.
//! The conditioners are instead evaluated with `pulp`, see [`EvalMlp`].

mod conditional;
mod layout;

pub(crate) use conditional::ConditionalModel;
use layout::{build_transform, LayoutModel};

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::{bail, Result};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde::Deserialize;
use smallvec::SmallVec;

use super::pattern::{Children, Pattern, Ragged};

/// Scratch space for one conditioner evaluation. Sized for the common case
/// (parent counts and hidden widths in the tens); wider maps spill to the heap.
pub(crate) type Scratch = SmallVec<[f64; 64]>;

/// Variables whose parents are all resolved, waiting to be picked up.
type ReadyList = SmallVec<[u32; 8]>;

/// An `f64` that one task writes and another reads.
///
/// Relaxed accesses compile to plain loads and stores, so this costs nothing
/// over a bare `f64`; it exists so that the concurrent access is expressed in
/// the memory model rather than asserted with raw pointers. Every cell is
/// written exactly once, and the ordering that makes a read of someone else's
/// cell meaningful comes from the acquire/release pair on the dependency
/// counters, not from the cell itself.
#[repr(transparent)]
#[derive(Debug)]
pub(crate) struct Cell(AtomicU64);

impl Cell {
    pub(crate) fn zeros(n: usize) -> Vec<Cell> {
        (0..n).map(|_| Cell(AtomicU64::new(0))).collect()
    }

    #[inline(always)]
    pub(crate) fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub(crate) fn set(&self, value: f64) {
        self.0.store(value.to_bits(), Ordering::Relaxed)
    }
}

/// How to walk the DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Schedule {
    /// Plain loop over levels, one thread. No atomics, no task overhead; the
    /// right answer whenever the DAG has no width to exploit, which includes
    /// every banded or dense blanket.
    Serial,
    /// Level-synchronous: each level's members in parallel, a barrier between
    /// levels. Levels below `min_parallel_work` run inline.
    Levels,
    /// Dataflow: a variable becomes runnable the moment its last parent lands,
    /// and the fork tree is built implicitly by `rayon::join` as the sweep
    /// fans out. See [`TriangularTransform::run_dataflow`].
    Dataflow,
    /// `Serial` if no level is worth parallelizing, `Dataflow` otherwise.
    Auto,
}

/// Everything the dataflow sweep shares between tasks.
struct Dataflow<'a> {
    x: &'a [f64],
    y: &'a [Cell],
    /// Per-variable log det contribution, summed in index order afterwards so
    /// that the total does not depend on how rayon happened to split the work.
    log_det: &'a [Cell],
    /// Unresolved parents remaining, per variable.
    pending: &'a [AtomicU32],
}

/// The sparse Jacobian of the forward map, recorded as the sweep runs.
///
/// `J = dy/dx` itself is dense -- it is the inverse of a sparse triangular
/// matrix -- but it is never needed. Writing the map as
/// `y_i = f_i(x_i, y_pa(i))` gives `J = D + A J`, so `J = (I - A)^-1 D` with
/// `D` diagonal and `A` strictly lower triangular in topological order. A
/// vector-Jacobian product is then one reverse sweep over the DAG followed by a
/// scale, and both factors are exactly as sparse as the blanket.
///
/// The log det rides along in the same structure: with
/// `ld_i = log(d f_i / d x_i)`, `grad = J^T(g + C^T 1) + b`, which folds into
/// one fused sweep. See `TriangularTransform::pullback`.
///
/// Everything here is sized once from the layout and overwritten in place, so a
/// steady-state leapfrog step allocates nothing.
pub(crate) struct Tape {
    /// Per edge, by edge id and interleaved:
    /// `d y_child / d y_parent` then `d ld_child / d y_parent`.
    pub(crate) edges: Vec<Cell>,
    /// `d y_i / d x_i`, the diagonal of `J`'s numerator.
    pub(crate) diag: Vec<Cell>,
    /// `d ld_i / d x_i`.
    pub(crate) ld_dx: Vec<Cell>,
    /// Whether a forward pass has recorded into this tape since the last
    /// pullback. A mismatched pair would yield a *wrong gradient* rather than
    /// an error, which is exactly the kind of bug that shows up weeks later as
    /// a bad acceptance rate.
    pub(crate) filled: bool,
}

impl Tape {
    fn new(n_variables: usize, n_edges: usize) -> Self {
        Self {
            edges: Cell::zeros(2 * n_edges),
            diag: Cell::zeros(n_variables),
            ld_dx: Cell::zeros(n_variables),
            filled: false,
        }
    }
}

/// One variable's evaluation, the model a [`TriangularTransform`] sweeps.
pub(crate) trait VariableModel: Sync {
    /// Per-task working buffers.
    type Pad: Send;
    fn new_pad(&self) -> Self::Pad;

    /// Variable `variable`, with parents `parents` (their values in `y`) on
    /// the edges `edges`, at base value `x`: its value and log determinant.
    ///
    /// Reads only cells belonging to variables at strictly earlier levels,
    /// which every schedule guarantees are already written. With `TAPE`, also
    /// records the variable's row of the sparse Jacobian in `tape`: its
    /// diagonal entries and, per edge, `d y / d y_parent` and `d log_det / d
    /// y_parent`.
    #[allow(clippy::too_many_arguments)]
    fn eval_variable<const TAPE: bool>(
        &self,
        variable: usize,
        parents: &[u32],
        edges: std::ops::Range<usize>,
        y: &[Cell],
        x: f64,
        pad: &mut Self::Pad,
        tape: &Tape,
    ) -> (f64, f64);
}

/// The generative direction of a sparse triangular map, `x -> y`, under one
/// of the [`Schedule`]s, with its sparse Jacobian recorded for the pullback.
/// What each variable computes is the model `M`'s.
pub struct TriangularTransform<M> {
    n_variables: usize,
    parents: Pattern<u32>,
    /// The transpose of `parents`, built here: the dataflow sweep pushes work
    /// forwards along edges, so it needs each variable's children, and the
    /// edge ids say where the tape keeps each edge's Jacobian entries.
    children: Children<u32>,
    /// The variables of each level; concatenated, an evaluation order.
    levels: Ragged<u32>,
    /// For [`Schedule::Levels`]: whether each level is worth handing to the
    /// thread pool. Levels are tiny far more often than not -- a banded blanket
    /// puts exactly one variable in each -- and rayon's per-call overhead would
    /// then dwarf the work several times over.
    level_parallel: Vec<bool>,
    schedule: Schedule,
    tape: Tape,
    model: M,
}

impl<M: VariableModel> TriangularTransform<M> {
    /// The sweep of `model` over the map with parents `parents`, in the
    /// levels `levels` (their estimated work in `level_work`).
    pub(crate) fn new(
        parents: Pattern<u32>,
        levels: Ragged<u32>,
        level_work: &[i64],
        min_parallel_work: i64,
        schedule: Schedule,
        model: M,
    ) -> Result<Self> {
        let n_variables = parents.n_var();
        if levels.values().len() != n_variables {
            bail!("level_vars must list every variable exactly once");
        }
        if levels.n_rows() != level_work.len() {
            bail!("level_ptr must have one more entry than level_work");
        }
        if levels.values().iter().any(|&v| v as usize >= n_variables) {
            bail!("level_vars contains an out of range variable");
        }
        let level_parallel = level_work
            .iter()
            .enumerate()
            .map(|(level, &work)| levels.row(level).len() > 1 && work >= min_parallel_work)
            .collect();
        let children = parents.children();
        let tape = Tape::new(n_variables, parents.n_edge());
        Ok(Self {
            n_variables,
            parents,
            children,
            levels,
            level_parallel,
            schedule,
            tape,
            model,
        })
    }

    pub(crate) fn n_variables(&self) -> usize {
        self.n_variables
    }

    #[inline(always)]
    fn eval_variable<const TAPE: bool>(
        &self,
        variable: usize,
        y: &[Cell],
        x: f64,
        pad: &mut M::Pad,
    ) -> (f64, f64) {
        self.model.eval_variable::<TAPE>(
            variable,
            self.parents.parents(variable),
            self.parents.edges(variable),
            y,
            x,
            pad,
            &self.tape,
        )
    }

    /// `x -> y`, returning `log|det dy/dx|`.
    ///
    /// With `record`, also fills the tape so that `pullback` can run.
    pub fn transform_and_log_det(&mut self, x: &[f64], record: bool) -> Result<(Vec<f64>, f64)> {
        if x.len() != self.n_variables {
            bail!(
                "expected an array of length {}, got {}",
                self.n_variables,
                x.len()
            );
        }

        let y = Cell::zeros(self.n_variables);
        let log_det = if record {
            self.sweep::<true>(x, &y)
        } else {
            self.sweep::<false>(x, &y)
        };
        self.tape.filled = record;
        Ok((y.iter().map(Cell::get).collect(), log_det))
    }

    fn sweep<const TAPE: bool>(&self, x: &[f64], y: &[Cell]) -> f64 {
        match self.effective_schedule() {
            Schedule::Serial => self.run_serial::<TAPE>(x, y),
            Schedule::Levels => self.run_levels::<TAPE>(x, y),
            Schedule::Dataflow | Schedule::Auto => self.run_dataflow_top::<TAPE>(x, y),
        }
    }

    /// Pull a cotangent on `(y, log_det)` back to one on `x`.
    ///
    /// `grad_x = J^T(grad_y + ld_bar * C^T 1) + ld_bar * b`, which is the
    /// fused reverse sweep below: each variable collects from its children,
    /// then scales by the Jacobian diagonal.
    ///
    /// Kept serial. It is two fused multiply-adds per edge against a tape a
    /// fraction the size of the weight blob, so it is memory bound and short;
    /// the expensive half of the VJP is the Jacobian construction, which the
    /// forward sweep already schedules.
    pub fn pullback(&self, grad_y: &[f64], ld_bar: f64) -> Result<Vec<f64>> {
        if !self.tape.filled {
            bail!("pullback called without a matching recorded forward pass");
        }
        if grad_y.len() != self.n_variables {
            bail!(
                "expected a cotangent of length {}, got {}",
                self.n_variables,
                grad_y.len()
            );
        }

        let mut w = vec![0.0; self.n_variables];
        let mut grad_x = vec![0.0; self.n_variables];

        for &variable in self.levels.values().iter().rev() {
            let variable = variable as usize;
            let mut total = grad_y[variable];
            for (child, edge) in self.children.entries(variable) {
                total += self.tape.edges[2 * edge].get() * w[child]
                    + ld_bar * self.tape.edges[2 * edge + 1].get();
            }
            w[variable] = total;
            grad_x[variable] =
                self.tape.diag[variable].get() * total + ld_bar * self.tape.ld_dx[variable].get();
        }

        Ok(grad_x)
    }

    fn effective_schedule(&self) -> Schedule {
        // Without threads there is no pool to hand levels to. rayon would fall
        // back to the current thread only if spawning reports `Unsupported`,
        // which emscripten does not.
        if !cfg!(feature = "parallel") {
            return Schedule::Serial;
        }
        match self.schedule {
            Schedule::Auto => {
                if self.level_parallel.iter().any(|p| *p) {
                    Schedule::Dataflow
                } else {
                    Schedule::Serial
                }
            }
            other => other,
        }
    }

    fn run_serial<const TAPE: bool>(&self, x: &[f64], y: &[Cell]) -> f64 {
        let mut pad = self.model.new_pad();
        let mut log_det = 0.0;
        for &variable in self.levels.values() {
            let variable = variable as usize;
            let (value, element) = self.eval_variable::<TAPE>(variable, y, x[variable], &mut pad);
            y[variable].set(value);
            log_det += element;
        }
        log_det
    }

    fn run_levels<const TAPE: bool>(&self, x: &[f64], y: &[Cell]) -> f64 {
        let mut pad = self.model.new_pad();
        let mut collected: Vec<(f64, f64)> = Vec::new();
        let mut log_det = 0.0;

        for level in 0..self.levels.n_rows() {
            let members = self.levels.row(level);
            if self.level_parallel[level] {
                members
                    .par_iter()
                    .map(|&variable| {
                        let mut pad = self.model.new_pad();
                        let variable = variable as usize;
                        self.eval_variable::<TAPE>(variable, y, x[variable], &mut pad)
                    })
                    .collect_into_vec(&mut collected);
                // Scatter and sum in member order, so the total does not depend
                // on how rayon split the level.
                for (&variable, &(value, element)) in members.iter().zip(&collected) {
                    y[variable as usize].set(value);
                    log_det += element;
                }
            } else {
                for &variable in members {
                    let variable = variable as usize;
                    let (value, element) =
                        self.eval_variable::<TAPE>(variable, y, x[variable], &mut pad);
                    y[variable].set(value);
                    log_det += element;
                }
            }
        }
        log_det
    }

    fn run_dataflow_top<const TAPE: bool>(&self, x: &[f64], y: &[Cell]) -> f64 {
        let log_det = Cell::zeros(self.n_variables);
        let pending: Vec<AtomicU32> = (0..self.n_variables)
            .map(|i| AtomicU32::new(self.parents.n_parent(i) as u32))
            .collect();

        let roots: ReadyList = (0..self.n_variables as u32)
            .filter(|&i| self.parents.n_parent(i as usize) == 0)
            .collect();

        self.run_dataflow::<TAPE>(
            roots,
            &Dataflow {
                x,
                y,
                log_det: &log_det,
                pending: &pending,
            },
        );

        log_det.iter().map(Cell::get).sum()
    }

    /// The dataflow sweep.
    ///
    /// Levels are a *static* schedule: they assume the useful unit of
    /// synchronization is "everything at depth d before anything at depth
    /// d + 1". That is stronger than the DAG requires, and it costs whenever
    /// the levels are uneven -- a level containing one 200-parent conditioner
    /// and two hundred 1-parent ones makes everybody wait for the slow one,
    /// and a long thin chain running beside a wide burst is stalled at every
    /// step even though nothing it needs is missing.
    ///
    /// Here each variable carries a count of unresolved parents instead.
    /// Finishing a variable decrements its children; a child whose count
    /// reaches zero is runnable *now*, whoever happens to be holding it. The
    /// task keeps one runnable variable for itself and forks the rest with
    /// `rayon::join`, so the fork tree is built implicitly from the DAG's own
    /// shape as the sweep unfolds -- no schedule is precomputed and no barrier
    /// is imposed.
    ///
    /// Two properties fall out of this that the level schedule needed
    /// heuristics for. A DAG with no width (a banded blanket: every variable
    /// has exactly one runnable successor) never forks at all and runs as a
    /// plain loop on one thread. And `rayon::join` only reaches the thread pool
    /// when a thread is actually free to steal, so a narrow burst costs a
    /// cheap failed steal rather than a scheduling round trip.
    ///
    /// The one thing `join` cannot express directly is the recursion the user
    /// might expect -- resolving a variable by joining on its *parents*. Those
    /// are shared between children, so that recursion is over a DAG, not a
    /// tree, and would need memoized nodes with tasks blocking on each other's
    /// subgoals. Pushing forward along edges keeps it a fork-join tree, which
    /// is exactly what `join` wants.
    fn run_dataflow<const TAPE: bool>(&self, mut ready: ReadyList, ctx: &Dataflow<'_>) {
        let mut pad = self.model.new_pad();

        // LIFO, so a task follows its own chain depth-first and keeps the
        // values it just wrote in cache.
        while let Some(variable) = ready.pop() {
            let variable = variable as usize;
            let (value, element) =
                self.eval_variable::<TAPE>(variable, ctx.y, ctx.x[variable], &mut pad);
            ctx.y[variable].set(value);
            ctx.log_det[variable].set(element);

            for &child in self.children.children(variable) {
                // AcqRel is what makes the gather in `eval_variable` legal: the
                // release half publishes the write of `y[variable]` just made,
                // and whichever thread sees the count reach zero acquires the
                // whole release sequence, i.e. every parent's write.
                if ctx.pending[child as usize].fetch_sub(1, Ordering::AcqRel) == 1 {
                    ready.push(child);
                }
            }

            if ready.len() > 1 {
                let split = ready.len() / 2;
                let rest: ReadyList = ready.drain(split..).collect();
                rayon::join(
                    || self.run_dataflow::<TAPE>(rest, ctx),
                    move || self.run_dataflow::<TAPE>(ready, ctx),
                );
                return;
            }
        }
    }
}

/// The whole flow `make_flow(kind="triangular")` builds, for the leapfrog
/// step: `z -> y` with its log determinant, and the pullback of a gradient in
/// `y` back to `z`.
///
/// That flow is `Chain([Sandwich(map, Permute(p)), Affine(loc, scale)])`, and
/// `Sandwich` is `Chain([outer, inner, Invert(outer)])`, so
///
/// ```text
/// u = z[p],   v = map(u),   w = v[p^-1],   y = loc + scale * w
/// ```
///
/// with log determinant `log_det(map) + sum(log |scale|)`. Built from the dict
/// `nutpie.triangular_rust.flow_transform_layout` returns; like
/// [`TriangularTransform`], only valid for the parameters it was built from.
/// A sparse triangular map's sweep, by where the map comes from.
pub enum Map {
    /// Flattened from a JAX map, see [`layout`].
    Layout(TriangularTransform<LayoutModel>),
    /// A flow built in Rust, see [`conditional`].
    Conditional(TriangularTransform<ConditionalModel>),
}

impl Map {
    fn n_variables(&self) -> usize {
        match self {
            Map::Layout(map) => map.n_variables(),
            Map::Conditional(map) => map.n_variables(),
        }
    }

    fn transform_and_log_det(&mut self, x: &[f64], record: bool) -> Result<(Vec<f64>, f64)> {
        match self {
            Map::Layout(map) => map.transform_and_log_det(x, record),
            Map::Conditional(map) => map.transform_and_log_det(x, record),
        }
    }

    fn pullback(&self, grad_y: &[f64], ld_bar: f64) -> Result<Vec<f64>> {
        match self {
            Map::Layout(map) => map.pullback(grad_y, ld_bar),
            Map::Conditional(map) => map.pullback(grad_y, ld_bar),
        }
    }
}

pub struct FlowTransform {
    /// `None` for the diagonal-only flow of the early windows, `Affine`
    /// alone: then `map` and the permutation are the identity.
    map: Option<Map>,
    permutation: Vec<usize>,
    inverse_permutation: Vec<usize>,
    loc: Vec<f64>,
    scale: Vec<f64>,
    log_det_affine: f64,
    buffer: Vec<f64>,
}

impl FlowTransform {
    pub fn from_layout(layout: &Bound<'_, PyDict>) -> Result<Self> {
        fn item<'py>(layout: &Bound<'py, PyDict>, key: &str) -> Result<Bound<'py, PyAny>> {
            layout
                .get_item(key)?
                .ok_or_else(|| anyhow::anyhow!("flow transform layout is missing {key:?}"))
        }
        fn ints(layout: &Bound<'_, PyDict>, key: &str) -> Result<Vec<i64>> {
            Ok(item(layout, key)?
                .extract::<PyReadonlyArray1<i64>>()
                .map_err(PyErr::from)?
                .as_slice()?
                .to_vec())
        }
        fn floats(layout: &Bound<'_, PyDict>, key: &str) -> Result<Vec<f64>> {
            Ok(item(layout, key)?
                .extract::<PyReadonlyArray1<f64>>()
                .map_err(PyErr::from)?
                .as_slice()?
                .to_vec())
        }

        let scale_skip = match layout.get_item("scale_skip_index")? {
            Some(index) if !index.is_none() => Some((
                floats(layout, "scale_skip_weight")?,
                index.extract::<i64>()?,
            )),
            _ => None,
        };
        let has_map = layout.contains("parent_indptr")?;
        let map = has_map
            .then(|| {
                build_transform(
                    &ints(layout, "parent_indptr")?,
                    &ints(layout, "parent_index")?,
                    &floats(layout, "blob")?,
                    &ints(layout, "blob_offset")?,
                    &ints(layout, "layer_out")?,
                    &floats(layout, "skip_weight")?,
                    item(layout, "skip_index")?.extract()?,
                    scale_skip.as_ref().map(|(w, i)| (w.as_slice(), *i)),
                    item(layout, "feature_degree")?.extract()?,
                    &floats(layout, "feature_params")?,
                    match layout.get_item("input_squash")? {
                        Some(value) => value.extract()?,
                        None => None,
                    },
                    &item(layout, "activation")?.extract::<String>()?,
                    &item(layout, "transformer")?,
                    &ints(layout, "level_ptr")?,
                    &ints(layout, "level_vars")?,
                    &ints(layout, "level_work")?,
                    item(layout, "min_parallel_work")?.extract()?,
                    &item(layout, "schedule")?.extract::<String>()?,
                )
                .map(Map::Layout)
            })
            .transpose()?;

        let loc = floats(layout, "loc")?;
        let scale = floats(layout, "scale")?;
        let permutation = if has_map {
            as_usize(&ints(layout, "permutation")?)?
        } else {
            (0..loc.len()).collect()
        };
        Self::new(map, permutation, loc, scale)
    }

    /// The flow from its parts; `map` `None` for the diagonal-only flow.
    pub(crate) fn new(
        map: Option<Map>,
        permutation: Vec<usize>,
        loc: Vec<f64>,
        scale: Vec<f64>,
    ) -> Result<Self> {
        let n = loc.len();
        if map.as_ref().is_some_and(|map| map.n_variables() != n) {
            bail!("loc must have one entry per variable of the map");
        }
        if permutation.len() != n || loc.len() != n || scale.len() != n {
            bail!("permutation, loc and scale must each have one entry per variable");
        }
        let mut inverse_permutation = vec![usize::MAX; n];
        for (position, &variable) in permutation.iter().enumerate() {
            if variable >= n || inverse_permutation[variable] != usize::MAX {
                bail!("permutation is not a permutation of 0..{n}");
            }
            inverse_permutation[variable] = position;
        }
        let log_det_affine = scale.iter().map(|s| s.abs().ln()).sum();

        Ok(Self {
            map,
            permutation,
            inverse_permutation,
            loc,
            scale,
            log_det_affine,
            buffer: vec![0.0; n],
        })
    }

    pub fn n_variables(&self) -> usize {
        self.loc.len()
    }

    /// `z -> y`, returning the log determinant. Records the tape `pullback`
    /// needs.
    pub fn transform_and_log_det(&mut self, z: &[f64], y: &mut [f64]) -> Result<f64> {
        for (slot, &source) in self.buffer.iter_mut().zip(&self.permutation) {
            *slot = z[source];
        }
        let Some(map) = self.map.as_mut() else {
            for (k, out) in y.iter_mut().enumerate() {
                *out = self.loc[k] + self.scale[k] * self.buffer[k];
            }
            return Ok(self.log_det_affine);
        };
        let (v, log_det) = map.transform_and_log_det(&self.buffer, true)?;
        for (k, out) in y.iter_mut().enumerate() {
            *out = self.loc[k] + self.scale[k] * v[self.inverse_permutation[k]];
        }
        Ok(log_det + self.log_det_affine)
    }

    /// Pull `grad_y` (with a unit log determinant cotangent) back to `z`,
    /// through the point the last `transform_and_log_det` recorded.
    pub fn pullback(&mut self, grad_y: &[f64], grad_z: &mut [f64]) -> Result<()> {
        // `w = v[p^-1]` gathers, so its pullback scatters back: `g_v = g_w[p]`.
        for (slot, &source) in self.buffer.iter_mut().zip(&self.permutation) {
            *slot = grad_y[source] * self.scale[source];
        }
        let Some(map) = self.map.as_ref() else {
            grad_z.copy_from_slice(&self.buffer);
            return Ok(());
        };
        let grad_u = map.pullback(&self.buffer, 1.0)?;
        // `u = z[p]`, likewise.
        for (k, &target) in self.permutation.iter().enumerate() {
            grad_z[target] = grad_u[k];
        }
        Ok(())
    }
}

/// Compiled forward transform for one `SparseTriangularMap`.
///
/// Built from the arrays of `nutpie.triangular_layout.extract_layout`, which
/// bake in the conditioner weights -- so an instance is only valid for the
/// parameter values it was created from. Use
/// `nutpie.triangular_rust.compile_transform` rather than calling this
/// constructor directly.
#[pyclass(name = "SparseTriangularTransform")]
pub struct PySparseTriangularTransform {
    inner: TriangularTransform<LayoutModel>,
}

#[pymethods]
impl PySparseTriangularTransform {
    #[new]
    #[pyo3(signature = (
        *,
        parent_indptr,
        parent_index,
        blob,
        blob_offset,
        layer_out,
        skip_weight,
        skip_index,
        feature_degree,
        feature_params,
        activation,
        transformer,
        level_ptr,
        level_vars,
        level_work,
        min_parallel_work,
        schedule,
        input_squash = None,
        scale_skip_weight = None,
        scale_skip_index = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: PyReadonlyArray1<'_, i64>,
        parent_index: PyReadonlyArray1<'_, i64>,
        blob: PyReadonlyArray1<'_, f64>,
        blob_offset: PyReadonlyArray1<'_, i64>,
        layer_out: PyReadonlyArray1<'_, i64>,
        skip_weight: PyReadonlyArray1<'_, f64>,
        skip_index: i64,
        feature_degree: i64,
        feature_params: PyReadonlyArray1<'_, f64>,
        activation: &str,
        transformer: &Bound<'_, PyAny>,
        level_ptr: PyReadonlyArray1<'_, i64>,
        level_vars: PyReadonlyArray1<'_, i64>,
        level_work: PyReadonlyArray1<'_, i64>,
        min_parallel_work: i64,
        schedule: &str,
        input_squash: Option<f64>,
        scale_skip_weight: Option<PyReadonlyArray1<'_, f64>>,
        scale_skip_index: Option<i64>,
    ) -> Result<Self> {
        let scale_skip_weight = scale_skip_weight
            .as_ref()
            .map(|w| w.as_slice())
            .transpose()?;
        let scale_skip = match (scale_skip_weight, scale_skip_index) {
            (Some(weight), Some(index)) => Some((weight, index)),
            (None, None) => None,
            _ => bail!("give both scale_skip_weight and scale_skip_index, or neither"),
        };
        Ok(Self {
            inner: build_transform(
                parent_indptr.as_slice()?,
                parent_index.as_slice()?,
                blob.as_slice()?,
                blob_offset.as_slice()?,
                layer_out.as_slice()?,
                skip_weight.as_slice()?,
                skip_index,
                scale_skip,
                feature_degree,
                feature_params.as_slice()?,
                input_squash,
                activation,
                transformer,
                level_ptr.as_slice()?,
                level_vars.as_slice()?,
                level_work.as_slice()?,
                min_parallel_work,
                schedule,
            )?,
        })
    }

    #[getter]
    fn n_variables(&self) -> usize {
        self.inner.n_variables()
    }

    /// The schedule that will actually run, with `auto` resolved.
    #[getter]
    fn schedule(&self) -> &'static str {
        match self.inner.effective_schedule() {
            Schedule::Serial => "serial",
            Schedule::Levels => "levels",
            Schedule::Dataflow => "dataflow",
            Schedule::Auto => unreachable!("resolved by effective_schedule"),
        }
    }

    /// Number of levels the `levels` schedule would run on the thread pool.
    #[getter]
    fn n_parallel_levels(&self) -> usize {
        self.inner.level_parallel.iter().filter(|p| **p).count()
    }

    /// `x -> (y, log|det dy/dx|)`.
    ///
    /// With `record=True` the sparse Jacobian is kept so that `pullback` can
    /// run; the tape lives in this object, so a given instance supports one
    /// forward/pullback pair at a time.
    #[pyo3(signature = (x, record = false))]
    fn transform_and_log_det<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        record: bool,
    ) -> Result<(Bound<'py, PyArray1<f64>>, f64)> {
        let x = x.as_slice()?.to_vec();
        let (y, log_det) = py.detach(|| self.inner.transform_and_log_det(&x, record))?;
        Ok((PyArray1::from_vec(py, y), log_det))
    }

    /// Pull a cotangent on `(y, log_det)` back to one on `x`.
    ///
    /// Must follow a `transform_and_log_det(..., record=True)` on the same
    /// object. `log_det_bar` is the cotangent on the scalar output, which the
    /// sampler always sets to 1.
    #[pyo3(signature = (grad_y, log_det_bar = 1.0))]
    fn pullback<'py>(
        &self,
        py: Python<'py>,
        grad_y: PyReadonlyArray1<'py, f64>,
        log_det_bar: f64,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let grad_y = grad_y.as_slice()?.to_vec();
        let grad_x = py.detach(|| self.inner.pullback(&grad_y, log_det_bar))?;
        Ok(PyArray1::from_vec(py, grad_x))
    }
}

/// The sampler's native flow transform, exposed for testing it against the
/// JAX bijection. Built from `nutpie.triangular_rust.flow_transform_layout`.
#[pyclass(name = "FlowTransform")]
pub struct PyFlowTransform {
    inner: FlowTransform,
}

impl PyFlowTransform {
    pub(crate) fn from_flow(inner: FlowTransform) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFlowTransform {
    #[new]
    fn new(layout: &Bound<'_, PyDict>) -> Result<Self> {
        Ok(Self {
            inner: FlowTransform::from_layout(layout)?,
        })
    }

    /// `z -> (y, log_det)`, recording the tape `pullback` needs.
    fn transform_and_log_det<'py>(
        &mut self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, f64>,
    ) -> Result<(Bound<'py, PyArray1<f64>>, f64)> {
        let z = z.as_slice()?.to_vec();
        let mut y = vec![0.0; z.len()];
        let log_det = py.detach(|| self.inner.transform_and_log_det(&z, &mut y))?;
        Ok((PyArray1::from_vec(py, y), log_det))
    }

    /// Pull `grad_y` (with a unit log determinant cotangent) back to `z`.
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        grad_y: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let grad_y = grad_y.as_slice()?.to_vec();
        let mut grad_z = vec![0.0; grad_y.len()];
        py.detach(|| self.inner.pullback(&grad_y, &mut grad_z))?;
        Ok(PyArray1::from_vec(py, grad_z))
    }
}

pub(crate) fn as_usize(values: &[i64]) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| anyhow::anyhow!("negative index {v} in layout")))
        .collect()
}

pub(crate) fn as_u32(values: &[i64]) -> Result<Vec<u32>> {
    values
        .iter()
        .map(|&v| u32::try_from(v).map_err(|_| anyhow::anyhow!("index {v} out of range")))
        .collect()
}
