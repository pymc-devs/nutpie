//! Per-draw arrays in tiles of one SIMD vector: the data, and the
//! linearization the operators read.

use std::sync::Arc;

use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

use super::simd::{load, store};

/// `count` values per draw, in tiles of `width` draws: laid out `(n_tile,
/// count, width)`, so a tile's values of one quantity (a variable or an edge)
/// are one SIMD vector.
#[derive(Debug, Clone)]
pub(crate) struct Tiled {
    values: Vec<f64>,
    /// Values per draw.
    count: usize,
    /// Draws per tile, the SIMD width.
    width: usize,
    n_tile: usize,
}

impl Tiled {
    pub(super) fn zeros(n_tile: usize, count: usize, width: usize) -> Self {
        Self {
            values: vec![0.0; n_tile * count * width],
            count,
            width,
            n_tile,
        }
    }

    /// From row-major `(n_draw, count)` values, `n_draw` a positive multiple
    /// of `width`.
    pub(super) fn from_rows(rows: &[f64], n_draw: usize, width: usize) -> Self {
        let count = rows.len() / n_draw;
        let mut out = Self::zeros(n_draw / width, count, width);
        for (draw, row) in rows.chunks_exact(count).enumerate() {
            let (tile, lane) = (draw / width, draw % width);
            for (k, &value) in row.iter().enumerate() {
                out.values[(tile * count + k) * width + lane] = value;
            }
        }
        out
    }

    /// Row-major `(n_draw, count)` values, the inverse of [`Self::from_rows`].
    pub(super) fn to_rows(&self) -> Vec<f64> {
        let (count, width) = (self.count, self.width);
        let mut out = vec![0.0; self.values.len()];
        for (draw, row) in out.chunks_exact_mut(count).enumerate() {
            let (tile, lane) = (draw / width, draw % width);
            for (k, value) in row.iter_mut().enumerate() {
                *value = self.values[(tile * count + k) * width + lane];
            }
        }
        out
    }

    pub(super) fn n_tile(&self) -> usize {
        self.n_tile
    }

    fn tile_len(&self) -> usize {
        self.count * self.width
    }

    pub(super) fn tile(&self, tile: usize) -> &[f64] {
        let len = self.tile_len();
        &self.values[tile * len..(tile + 1) * len]
    }

    /// Every tile, mutably, to write them in parallel. Unlike
    /// `chunks_exact_mut`, also for `count = 0`.
    pub(super) fn tiles_mut(&mut self) -> Vec<&mut [f64]> {
        let len = self.tile_len();
        let mut out = Vec::with_capacity(self.n_tile);
        let mut rest = self.values.as_mut_slice();
        for _ in 0..self.n_tile {
            let (head, tail) = rest.split_at_mut(len);
            out.push(head);
            rest = tail;
        }
        out
    }

    /// Value `k` of tile `tile`, as one vector.
    #[inline(always)]
    pub(super) fn value<S: Simd>(&self, simd: S, tile: usize, k: usize) -> S::f64s {
        let start = (tile * self.count + k) * self.width;
        S::f64s::from_slice(simd, &self.values[start..start + self.width])
    }

    /// Tile `tile` as vectors, into `dst`.
    #[inline(always)]
    pub(super) fn load<S: Simd>(&self, simd: S, tile: usize, dst: &mut [S::f64s]) {
        load(simd, self.tile(tile), dst);
    }
}

/// The draws the residuals are fitted to.
pub(super) struct Data {
    pub(super) n_draw: usize,
    /// The map's inputs and the gradients there, `n_var` per draw.
    pub(super) y: Tiled,
    pub(super) g: Tiled,
}

impl Data {
    /// The residuals' factor `1 / sqrt(n_draw)`.
    pub(super) fn scale(&self) -> f64 {
        1.0 / (self.n_draw as f64).sqrt()
    }

    pub(super) fn n_tile(&self) -> usize {
        self.y.n_tile()
    }
}

/// The residuals at `theta` as every derivative there needs them: the data
/// they were computed from, and what the triangular solves couple across
/// variables.
pub(crate) struct Linearization {
    pub(super) theta: Vec<f64>,
    pub(super) data: Arc<Data>,
    /// `A`, per edge: the strictly lower part of `J`.
    pub(super) edge_a: Tiled,
    /// `delta`, per variable: the diagonal of `J`.
    pub(super) delta: Tiled,
    /// `x` and `w`, per variable.
    pub(super) x: Tiled,
    pub(super) w: Tiled,
}

impl Linearization {
    pub(super) fn zeros(theta: &[f64], data: Arc<Data>, n_var: usize, n_edge: usize) -> Self {
        let (n_tile, width) = (data.n_tile(), data.y.width);
        Self {
            theta: theta.to_vec(),
            data,
            edge_a: Tiled::zeros(n_tile, n_edge, width),
            delta: Tiled::zeros(n_tile, n_var, width),
            x: Tiled::zeros(n_tile, n_var, width),
            w: Tiled::zeros(n_tile, n_var, width),
        }
    }

    /// Every tile, mutably, for the primal to write them in parallel.
    pub(super) fn tiles_mut(&mut self) -> Vec<LinearTileMut<'_>> {
        let edge_a = self.edge_a.tiles_mut();
        let delta = self.delta.tiles_mut();
        let x = self.x.tiles_mut();
        let w = self.w.tiles_mut();
        edge_a
            .into_iter()
            .zip(delta)
            .zip(x)
            .zip(w)
            .map(|(((edge_a, delta), x), w)| LinearTileMut {
                edge_a,
                delta,
                x,
                w,
            })
            .collect()
    }
}

/// One tile of a [`Linearization`]'s arrays, being written by the primal.
pub(super) struct LinearTileMut<'a> {
    edge_a: &'a mut [f64],
    delta: &'a mut [f64],
    x: &'a mut [f64],
    w: &'a mut [f64],
}

/// One tile of a [`Linearization`]'s arrays, as vectors.
pub(super) struct LinearTile<S: Simd> {
    pub(super) edge_a: Vec<S::f64s>,
    pub(super) delta: Vec<S::f64s>,
    pub(super) x: Vec<S::f64s>,
    pub(super) w: Vec<S::f64s>,
}

impl<S: Simd> LinearTile<S> {
    pub(super) fn new(simd: S, n_var: usize, n_edge: usize) -> Self {
        let zero = S::f64s::splat(simd, 0.0);
        Self {
            edge_a: vec![zero; n_edge],
            delta: vec![zero; n_var],
            x: vec![zero; n_var],
            w: vec![zero; n_var],
        }
    }

    #[simd]
    pub(super) fn load(&mut self, simd: S, lin: &Linearization, tile: usize) {
        lin.edge_a.load(simd, tile, &mut self.edge_a);
        lin.delta.load(simd, tile, &mut self.delta);
        lin.x.load(simd, tile, &mut self.x);
        lin.w.load(simd, tile, &mut self.w);
    }

    #[simd]
    pub(super) fn store(&self, simd: S, out: LinearTileMut<'_>) {
        store(simd, &self.edge_a, out.edge_a);
        store(simd, &self.delta, out.delta);
        store(simd, &self.x, out.x);
        store(simd, &self.w, out.w);
    }
}
