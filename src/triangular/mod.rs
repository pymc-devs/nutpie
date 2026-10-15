//! The sparse triangular transport map.
//!
//! - [`pattern`], [`symbolic`]: the map's parents, and the ordering and fill
//!   that choose them;
//! - [`transformer`]: the elementwise bijection of one variable, given its
//!   coefficients (with its layers and the jets it is differentiated with);
//! - [`conditioner`]: the coefficients as functions of the parents;
//! - [`conditional`]: a variable's conditioners and transformer together;
//! - [`flow`]: the whole flow, with its parameters, permutation and affine;
//! - [`lm`]: the Levenberg-Marquardt fit of the conditioners;
//! - [`sampler`]: the flow's transform for the sampler's leapfrog.

pub(crate) mod conditional;
pub(crate) mod conditioner;
pub(crate) mod flow;
pub(crate) mod lm;
pub(crate) mod pattern;
pub(crate) mod sampler;
pub(crate) mod simd;
pub(crate) mod symbolic;
pub(crate) mod transformer;
