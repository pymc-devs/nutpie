//! The sparse triangular transport map: its forward and inverse transform,
//! the Levenberg-Marquardt fit of its conditioners, and the symbolic
//! factorization that chooses its parents.

pub(crate) mod jet;
pub(crate) mod layers;
pub(crate) mod lm;
pub(crate) mod pattern;
pub(crate) mod symbolic;
pub(crate) mod transform;
