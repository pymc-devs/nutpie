//! A variable's conditional: its conditioners and its transformer, the model
//! every variable of a flow shares. The LM's kernels on it are in
//! [`crate::triangular::lm`].

use anyhow::{bail, Result};

use crate::triangular::conditioner::Conditioners;
use crate::triangular::transformer::{AnyTransformer, Transformer};

/// A variable's conditioners and transformer. Every variable has the same;
/// its weights depend on its number of parents.
#[derive(Debug, Clone)]
pub(crate) struct Conditional {
    pub(crate) conditioners: Conditioners,
    pub(crate) transformer: AnyTransformer,
}

impl Conditional {
    pub(crate) fn new(conditioners: Conditioners, transformer: AnyTransformer) -> Result<Self> {
        if conditioners.n_coefs() != transformer.n_coefs() {
            bail!("the conditioners were built for a different transformer");
        }
        Ok(Self {
            conditioners,
            transformer,
        })
    }

    pub(crate) fn n_coefs(&self) -> usize {
        self.transformer.n_coefs()
    }

    pub(crate) fn n_weights(&self, n_parent: usize) -> usize {
        self.conditioners.n_weights(n_parent)
    }
}
