//! Python objects to and from serde types, through Python's `json` module.
//!
//! For config-sized data. Python values have to be what `json.dumps` accepts,
//! so no `NaN` or `inf`, and no numpy scalars; Rust values leave non-finite
//! floats as `null`.

use anyhow::Result;
use pyo3::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

/// Read `obj` (dicts, lists, strings, numbers, bools and `None`) as a `T`.
pub(crate) fn from_py<T: DeserializeOwned>(obj: &Bound<'_, PyAny>) -> Result<T> {
    let text: String = obj
        .py()
        .import("json")?
        .call_method1("dumps", (obj,))?
        .extract()?;
    Ok(serde_json::from_str(&text)?)
}

/// `value` as plain Python dicts, lists, strings, numbers, bools and `None`.
pub(crate) fn to_py<'py, T: Serialize>(py: Python<'py>, value: &T) -> Result<Bound<'py, PyAny>> {
    let text = serde_json::to_string(value)?;
    Ok(py.import("json")?.call_method1("loads", (text,))?)
}
