mod common;
mod pymc;
mod transform;
#[cfg(not(target_os = "emscripten"))]
mod progress;
#[cfg(not(target_os = "emscripten"))]
mod pyfunc;
#[cfg(not(target_os = "emscripten"))]
mod stan;
#[cfg(not(target_os = "emscripten"))]
mod wrapper;
#[cfg(target_os = "emscripten")]
mod wasm;

#[cfg(not(target_os = "emscripten"))]
pub use wrapper::_lib;
#[cfg(target_os = "emscripten")]
pub use wasm::_lib;
