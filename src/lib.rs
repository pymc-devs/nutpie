mod common;
#[cfg(not(target_os = "emscripten"))]
mod progress;
#[cfg(not(target_os = "emscripten"))]
mod pyfunc;
mod pymc;
#[cfg(not(target_os = "emscripten"))]
mod stan;
mod transform;
#[cfg(target_os = "emscripten")]
mod wasm;
#[cfg(not(target_os = "emscripten"))]
mod wrapper;

#[cfg(target_os = "emscripten")]
pub use wasm::_lib;
#[cfg(not(target_os = "emscripten"))]
pub use wrapper::_lib;
