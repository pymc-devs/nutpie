# nuts-py: A python wrapper for nuts-rs

Still work-in-progress

To install, clone nuts-rs (https://github.com/aseyboldt/nuts-rs) into the parent directory
of nuts-py, and run
```
maturin develop --release
```
If you want to use the nightly simd implementation for some of the math functions,
switch to rust nightly and then install with the `simd_support` feature in the nuts-py dir:

```
rustup override set nightly
maturin develop --release --extra-cargo-args="--features=simd_support"
```

For usage with pymc, see the notebook aesara_logp.
