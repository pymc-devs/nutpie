# nutpie: A python wrapper for nuts-rs

Install a rust compiler (ie using rustup), install the python package maturin and run
```
maturin develop --release
```
If you want to use the nightly simd implementation for some of the math functions,
switch to rust nightly and then install with the `simd_support` feature in the nutpie dir:

```
rustup override set nightly
maturin develop --release --extra-cargo-args="--features=simd_support"
```

For usage, the the notebook aesara_logp.

The stan support currently requires a patched version of httpstan, which can be found
[here](https://github.com/stan-dev/httpstan/pull/600). Make sure to follow the development
[installation instructions for httpstan](https://httpstan.readthedocs.io/en/latest/installation.html#installation-from-source).
