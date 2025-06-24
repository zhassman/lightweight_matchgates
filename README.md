maturin develop --release

USE_RUST_SIMULATOR=1 pytest # imports and uses emsim

USE_RUST_SIMULATOR=1 pytest --run-expensive # runs a test to generate samples and verify a distribution

USE_RUST_SIMULATOR=0 pytest # skips emsim and stays in pure-Python
