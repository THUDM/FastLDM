# FastLDM

We focus on inference speed-up for [ldm](https://github.com/CompVis/stable-diffusion).

## TensorRT

We are now trying to convert model of stable-diffusion into an optimized TensorRT engine.

* `test.py` is about timestep embedding transformation.
* `test_plugin.py` is about attention plugin transformation.

Code structure:

* We provide parameter mapping functions from a module to another one in utils.mapping
* We provide TensorRT-friendly modules in utils.modules
* We provide benchmarking functions in utils.benchmark
* We provide experiment functions to compare different implemented models with same interface

