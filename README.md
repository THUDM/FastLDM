# FastLDM

We focus on inference speed-up for [ldm](https://github.com/CompVis/stable-diffusion).

## TensorRT

We are now trying to convert model of stable-diffusion into an optimized TensorRT engine.

* `test_ldm.py` is about unet transformation.
* `test_plugin.py` is about attention plugin transformation.

We provide:

* parameter mapping functions from a module to another one in utils.mapping
* TensorRT-friendly modules in utils.plugins and utils.modules
* benchmarking functions in utils.benchmark
* experiment functions to compare different implemented models with same interface

