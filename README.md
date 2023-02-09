# FastLDM

We focus on inference speed-up for [ldm](https://github.com/CompVis/stable-diffusion).

## TensorRT

We are now trying to convert model of stable-diffusion into an optimized TensorRT engine.

* `test.py` is about timestep embedding transformation.
* `test_plugin.py` is about attention plugin transformation.

Codes are not tidy yet.