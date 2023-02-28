# FastLDM

We focus on inference speed-up for [ldm](https://github.com/CompVis/stable-diffusion).

## TensorRT

We are now trying to convert model of stable-diffusion into an optimized TensorRT engine.

* `test_ldm.py` is about unet transformation.
* `test_plugin.py` is about attention plugin transformation.

When we transform a PyTorch model to TensorRT engine, we usually follow "torch->onnx->trt" footprint. It is convenient for most cases, except when we want to use [plugins](https://github.com/NVIDIA/TensorRT/tree/main/plugin). If we want to use plugins, we usually need to modify the exported onnx file based on [TensorRT official demo](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion). The core idea of this repo is to provide a shortcut. We provide many plugin-friendly PyTorch modules, which can plug into your normal PyTorch model and export onnx directly able to be optimized by TensorRT with plugins.

The advantages of this repo are:

* Seamlessly transform from PyTorch to TensorRT with plugins.
* Seamlessly export onnx to run directly in onnxruntime.

More specifically, we provide:

* parameter mapping functions from a module to another one in utils.mapping
* TensorRT-friendly modules in utils.plugins and utils.modules
* benchmarking functions in utils.benchmark
* experiment functions to compare different implemented models with same interface

We recommend to run scripts using CUDA_VISIBLE_DEVICES=0,1 (or any other devices not less than 2), because TensorRT will constantly use cuda:0. We should leave as much space as possible for TensorRT, so we'd better use cuda:1 in pytorch.

## Install

```
pip install -e .
```