# FastLDM

We focus on inference speed-up for [ldm](https://github.com/CompVis/stable-diffusion).

## TensorRT

We are now trying to convert model of stable-diffusion into an optimized TensorRT engine. See `examples/` folder for use cases.

When we transform a PyTorch model to TensorRT engine, we usually follow "torch->onnx->trt" footprint. It is convenient for most cases, except when we want to use [plugins](https://github.com/NVIDIA/TensorRT/tree/main/plugin). If we want to use plugins, we usually need to modify the exported onnx file based on [TensorRT official demo](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion). The core idea of this repo is to provide a shortcut. We provide many plugin-friendly PyTorch modules, which can plug into your normal PyTorch model and export onnx directly able to be optimized by TensorRT with plugins.

The advantages of this repo are:

* Seamlessly transform from PyTorch to TensorRT with plugins.
* Seamlessly export onnx to run directly in onnxruntime.

More specifically, we provide:

* parameter mapping functions from a module to another one in utils.mapping and utils.modifier
* TensorRT-friendly modules in utils.plugins and utils.modules
* benchmarking functions in utils.benchmark
* experiment functions to compare different implemented models with same interface

## Install

```
pip install -e .
```

or

```
pip install git+https://github.com/THUDM/FastLDM.git
```

## Environment

We rely on the docker environment `nvcr.io/nvidia/pytorch:22.12-py3`. (For newer version of TensorRT, this repo may not work because of removement of plugins.)
