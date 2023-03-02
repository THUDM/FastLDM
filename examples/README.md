For `transform_stable_diffusion_unet.py`, use this fork: https://github.com/1049451037/stable-diffusion

We recommend to run scripts using CUDA_VISIBLE_DEVICES=0,1 (or any other devices not less than 2), because TensorRT will constantly use cuda:0. We should leave as much space as possible for TensorRT, so we'd better use cuda:1 in pytorch.

The engine will speed up 4x to the original torch module, and 2.8x to the autocast context for RTX 3090. You can run `benchmark_unet.py` to benchmark it.

For `transform_diffusers_unet.py`, you may meet [this issue](https://github.com/pytorch/pytorch/issues/93937). It's a pytorch bug so we don't provide detailed plugin example here. If you want to use attention plugins, operations are similar to `transform_stable_diffusion_unet.py`.
