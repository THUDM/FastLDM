from omegaconf import OmegaConf
config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
config = config.model.params.unet_config
# from ldm.util import instantiate_from_config
# unet = instantiate_from_config(config)
import torch
from ldm.modules.diffusionmodules.openaimodel import UNetModel

unet = UNetModel(**config.params)
ckpt = torch.load('stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt', map_location='cpu')
state_dict = {}
for k in ckpt['state_dict']:
    if 'model.diffusion_model.' in k:
        state_dict[k[len('model.diffusion_model.'):]] = ckpt['state_dict'][k]
unet.load_state_dict(state_dict)
model = unet.cuda(1)
input_0 = torch.randn(6, 4, 64, 64, dtype=torch.float32).cuda(1)
input_1 = torch.tensor([1, 3, 7, 8, 9, 23], dtype=torch.int32).cuda(1)
input_2 = torch.randn(6, 77, 768, dtype=torch.float32).cuda(1)

from fastldm.benchmark import benchmark, benchmark_trt
time_origin, outputs_origin = benchmark(model, (input_0, input_1, input_2), {}, 100)
time_ac, outputs_ac = benchmark(model, (input_0, input_1, input_2), {}, 100, use_autocast=True)
time_trt, outputs_trt = benchmark_trt('trt/UNetModel_ONNX_ONLY_False_.trt', (input_0, input_1, input_2), 100)

print('time origin', time_origin)
print('time autocast', time_ac)
print('time tensorrt', time_trt)

from fastldm.helper import profile_outdiff
measure = profile_outdiff(outputs_origin, outputs_trt['output_0'])
import pprint 
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(measure)