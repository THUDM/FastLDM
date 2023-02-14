"""
Todo:

1. study the structure of ldm unet
    * components
    * input
    * output
2. restructure the unet code into
    * onnx friendly
    * plugin replacement
3. benchmark the transformed model
"""

from omegaconf import OmegaConf

config = OmegaConf.load("../../stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
config = config.model.params.unet_config

# from ldm.util import instantiate_from_config

# unet = instantiate_from_config(config)
import torch
from ldm.modules.diffusionmodules.openaimodel import UNetModel
# The CrossAtttion modules are replaced with utils.modules.ldmSelfAttn&ldmCrossAttn
# checkpointing is switched off

unet = UNetModel(**config.params) #, use_fp16=True)
# breakpoint()
ckpt = torch.load('../../stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt')
state_dict = {}
for k in ckpt['state_dict']:
    if 'model.diffusion_model.' in k:
        state_dict[k[len('model.diffusion_model.'):]] = ckpt['state_dict'][k]
print(state_dict.keys())
unet.load_state_dict(state_dict)

model = unet.eval()#.half()
# model.convert_to_fp16()
model = model.cuda()
input_0 = torch.randn(6, 4, 64, 64, dtype=torch.float32).cuda()
input_1 = torch.tensor([1, 3, 7, 8, 9, 23], dtype=torch.int32).cuda()
input_2 = torch.randn(6, 77, 768, dtype=torch.float32).cuda()
from utils.experiment import generate_trt, experiment
trt_name = generate_trt(unet, (input_0, input_1, input_2))
measure_dict, var, outputs_dict = experiment([model], [trt_name], (input_0, input_1, input_2))
for k in measure_dict:
    print(k, measure_dict[k])
print(var)
breakpoint()