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

import fastldm.modules as fm
from fastldm.modifier import modify, MODIFIER, post_transform
from ldm.modules.diffusionmodules.util import GroupNorm32
MODIFIER.register(GroupNorm32, fm.GroupNorm)(MODIFIER.get(torch.nn.GroupNorm, fm.GroupNorm))

from ldm.modules.attention import CrossAttention
@MODIFIER.register(CrossAttention, fm.ldmCrossAttn)
def modifier_crossattn(src_instance, dst_type):
    dst_instance = dst_type(src_instance.to_q.weight.shape[1], context_dim=src_instance.to_k.weight.shape[1], heads=src_instance.heads, dim_head=src_instance.to_q.weight.shape[0]//src_instance.heads)
    return post_transform(src_instance, dst_instance)

MODIFIER.register(CrossAttention, fm.ldmSelfAttn)(MODIFIER.get(CrossAttention, fm.ldmCrossAttn))

map_dict = {
    torch.nn.LayerNorm: fm.LayerNorm,
    torch.nn.GroupNorm: fm.GroupNorm,
    GroupNorm32: fm.GroupNorm,
    CrossAttention: {'attn1': fm.ldmSelfAttn, 'attn2': fm.ldmCrossAttn}
}
model = modify(model, map_dict)

from fastldm.experiment import experiment_onnx, experiment_trt
from fastldm.environ import ONNX_ONLY
if ONNX_ONLY:
    measure, var, outputs = experiment_onnx(model, (input_0, input_1, input_2))
else:
    measure, var, outputs = experiment_trt(model, (input_0, input_1, input_2))

print('output maximum absolute error:', var)