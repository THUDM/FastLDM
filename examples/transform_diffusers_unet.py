import torch
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",
                                            torch_dtype=torch.float16,
											revision="fp16",
                                            subfolder="unet")
unet.cuda(1)
inputs = torch.randn(2, 4, 64, 64, dtype=torch.half, device='cuda:1'), torch.tensor([1, 3], dtype=torch.int32, device='cuda:1'), torch.randn(2, 77, 768, dtype=torch.half, device='cuda:1')

import fastldm.modules as fm
from fastldm.modifier import modify, MODIFIER
map_dict = {
    torch.nn.LayerNorm: fm.LayerNorm,
    torch.nn.GroupNorm: fm.GroupNorm,
}
unet = modify(unet, map_dict)

from fastldm.experiment import experiment_onnx, experiment_trt
from fastldm.environ import ONNX_ONLY
if ONNX_ONLY:
    measure, var, outputs = experiment_onnx(unet, inputs)
else:
    measure, var, outputs = experiment_trt(unet, inputs)

print(var)
breakpoint()
for i in range(len(outputs[type(unet).__name__])):
    out_model = outputs[type(unet).__name__][i].cpu()
    out_ort = outputs['TRTModule'][i].cpu()

    from fastldm.helper import profile_outdiff
    measure = profile_outdiff(out_model, out_ort)
    # print(measure)
    import pprint 
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(measure)
    breakpoint()