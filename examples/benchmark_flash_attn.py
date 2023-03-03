import torch
from fastldm.modules import FlashCrossAttnWG, ldmCrossAttnWG
model = ldmCrossAttnWG(768, 768, 8, 64).half().cuda()
model_flash = FlashCrossAttnWG(768, 768, 8, 64).half().cuda()
from fastldm.mapping import MAPPING
MAPPING.transform(model, model_flash)
x = torch.randn(4, 128, 768).half().cuda()
context = torch.randn(4, 88, 768).half().cuda()
mask = (torch.randn(4, 8, 128, 88) > 0.1).cuda()
from fastldm.experiment import experiment
from fastldm.benchmark import benchmark_backward

measure, var, outputs = experiment([model, model_flash], [], (x, context, mask))
print(measure)
print(var)
measure, var, outputs = experiment([model, model_flash], [], (x, context, mask), forward_only=False)
print(measure)
print(var)
measure, var, outputs = experiment([model, model_flash], [], (x, context, mask), benchmark_func=benchmark_backward)
print(measure)
print(var)
breakpoint()