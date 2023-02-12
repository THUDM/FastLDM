from utils.modules import qkvLinearSlow, TRTSelfAttn, FlashSelfAttn
import torch
import torch.nn as nn
from utils.mapping import MAPPING

lin = qkvLinearSlow(768, 8)
model = TRTSelfAttn(768, 8)
MAPPING.get(lin, model.projection)(lin, model.projection)
model = model.cuda().half()
flash = FlashSelfAttn(768, 8)
MAPPING.get(lin, flash.projection)(lin, flash.projection)
flash = flash.cuda().half()
input = torch.randn(512, 2, 768).cuda().half() # seq_len 128 bug

outputs = model(input)
from torch.nn import MultiheadAttention
th_model = MultiheadAttention(768, 8)
MAPPING.get(lin, th_model)(lin, th_model)
th_model = th_model.cuda().half()

from torch.onnx import OperatorExportTypes
torch.onnx.export(model, (input, ), 'test.onnx', operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH, input_names=['input_0'], output_names=['output_0'])
import os
os.system("trtexec --onnx=test.onnx --saveEngine=test.trt --fp16")

from utils import benchmark, benchmark_trt
inputs_h = {
    'input_0': input
}
print(inputs_h['input_0'].dtype)
measure_trt, outputs_trt = benchmark_trt('test.trt', inputs_h, 100)
measure_th, outputs_th = benchmark(th_model, (input,input,input), {}, 100)
measure_my, outputs_my = benchmark(model, (input,), {}, 100)
measure_f, outputs_f = benchmark(flash, (input,), {}, 100)
print(measure_trt)
print(measure_th)
print(measure_my)
print(measure_f)
breakpoint()