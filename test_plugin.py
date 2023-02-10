from utils.modules import qkvLinear, qkvLinearSlow, CustomQKVToContextPluginDynamic, FlashSelfAttn
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, Wq=None, Wk=None, Wv=None):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.projection = qkvLinear(hidden_size, num_heads, Wq=Wq, Wk=Wk, Wv=Wv)
    def forward(self, x):
        # shape of x (seq_len, batch_size, hidden_size)
        # shape of i_mask (batch_size)
        # output (seq_len, batch_size, hidden_size)
        qkv = self.projection(x)
        return CustomQKVToContextPluginDynamic.apply(qkv, self.hidden_size, self.num_heads).select(-1, 0).select(-1, 0)

lin = qkvLinearSlow(768, 8)
model = SelfAttention(768, 8, lin.Wq, lin.Wk, lin.Wv).cuda().half()
flash = FlashSelfAttn(768, 8, lin.Wq, lin.Wk, lin.Wv).cuda().half()
input = torch.randn(512, 2, 768).cuda().half() # seq_len 128 bug

outputs = model(input)
from torch.nn import MultiheadAttention
th_model = MultiheadAttention(768, 8)
th_model.in_proj_weight.data = torch.cat([lin.Wq.weight.data, lin.Wk.weight.data, lin.Wv.weight.data]) #model.projection.Wqkv.weight.data
th_model.in_proj_bias.data = torch.cat([lin.Wq.bias.data, lin.Wk.bias.data, lin.Wv.bias.data]) #model.projection.Wqkv.bias.data
th_model.out_proj.weight.data = torch.eye(768)
th_model.out_proj.bias.data = torch.zeros(768)
th_model = th_model.cuda().half()
outputs_th = th_model(input, input, input)
# breakpoint()

# i_mask = torch.tensor([[64], [78]], dtype=torch.float32).cuda()

from torch.onnx import OperatorExportTypes
torch.onnx.export(model, (input, ), 'test.onnx', operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH, input_names=['input_0'], output_names=['output_0'])
import os
os.system("trtexec --onnx=test.onnx --saveEngine=test.trt --fp16")

from utils import benchmark, benchmark_trt
inputs_h = {
    'input_0': input.cpu().numpy()
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

# lin_slow = qkvLinearSlow(768, 8)
# lin = qkvLinear(768, 8, lin_slow.Wq, lin_slow.Wk, lin_slow.Wv)
# x = torch.randn(128, 2, 768)
# out_slow = lin_slow(x)
# out = lin(x)
# breakpoint()