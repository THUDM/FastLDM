import torch
from utils import benchmark, benchmark_trt

model_channels = 320
timesteps = torch.tensor([320, 452, 777], dtype=torch.int64).cuda()
max_period = torch.tensor([10000], dtype=torch.int64).cuda()

from ts import timestep_embedding
print('origin func', benchmark(timestep_embedding, (timesteps, model_channels), {'repeat_only':False}, 100))
# t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False)

from ts import TS
tsmodel = TS(model_channels).cuda()
# t_emb2 = tsmodel(timesteps, max_period)
print('nn.module', benchmark(tsmodel, (timesteps, max_period), {}, 100))

tsm = torch.jit.script(tsmodel, (timesteps, max_period))
print('torchscript', benchmark(tsm, (timesteps, max_period), {}, 100))
torch.onnx.export(tsm, (timesteps, max_period), "origin.onnx", input_names = ['input_0', 'input_1'], output_names = ['output_0'])
# import os
# os.system("trtexec --onnx=origin.onnx --saveEngine=origin.trt")

inputs_h = {
    'input_0': timesteps.type(torch.int32).cpu().numpy(),
    'input_1': max_period.type(torch.int32).cpu().numpy()
}
print('trt engine', benchmark_trt('origin.trt', inputs_h, 100))

# os.system("torchtrtc origin.trt origin.ts --embed-engine --device-type=gpu")

import torch_tensorrt
model = torch.jit.load('origin.ts')
print('torchtrt', benchmark(model, (timesteps.type(torch.int32), max_period.type(torch.int32)), {}, 100))