import os
import torch
import torch.nn as nn
import tensorrt as trt
import ctypes
from collections import defaultdict

def list_or_tuple(x):
    return isinstance(x, (list, tuple))

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
from .environ import PLUGINS
for path in PLUGINS:
    ctypes.cdll.LoadLibrary(path)

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

from torch.testing._internal.common_utils import numpy_to_torch_dtype_dict
numpy_to_torch_dtype_dict[bool] = torch.bool
def get_trt_stuff(engine):
    context = engine.create_execution_context()
    inputs_dict = {}
    outputs_dict = {}
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        # size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        shape = tuple(context.get_binding_shape(binding_idx))
        if engine.binding_is_input(binding):
            inputs_dict[binding] = torch.empty(*shape, dtype=numpy_to_torch_dtype_dict[dtype], device='cuda')
            bindings.append(int(inputs_dict[binding].data_ptr()))
        else:
            outputs_dict[binding] = torch.empty(*shape, dtype=numpy_to_torch_dtype_dict[dtype], device='cuda')
            bindings.append(int(outputs_dict[binding].data_ptr()))
    return context, bindings, inputs_dict, outputs_dict

class TRTModule(nn.Module):
    def __init__(self, engine_path, num_worker):
        """
        Only support running engine on cuda:0 for now
        """
        super().__init__()
        self.num_worker = num_worker
        self.engine = load_engine(engine_path)
        self.context = []
        self.bindings = []
        self.inputs_dict = []
        self.outputs_dict = []
        self.stream = []
        for i in range(num_worker):
            context, bindings, inputs_dict, outputs_dict = get_trt_stuff(self.engine)
            self.context.append(context)
            self.bindings.append(bindings)
            self.inputs_dict.append(inputs_dict)
            self.outputs_dict.append(outputs_dict)
            self.stream.append(torch.cuda.Stream(0))
    def move_to_engine(self, inputs_h):
        for i in range(self.num_worker):
            for k in inputs_h:
                self.inputs_dict[i][k].copy_(inputs_h[k][i])
        torch.cuda.default_stream().synchronize()
    def run_engine(self):
        for context, bindings, stream in zip(self.context, self.bindings, self.stream):
            state = context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
            if not state:
                raise Exception("trt engine execution failed")
        for stream in self.stream:
            stream.synchronize()
    def merge_output(self):
        final_outputs_dict = defaultdict(list)
        for outputs_dict in self.outputs_dict:
            for k in outputs_dict:
                final_outputs_dict[k].append(outputs_dict[k])
        for k in final_outputs_dict:
            final_outputs_dict[k] = torch.cat(final_outputs_dict[k])
        return final_outputs_dict
    def forward(self, *inputs, **kw_args):
        inputs_h = {}
        device = 'cpu'
        for i, inp in enumerate(inputs):
            inputs_h['input_{}'.format(i)] = torch.chunk(inp, self.num_worker)
            device = inp.device
        shift = len(inputs)
        for k in kw_args:
            inputs_h['input_{}'.format(shift)] = torch.chunk(kw_args[k], self.num_worker)
            shift += 1
        self.move_to_engine(inputs_h)
        self.run_engine()
        outputs_dict = self.merge_output()
        outputs = []
        for i in range(len(outputs_dict)):
            outputs.append(outputs_dict['output_{}'.format(i)].to(device))
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

import onnxruntime as ort

def get_ort_stuff(onnx_path, providers):
    return ort.InferenceSession(onnx_path, providers=providers)

class ORTModule(nn.Module):
    def __init__(self, onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        super().__init__()
        self.sess = get_ort_stuff(onnx_path, providers)
    def forward(self, *inputs, **kw_args):
        device = 'cpu'
        for inp in inputs:
            device = inp.device
        for k in kw_args:
            device = kw_args[k].device
        inputs_dict = {'input_{}'.format(i):x.cpu().numpy() if isinstance(x, torch.Tensor) else x for i, x in enumerate(inputs)}
        shift = len(inputs_dict)
        for k in kw_args:
            inputs_dict['input_{}'.format(shift)] = kw_args[k].cpu().numpy()
            shift += 1
        outputs = self.sess.run(None, inputs_dict)
        outputs = [torch.from_numpy(x).to(device) for x in outputs]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

from collections import OrderedDict

def profile_matrix(m):
    mi = m.min()
    ma = m.max()
    mm = m.mean()
    profile = OrderedDict([
        ("shape", m.shape),
        ("min", (mi, (m==mi).nonzero()[0])),
        ("max", (ma, (m==ma).nonzero()[0])),
        ("mean", (mm, ))
    ])
    return profile

def profile_outdiff(o1, o2):
    o1 = o1.cpu().type(torch.float32)
    o2 = o2.cpu().type(torch.float32)
    error = (o1-o2).abs()
    measure_dict = OrderedDict([
        ("o1", profile_matrix(o1)),
        ("o2", profile_matrix(o2)),
        ("|o1|", profile_matrix(o1.abs())),
        ("|o2|", profile_matrix(o2.abs())),
        ("absolute error", profile_matrix(error)),
        ("relative error", profile_matrix(error / torch.max(o1.abs(), o2.abs()))),
        ("norm relative error", profile_matrix(error / o1.abs().mean()))
    ])
    for i in ["absolute error", "relative error", "norm relative error"]:
        for j in ["min", "max"]:
            ind = measure_dict[i][j][-1]
            measure_dict[i][j] += (o1[tuple(ind)], o2[tuple(ind)])
    return measure_dict

