import os
import torch
import torch.nn as nn
import tensorrt as trt

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

from torch.testing._internal.common_utils import numpy_to_torch_dtype_dict
def get_trt_stuff(engine_path):
    engine = load_engine(engine_path)
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

def run_trt(context, bindings, stream=None):
    if stream is None:
        stream = torch.cuda.default_stream()
    state = context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
    stream.synchronize()
    return state

class TRTModule(nn.Module):
    def __init__(self, engine_path):
        """
        Only support running engine on cuda:0 for now
        """
        super().__init__()
        self.context, self.bindings, self.inputs_dict, self.outputs_dict = get_trt_stuff(engine_path)
    def forward(self, *inputs, **kw_args):
        device = 'cpu'
        for i, inp in enumerate(inputs):
            self.inputs_dict['input_{}'.format(i)].copy_(inp)
            device = inp.device
        shift = len(inputs)
        for k in kw_args:
            self.inputs_dict['input_{}'.format(shift)].copy_(kw_args[k])
            shift += 1
        state = run_trt(self.context, self.bindings)
        if not state:
            raise Exception("trt engine execution failed")
        outputs = []
        for i in range(len(self.outputs_dict)):
            outputs.append(self.outputs_dict['output_{}'.format(i)].cpu().to(device))
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
        ("min", (mi, (m==mi).nonzero()[:1])),
        ("max", (ma, (m==ma).nonzero()[:1])),
        ("mean", (mm, ))
    ])
    return profile

def profile_outdiff(o1, o2):
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
    return measure_dict

