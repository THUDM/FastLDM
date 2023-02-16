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
    def forward(self, *inputs):
        for i, inp in enumerate(inputs):
            self.inputs_dict['input_{}'.format(i)].copy_(inp)
        state = run_trt(self.context, self.bindings)
        if not state:
            raise Exception("trt engine execution failed")
        outputs = []
        for i in range(len(self.outputs_dict)):
            outputs.append(self.outputs_dict['output_{}'.format(i)].clone())
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
    def forward(self, *inputs):
        inputs_dict = {'input_{}'.format(i):x.cpu().numpy() if isinstance(x, torch.Tensor) else x for i, x in enumerate(inputs)}
        outputs = self.sess.run(None, inputs_dict)
        outputs = [torch.from_numpy(x) for x in outputs]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs