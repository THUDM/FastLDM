import time
import torch
import numpy as np
from tqdm import tqdm

def benchmark(model, inputs, kwargs, n_iter, func_name=None, warmup_step=5):
    if hasattr(model, 'eval') and callable(model.eval):
        model.eval()
        print('eval mode...')
    if func_name is not None:
        func = getattr(model, func_name)
    else:
        func = model
    print('start warming up...')
    with torch.no_grad():
        for i in tqdm(range(warmup_step)):
            outputs = func(*inputs, **kwargs)
            assert outputs is not None
    print('start timing...')
    time_list = []
    with torch.no_grad():
        for i in tqdm(range(n_iter)):
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = func(*inputs, **kwargs)
            torch.cuda.synchronize()
            end_time = time.time()
            assert outputs is not None
            time_list.append(end_time - start_time)
    times = np.array(time_list)
    measurements = {'average': times.mean(), 'min': times.min()}
    return measurements, outputs

import os
import tensorrt as trt
from torch.testing._internal.common_utils import numpy_to_torch_dtype_dict

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_trt(engine_path, inputs_dict, n_iter, warmup_step=5):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    outputs_dict = {}
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        # size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            bindings.append(int(inputs_dict[binding].data_ptr()))
        else:
            shape = tuple(context.get_binding_shape(binding_idx))
            outputs_dict[binding] = torch.empty(*shape, dtype=numpy_to_torch_dtype_dict[dtype], device='cuda')
            bindings.append(int(outputs_dict[binding].data_ptr()))
    stream = torch.cuda.Stream()
    def func():
        state = context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        stream.synchronize()
        return state
    measurement, state = benchmark(func, (), {}, n_iter, warmup_step=warmup_step)
    assert state is True
    return measurement, outputs_dict
