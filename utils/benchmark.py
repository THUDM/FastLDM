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
    return {'average': times.mean(), 'min': times.min()}

import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger()

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_trt(engine_path, inputs_dict, n_iter, warmup_step=5):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs_h = inputs_dict
    inputs_d = {}
    outputs_h = {}
    outputs_d = {}
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        # size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            inputs_d[binding] = cuda.mem_alloc(inputs_h[binding].nbytes)
            bindings.append(int(inputs_d[binding]))
        else:
            shape = tuple(context.get_binding_shape(binding_idx))
            outputs_h[binding] = cuda.pagelocked_empty(shape, dtype)
            outputs_d[binding] = cuda.mem_alloc(outputs_h[binding].nbytes)
            bindings.append(int(outputs_d[binding]))
    def func():
        stream = cuda.Stream()
        for k in inputs_h:
            cuda.memcpy_htod_async(inputs_d[k], inputs_h[k], stream)
        # stream.synchronize()
        # from utils import benchmark
        # print(benchmark(context, (), {'bindings': bindings, 'stream_handle':stream.handle}, 100, 'execute_async_v2'))
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for k in outputs_d:
            cuda.memcpy_dtoh_async(outputs_h[k], outputs_d[k], stream)
        stream.synchronize()
        return outputs_h
    return benchmark(func, (), {}, n_iter, warmup_step=warmup_step)
