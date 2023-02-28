import time
import torch
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
from .helper import list_or_tuple

def benchmark(model, inputs, kwargs, n_iter, func_name=None, warmup_step=5, use_autocast=False):
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
        context = torch.autocast("cuda") if use_autocast else nullcontext()
        with context:
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

from .helper import get_trt_stuff
from .helper import load_engine
def benchmark_trt(engine_path, inputs_h, n_iter, warmup_step=5):
    engine = load_engine(engine_path)
    context, bindings, inputs_dict, outputs_dict = get_trt_stuff(engine)
    if list_or_tuple(inputs_h):
        inputs_h = {'input_{}'.format(i):x for i, x in enumerate(inputs_h)}
    for k in inputs_h:
        inputs_dict[k].copy_(inputs_h[k])
    stream = torch.cuda.default_stream()
    def func():
        state = context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        stream.synchronize()
        return state
    measurement, state = benchmark(func, (), {}, n_iter, warmup_step=warmup_step)
    assert state is True
    return measurement, outputs_dict


def benchmark_trt_np(engine_path, inputs_dict, n_iter, warmup_step=5):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
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
    stream = cuda.Stream()
    for k in inputs_h:
        cuda.memcpy_htod_async(inputs_d[k], inputs_h[k], stream)
    stream.synchronize()
    def func():
        state = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        return state
    # measurement, _ = benchmark(context, (), {'bindings': bindings, 'stream_handle':stream.handle}, n_iter, func_name='execute_async_v2', warmup_step=warmup_step)
    measurement, state = benchmark(func, (), {}, n_iter, warmup_step=warmup_step)
    assert state is True
    for k in outputs_d:
        cuda.memcpy_dtoh_async(outputs_h[k], outputs_d[k], stream)
    stream.synchronize()
    return measurement, outputs_h