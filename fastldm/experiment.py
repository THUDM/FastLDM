import os
from torch.onnx import OperatorExportTypes
from .benchmark import benchmark, benchmark_trt
import numpy as np
from collections import OrderedDict
import torch
from .helper import ORTModule, TRTModule, list_or_tuple
from .environ import ONNX_ONLY, PLUGINS
from collections import Counter

PLUGIN_config = ' '.join(['--plugins={}'.format(p) for p in PLUGINS])

def generate_trt(model, inputs, kw_args={}, experiment_name='', onnx_only=False, use_fp16=True, skip=False):
    os.makedirs('./onnx/', exist_ok=True)
    os.makedirs('./trt/', exist_ok=True)
    name = '{}_ONNX_ONLY_{}_{}'.format(type(model).__name__, ONNX_ONLY, experiment_name)
    if skip:
        if onnx_only:
            return 'onnx/{}.onnx'.format(name)
        return 'onnx/{}.onnx'.format(name), 'trt/{}.trt'.format(name)
    model.eval()
    with torch.no_grad():
        outputs = model(*inputs, **kw_args)
        num_output = 1 if not list_or_tuple(outputs) else len(outputs)
        os.system("rm onnx/{}.onnx".format(name))
        torch.onnx.export(model, tuple(inputs)+(kw_args,), 'onnx/{}.onnx'.format(name), operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH, input_names=['input_{}'.format(i) for i in range(len(inputs)+len(kw_args))], output_names=['output_{}'.format(i) for i in range(num_output)])
        if onnx_only:
            return 'onnx/{}.onnx'.format(name)
    os.system("rm trt/{}.trt".format(name))
    os.system("trtexec --onnx=onnx/{}.onnx --saveEngine=trt/{}.trt --buildOnly {} {}".format(name, name, '--fp16' if use_fp16 else '', PLUGIN_config))
    return 'onnx/{}.onnx'.format(name), 'trt/{}.trt'.format(name)

def experiment(models, trt_models, inputs, kw_args={}, n_iter=100, warm_up_step=5, forward_only=True, benchmark_func=benchmark):
    measure_dict = OrderedDict()
    outputs_dict = OrderedDict()
    name_count = Counter()
    len_out = 0
    for model in models:
        name = type(model).__name__ + str(name_count[type(model).__name__])
        measure, outputs = benchmark_func(model, inputs, kw_args, n_iter, warmup_step=warm_up_step, forward_only=forward_only)
        if not list_or_tuple(outputs):
            outputs = [outputs]
        measure_dict[name] = measure
        outputs_dict[name] = outputs
        len_out = len(outputs)
        name_count[type(model).__name__] += 1
    inputs_h = {'input_{}'.format(i): inputs[i] for i in range(len(inputs))}
    shift = len(inputs)
    for k in kw_args:
        inputs_h['input_{}'.format(shift)] = kw_args[k]
        shift += 1
    for trt in trt_models:
        measure, outputs = benchmark_trt(trt, inputs_h, n_iter, warmup_step=warm_up_step)
        measure_dict[trt] = measure
        outputs = [outputs['output_{}'.format(i)] for i in range(len(outputs))]
        outputs_dict[trt] = outputs
        len_out = len(outputs)
    num = len(outputs_dict)
    var = np.zeros((len_out, num, num))
    for i, k1 in enumerate(outputs_dict):
        for j, k2 in enumerate(outputs_dict):
            if i == j:
                continue
            for k, (o1, o2) in enumerate(zip(outputs_dict[k1], outputs_dict[k2])):
                var[k, i, j] = (o1.cpu().type(torch.float32)-o2.cpu().type(torch.float32)).abs().max().item()
    return measure_dict, var, outputs_dict

def experiment_onnx(model, inputs, kw_args={}, new_inputs=None, new_kw_args=None):
    onnx_path = generate_trt(model, inputs, kw_args=kw_args, onnx_only=True)
    ortmodel = ORTModule(onnx_path)
    if new_inputs is None:
        new_inputs = inputs
    if new_kw_args is None:
        new_kw_args = kw_args
    return experiment([model, ortmodel], [], new_inputs, kw_args=new_kw_args, n_iter=1, warm_up_step=0)

def experiment_trt(model, inputs, kw_args={}, new_inputs=None, new_kw_args=None, use_fp16=True, skip=False):
    _, trt_path = generate_trt(model, inputs, kw_args=kw_args, use_fp16=use_fp16, skip=skip)
    trtmodel = TRTModule(trt_path, 1)
    if new_inputs is None:
        new_inputs = inputs
    if new_kw_args is None:
        new_kw_args = kw_args
    return experiment([model, trtmodel], [], new_inputs, kw_args=new_kw_args, n_iter=1, warm_up_step=0)

def experiment_onnx_trt(model, inputs, kw_args={}, new_inputs=None, new_kw_args=None):
    onnx_path, trt_path = generate_trt(model, inputs, kw_args=kw_args)
    ortmodel = ORTModule(onnx_path)
    trtmodel = TRTModule(trt_path)
    if new_inputs is None:
        new_inputs = inputs
    if new_kw_args is None:
        new_kw_args = kw_args
    return experiment([model, ortmodel, trtmodel], [], new_inputs, kw_args=new_kw_args, n_iter=1, warm_up_step=0)