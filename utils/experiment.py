import os
from torch.onnx import OperatorExportTypes
from .benchmark import benchmark, benchmark_trt
import numpy as np
from collections import OrderedDict
import torch
from .helper import ORTModule, TRTModule

def generate_trt(model, inputs, kw_args={}, experiment_name='', onnx_only=False):
    os.makedirs('./onnx/', exist_ok=True)
    os.makedirs('./trt/', exist_ok=True)
    name = type(model).__name__ + '_' + experiment_name
    model.eval()
    with torch.no_grad():
        outputs = model(*inputs, **kw_args)
        num_output = 1 if type(outputs) is not list and type(outputs) is not tuple else len(outputs)
        os.system("rm onnx/{}.onnx".format(name))
        torch.onnx.export(model, tuple(inputs)+(kw_args,), 'onnx/{}.onnx'.format(name), operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH, input_names=['input_{}'.format(i) for i in range(len(inputs)+len(kw_args))], output_names=['output_{}'.format(i) for i in range(num_output)])
        if onnx_only:
            return 'onnx/{}.onnx'.format(name)
    os.system("rm trt/{}.trt".format(name))
    os.system("trtexec --onnx=onnx/{}.onnx --saveEngine=trt/{}.trt --fp16 --buildOnly".format(name, name))
    return 'onnx/{}.onnx'.format(name), 'trt/{}.trt'.format(name)

def experiment(models, trt_models, inputs, kw_args={}, n_iter=100):
    measure_dict = OrderedDict()
    outputs_dict = OrderedDict()
    for model in models:
        name = type(model).__name__
        measure, outputs = benchmark(model, inputs, kw_args, n_iter)
        measure_dict[name] = measure
        outputs_dict[name] = outputs
    inputs_h = {'input_{}'.format(i): inputs[i] for i in range(len(inputs))}
    shift = len(inputs)
    for k in kw_args:
        inputs_h['input_{}'.format(shift)] = kw_args[k]
        shift += 1
    for trt in trt_models:
        measure, outputs = benchmark_trt(trt, inputs_h, n_iter)
        measure_dict[trt] = measure
        outputs = [outputs['output_{}'.format(i)] for i in range(len(outputs))]
        if len(outputs) == 1:
            outputs = outputs[0]
        outputs_dict[trt] = outputs
    num = len(outputs_dict)
    var = np.zeros((num, num))
    for i, k1 in enumerate(outputs_dict):
        for j, k2 in enumerate(outputs_dict):
            if i == j:
                continue
            ma = 0.
            for o1, o2 in zip(outputs_dict[k1], outputs_dict[k2]):
                ma = max(ma, (o1.cpu()-o2.cpu()).abs().max().item())
            var[i, j] = ma
    return measure_dict, var, outputs_dict

def experiment_onnx(model, inputs, kw_args={}, new_inputs=None, new_kw_args=None):
    onnx_path = generate_trt(model, inputs, kw_args=kw_args, onnx_only=True)
    ortmodel = ORTModule(onnx_path)
    if new_inputs is None:
        new_inputs = inputs
    if new_kw_args is None:
        new_kw_args = kw_args
    return experiment([model, ortmodel], [], new_inputs, kw_args=new_kw_args, n_iter=1)

def experiment_onnx_trt(model, inputs, kw_args={}, new_inputs=None, new_kw_args=None):
    onnx_path, trt_path = generate_trt(model, inputs, kw_args=kw_args)
    ortmodel = ORTModule(onnx_path)
    trtmodel = TRTModule(trt_path)
    if new_inputs is None:
        new_inputs = inputs
    if new_kw_args is None:
        new_kw_args = kw_args
    return experiment([model, ortmodel, trtmodel], [], new_inputs, kw_args=new_kw_args, n_iter=1)