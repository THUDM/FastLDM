import os
from torch.onnx import OperatorExportTypes
from .benchmark import benchmark, benchmark_trt
import numpy as np
from collections import OrderedDict
import torch

def generate_trt(model, inputs, experiment_name=''):
    os.makedirs('./onnx/', exist_ok=True)
    os.makedirs('./trt/', exist_ok=True)
    name = type(model).__name__ + '_' + experiment_name
    model.eval()
    with torch.no_grad():
        outputs = model(*inputs)
        num_output = 1 if type(outputs) is not list and type(outputs) is not tuple else len(outputs)
        del outputs
        os.system("rm onnx/{}.onnx".format(name))
        torch.onnx.export(model, inputs, 'onnx/{}.onnx'.format(name), operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH, input_names=['input_{}'.format(i) for i in range(len(inputs))], output_names=['output_{}'.format(i) for i in range(num_output)])
    os.system("rm trt/{}.trt".format(name))
    os.system("trtexec --onnx=onnx/{}.onnx --saveEngine=trt/{}.trt --fp16".format(name, name))
    return 'trt/'+name+'.trt'

def experiment(models, trt_models, inputs, n_iter=100):
    measure_dict = OrderedDict()
    outputs_dict = OrderedDict()
    for model in models:
        name = type(model).__name__
        measure, outputs = benchmark(model, inputs, {}, n_iter)
        measure_dict[name] = measure
        outputs_dict[name] = outputs
    for trt in trt_models:
        inputs_h = {'input_{}'.format(i): inputs[i] for i in range(len(inputs))}
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
                ma = max(ma, (o1-o2).abs().max().item())
            var[i, j] = ma
    return measure_dict, var, outputs_dict