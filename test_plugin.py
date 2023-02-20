import torch
import torch.nn as nn
from fastldm.mapping import MAPPING
from fastldm.experiment import generate_trt, experiment

def experiment_self_attn():
    from fastldm.modules import qkvLinearSlow, TRTSelfAttn, FlashSelfAttn, TorchSelfAttn
    x = torch.randn(512, 2, 768).cuda().half() # seq_len 128 bug
    lin = qkvLinearSlow(768, 8)
    model = TRTSelfAttn(768, 8)
    MAPPING.get(lin, model.projection)(lin, model.projection)
    model = model.cuda().half()
    flash = FlashSelfAttn(768, 8)
    MAPPING.get(lin, flash.projection)(lin, flash.projection)
    flash = flash.cuda().half()
    th_model = TorchSelfAttn(768, 8)
    MAPPING.get(lin, th_model.mha)(lin, th_model.mha)
    th_model = th_model.cuda().half()
    _, trt_name = generate_trt(model, (x,))
    measure_dict, var, outputs_dict = experiment([model, flash, th_model], [trt_name], (x,))
    for k in measure_dict:
        print(k, measure_dict[k])
    print(var)

def experiment_ldm_attn(seq_len):
    from fastldm.modules import ldmSelfAttn, ldmCrossAttn
    from ldm.modules.attention import CrossAttention
    x = torch.randn(2, seq_len, 320).half().cuda()
    # model = ldmSelfAttn(320, heads=8, dim_head=40).half()
    model = ldmCrossAttn(320, heads=8, dim_head=40).half()
    # ldm_model = CrossAttention(320, heads=8, dim_head=40).half()
    # for src, dst in zip(model.parameters(), ldm_model.parameters()):
    #     dst.data = src.data
    model = model.cuda()
    # ldm_model = ldm_model.cuda()
    _, trt_name = generate_trt(model, (x,))
    measure_dict, var, outputs_dict = experiment([model], [trt_name], (x,))
    # for k in measure_dict:
    #     print(k, measure_dict[k])
    # print(var)
    return var[0, 1]

def experiment_flash_attn():
    from fastldm.modules import ldmSelfAttn
    from ldm.modules.attention import CrossAttention
    x = torch.randn(2, 4096, 320).half().cuda()
    model = ldmSelfAttn(320, heads=8, dim_head=40).half()
    ldm_model = CrossAttention(320, heads=8, dim_head=40).half()
    MAPPING.get(None, None)(model, ldm_model)
    model = model.cuda()
    ldm_model = ldm_model.cuda()
    _, trt_name = generate_trt(model, (x,))
    measure_dict, var, outputs_dict = experiment([model, ldm_model], [trt_name], (x,))
    for k in measure_dict:
        print(k, measure_dict[k])
    print(var)

def experiment_ldm_crossattn():
    from fastldm.modules import ldmCrossAttn
    from ldm.modules.attention import CrossAttention
    x = torch.randn(2, 4096, 320).cuda()
    context = torch.randn(2, 77, 768).cuda()
    model = ldmCrossAttn(320, context_dim=768, heads=8, dim_head=40)
    ldm_model = CrossAttention(320, context_dim=768, heads=8, dim_head=40)
    MAPPING.get(None, None)(model, ldm_model)
    model = model.cuda()
    ldm_model = ldm_model.cuda()
    _, trt_name = generate_trt(model, (x, context))
    measure_dict, var, outputs_dict = experiment([model, ldm_model], [trt_name], (x, context))
    for k in measure_dict:
        print(k, measure_dict[k])
    print(var)

def experiment_var():
    x = []
    y = []
    for l in range(64, 1024+1, 64):
        x.append(l)
        v = experiment_ldm_attn(l)
        y.append(v)
    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.savefig('var.png')


if __name__ == '__main__':
    # experiment_self_attn()
    # experiment_var()
    # experiment_ldm_crossattn()
    experiment_flash_attn()