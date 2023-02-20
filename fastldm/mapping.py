from .registry import Registry
import torch

MAPPING = Registry('mapping')

@MAPPING.register(type(None), type(None))
def mapping_default(src, dst):
    for s, d in zip(src.parameters(), dst.parameters()):
        d.data = s.data

from .modules import qkvLinearSlow, qkvLinear
@MAPPING.register(qkvLinearSlow, qkvLinear)
def mapping_qkv_linear(src, dst):
    wq = src.Wq.weight.data
    bq = src.Wq.bias.data
    wk = src.Wk.weight.data
    bk = src.Wk.bias.data
    wv = src.Wv.weight.data
    bv = src.Wv.bias.data
    wqkv = torch.cat([wq, wk, wv]).view(3, src.num_heads, src.size_per_head, src.hidden_size).transpose(0, 1).contiguous().view(3*src.hidden_size, src.hidden_size)
    bqkv = torch.cat([bq, bk, bv]).view(3, src.num_heads, src.size_per_head).transpose(0, 1).contiguous().view(3*src.hidden_size)
    dst.Wqkv.weight.data = wqkv
    dst.Wqkv.bias.data = bqkv


from torch.nn import MultiheadAttention
@MAPPING.register(qkvLinearSlow, MultiheadAttention)
def mapping_qkv_linear_mha(src, dst):
    dst.in_proj_weight.data = torch.cat([src.Wq.weight.data, src.Wk.weight.data, src.Wv.weight.data])
    dst.in_proj_bias.data = torch.cat([src.Wq.bias.data, src.Wk.bias.data, src.Wv.bias.data])
    dst.out_proj.weight.data = torch.eye(dst.out_proj.weight.data.shape[0], dtype=dst.out_proj.weight.data.dtype, device=dst.out_proj.weight.data.device)
    dst.out_proj.bias.data = torch.zeros(dst.out_proj.bias.data.shape[0], dtype=dst.out_proj.bias.data.dtype, device=dst.out_proj.bias.data.device)