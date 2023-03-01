from .registry import Registry
import torch
from .mapping import MAPPING

MODIFIER = Registry('modifier')

def modify(model, map_dict):
    if type(model) in map_dict:
        return MODIFIER.transform(model, map_dict[type(model)])
    flag = True
    names = set()
    while flag:
        flag = False
        for name, child in model.named_children():
            if name not in names:
                setattr(model, name, modify(child, map_dict))
                names.add(name)
                flag = True
    return model

@MODIFIER.register(type(None), type(None))
def modifier_default(src_instance, dst_type):
    raise Exception("Related modifier from {} to {} is not implemented yet".format(type(src_instance).__name__, dst_type.__name__))

def post_transform(src_instance, dst_instance):
    param = src_instance.parameters().__next__()
    dst_instance = dst_instance.to(param.dtype).to(param.device)
    MAPPING.transform(src_instance, dst_instance)
    return dst_instance

from .modules import qkvLinearSlow, qkvLinear
@MODIFIER.register(qkvLinearSlow, qkvLinear)
def modifier_qkv_linear(src_instance, dst_type):
    dst_instance = dst_type(src_instance.hidden_size, src_instance.num_heads)
    return post_transform(src_instance, dst_instance)

from .modules import GroupNorm
@MODIFIER.register(torch.nn.GroupNorm, GroupNorm)
def modifier_groupnorm(src_instance, dst_type):
    dst_instance = dst_type(src_instance.num_groups, src_instance.num_channels, src_instance.eps)
    return post_transform(src_instance, dst_instance)

from ldm.modules.diffusionmodules.util import GroupNorm32
@MODIFIER.register(GroupNorm32, GroupNorm)
def modifier_groupnorm(src_instance, dst_type):
    dst_instance = dst_type(src_instance.num_groups, src_instance.num_channels, src_instance.eps)
    return post_transform(src_instance, dst_instance)

from .modules import NewLayerNorm
@MODIFIER.register(torch.nn.LayerNorm, NewLayerNorm)
def modifier_layernorm(src_instance, dst_type):
    dst_instance = dst_type(src_instance.normalized_shape[0], src_instance.eps)
    return post_transform(src_instance, dst_instance)