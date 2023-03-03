import torch
import torch.nn.functional as F
from .environ import ONNX_ONLY

class BaseApply:
    @classmethod
    def apply(cls, *inputs, **kw_args):
        return cls.forward(None, *inputs, **kw_args)

BasePlugin = BaseApply if ONNX_ONLY else torch.autograd.Function

class CustomQKVToContextPluginDynamic(BasePlugin):
    # https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/bertQKVToContextPlugin
    # Restriction of this plugin: https://github.com/NVIDIA/TensorRT/issues/2653
    @staticmethod
    def forward(ctx, qkv, hidden_size, num_heads, type_id):
        # Now I get qkv with shape (seq_len, batch_size, 3*hidden_size, 1, 1)
        # I need to do attention to it
        # The layout of 3*hidden_size dimension is (num_heads, 3, size_per_head)
        size_per_head = hidden_size // num_heads
        seq_len = qkv.size(0)
        batch_size = qkv.size(1)
        qkv = qkv.view(seq_len, batch_size, num_heads, 3, size_per_head).transpose(0, 2)
        # q, k, v = torch.chunk(qkv, 3, dim=3)
        q = qkv.select(-2, 0)
        k = qkv.select(-2, 1)
        v = qkv.select(-2, 2) # (num_heads, batch_size, seq_len, size_per_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (size_per_head**-0.5)
        scores = F.softmax(scores, -1)
        result = torch.matmul(scores, v).transpose(0, 2).contiguous().view(seq_len, batch_size, hidden_size, 1, 1)
        return result
    @staticmethod
    def symbolic(g, qkv, hidden_size, num_heads, type_id):
        return g.op("CustomQKVToContextPluginDynamic", qkv, plugin_version_s='1', type_id_i=type_id, hidden_size_i=hidden_size, num_heads_i=num_heads, has_mask_i=False)

class fMHCA(BasePlugin):
    # https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/multiHeadCrossAttentionPlugin
    # Potential issue of this plugin: https://github.com/NVIDIA/TensorRT/issues/2674
    @staticmethod
    def forward(ctx, q, kv):
        """
        q: (batch_size, seq_len, num_head, size_per_head)
        kv: (batch_size, seq_len, num_head, 2, size_per_head)
        output: like q
        """
        size_per_head = q.size(3)
        batch_size = q.size(0)
        seq_len = q.size(1)
        num_head = q.size(2)
        q = q.transpose(1, 2).contiguous()
        kv = kv.transpose(1, 2).contiguous()
        k = kv.select(-2, 0)
        v = kv.select(-2, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (size_per_head**-0.5)
        scores = F.softmax(scores, -1)
        result = torch.matmul(scores, v).transpose(1, 2).contiguous().view(batch_size, seq_len, num_head, size_per_head)
        return result
    @staticmethod
    def symbolic(g, q, kv):
        return g.op("fMHCA", q, kv, plugin_version_s='1')

class fMHA_V2(BasePlugin):
    # https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/multiHeadFlashAttentionPlugin
    @staticmethod
    def forward(ctx, qkv):
        """
        qkv: (batch_size, seq_len, num_head, 3, size_per_head)
        output: (batch_size, seq_len, num_head, size_per_head)
        """
        size_per_head = qkv.size(4)
        batch_size = qkv.size(0)
        seq_len = qkv.size(1)
        num_head = qkv.size(2)
        qkv = qkv.transpose(1, 2).contiguous()
        q = qkv.select(-2, 0)
        k = qkv.select(-2, 1)
        v = qkv.select(-2, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (size_per_head**-0.5)
        scores = F.softmax(scores, -1)
        result = torch.matmul(scores, v).transpose(1, 2).contiguous().view(batch_size, seq_len, num_head, size_per_head)
        return result
    @staticmethod
    def symbolic(g, qkv):
        return g.op("fMHA_V2", qkv, plugin_version_s='1')

class GroupNormalizationPlugin(BasePlugin):
    # https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/groupNormalizationPlugin
    @staticmethod
    def forward(ctx, x, scale, bias, num_groups, eps):
        return F.group_norm(x, num_groups, weight=scale, bias=bias, eps=eps)
    @staticmethod
    def symbolic(g, x, scale, bias, num_groups, eps):
        return g.op("GroupNormalizationPlugin", x, scale, bias, plugin_version_s='1', eps_f=eps, num_groups_i=num_groups)

class LayerNormPlugin(BasePlugin):
    # https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/layerNormPlugin
    # Potential issue of this plugin: https://github.com/NVIDIA/TensorRT/issues/2707
    @staticmethod
    def forward(ctx, x, scale, bias, channels, eps):
        return F.layer_norm(x, [channels], scale, bias, eps)
    @staticmethod
    def symbolic(g, x, scale, bias, channels, eps):
        return g.op("LayerNorm", x, scale, bias, plugin_version_s='1', epsilon_f=eps, axis_i=-1)

class NewLayerNormPlugin(BasePlugin):
    # https://github.com/1049451037/tensorrt-layernorm-plugin
    @staticmethod
    def forward(ctx, x, scale, bias, channels, eps):
        return F.layer_norm(x, [channels], scale, bias, eps)
    @staticmethod
    def symbolic(g, x, scale, bias, channels, eps):
        return g.op("LayerNormalizationPlugin", x, scale, bias, plugin_version_s='1', eps_f=eps)

class MHCAWG(BaseApply):
    @staticmethod
    def forward(ctx, q, kv, bias):
        """
        q: (batch_size, seq_len, num_head, size_per_head)
        kv: (batch_size, seq_len, num_head, 2, size_per_head)
        output: like q
        """
        size_per_head = q.size(3)
        batch_size = q.size(0)
        seq_len = q.size(1)
        num_head = q.size(2)
        q = q.transpose(1, 2).contiguous()
        kv = kv.transpose(1, 2).contiguous()
        k = kv.select(-2, 0)
        v = kv.select(-2, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (size_per_head**-0.5) + bias
        scores = F.softmax(scores, -1)
        result = torch.matmul(scores, v).transpose(1, 2).contiguous().view(batch_size, seq_len, num_head, size_per_head)
        return result