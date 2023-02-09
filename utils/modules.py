import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/bertQKVToContextPlugin
# The yaml file says that version 3 is not supported yet.

class CustomQKVToContextPluginDynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, hidden_size, num_heads):
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
        # / sqrt(d) is what d? size_per_head? or hidden_size?
        scores = torch.matmul(q, k.transpose(-2, -1)) / (size_per_head**0.5)
        scores = F.softmax(scores, -1)
        result = torch.matmul(scores, v).transpose(0, 2).contiguous().view(seq_len, batch_size, hidden_size, 1, 1)
        return result
    @staticmethod
    def symbolic(g, qkv, hidden_size, num_heads):
        return g.op("CustomQKVToContextPluginDynamic", qkv, plugin_version_s='1', type_id_i=0, hidden_size_i=hidden_size, num_heads_i=num_heads, has_mask_i=False)

class qkvLinearSlow(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.size_per_head = hidden_size // num_heads
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        qkv = torch.cat([Q, K, V], dim=2)
        qkv = qkv.view(x.size(0), x.size(1), 3, self.num_heads, self.size_per_head)
        qkv = qkv.transpose(2, 3).contiguous().view(x.size(0), x.size(1), 3*self.hidden_size, 1, 1)
        return qkv

class qkvLinear(nn.Module):
    def __init__(self, hidden_size, num_heads, Wq=None, Wk=None, Wv=None):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.size_per_head = hidden_size // num_heads
        self.Wqkv = nn.Linear(hidden_size, 3*hidden_size)
        if Wq is not None and Wk is not None and Wv is not None:
            wq = Wq.weight.data
            bq = Wq.bias.data
            wk = Wk.weight.data
            bk = Wk.bias.data
            wv = Wv.weight.data
            bv = Wv.bias.data
            wqkv = torch.cat([wq, wk, wv]).view(3, num_heads, self.size_per_head, hidden_size).transpose(0, 1).contiguous().view(3*hidden_size, hidden_size)
            bqkv = torch.cat([bq, bk, bv]).view(3, num_heads, self.size_per_head).transpose(0, 1).contiguous().view(3*hidden_size)
            self.Wqkv.weight.data = wqkv
            self.Wqkv.bias.data = bqkv

    def forward(self, x):
        return self.Wqkv(x).view(x.size(0), x.size(1), 3*self.hidden_size, 1, 1)