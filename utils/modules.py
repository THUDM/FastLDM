import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomQKVToContextPluginDynamic(torch.autograd.Function):
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
        scores = torch.matmul(q, k.transpose(-2, -1)) / (size_per_head**0.5)
        scores = F.softmax(scores, -1)
        result = torch.matmul(scores, v).transpose(0, 2).contiguous().view(seq_len, batch_size, hidden_size, 1, 1)
        return result
    @staticmethod
    def symbolic(g, qkv, hidden_size, num_heads, type_id):
        return g.op("CustomQKVToContextPluginDynamic", qkv, plugin_version_s='1', type_id_i=type_id, hidden_size_i=hidden_size, num_heads_i=num_heads, has_mask_i=False)

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
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.size_per_head = hidden_size // num_heads
        self.Wqkv = nn.Linear(hidden_size, 3*hidden_size)

    def forward(self, x):
        return self.Wqkv(x).view(x.size(0), x.size(1), 3*self.hidden_size, 1, 1)

class TRTSelfAttn(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.projection = qkvLinear(hidden_size, num_heads)
    def forward(self, x):
        # shape of x (seq_len, batch_size, hidden_size)
        # shape of i_mask (batch_size)
        # output (seq_len, batch_size, hidden_size)
        qkv = self.projection(x)
        type_id = 0 if qkv.dtype == torch.float32 else 1
        return CustomQKVToContextPluginDynamic.apply(qkv, self.hidden_size, self.num_heads, type_id).select(-1, 0).select(-1, 0)

from flash_attn.flash_attn_interface import _flash_attn_forward
class FlashSelfAttn(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.size_per_head = hidden_size // num_heads
        self.scale = 1 / self.size_per_head**0.5
        self.projection = qkvLinear(hidden_size, num_heads)
    def forward(self, x):
        # shape of x (seq_len, batch_size, hidden_size)
        # shape of i_mask (batch_size)
        # output (seq_len, batch_size, hidden_size)
        seq_len = x.size(0)
        batch_size = x.size(1)
        qkv = self.projection(x).view(seq_len, batch_size, self.num_heads, 3, self.size_per_head).transpose(0, 1).contiguous().view(seq_len*batch_size, self.num_heads, 3, self.size_per_head)
        q = qkv.select(-2, 0)
        k = qkv.select(-2, 1)
        v = qkv.select(-2, 2)
        cu_seqlen = torch.arange(start=0, end=batch_size*seq_len, step=seq_len, dtype=torch.int32, device=q.device)
        max_seqlen = seq_len
        out = torch.empty_like(v)
        return _flash_attn_forward(q, k, v, out, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, 0., self.scale, False, False)[0].view(seq_len, batch_size, self.hidden_size)
