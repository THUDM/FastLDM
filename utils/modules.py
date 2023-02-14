import torch
import torch.nn as nn

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

from .plugins import CustomQKVToContextPluginDynamic
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

from torch.nn import MultiheadAttention
class TorchSelfAttn(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.mha = MultiheadAttention(hidden_size, num_heads)
    def forward(self, x):
        return self.mha(x, x, x)[0]

from .plugins import fMHCA
from einops import rearrange
class ldmCrossAttn(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=40):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim))

    def forward(self, x, context=None):
        """
        x: (batch_size, seq_len_q, query_dim)
        context: (batch_size, seq_len_kv, context_dim)
        out: (batch_size, seq_len, seq_len_q, query_dim)
        """
        h = self.heads

        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b n (h d) -> b n h d', h=h)
        k = rearrange(k, 'b n (h d) -> b n h 1 d', h=h)
        v = rearrange(v, 'b n (h d) -> b n h 1 d', h=h)
        kv = torch.cat([k, v], dim=-2)

        out = fMHCA.apply(q, kv)
        out = rearrange(out, 'b n h d -> b n (h d)', h=h)
        return self.to_out(out)

from .plugins import fMHA_V2
from einops import rearrange
class ldmSelfAttn(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=40):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim))

    def forward(self, x, context=None):
        """
        x: (batch_size, seq_len_q, query_dim)
        context: (batch_size, seq_len_kv, context_dim)
        out: (batch_size, seq_len, seq_len_q, query_dim)
        """
        h = self.heads

        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b n (h d) -> b n h 1 d', h=h)
        k = rearrange(k, 'b n (h d) -> b n h 1 d', h=h)
        v = rearrange(v, 'b n (h d) -> b n h 1 d', h=h)
        qkv = torch.cat([q, k, v], dim=-2)

        out = fMHA_V2.apply(qkv)
        out = rearrange(out, 'b n h d -> b n (h d)', h=h)
        return self.to_out(out)