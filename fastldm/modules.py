import torch
import torch.nn as nn
from einops import rearrange

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

from flash_attn.modules.mha import FlashSelfAttention
class FlashSelfAttnWG(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.flash = FlashSelfAttention(softmax_scale=self.scale)

    def forward(self, x, context=None, mask=None, bias=None):
        assert context is None and mask is None and bias is None
        h = self.heads
        context = x

        inputx_shape = x.size()
        if len(x.shape) == 3 and len(context.shape) == 3: # normal
            pass
        elif len(x.shape) == 4 and len(context.shape) == 3:
            # BLOCK x but shared context for cross attention
            x = rearrange(x, 'b k n d -> b (k n) d')
        elif len(x.shape) == 4 and len(context.shape) == 4:
            # BLOCK attention
            assert x.shape[1] == context.shape[1], 'num of BLOCK not the same'
            # b dim later will be "batch * nblk * nheads"
            # assert not exists(mask) and not exists(bias), 'not implemented yet'
        else:
            raise ValueError(f'x shape: {x.shape} , context shape: {context.shape}')

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, '... n (h d) -> (...) n h d', h=h), (q, k, v))

        out = self.flash(torch.stack([q, k, v], dim=2))
            
        out = rearrange(out, 'b n h d -> b n (h d)', h=h).view(inputx_shape)
        return self.to_out(out)

from flash_attn.flash_attn_triton import FlashAttnFunc
class FlashCrossAttnWG(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        assert context_dim is not None

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        assert context is not None

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q, k, v))

        sim = torch.zeros(q.shape[0], q.shape[2], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device)
        max_neg_value = -torch.finfo(sim.dtype).max / 2
        sim = sim.masked_fill_(~mask, max_neg_value)
        sim = sim.view(q.shape[0], q.shape[2], q.shape[1], k.shape[1])

        out = FlashAttnFunc.apply(q, k, v, sim, False, self.scale)

        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)

from .plugins import MHCAWG
class ldmCrossAttnWG(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        assert context_dim is not None

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        x: (batch_size, seq_len_q, query_dim)
        context: (batch_size, seq_len_kv, context_dim)
        out: (batch_size, seq_len, seq_len_q, query_dim)
        """
        h = self.heads

        q = self.to_q(x)
        assert context is not None
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b n (h d) -> b n h d', h=h)
        k = rearrange(k, 'b n (h d) -> b n h 1 d', h=h)
        v = rearrange(v, 'b n (h d) -> b n h 1 d', h=h)
        kv = torch.cat([k, v], dim=-2)

        sim = torch.zeros(q.shape[0], q.shape[2], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device)
        max_neg_value = -torch.finfo(sim.dtype).max / 2
        sim = sim.masked_fill_(~mask, max_neg_value)
        sim = sim.view(q.shape[0], q.shape[2], q.shape[1], k.shape[1])

        out = MHCAWG.apply(q, kv, sim)
        out = rearrange(out, 'b n h d -> b n (h d)', h=h)
        return self.to_out(out)

from torch.nn import MultiheadAttention
class TorchSelfAttn(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.mha = MultiheadAttention(hidden_size, num_heads)
    def forward(self, x):
        return self.mha(x, x, x)[0]

from .plugins import fMHCA
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

class LinearConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x.transpose(1, -1)).transpose(1, -1).contiguous()

from .plugins import GroupNormalizationPlugin
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.eps = eps
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    def forward(self, x):
        return GroupNormalizationPlugin.apply(x, self.weight, self.bias, self.num_groups, self.eps)

from .plugins import LayerNormPlugin
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.channels = channels
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        return LayerNormPlugin.apply(x, self.weight, self.bias, self.channels, self.eps)

from .plugins import NewLayerNormPlugin
class NewLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.channels = channels
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        return NewLayerNormPlugin.apply(x, self.weight, self.bias, self.channels, self.eps)
