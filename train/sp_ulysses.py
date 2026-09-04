#!/usr/bin/env python3
"""Ulysses sequence parallelism for hybrid Qwen models under FSDP2.

This module applies the GDN-Ulysses forward inside the model used by
`driver.py`. It provides:

  * build_process_groups(rank, world, sp_size)
        -> intra-group SP collectives and strided DP groups for FSDP2.

  * gather_seq_scatter_heads / gather_heads_scatter_seq
        -> the standard Ulysses all-to-all collectives (seq-sharded all-heads <->
        seq-gathered head-sharded), implemented as differentiable autograd
        Functions (torch's all-to-all is not autograd-aware).

  * patch_model_for_sp(model, sp_group, sp_rank, sp_size)
        -> monkey-patch EVERY decoder layer of the loaded model:
            - linear_attention layers (Qwen3_5GatedDeltaNet): replace .forward with
              the Ulysses-aware GDN forward (all-to-all heads -> full-seq chunked GDN
              with per-rank-sliced conv1d weight + head-sharded gating/norm -> all-to-all
              back). Faithful to the real HF forward (use_qk_l2norm_in_kernel=True, GQA
              repeat_interleave, gated RMSNorm with head-sharded z).
            - full_attention layers (Qwen3_5Attention): replace .forward with the
              Ulysses-aware softmax forward (rotary on local seq shard -> all-to-all
              heads -> full-seq causal SDPA -> all-to-all back -> per-token gate).
        Returns (n_gdn_patched, n_softmax_patched).

SEQUENCE-SHARDING CONTRACT (enforced by the caller, driver.py):
  The model is fed input_ids/attention_mask ALREADY sliced to this rank's seq shard
  [B, T/sp] (Step 5), AND position_ids = the GLOBAL positions of this rank's shard
  (so the internal rotary builds correct cos/sin per-rank). The patched layers then
  gather the full sequence per-head via all-to-all, so every token mixer sees the
  whole 128k sequence; only the per-rank activation footprint is 1/sp.

WHY a clean full-seq causal mask in the softmax patch (rather than the passed mask):
  TextModel builds `causal_mask` from the LOCAL T/sp shard; that mask is wrong for the
  all-to-all'd FULL-seq attention. For our right-padded training sequences (real tokens
  contiguous from the left, padding on the right, response_mask zeroes padded loss) a
  plain full-seq causal mask is correct: a real token at position p attends only to
  positions <= p, all of which are real. So we use F.scaled_dot_product_attention
  is_causal=True on the gathered full sequence.
"""
from __future__ import annotations

import datetime
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

# The GDN sequence-parallel forward needs fla's Triton causal-conv1d kernel. fla is an
# optional dependency, so import it once here and defer the hard failure to
# patch_model_for_sp(): a missing install then aborts at SP-setup time with a clear,
# actionable message instead of a cryptic ImportError raised deep inside the patched
# forward on every rank of a multinode run.
try:
    from fla.modules.convolution import causal_conv1d
except ImportError as e:  # pragma: no cover - environment-dependent
    causal_conv1d = None
    _FLA_IMPORT_ERROR = e
else:
    _FLA_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# process groups (SP intra-node NVLink, DP cross-node IB)
# ---------------------------------------------------------------------------
def build_process_groups(rank, world, sp_size, timeout_minutes=120):
    """SP/DP process-group partition for Ulysses + FSDP2.

    SP groups contain contiguous ranks. DP groups contain ranks with the same
    SP position across those groups.

    EVERY rank must call new_group() for EVERY group in the SAME order (collective
    creation), even groups it does not belong to, or NCCL deadlocks.

    Sub-groups do not inherit the main process-group timeout. The caller passes
    the same derived timeout used by the main group so rollout broadcasts and
    optimizer collectives share one explicit failure budget.
    """
    assert world % sp_size == 0, f"world={world} not divisible by sp_size={sp_size}"
    _to = datetime.timedelta(minutes=timeout_minutes)
    # SP group: the sp_size contiguous ranks that share one sequence-shard pair.
    sp_groups = [dist.new_group(list(range(i, i + sp_size)), timeout=_to)
                 for i in range(0, world, sp_size)]
    sp_group = sp_groups[rank // sp_size]
    sp_rank = rank % sp_size

    # DP group: same sp_rank position (stride sp_size) across all SP pairs.
    dp_groups = [dist.new_group(list(range(j, world, sp_size)), timeout=_to)
                 for j in range(sp_size)]
    dp_group = dp_groups[sp_rank]
    dp_rank = rank // sp_size
    dp_world = world // sp_size
    return sp_group, sp_rank, dp_group, dp_rank, dp_world


# ---------------------------------------------------------------------------
# Ulysses all-to-all collectives. The reshape/all_to_all logic is validated
# bit-exact (round-trip identity). BUT the raw torch.distributed.all_to_all_single
# is NOT autograd-aware (torch 2.11: it raises "element 0 ... does not require grad"
# on backward). Since the LoRA
# params (softmax q/k/v_proj) sit BEFORE the all-to-all in the patched forward, the
# gradient MUST flow back through it, so we wrap each collective in an autograd
# Function. The all-to-all is permutation-like (orthogonal): the backward of
# gather_seq_scatter_heads is exactly gather_heads_scatter_seq on the grad, and vice
# versa. We implement that with the _raw helpers (no autograd) inside backward.
# ---------------------------------------------------------------------------
def _raw_gather_seq_scatter_heads(x, group, sp_size):
    """[B,Sloc,H,Dh] -> [B,Sfull,Hloc,Dh] (no autograd; reshape logic from the probe)."""
    B, Sloc, H, Dh = x.shape
    assert H % sp_size == 0, f"H={H} not divisible by sp_size={sp_size}"
    Hloc = H // sp_size
    x = x.reshape(B, Sloc, sp_size, Hloc, Dh).permute(2, 0, 1, 3, 4).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    out = out.permute(1, 0, 2, 3, 4).reshape(B, sp_size * Sloc, Hloc, Dh).contiguous()
    return out


def _raw_gather_heads_scatter_seq(x, group, sp_size):
    """[B,Sfull,Hloc,Dh] -> [B,Sloc,H,Dh] (no autograd; reshape logic from the probe)."""
    B, Sfull, Hloc, Dh = x.shape
    assert Sfull % sp_size == 0, f"Sfull={Sfull} not divisible by sp_size={sp_size}"
    Sloc = Sfull // sp_size
    x = x.reshape(B, sp_size, Sloc, Hloc, Dh).permute(1, 0, 2, 3, 4).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    out = out.permute(1, 2, 0, 3, 4).reshape(B, Sloc, sp_size * Hloc, Dh).contiguous()
    return out


class _GatherSeqScatterHeads(torch.autograd.Function):
    """Differentiable gather_seq_scatter_heads. backward = gather_heads_scatter_seq(grad)."""
    @staticmethod
    def forward(ctx, x, group, sp_size):
        ctx.group = group
        ctx.sp_size = sp_size
        return _raw_gather_seq_scatter_heads(x, group, sp_size)

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [B,Sfull,Hloc,Dh] -> inverse all-to-all -> [B,Sloc,H,Dh]
        return _raw_gather_heads_scatter_seq(grad_out, ctx.group, ctx.sp_size), None, None


class _GatherHeadsScatterSeq(torch.autograd.Function):
    """Differentiable gather_heads_scatter_seq. backward = gather_seq_scatter_heads(grad)."""
    @staticmethod
    def forward(ctx, x, group, sp_size):
        ctx.group = group
        ctx.sp_size = sp_size
        return _raw_gather_heads_scatter_seq(x, group, sp_size)

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [B,Sloc,H,Dh] -> all-to-all -> [B,Sfull,Hloc,Dh]
        return _raw_gather_seq_scatter_heads(grad_out, ctx.group, ctx.sp_size), None, None


def gather_seq_scatter_heads(x, group, sp_size):
    """All-to-all (differentiable): seq-sharded+all-heads -> seq-gathered+head-sharded. [B,Sloc,H,Dh]->[B,Sfull,Hloc,Dh]."""
    return _GatherSeqScatterHeads.apply(x, group, sp_size)


def gather_heads_scatter_seq(x, group, sp_size):
    """Inverse all-to-all (differentiable): seq-gathered+head-sharded -> seq-sharded+all-heads. [B,Sfull,Hloc,Dh]->[B,Sloc,H,Dh]."""
    return _GatherHeadsScatterSeq.apply(x, group, sp_size)


def _get_local_conv1d_weight(conv_w_full, key_dim, value_dim, sp_rank, sp_size):
    """Slice depthwise conv1d weight to match head-sharded mixed_qkv channels (VeOmni logic).

    conv_w_full: [conv_dim, K]. channel order = [q(key_dim) | k(key_dim) | v(value_dim)].
    Each rank keeps its 1/sp slice of q-channels, k-channels, v-channels.
    """
    assert conv_w_full.shape[0] == key_dim * 2 + value_dim
    local_key_dim = key_dim // sp_size
    local_value_dim = value_dim // sp_size
    ko = sp_rank * local_key_dim
    vo = sp_rank * local_value_dim
    w_q = conv_w_full[ko: ko + local_key_dim]
    w_k = conv_w_full[key_dim + ko: key_dim + ko + local_key_dim]
    w_v = conv_w_full[2 * key_dim + vo: 2 * key_dim + vo + local_value_dim]
    return torch.cat([w_q, w_k, w_v], dim=0)


# ---------------------------------------------------------------------------
# Ulysses-aware GDN forward (Qwen3_5GatedDeltaNet). FAITHFUL to the real HF forward:
#   * causal_conv1d on the FULL seq with per-rank-sliced conv weight
#   * use_qk_l2norm_in_kernel=True (the kernel L2-norms q/k internally)
#   * GQA repeat_interleave of q/k to num_v heads BEFORE the kernel
#   * gated RMSNorm self.norm(core, z) with HEAD-SHARDED z
# self: the GDN layer. hidden_states: [B, T/sp, D] (seq-sharded, all-heads). Returns [B,T/sp,D].
# ---------------------------------------------------------------------------
def gdn_sp_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    sp_group, sp_rank, sp_size = self._sp_group, self._sp_rank, self._sp_size
    B, Sloc, _ = hidden_states.shape
    key_dim, value_dim = self.key_dim, self.value_dim
    num_k, num_v = self.num_k_heads, self.num_v_heads
    Dk, Dv = self.head_k_dim, self.head_v_dim

    # --- input projections on the LOCAL seq shard (seq-sharded, all heads) ---
    mixed = self.in_proj_qkv(hidden_states)                       # [B, Sloc, conv_dim]
    z = self.in_proj_z(hidden_states).reshape(B, Sloc, num_v, Dv)  # [B, Sloc, num_v, Dv]
    b = self.in_proj_b(hidden_states)                            # [B, Sloc, num_v]
    a = self.in_proj_a(hidden_states)                            # [B, Sloc, num_v]

    q, k, v = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
    q = q.reshape(B, Sloc, num_k, Dk)
    k = k.reshape(B, Sloc, num_k, Dk)
    v = v.reshape(B, Sloc, num_v, Dv)

    # --- all-to-all: gather full seq, scatter heads ---
    q = gather_seq_scatter_heads(q, sp_group, sp_size)           # [B, Sfull, num_k/sp, Dk]
    k = gather_seq_scatter_heads(k, sp_group, sp_size)
    v = gather_seq_scatter_heads(v, sp_group, sp_size)           # [B, Sfull, num_v/sp, Dv]
    z = gather_seq_scatter_heads(z, sp_group, sp_size)           # [B, Sfull, num_v/sp, Dv]
    # gate params: [B, Sloc, num_v] -> head-sharded full-seq [B, Sfull, num_v/sp]
    b = gather_seq_scatter_heads(b.reshape(B, Sloc, num_v, 1), sp_group, sp_size).reshape(
        B, sp_size * Sloc, num_v // sp_size)
    a = gather_seq_scatter_heads(a.reshape(B, Sloc, num_v, 1), sp_group, sp_size).reshape(
        B, sp_size * Sloc, num_v // sp_size)

    Sfull = q.shape[1]
    lk_heads, lv_heads = num_k // sp_size, num_v // sp_size
    lkd, lvd = lk_heads * Dk, lv_heads * Dv

    # --- causal conv1d on the FULL seq with per-rank-sliced weight (channel-last fla) ---
    w_local = _get_local_conv1d_weight(self.conv1d.weight.squeeze(1), key_dim, value_dim,
                                       sp_rank, sp_size)
    qkv = torch.cat([q.reshape(B, Sfull, lkd), k.reshape(B, Sfull, lkd),
                     v.reshape(B, Sfull, lvd)], dim=-1)
    qkv = causal_conv1d(x=qkv, weight=w_local, bias=None, activation=self.activation,
                        backend="triton")[0]
    q2, k2, v2 = torch.split(qkv, [lkd, lkd, lvd], dim=-1)
    q2 = q2.reshape(B, Sfull, lk_heads, Dk)
    k2 = k2.reshape(B, Sfull, lk_heads, Dk)
    v2 = v2.reshape(B, Sfull, lv_heads, Dv)

    # --- gating (head-sharded) + GQA repeat to value heads, then chunked GDN ---
    beta = b.sigmoid()                                            # [B, Sfull, num_v/sp]
    s, e = sp_rank * lv_heads, (sp_rank + 1) * lv_heads
    A_log = self.A_log[s:e]                                       # [num_v/sp]
    dt_bias = self.dt_bias[s:e]                                   # [num_v/sp]
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)    # [B, Sfull, num_v/sp]
    if num_v // num_k > 1:
        q2 = q2.repeat_interleave(num_v // num_k, dim=2)         # -> [B, Sfull, num_v/sp, Dk]
        k2 = k2.repeat_interleave(num_v // num_k, dim=2)
    core, _ = self.chunk_gated_delta_rule(
        q2, k2, v2, g=g, beta=beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, cu_seqlens=None,
    )                                                            # [B, Sfull, num_v/sp, Dv]

    # --- gated RMSNorm with head-sharded z (norm is per head_v_dim) ---
    core = core.reshape(-1, Dv)
    z = z.reshape(-1, Dv)
    core = self.norm(core, z)
    core = core.reshape(B, Sfull, lv_heads, Dv)

    # --- all-to-all back: scatter heads -> seq-sharded all-heads ---
    core = gather_heads_scatter_seq(core, sp_group, sp_size)      # [B, Sloc, num_v, Dv]
    core = core.reshape(B, Sloc, num_v * Dv)
    return self.out_proj(core)


# ---------------------------------------------------------------------------
# Ulysses-aware softmax forward (Qwen3_5Attention). rotary on local shard ->
# all-to-all heads -> full-seq causal SDPA -> all-to-all back -> per-token gate.
# self: the attention layer. hidden_states: [B, T/sp, D] (seq-sharded). Returns ([B,T/sp,D], None).
# ---------------------------------------------------------------------------
def softmax_sp_forward(self, hidden_states, position_embeddings, attention_mask=None,
                       past_key_values=None, **kwargs):
    sp_group, sp_rank, sp_size = self._sp_group, self._sp_rank, self._sp_size
    B, Sloc, _ = hidden_states.shape
    Hd = self.head_dim
    nH = self.config.num_attention_heads          # 24
    nKV = self.config.num_key_value_heads         # 4

    # --- projections + q/k norm + gate (all on the LOCAL seq shard) ---
    q_full, gate = torch.chunk(
        self.q_proj(hidden_states).view(B, Sloc, -1, Hd * 2), 2, dim=-1)  # [B,Sloc,nH,Hd]
    gate = gate.reshape(B, Sloc, -1)                                      # [B,Sloc,nH*Hd] seq-sharded
    q = self.q_norm(q_full).transpose(1, 2)                              # [B,nH,Sloc,Hd]
    k = self.k_norm(self.k_proj(hidden_states).view(B, Sloc, -1, Hd)).transpose(1, 2)  # [B,nKV,Sloc,Hd]
    val = self.v_proj(hidden_states).view(B, Sloc, -1, Hd).transpose(1, 2)             # [B,nKV,Sloc,Hd]

    # --- rotary on the LOCAL seq shard (cos/sin already carry this rank's global positions) ---
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)                          # [B,nH,Sloc,Hd] / [B,nKV,Sloc,Hd]

    # --- all-to-all: gather full seq, scatter heads. a2a expects [B,Sloc,H,Dh]. ---
    q = gather_seq_scatter_heads(q.transpose(1, 2), sp_group, sp_size)   # [B,Sfull,nH/sp,Hd]
    k = gather_seq_scatter_heads(k.transpose(1, 2), sp_group, sp_size)   # [B,Sfull,nKV/sp,Hd]
    val = gather_seq_scatter_heads(val.transpose(1, 2), sp_group, sp_size)  # [B,Sfull,nKV/sp,Hd]
    Sfull = q.shape[1]

    # --- full-seq causal SDPA on head-sharded q/k/v (GQA via enable_gqa) ---
    q = q.transpose(1, 2)                                                 # [B,nH/sp,Sfull,Hd]
    k = k.transpose(1, 2)                                                 # [B,nKV/sp,Sfull,Hd]
    val = val.transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q, k, val, attn_mask=None, dropout_p=0.0, is_causal=True,
        scale=self.scaling, enable_gqa=(nH != nKV))                      # [B,nH/sp,Sfull,Hd]

    # --- all-to-all back: scatter heads -> seq-sharded all-heads ---
    attn = gather_heads_scatter_seq(attn.transpose(1, 2), sp_group, sp_size)  # [B,Sloc,nH,Hd]
    attn = attn.reshape(B, Sloc, -1).contiguous()                        # [B,Sloc,nH*Hd]
    attn = attn * torch.sigmoid(gate)                                    # per-token gate (seq-sharded)
    return self.o_proj(attn), None


# ---------------------------------------------------------------------------
# monkey-patch every decoder layer of the (already loaded) model
# ---------------------------------------------------------------------------
def _iter_decoder_layers(model):
    m = getattr(model, "base_model", model)
    m = getattr(m, "model", m)
    m = getattr(m, "model", m)
    layers = getattr(m, "layers", None)
    if layers is None:
        raise RuntimeError("could not locate decoder layers for SP patching")
    return list(layers)


def patch_model_for_sp(model, sp_group, sp_rank, sp_size):
    """Replace the token-mixer forward of every decoder layer with its SP-aware version.

    Patches the inner module (.linear_attn / .self_attn) of each Qwen3_5DecoderLayer in
    place via types.MethodType so the bound `self` is the mixer module; injects
    _sp_group/_sp_rank/_sp_size as attributes on each patched module.
    """
    if causal_conv1d is None:
        raise ImportError(
            "Ulysses sequence parallelism (sp_size>1) needs the flash-linear-attention "
            "(fla) Triton causal-conv1d kernel for the GDN forward, but fla failed to "
            "import. Install it on every node (`pip install flash-linear-attention` plus a "
            "built `causal-conv1d`), or run with --sp-size 1. "
            f"Original import error: {_FLA_IMPORT_ERROR!r}"
        )
    n_gdn = n_softmax = 0
    for layer in _iter_decoder_layers(model):
        # Transformers renamed this discriminator from layer_type to block_type.
        # Accept both so SP patching remains compatible across Qwen3.5 revisions.
        lt = getattr(layer, "layer_type", None)
        if lt is None:
            lt = getattr(layer, "block_type", None)
        if lt == "linear_attention" and hasattr(layer, "linear_attn"):
            mod = layer.linear_attn
            mod._sp_group, mod._sp_rank, mod._sp_size = sp_group, sp_rank, sp_size
            mod.forward = types.MethodType(gdn_sp_forward, mod)
            n_gdn += 1
        elif lt == "full_attention" and hasattr(layer, "self_attn"):
            mod = layer.self_attn
            mod._sp_group, mod._sp_rank, mod._sp_size = sp_group, sp_rank, sp_size
            mod.forward = types.MethodType(softmax_sp_forward, mod)
            n_softmax += 1
    return n_gdn, n_softmax
