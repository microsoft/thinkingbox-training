#!/usr/bin/env python3
"""End-to-end verl-FSDP2 DAPO/GRPO trainer for hybrid Qwen models with LoRA.

The RL training loop:
  real multi-turn MCP + user-sim rollouts -> configured reward ->
  verl-native group-relative advantage + clipped loss -> FSDP2 backward + optimizer
  step -> gather the LoRA adapter + hot-swap it into the rollout vLLM -> repeat.

Three pieces:
  * FSDP2 build + device_id multinode fix + per-chunk torch.utils.checkpoint
    logprob (mandatory at 64k) + verl-native compute_grpo_outcome_advantage
    + POLICY_LOSS_REGISTRY['vanilla'].
  * train.rollout.RolloutRunner -> the real agentic rollout.
  * fsdp_lora_sync.fsdp_sync_vllm_lora -> FSDP2 LoRA gather and runtime
    adapter reload.

DATA-PARALLEL LAYOUT:
  Each rank selects the SAME global prompt pool (validated absolute-step schedule
  when supplied, otherwise step-seeded RNG), takes its stripe
  picks[rank::world], rolls out g samples/prompt against the SHARED vLLM endpoint
  or its configured node-local endpoint, builds its LOCAL batch of complete
  g-groups, computes the selected loss on its local trajectories, and FSDP2
  reduces gradients across the configured data-parallel ranks.
  Complete groups per rank => group_mean advantage is correct locally (no cross-rank
  advantage coupling needed). n_prompts must be divisible by world_size.

See `scripts/launch_training.sh` for the parameter-complete launcher.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.checkpoint as _ckpt
import verl
import yaml
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.trainer.ppo import core_algos as ca
from verl.workers.config.actor import ActorConfig

from thinkingbox.common.config_types import ConfigFile
from train.data_pipeline import hydrate, load_test_list, require_materialized_input
from train.eval_loop import run_eval
from train.lora_sync import (
    LORA_TARGET_PROFILES, DEFAULT_TARGET_PROFILE,
    resolve_lora_target_modules, VLLMLoraClient)
from train.rewards import apply_overlong_penalty, score_rollouts
from train.rollout import RolloutRunner
from train.tokenize_chat import render_with_assistant_mask
from train.token_replay import (
    TokenReplayError,
    build_token_training_segments,
    validate_segment_length,
)
from train.wandb_logger import WandbLogger
from train.algorithm import (
    build_prompt_group_pool,
    cap_training_agent_turns,
    effective_algorithm_settings,
    filter_complete_rollout_groups,
    has_sufficient_complete_groups,
    is_clean_rollout_batch,
    is_timeout_censored_rollout_batch,
    policy_loss_weight,
    resolve_algorithm_defaults,
    rollout_response_token_totals,
    step_optimizer_if_advantage,
    validate_training_args,
)
from train.checkpoint_consistency import (
    compute_adapter_payload_digest,
    optimizer_status_code,
    resolve_recovery_resume,
    validate_optimizer_resume_consensus,
)
from train.fsdp_lora_sync import (
    _is_vllm_writer,
    fsdp_sync_vllm_lora,
    load_vllm_adapter_for_training,
    validate_vllm_writer_topology,
)
from train.prompt_schedule import (
    load_prompt_schedule,
    select_global_prompt_uids,
    stripe_global_prompt_uids,
)
from train.sp_ulysses import build_process_groups, patch_model_for_sp


# ---------------------------------------------------------------------------
# DTensor DeviceMesh resilience patch (must run before any mesh is built)
# ---------------------------------------------------------------------------
def _patch_device_mesh_resilience():
    """Tolerate incomplete DeviceMesh state loaded with optimizer DTensors.

    Some Torch versions deserialize a DeviceMesh without all private layout
    attributes used by equality, hashing, representation, and redistribution
    cost helpers. This trainer constructs one DP mesh topology, so a complete
    live mesh and its incomplete deserialized counterpart represent the same
    topology.

    The patch supplies fallbacks only after an AttributeError. Normal
    DeviceMesh instances retain upstream behavior. Cost-estimation bindings
    imported by name are patched in each importing module because rebinding
    the defining module does not update those copies.
    """
    import torch.distributed.device_mesh as device_mesh_mod  # noqa: PLC0415
    import torch.distributed.tensor._collective_utils as cu_mod  # noqa: PLC0415

    # Torch <=2.6 uses public `mesh` / `device_type` attributes and has no
    # `_MeshLayout`; it does not have the pickle reconstruction failure this
    # patch targets. Replacing its public attributes with 2.11-style
    # properties would break DeviceMesh.__init__, so leave that implementation
    # completely untouched.
    if not hasattr(device_mesh_mod, "_MeshLayout"):
        return

    def _safe_device_type(self):
        try:
            return object.__getattribute__(self, "_device_type")
        except AttributeError:
            return "cuda"

    def _safe_ndim(self):
        try:
            return len(object.__getattribute__(self, "_layout"))
        except AttributeError:
            return 1

    _GHOST_SENTINEL_ATTRS = (
        "_layout", "_device_type", "_mesh_dim_names", "_flatten_rank_map", "_thread_id")

    def _is_ghost_mesh(self):
        # A DeviceMesh that went through normal __init__ ALWAYS has ALL five
        # of these set directly in its own __dict__. Checking __dict__
        # membership (not getattr/hasattr, which would trigger -- and be
        # satisfied by -- __getattr__'s synthesis below) is the only way to
        # distinguish a real mesh from a "ghost" BEFORE that synthesis
        # happens. Checking ANY (not just _layout) matters: __getattr__
        # caches _layout on first access (e.g. from an earlier ndim/size()
        # call in the same op), so by the time __eq__ runs, _layout alone
        # may already look present on a mesh that is still a ghost in every
        # other respect.
        d = self.__dict__
        return any(attr not in d for attr in _GHOST_SENTINEL_ATTRS)

    _orig_repr = device_mesh_mod.DeviceMesh.__repr__
    _orig_eq = device_mesh_mod.DeviceMesh.__eq__
    _orig_hash = device_mesh_mod.DeviceMesh.__hash__

    def _safe_repr(self):
        try:
            return _orig_repr(self)
        except AttributeError:
            return (f"DeviceMesh(device_type={_safe_device_type(self)!r}, "
                    f"repr_incomplete=True)")

    def _safe_eq(self, other):
        if self is other:
            return True
        if not isinstance(other, device_mesh_mod.DeviceMesh):
            return False
        if _is_ghost_mesh(self) or _is_ghost_mesh(other):
            # 6th live crash (a plain "return False" here was the ORIGINAL
            # choice): common_pointwise_strategy legitimately raises
            # ValueError("Could not run pointwise computation across
            # different mesh") once __eq__ stopped crashing and started
            # reporting "not equal" for this pair -- a healthy
            # DeviceMesh((3,), 'cuda', ...) vs the same broken/attribute-
            # missing "ghost" mesh seen throughout this investigation.
            # 8th live crash: patching __getattr__ to synthesize a
            # degenerate `_layout` (see below) so ndim/size()/shape stop
            # raising ALSO made `_orig_eq`'s own `self._layout ==
            # other._layout` comparison succeed normally -- but comparing a
            # SYNTHESIZED (1,)-shaped placeholder against the real (3,)
            # mesh is a legitimately-computed (not exception-driven) False,
            # so the AttributeError-only fallback below never even fires.
            # Ghost status must therefore be checked PROACTIVELY (via
            # __dict__, before any attribute access can trigger that
            # synthesis), not reactively via try/except.
            # torch.save()/torch.load() of DTensor optimizer state
            # (exp_avg/exp_avg_sq in the resumed AdamW checkpoint) pickles
            # each tensor's embedded DeviceMesh, whose actual identity is a
            # live ProcessGroup/NCCL-communicator handle that CANNOT be
            # faithfully reconstructed by unpickling alone in a new process
            # -- the private layout state that would normally resolve it is
            # exactly what comes back missing (the "ghost"). This trainer
            # builds exactly ONE mesh topology for its whole lifetime
            # (DeviceMesh.from_group(dp_group, "cuda"), driver.py),
            # so a ghost mesh reaching this comparison necessarily
            # represents that SAME conceptual mesh, differing only by
            # Python object identity after this deserialization gap --
            # "equal" is the CORRECT answer, not merely a safe fallback.
            # device_type is checked as a cheap sanity gate (agrees even in
            # the fully-degenerate case, since _safe_device_type's own
            # fallback is "cuda" on both sides) rather than True
            # unconditionally.
            return _safe_device_type(self) == _safe_device_type(other)
        try:
            return _orig_eq(self, other)
        except AttributeError:
            return _safe_device_type(self) == _safe_device_type(other)

    def _safe_hash(self):
        # _safe_eq deliberately treats a deserialized ghost as equal to its
        # live mesh when device types agree. Python therefore requires both
        # objects to share a hash. Hash every patched DeviceMesh on that same
        # invariant; different healthy topologies may collide, but __eq__
        # still distinguishes them and dict correctness is preserved.
        return hash(_safe_device_type(self))

    _orig_getattr = getattr(device_mesh_mod.DeviceMesh, "__getattr__", None)


    def _safe_getattr(self, name):
        # 7th live crash: DeviceMesh.size() (device_mesh.py:1126, ->
        # self._layout[mesh_dim]) hit the SAME _layout-missing ghost via YET
        # ANOTHER call site (_dtensor_spec.py::num_shards, a strategy-
        # counting heuristic, not the eq/hash/ndim/redistribute_cost paths
        # patched above). DeviceMesh has many public methods that read one
        # of these private attributes directly (get_group, __getitem__,
        # shape, size, ...); patching them one at a time has now taken
        # SEVEN rounds. __getattr__ is only invoked by Python when normal
        # attribute lookup (instance __dict__ + class descriptors) already
        # failed -- i.e. precisely this missing-private-state case -- so
        # supplying safe values here covers every current AND future
        # accessor of these attributes in one place, instead of continuing
        # to chase individual methods. Coverage includes not only the five
        # attributes __eq__/__hash__ read (_layout, _device_type,
        # _mesh_dim_names, _flatten_rank_map, _thread_id) but also
        # _rank/_dim_group_names/_flatten_mapping/_pg_registry, found by
        # auditing every `self._xxx` in device_mesh.py -- _pg_registry in
        # particular is CONFIRMED involved in the real mechanism: DeviceMesh
        # .__getstate__ explicitly excludes it from pickling ("ProcessGroup
        # objects can't be pickled") and __setstate__ tries to reconstruct
        # it from _dim_group_names, which is exactly the lossy
        # torch.save/load round-trip this whole patch exists for. The
        # values mirror what a degenerate 1-device mesh would hold;
        # correctness for this trainer only ever depends on _safe_eq
        # treating the ghost as equal to the live mesh, never on these
        # being topologically accurate for the ghost itself (see
        # _safe_eq's docstring comment).
        if name == "_device_type":
            return "cuda"
        if name == "_mesh_dim_names":
            return None
        if name == "_thread_id":
            return 0
        if name == "_flatten_rank_map":
            return None
        if name == "_root_mesh":
            return None
        if name == "_rank":
            return 0
        if name == "_dim_group_names":
            return []
        if name == "_flatten_mapping":
            return {}
        if name == "_pg_registry":
            # DeviceMesh.__getstate__ EXPLICITLY excludes this from pickling
            # ("ProcessGroup objects can't be pickled") and __setstate__
            # tries to reconstruct it from _dim_group_names -- the exact
            # mechanism confirming the "ghost mesh" theory throughout this
            # patch's docstring. An empty registry is the correct
            # degenerate answer when that reconstruction has nothing to
            # work from.
            return {}
        if name == "_rank_map":
            rank_map = torch.zeros(1, dtype=torch.int64)
            object.__setattr__(self, "_rank_map", rank_map)
            return rank_map
        if name == "_layout":
            degenerate = torch.zeros(1, dtype=torch.int64)
            layout = device_mesh_mod._MeshLayout(degenerate.size(), degenerate.stride())
            object.__setattr__(self, "_layout", layout)
            return layout
        if _orig_getattr is not None:
            return _orig_getattr(self, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    device_mesh_mod.DeviceMesh.device_type = property(_safe_device_type)
    device_mesh_mod.DeviceMesh.ndim = property(_safe_ndim)
    device_mesh_mod.DeviceMesh.__repr__ = _safe_repr
    device_mesh_mod.DeviceMesh.__eq__ = _safe_eq
    device_mesh_mod.DeviceMesh.__hash__ = _safe_hash
    device_mesh_mod.DeviceMesh.__getattr__ = _safe_getattr

    _orig_redistribute_cost = cu_mod.redistribute_cost
    _orig_one_step_redistribute_cost = getattr(
        cu_mod, "one_step_redistribute_cost", None)

    def _safe_redistribute_cost(current_spec, target_spec):
        try:
            return _orig_redistribute_cost(current_spec, target_spec)
        except AttributeError:
            return 0.0

    if _orig_one_step_redistribute_cost is not None:
        def _safe_one_step_redistribute_cost(current_spec, target_spec):
            try:
                return _orig_one_step_redistribute_cost(
                    current_spec, target_spec)
            except AttributeError:
                return 0.0
    else:
        _safe_one_step_redistribute_cost = None

    # Patch the definitions in their home module (covers MeshTopoInfo.build_from_mesh
    # callers inside _collective_utils.py itself, and any dotted-attribute access
    # elsewhere), plus every module that imported the original by name.
    cu_mod.redistribute_cost = _safe_redistribute_cost
    if _safe_one_step_redistribute_cost is not None:
        cu_mod.one_step_redistribute_cost = _safe_one_step_redistribute_cost
    try:
        import torch.distributed.tensor._utils as dt_utils_mod  # noqa: PLC0415
        dt_utils_mod.redistribute_cost = _safe_redistribute_cost
    except ImportError:
        pass
    try:
        import torch.distributed.tensor._ops.utils as dt_ops_utils_mod  # noqa: PLC0415
        dt_ops_utils_mod.redistribute_cost = _safe_redistribute_cost
    except ImportError:
        pass
    try:
        import torch.distributed.tensor._redistribute as dt_redistribute_mod  # noqa: PLC0415
        if (_safe_one_step_redistribute_cost is not None
                and hasattr(dt_redistribute_mod,
                            "one_step_redistribute_cost")):
            dt_redistribute_mod.one_step_redistribute_cost = (
                _safe_one_step_redistribute_cost)
    except ImportError:
        pass


_patch_device_mesh_resilience()


# ---------------------------------------------------------------------------
# distributed setup (device_id fix is CRITICAL for heterogeneous 6+8 per-node)
# ---------------------------------------------------------------------------
# The NCCL watchdog aborts the whole process when a collective outlives its
# timeout. Under Ulysses SP the sp_rank!=0 partner sits inside
# broadcast_object_list for its roller's ENTIRE rollout, so the process-group
# timeout must exceed the worst-case ROLLOUT budget, not the update time.
PG_TIMEOUT_SAFETY = 1.5          # multiple of the worst-case rollout budget
PG_TIMEOUT_HEADROOM_S = 600.0    # absolute slack for tokenize/segment/GC/IO
PG_TIMEOUT_FLOOR_MIN = 120       # never go below the historical default


def resolve_pg_timeout_minutes(rollout_timeout_s, dynamic_sampling_max_rounds,
                               env=None):
    """Derive the NCCL process-group timeout from the real rollout budget.

    A rank that is not rolling still blocks in a collective for as long as its
    peer rolls. Dynamic sampling runs `max_rounds + 1` rounds, each bounded by
    `--rollout-timeout`. Deriving one timeout from that complete budget avoids
    an independently configured watchdog expiring before a valid rollout.

    NCCL_PG_TIMEOUT_MIN overrides the result (still floored) for operators who
    need a different detection latency.
    """
    env = os.environ if env is None else env
    rounds = max(1, int(dynamic_sampling_max_rounds) + 1)
    budget_s = max(0.0, float(rollout_timeout_s)) * rounds
    required_min = math.ceil(
        (budget_s * PG_TIMEOUT_SAFETY + PG_TIMEOUT_HEADROOM_S) / 60.0)

    override = env.get("NCCL_PG_TIMEOUT_MIN")
    if override:
        chosen = int(override)
        if chosen < required_min:
            logging.getLogger("train.driver").warning(
                "NCCL_PG_TIMEOUT_MIN=%d is below the %d min needed to cover a "
                "%.0fs rollout budget (%d rounds x %.0fs); a slow step can abort "
                "every rank", chosen, required_min, budget_s, rounds,
                float(rollout_timeout_s))
    else:
        chosen = required_min
    return max(PG_TIMEOUT_FLOOR_MIN, chosen)


def setup_dist(pg_timeout_minutes=PG_TIMEOUT_FLOOR_MIN):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        # Without device_id, torch guesses rank->GPU from the GLOBAL rank assuming a
        # uniform per-node count; with 6+8 that guess is wrong on node-1 and FSDP2's
        # first foreach_all_gather DEADLOCKS. Pin the local GPU.
        dev = torch.device(f"cuda:{local}") if torch.cuda.is_available() else None
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=pg_timeout_minutes),
            device_id=dev)
    return rank, world, local


def is_rank0(rank):
    return rank == 0


def log(rank, world, local, msg):
    print(f"[rank {rank}/{world} gpu{local}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# FSDP2 model build
# ---------------------------------------------------------------------------
def _iter_blocks(model):
    m = getattr(model, "base_model", model)
    m = getattr(m, "model", m)
    m = getattr(m, "model", m)
    layers = getattr(m, "layers", None)
    return list(layers) if layers is not None else []


def _lm_head_of(model):
    m = getattr(model, "base_model", model)
    m = getattr(m, "model", m)
    lm = getattr(m, "lm_head", None)
    if lm is None:
        lm = getattr(getattr(m, "model", m), "lm_head", None)
    if lm is None:
        raise RuntimeError("could not locate lm_head")
    return lm


def load_resume_adapter_distributed(model, resume_lora_dir, rank, world, local):
    """Load the resume adapter on every rank and agree on the outcome collectively.

    Mirrors ``load_optimizer_checkpoint_distributed``. Checkpoint directories are
    node-local, and the load validates rank/alpha/targets/base-model plus tensor
    integrity, so it can fail on ONE node (a torn safetensors, an incomplete
    node-local copy, a mismatched adapter) while its peers succeed. Raising
    straight out of the per-rank load let the healthy ranks walk into the next
    NCCL collective and block until the job hit a multi-hour timeout, hiding the
    real cause. Hold the local exception until after a global MIN vote so every
    rank aborts together with the originating error attached.
    """
    n_loaded = 0
    local_digest = None
    local_error = None
    try:
        local_digest = compute_adapter_payload_digest(resume_lora_dir)
        n_loaded = load_vllm_adapter_for_training(model, resume_lora_dir)
    except Exception as exc:  # noqa: BLE001 — re-raised after the collective vote
        local_error = exc

    device = (torch.device(f"cuda:{local}") if torch.cuda.is_available()
              else torch.device("cpu"))
    ok = torch.tensor(0 if local_error is not None else 1,
                      dtype=torch.uint8, device=device)
    if world > 1 and dist.is_initialized():
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not bool(ok.item()):
        if local_error is not None:
            detail = f"{type(local_error).__name__}: {local_error}"
        else:
            detail = "another rank failed to load the resume adapter"
        raise RuntimeError(
            f"adapter resume failed collectively: {detail}") from local_error
    assert local_digest is not None
    if world > 1 and dist.is_initialized():
        digest_values = torch.tensor(
            list(bytes.fromhex(local_digest)),
            dtype=torch.long,
            device=device,
        )
        digest_min = digest_values.clone()
        digest_max = digest_values.clone()
        dist.all_reduce(digest_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(digest_max, op=dist.ReduceOp.MAX)
        if not torch.equal(digest_min, digest_max):
            raise RuntimeError(
                "adapter resume failed collectively: node-local adapter "
                "payload digest mismatch across ranks")
    return n_loaded


def build_fsdp2_lora(model_path, lora_rank, lora_alpha, lr, rank, world, local,
                     dp_group=None, target_modules=None, resume_lora_dir=None,
                     init_lora_dir=None):
    t0 = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    targets = list(resolve_lora_target_modules(DEFAULT_TARGET_PROFILE)
                   if target_modules is None else target_modules)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("LoRA target modules must be non-empty and unique")
    lconf = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=targets,
        task_type="CAUSAL_LM")
    # peft's get_peft_model defaults to autocast_adapter_dtype=True, which upcasts the
    # LoRA A/B weights to fp32 for stability. Under FSDP2 fully_shard (applied per decoder
    # layer below), each shard group then holds BOTH the frozen bf16 base projections AND
    # the fp32 LoRA params -> FSDP2._init_mp_dtypes asserts "uniform original parameter
    # dtype but got {bfloat16, float32}". The base is bf16 (27B fp32 = 108GB won't fit),
    # so the adapters must also be bf16. Disable the upcast where supported, then hard-cast
    # any remaining fp32 trainable (LoRA) params to bf16 so every group is uniformly bf16.
    try:
        model = get_peft_model(base, lconf, autocast_adapter_dtype=False)
    except TypeError:  # older peft without the kwarg
        model = get_peft_model(base, lconf)
    if resume_lora_dir and init_lora_dir:
        raise ValueError(
            "resume_lora_dir and init_lora_dir are mutually exclusive")
    adapter_source = resume_lora_dir or init_lora_dir
    if adapter_source:
        n_loaded = load_resume_adapter_distributed(
            model, adapter_source, rank, world, local)
        load_kind = ("resume" if resume_lora_dir
                     else "fresh optimizer warm start")
        log(rank, world, local,
            f"adapter {load_kind}: restored {n_loaded} LoRA tensors from "
            f"{adapter_source}")
    _ncast = 0
    for _n, _p in model.named_parameters():
        if _p.requires_grad and _p.dtype != torch.bfloat16:
            _p.data = _p.data.to(torch.bfloat16)
            _ncast += 1
    if _ncast:
        log(rank, world, local,
            f"cast {_ncast} fp32 LoRA param tensors -> bf16 for FSDP2 uniform-dtype")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(rank, world, local,
        f"LoRA targets={targets} -> {n_train:,} trainable params (r={lora_rank})")
    # --- GATE 1: enumerate the modules PEFT ACTUALLY wrapped and CONFIRM the
    # Gated-DeltaNet (linear-attn) projections are among them with nonzero trainable
    # params. The whole point of gdn_hybrid is to adapt in_proj_qkv/z/a/b + out_proj (the
    # GDN token mixer) in ADDITION to attn+MLP. If the profile is gdn_hybrid but zero GDN
    # modules got wrapped (e.g. a name mismatch silently fell back to attn-only), ABORT
    # rather than waste the whole run on an attention-only adapter.
    _GDN_LEAVES = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                   "in_proj_qkvz", "in_proj_ba", "out_proj"}
    _wrapped_leaves: dict[str, int] = {}
    _gdn_params = 0
    for _mn, _m in model.named_modules():
        _la = getattr(_m, "lora_A", None)
        if isinstance(_la, torch.nn.ModuleDict) and len(_la) > 0:
            _leaf = _mn.split(".")[-1]
            _wrapped_leaves[_leaf] = _wrapped_leaves.get(_leaf, 0) + 1
            if _leaf in _GDN_LEAVES or "linear_attn" in _mn:
                _gdn_params += sum(p.numel() for p in _m.parameters() if p.requires_grad)
    _gdn_leaves_hit = sorted(l for l in _wrapped_leaves if l in _GDN_LEAVES)
    _gdn_active = bool(_gdn_leaves_hit) and _gdn_params > 0
    log(rank, world, local,
        f"[GATE1] wrapped LoRA leaves={dict(sorted(_wrapped_leaves.items()))} "
        f"| GDN leaves wrapped={_gdn_leaves_hit} | GDN trainable params={_gdn_params:,} "
        f"| GDN_LORA_ACTIVE={_gdn_active}")
    _profile_is_gdn = any(l in _GDN_LEAVES for l in targets)
    if _profile_is_gdn and not _gdn_active:
        raise RuntimeError(
            "[GATE1] gdn_hybrid requested but ZERO Gated-DeltaNet projections were wrapped "
            f"(wrapped leaves={sorted(_wrapped_leaves)}). Refusing to train an attn-only "
            "adapter under gdn_hybrid. Check the model's linear_attn module names.")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # Under Ulysses SP the params must be sharded/reduce-scattered ONLY across the DP
    # group (the cross-node IB ranks that hold replicated weights), NOT the full world
    # (which now mixes DP and SP). torch 2.11 fully_shard takes a `mesh` (DeviceMesh),
    # not a raw process_group: build a 1-D mesh FROM dp_group. The device_id heterogeneous
    # fix lives in setup_dist (init_process_group device_id + set_device) and is preserved;
    # here we only redirect the FSDP sharding mesh. dp_group=None (sp_size=1) -> mesh=None
    # -> FSDP shards over the default world group == existing behaviour, unchanged.
    mesh = None
    if dp_group is not None:
        mesh = DeviceMesh.from_group(dp_group, "cuda")
    nblk = 0
    for layer in _iter_blocks(model):
        fully_shard(layer, mp_policy=mp, mesh=mesh)
        nblk += 1
    fully_shard(model, mp_policy=mp, mesh=mesh)
    log(rank, world, local,
        f"FSDP2 fully_shard on {nblk} decoder layers + root "
        f"(mesh={'dp_group' if mesh is not None else 'world'} build {time.monotonic()-t0:.1f}s)")
    # weight_decay=0.0 explicitly (torch AdamW default is 0.01): decoupled decay would
    # shrink the LoRA weights on EVERY opt.step() incl. all-dummy zero-grad ranks; LoRA
    # RL wants no decay. [fix #7]
    # foreach=False keeps DTensor parameters out of the multi-tensor path,
    # where sharding propagation can encounter incompletely deserialized
    # DeviceMesh state. Adam's math is unchanged; only the batched-kernel
    # optimization is disabled. The DeviceMesh compatibility patch remains
    # defense in depth for other callers.
    # (e.g. third-party code), but this trainer's own two call sites (here and
    # clip_grad_norm_) no longer do.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0, foreach=False)
    return model, opt


def chunked_target_logprob(hidden, lm_head, input_ids, chunk):
    """Vocab-free target-token logprob, chunked + checkpointed (mandatory at 64k)."""
    tgt = input_ids[:, 1:]
    pred = hidden[:, :-1, :]
    outs = []
    P = pred.shape[1]

    def _chunk_logp(pred_chunk, tgt_chunk):
        logits = lm_head(pred_chunk).float()
        lsm = torch.log_softmax(logits, dim=-1)
        return lsm.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)

    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        g = _ckpt.checkpoint(_chunk_logp, pred[:, s:e, :], tgt[:, s:e],
                             use_reentrant=False)
        outs.append(g)
    return torch.cat(outs, dim=1)


def _transformer_body_of(model):
    """Return the text transformer body without bypassing the outer forward."""
    if hasattr(model, "get_base_model"):
        causal_lm = model.get_base_model()
    else:
        wrapped = getattr(model, "base_model", model)
        causal_lm = getattr(wrapped, "model", wrapped)
    body = getattr(causal_lm, "model", None)
    if body is None:
        raise RuntimeError("could not locate transformer body")
    return body


def _body_hidden(
    model,
    input_ids,
    attn,
    position_ids=None,
    activation_offload=False,
):
    # position_ids: under Ulysses SP, input_ids is THIS rank's seq shard [B, T/sp] and
    # position_ids carries the GLOBAL positions of that shard so the model's internal
    # rotary builds correct cos/sin per-rank. None (sp_size=1) -> model builds
    # arange(T) itself (unchanged behaviour).
    # output_hidden_states=True retains every layer output. Capture only the
    # transformer-body result while preserving the outer PEFT/FSDP forward.
    captured = []
    body = _transformer_body_of(model)

    def _capture(_module, _inputs, output):
        hidden = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output[0]
        )
        captured.append(hidden)

    handle = body.register_forward_hook(_capture)
    try:
        offload = (
            torch.autograd.graph.save_on_cpu(
                pin_memory=True,
                device_type="cuda",
            )
            if activation_offload
            else contextlib.nullcontext()
        )
        with offload:
            model(
                input_ids=input_ids,
                attention_mask=attn,
                position_ids=position_ids,
                output_hidden_states=False,
                logits_to_keep=1,
                use_cache=False,
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"transformer body hook captured {len(captured)} outputs; "
            "expected exactly one"
        )
    return captured[0]


class _AllGatherSeq(torch.autograd.Function):
    """DIFFERENTIABLE all-gather of a seq-sharded hidden across the SP group.

    forward:  each rank's [B, T/sp, D] shard -> full [B, T, D] (concat in rank order).
    backward: every SP rank computes the SAME full-sequence loss from the gathered hidden,
              so dL/d(gathered) is IDENTICAL on all ranks. This rank's input (its shard)
              only needs ITS OWN slice of that gradient -> backward simply slices out
              [:, sp_rank*chunk : (sp_rank+1)*chunk, :]. (NOT a reduce_scatter-sum, which
              would scale the gradient by sp_size since the per-rank losses are identical.)

    This is the fix for the `element 0 ... does not require grad` crash: the plain
    dist.all_gather list API is NOT autograd-aware, so it detached hidden from the graph
    and loss.backward() had no grad_fn. This Function restores the gradient path."""
    @staticmethod
    def forward(ctx, hidden_local, sp_group, sp_rank, sp_size):
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size
        ctx.Sloc = hidden_local.shape[1]
        parts = [torch.empty_like(hidden_local) for _ in range(sp_size)]
        dist.all_gather(parts, hidden_local.contiguous(), group=sp_group)
        return torch.cat(parts, dim=1)  # [B, sp_size*Sloc = T, D]

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [B, T, D] (same on every SP rank). Take THIS rank's seq slice.
        s = ctx.sp_rank * ctx.Sloc
        grad_local = grad_out[:, s:s + ctx.Sloc, :].contiguous()
        return grad_local, None, None, None


def _all_gather_seq(hidden_local, sp_group, sp_rank, sp_size):
    """Differentiable all-gather of a seq-sharded hidden [B, T/sp, D] -> full [B, T, D].

    The body forward returns hidden seq-sharded (each rank owns its T/sp slice, contiguous,
    rank r = slice r). Gathering along seq reconstructs the full sequence so the lm_head
    logprob is computed over the WHOLE sequence (correct at shard boundaries; the next-token
    target for rank r's last token lives in rank r+1's shard). The loss then runs on the full
    sequence == identical to the no-SP path. lm_head logprob is chunk-checkpointed so its peak
    stays bounded by the chunk, not the full T. Gradients flow back to each rank's shard via
    _AllGatherSeq.backward (slice), keeping the graph intact for loss.backward()."""
    return _AllGatherSeq.apply(hidden_local, sp_group, sp_rank, sp_size)


# ---------------------------------------------------------------------------
# real rollout â€” per-rank stripe of prompts via RolloutRunner (mirrors run_one_step)
# ---------------------------------------------------------------------------
def build_runner(tb_cfg, lora_name, concurrency, mcp_urls=None):
    """A RolloutRunner whose agent model points at the current LoRA adapter name.

    mcp_urls: optional list of MCP endpoint URLs to round-robin (load balance)
    across. None -> single endpoint from config (today's behaviour)."""
    cfg = tb_cfg.model_copy(deep=True)
    if lora_name:
        cfg.orchestrator.agent_model.deployment = lora_name
    return RolloutRunner(config=cfg, concurrency=concurrency, dump_raw=False,
                         mcp_urls=mcp_urls)


def _trajectory_to_tokens(messages, tools, tokenizer, max_seq_len):
    tok = render_with_assistant_mask(messages, tokenizer, tools=tools)
    ids, mask = tok.input_ids, tok.assistant_mask
    if len(ids) > max_seq_len:
        ids, mask = ids[:max_seq_len], mask[:max_seq_len]
    return ids, mask


def _trajectory_to_segments(rollout, reward, max_seq_len):
    """Apply token replay to build independent causal training segments.

    P1 fix: the PPO ratio needs the vLLM behavior logprob as its denominator,
    which is only defined against the EXACT sampled tokens. Each serving generation
    becomes one ``(ids, assistant_mask, behavior_logp)`` tuple where
    ``behavior_logp[i]`` is the sampled-token logprob (0.0 on context/pad positions,
    which are masked out of the loss). Segments whose sampled actions don't fit in
    ``max_seq_len`` are dropped (never truncated). Returns [] when the rollout has no
    captured token trace; such rollouts cannot be used for training."""
    if rollout.capture_integrity_error is not None or rollout.behavior_generations is None:
        return []
    try:
        segments = build_token_training_segments(
            rollout.behavior_generations
        )
    except TokenReplayError:
        return []
    out = []
    for segment in segments:
        disp = validate_segment_length(
            segment, reward=float(reward), max_seq_len=max_seq_len)
        if not disp.is_valid:
            continue
        exact = disp.segment
        behavior_logp = [0.0 if v is None else v for v in exact.behavior_logprobs]
        out.append((exact.input_ids, exact.assistant_mask, behavior_logp))
    return out


def rollout_local_stripe(runner, by_name, rng, n_prompts, world, rank, g,
                         tokenizer, max_seq_len, reward_fn, overlong_penalty,
                         timeout_s, global_picks=None):
    """Sample the global prompt pool (RNG identical across ranks), roll this rank's
    stripe, score, return [{ids, mask, reward, uid}] for COMPLETE g-groups.

    NOTE: `world`/`rank` here are the DATA-PARALLEL stripe coordinates (dp_world/dp_rank),
    NOT the raw world/rank. Under Ulysses SP the sp_size ranks of one SP pair share ONE DP
    stripe (same prompts, same trajectories) and only sp_rank 0 actually rolls; the caller
    broadcasts the rolled `samples` to the SP partner. With sp_size=1, dp == world (unchanged)."""
    pool = sorted(by_name.keys())
    picks = (
        list(global_picks)
        if global_picks is not None
        else select_global_prompt_uids(pool, rng, n_prompts)
    )
    local_picks = stripe_global_prompt_uids(
        picks, dp_rank=rank, dp_world=world)
    batch_tcs = [by_name[u] for u in local_picks]
    if not batch_tcs:
        return [], 0, []
    by_uid, _ = asyncio.run(
        _bounded_rollout(
            runner, batch_tcs, g, timeout_s,
            group_ids=local_picks))

    samples = []
    n_sys_err = 0
    for uid, rolls in by_uid.items():
        rewards = score_rollouts(rolls, name=reward_fn)
        rewards = apply_overlong_penalty(rolls, rewards, penalty=overlong_penalty)
        for r, rew in zip(rolls, rewards):
            if r.is_system_error or not r.messages:
                n_sys_err += 1
                continue
            if r.capture_integrity_error is not None:
                # A rollout without an exact token trace is unusable for
                # policy-gradient training. Book it explicitly as a system
                # error instead of letting _trajectory_to_segments return []
                # and making the rollout disappear from every counter.
                n_sys_err += 1
                continue
            for ids, mask, behavior_logp in _trajectory_to_segments(r, rew, max_seq_len):
                if sum(mask) == 0:
                    continue
                samples.append({"ids": ids, "mask": mask, "behavior_logp": behavior_logp,
                                "reward": float(rew), "uid": uid,
                                "rollout_id": f"{uid}:{r.sample_idx}"})
    return samples, n_sys_err, local_picks


async def _bounded_rollout(runner, batch_tcs, g, timeout_s, group_ids=None):
    """Preserve completed samples while bounding slow rollout tasks."""
    return await runner.bounded_rollout_many(
        batch_tcs, n_samples=g, timeout_s=timeout_s,
        group_ids=group_ids)


def stub_local_stripe(
        by_name, rng, n_prompts, world, rank, g, tokenizer, global_picks=None):
    """FAST stub: g short trajectories/prompt with Bernoulli(0.5) reward (guaranteed
    intra-group variance -> non-zero DAPO gradient). Validates the training machinery
    (micro-batch FSDP2 fwd/bwd + hot-swap) WITHOUT the multi-minute MCP rollouts.
    Used only with --stub-rollout for a fast end-to-end machinery check."""
    pool = sorted(by_name.keys())
    picks = (
        list(global_picks)
        if global_picks is not None
        else select_global_prompt_uids(pool, rng, n_prompts)
    )
    local_picks = stripe_global_prompt_uids(
        picks, dp_rank=rank, dp_world=world)
    base = tokenizer.encode("<|im_start|>user\nhi<|im_end|>\n", add_special_tokens=False)
    hdr = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    vocab = tokenizer.vocab_size
    samples = []
    for uid in local_picks:
        for j in range(g):
            ans = [rng.randrange(0, vocab) for _ in range(rng.randint(4, 24))]
            ids = base + hdr + ans + [im_end]
            mask = [0] * (len(base) + len(hdr)) + [1] * (len(ans) + 1)
            samples.append({"ids": ids, "mask": mask,
                            "reward": float(rng.random() < 0.5), "uid": uid,
                            "rollout_id": f"{uid}:{j}", "is_stub": True})
    return samples, 0, local_picks


# ---------------------------------------------------------------------------
# DAPO dynamic sampling: oversample groups across bounded rounds, keep only
# reward-variance groups, and resample fresh prompts.
# ---------------------------------------------------------------------------
def _group_has_variance(rewards) -> bool:
    """A group is 'informative' iff its rewards are not all equal. A degenerate
    group (all-same reward, or a lone survivor of the per-rollout timeout) gives
    zero group-relative advantage -> zero DAPO gradient, so it is dropped."""
    if len(rewards) < 2:
        return False
    return min(rewards) != max(rewards)


def _sample_local_picks(by_name, rng, n_prompts, dp_world, dp_rank, used):
    """Deterministically sample n_prompts fresh prompt uids (excluding `used`)
    from the shared pool and return THIS DP rank's stripe. Every rank shares the
    same rng + `used`, so `picks` is identical everywhere and the stripe
    partitions cleanly; `used` is updated in place. Returns [] when exhausted."""
    pool = sorted(set(by_name.keys()) - used)
    if not pool:
        return []
    picks = select_global_prompt_uids(pool, rng, n_prompts)
    used.update(picks)
    return stripe_global_prompt_uids(
        picks, dp_rank=dp_rank, dp_world=dp_world)


async def _roll_round(runner, by_name, local_picks, g, tokenizer, max_seq_len,
                      reward_fn, overlong_penalty, timeout_s,
                      complete_groups_only=False):
    """Roll g samples for each local prompt, score + tokenize, and return
    grouped samples plus system-error, unfinished, scheduled, and completeness
    telemetry. GRPO callers censor incomplete groups before advantage computation."""
    batch_tcs = [by_name[u] for u in local_picks]
    if not batch_tcs:
        return {}, 0, 0, 0, {
            "complete_groups": 0,
            "informative_groups": 0,
            "incomplete_groups": 0,
            "incomplete_samples": 0,
            "timed_out_samples": 0,
            "valid_rollouts": 0,
            "valid_reward_sum": 0.0,
            "valid_reward_sq_sum": 0.0,
        }
    by_uid, n_unfinished = await _bounded_rollout(
        runner, batch_tcs, g, timeout_s,
        group_ids=local_picks)
    groups: dict = {}
    n_sys_err = 0
    n_timed_out = 0
    for uid, rolls in by_uid.items():
        rewards = score_rollouts(rolls, name=reward_fn)
        rewards = apply_overlong_penalty(rolls, rewards, penalty=overlong_penalty)
        for r, rew in zip(rolls, rewards):
            metadata = getattr(r, "metadata", None) or {}
            if metadata.get("rollout_timeout"):
                n_timed_out += 1
            if r.is_system_error or not r.messages:
                n_sys_err += 1
                continue
            if r.capture_integrity_error is not None:
                n_sys_err += 1
                continue
            for ids, mask, behavior_logp in _trajectory_to_segments(r, rew, max_seq_len):
                if sum(mask) == 0:
                    continue
                groups.setdefault(uid, []).append(
                    {"ids": ids, "mask": mask, "behavior_logp": behavior_logp,
                     "reward": float(rew), "uid": uid,
                     "rollout_id": f"{uid}:{r.sample_idx}"})
    valid_rewards: dict[str, float] = {}
    for rows in groups.values():
        for sample in rows:
            valid_rewards.setdefault(
                str(sample["rollout_id"]), float(sample["reward"]))
    groups, group_stats = filter_complete_rollout_groups(
        groups,
        candidate_group_ids=local_picks,
        expected_g=g,
        drop_incomplete=complete_groups_only,
    )
    group_stats["timed_out_samples"] = n_timed_out
    group_stats["valid_rollouts"] = len(valid_rewards)
    group_stats["valid_reward_sum"] = sum(valid_rewards.values())
    group_stats["valid_reward_sq_sum"] = sum(
        reward * reward for reward in valid_rewards.values())
    return (
        groups,
        n_sys_err,
        n_unfinished,
        len(batch_tcs) * g,
        group_stats,
    )


async def _rollout_local_stripe_dynamic(
        runner, by_name, rng, n_prompts, dp_world, dp_rank, g,
        tokenizer, max_seq_len, reward_fn, overlong_penalty,
        timeout_s, max_rounds, device, red_group, rank, world, local,
        complete_groups_only=False, filter_variance_groups_enabled=True,
        scheduled_global_picks=None):
    """DAPO dynamic sampling over the DP-stripe model.

    Each round samples a fresh global prompt batch (excluding already-used
    prompts), rolls this rank's stripe, KEEPS groups that have reward variance
    (non-zero DAPO gradient) and discards zero-variance ones, until every DP rank
    has `local_target` informative groups or the round budget is spent.

    Cross-rank lockstep: the round count is governed by an all_reduce(MAX) of each
    rank's remaining `need` over the DP group (`red_group`; None == world when
    sp_size=1), so EVERY rank runs the same number of rounds and none reaches the
    update-phase collectives while another is still rolling. Backfill from the
    discard pool keeps the final group count == local_target (predictable
    micro-batch packing). Returns (samples, n_sys_err, kept_uids) â€” same shape as
    rollout_local_stripe.

    NOTE (SP): when sp_size>1 only sp_rank 0 (the roller) calls this, and the
    `need` reduction runs over dp_group (the rollers). SP partners receive the
    final payload via the existing broadcast. This path is exercised today only
    at sp_size=1 (dp_group=None -> world); validate the dp_group reduction before
    enabling dynamic sampling on an SP run. Returns samples, system errors, kept
    uids, unfinished/scheduled sample counts, and group telemetry."""
    if scheduled_global_picks is not None:
        if max_rounds != 0:
            raise ValueError(
                "scheduled prompt batches require dynamic sampling rounds = 0"
            )
        if len(scheduled_global_picks) != n_prompts:
            raise ValueError(
                "scheduled prompt batch size does not match --n-prompts"
            )
        if len(set(scheduled_global_picks)) != len(scheduled_global_picks):
            raise ValueError("scheduled prompt batch repeats a group UID")
        unknown_scheduled = set(scheduled_global_picks) - set(by_name)
        if unknown_scheduled:
            raise ValueError(
                "scheduled prompt batch contains UIDs absent from the prompt pool"
            )
        base_pool_size = len(scheduled_global_picks)
    else:
        base_pool_size = min(n_prompts, len(by_name))
    local_target = len(range(dp_rank, base_pool_size, dp_world))  # == single-round stripe size

    used: set = set()
    kept: dict = {}       # informative groups (reward variance)
    discard: dict = {}    # zero-variance groups (backfill pool)
    n_sys_err_total = 0
    n_unfinished_total = 0
    n_scheduled_total = 0
    group_stats_total = {
        "complete_groups": 0,
        "informative_groups": 0,
        "incomplete_groups": 0,
        "incomplete_samples": 0,
        "timed_out_samples": 0,
        "valid_rollouts": 0,
        "valid_reward_sum": 0.0,
        "valid_reward_sq_sum": 0.0,
    }
    round_idx = 0
    while True:
        if scheduled_global_picks is None:
            local_picks = _sample_local_picks(
                by_name, rng, n_prompts, dp_world, dp_rank, used)
        else:
            local_picks = stripe_global_prompt_uids(
                scheduled_global_picks, dp_rank=dp_rank, dp_world=dp_world)
        if local_picks:
            (
                groups,
                n_sys,
                n_unfinished,
                n_scheduled,
                group_stats,
            ) = await _roll_round(
                runner, by_name, local_picks, g, tokenizer,
                max_seq_len, reward_fn, overlong_penalty, timeout_s,
                complete_groups_only=complete_groups_only)
            n_sys_err_total += n_sys
            n_unfinished_total += n_unfinished
            n_scheduled_total += n_scheduled
            for key in group_stats_total:
                group_stats_total[key] += group_stats[key]
            for uid, samps in groups.items():
                informative = _group_has_variance(
                    [s["reward"] for s in samps])
                if len(kept) < local_target and (
                        informative or not filter_variance_groups_enabled):
                    kept[uid] = samps
                else:
                    discard[uid] = samps
        # global remaining need (MAX over DP ranks) -> identical round count everywhere
        need = max(0, local_target - len(kept))
        if (red_group is not None) or (dist.is_initialized() and world > 1):
            nt = torch.tensor(need, device=device, dtype=torch.long)
            dist.all_reduce(nt, op=dist.ReduceOp.MAX, group=red_group)
            global_need = int(nt.item())
        else:
            global_need = need
        if global_need == 0 or round_idx >= max_rounds:
            break
        round_idx += 1
    # backfill from the discard pool so each rank reaches local_target (or pool empty)
    for uid in list(discard.keys()):
        if len(kept) >= local_target:
            break
        kept[uid] = discard.pop(uid)
    samples = [s for samps in kept.values() for s in samps]
    n_informative = sum(1 for samps in kept.values()
                        if _group_has_variance([s["reward"] for s in samps]))
    if is_rank0(rank):
        log(rank, world, local,
            f"dynamic_sampling: {n_informative}/{len(kept)} kept groups informative "
            f"(target {local_target}, backfill {len(kept) - n_informative}) "
            f"after {round_idx + 1} round(s)")
    return (
        samples,
        n_sys_err_total,
        sorted(kept.keys()),
        n_unfinished_total,
        n_scheduled_total,
        group_stats_total,
    )


def rollout_local_stripe_dynamic(
        runner, by_name, rng, n_prompts, dp_world, dp_rank, g,
        tokenizer, max_seq_len, reward_fn, overlong_penalty,
        timeout_s, max_rounds, device, red_group, rank, world, local,
        complete_groups_only=False, filter_variance_groups_enabled=True,
        scheduled_global_picks=None):
    """Run every dynamic-sampling round on one event loop.

    RolloutRunner owns loop-bound semaphores and HTTP clients. Repeated
    ``asyncio.run`` calls reused those objects from a closed loop and crashed
    round two with "bound to a different event loop".
    """
    return asyncio.run(_rollout_local_stripe_dynamic(
        runner, by_name, rng, n_prompts, dp_world, dp_rank, g,
        tokenizer, max_seq_len, reward_fn, overlong_penalty,
        timeout_s, max_rounds, device, red_group, rank, world, local,
        complete_groups_only=complete_groups_only,
        filter_variance_groups_enabled=filter_variance_groups_enabled,
        scheduled_global_picks=scheduled_global_picks))


# ---------------------------------------------------------------------------
# pack a local sample list -> tensors for the policy update (right-padded [B,T])
# ---------------------------------------------------------------------------
def pack_local_batch(samples, pad_id, device):
    B = len(samples)
    T = max((len(s["ids"]) for s in samples), default=1)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    resp_mask = torch.zeros((B, T), dtype=torch.float32)
    behavior_logp = torch.zeros((B, T), dtype=torch.float32)
    behavior_valid = torch.zeros((B,), dtype=torch.bool)
    scores = torch.zeros((B, T), dtype=torch.float32)
    uids = []
    rollout_uids = []
    for i, s in enumerate(samples):
        ids, am, L = s["ids"], s["mask"], len(s["ids"])
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = 1
        resp_mask[i, :L] = torch.tensor(am, dtype=torch.float32)
        raw_behavior = s.get("behavior_logp")
        if raw_behavior is not None:
            if len(raw_behavior) != L:
                raise ValueError("behavior logprobs must align with token IDs")
            behavior_logp[i, :L] = torch.tensor(raw_behavior, dtype=torch.float32)
            behavior_valid[i] = True
        elif not s.get("is_stub", False):
            raise ValueError("non-stub sample is missing behavior logprobs")
        last = max((j for j in range(L) if am[j] == 1), default=L - 1)
        scores[i, last] = float(s["reward"])
        uids.append(s["uid"])
        rollout_uids.append(s["rollout_id"])
    return (input_ids.to(device), attn.to(device), resp_mask.to(device),
            behavior_logp.to(device), behavior_valid.to(device),
            scores.to(device), np.array([str(u) for u in uids]),
            np.array([str(u) for u in rollout_uids]))


def select_mb_budget_rows(n_local, row_lens, group_ids, rollout_ids, rewards,
                          token_budget, row_cap, device, sp_size, sp_rank,
                          sp_group, rank, allow_informative_pair=True):
    """Select a bounded, still-valid group-relative batch before advantages.

    Token replay expands one rollout into multiple segment rows. The old cap sampled
    rows after computing advantages, which could leave a prompt with a singleton or
    an off-centre subset. This selector instead operates on complete rollouts:

      1. Prefer complete prompt groups while they fit the row/token budget.
      2. If no full prompt fits and ``allow_informative_pair`` is true (DAPO),
         choose the cheapest pair of complete rollouts with different rewards.
         GRPO disables this fallback because it requires all expected g rollouts.
      3. If no informative pair fits, select nothing; callers run lockstep dummies
         rather than silently training a biased singleton.

    Advantages MUST be recomputed after applying the returned indices. The token
    budget is an update-work budget (sum of unpadded row lengths), not a peak-memory
    guarantee. Peak memory is controlled by sequence parallelism because each update
    micro-batch is [1, T_global].
    """
    if n_local <= 0 or (token_budget <= 0 and row_cap <= 0):
        return None

    keep_mask = torch.zeros(n_local, dtype=torch.bool, device=device)
    if sp_size == 1 or sp_rank == 0:
        prompts: dict = {}
        for i in range(n_local):
            uid = str(group_ids[i])
            rid = str(rollout_ids[i])
            entry = prompts.setdefault(uid, {})
            roll = entry.setdefault(rid, {"rows": [], "reward": float(rewards[i])})
            roll["rows"].append(i)
            if float(rewards[i]) != roll["reward"]:
                raise ValueError(f"rollout {rid} has inconsistent rewards")

        def cost(rows):
            return len(rows), sum(int(row_lens[i]) for i in rows)

        def fits(rows, used_rows=0, used_tokens=0):
            n_rows, n_tokens = cost(rows)
            return ((row_cap <= 0 or used_rows + n_rows <= row_cap)
                    and (token_budget <= 0 or used_tokens + n_tokens <= token_budget))

        keys = sorted(prompts)
        chosen = []
        used_rows = used_tokens = 0
        for gi in torch.randperm(len(keys), device=device).tolist():
            uid = keys[gi]
            rows = [i for roll in prompts[uid].values() for i in roll["rows"]]
            if fits(rows, used_rows, used_tokens):
                chosen.extend(rows)
                nr, nt = cost(rows)
                used_rows += nr
                used_tokens += nt

        if not chosen and allow_informative_pair:
            # Full G groups can exceed the cap because token replay creates many
            # rows. Find the cheapest complete positive/negative rollout pair and
            # recompute a valid G=2 baseline on it rather than truncating rows.
            pairs = []
            for uid, rolls_by_id in prompts.items():
                rolls = list(rolls_by_id.values())
                for i in range(len(rolls)):
                    for j in range(i + 1, len(rolls)):
                        if rolls[i]["reward"] == rolls[j]["reward"]:
                            continue
                        rows = rolls[i]["rows"] + rolls[j]["rows"]
                        nr, nt = cost(rows)
                        if fits(rows):
                            pairs.append((nt, nr, uid, sorted(rows)))
            if pairs:
                chosen = min(pairs)[3]

        if chosen:
            keep_mask[torch.tensor(sorted(chosen), dtype=torch.long, device=device)] = True
    if sp_size > 1:
        dist.broadcast(keep_mask, src=rank - sp_rank, group=sp_group)
    return keep_mask.nonzero(as_tuple=False).flatten()


def resolve_vllm_health_urls(vllm_base_url):
    """Resolve rollout health endpoints without embedding pod-specific addresses."""
    configured = os.environ.get("VLLM_HEALTH_URLS")
    if configured is None:
        urls = [f"{vllm_base_url.rstrip('/')}/health"]
    else:
        urls = [url.strip() for url in configured.split(",") if url.strip()]
    if not urls:
        raise ValueError(
            "VLLM_HEALTH_URLS is set but contains no non-empty URLs")
    invalid = [url for url in urls
               if not url.startswith(("http://", "https://"))]
    if invalid:
        raise ValueError(
            f"VLLM health URLs must use http:// or https://; got {invalid}")
    return urls


def validate_sp1_update_limit(max_seq_len, sp_size, sp1_max_update_t):
    """Reject a rollout limit that the configured update topology cannot train."""
    if (sp_size == 1 and sp1_max_update_t > 0
            and max_seq_len > sp1_max_update_t):
        raise ValueError(
            f"--max-seq-len={max_seq_len} exceeds "
            f"SP1_MAX_UPDATE_T={sp1_max_update_t} with --sp-size=1; "
            f"lower --max-seq-len or use --sp-size 2")


def save_optimizer_checkpoint(opt, adapter_dir, rank, world, sp_size, step, lr,
                              barrier_fn=None):
    """Atomically save one FSDP2 optimizer shard and a node-local manifest.

    Checkpoint directories are node-local in the supported multi-node topology.
    Every rank writes its local shard, then a world barrier proves that all ranks
    completed the save phase. Local rank 0 on EACH node publishes the same
    META.json, so every node can later load its own rank shards without first
    copying a manifest from global rank 0's filesystem.

    Unique temp names keep the identical manifest writes safe if the checkpoint
    directory happens to be shared instead. Failures propagate: silently losing
    AdamW moments invalidates a reward-slope experiment.
    """
    opt_dir = Path(adapter_dir) / "optimizer"
    opt_dir.mkdir(parents=True, exist_ok=True)
    shard = opt_dir / f"opt_rank{rank:03d}.pt"
    tmp = opt_dir / f".{shard.name}.tmp"
    torch.save(
        {"world": world, "rank": rank, "step": step, "sp_size": sp_size,
         "lr": lr, "state_dict": opt.state_dict()},
        tmp)
    os.replace(tmp, shard)
    if barrier_fn is not None:
        barrier_fn()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if local_rank == 0:
        meta = {"world": world, "sp_size": sp_size, "step": step,
                "lr": lr, "adapter": Path(adapter_dir).name}
        meta_tmp = opt_dir / f".META.rank{rank:03d}.json.tmp"
        meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        os.replace(meta_tmp, opt_dir / "META.json")
    if barrier_fn is not None:
        barrier_fn()
    return opt_dir


def load_optimizer_checkpoint(opt, resume_lora_dir, rank, world, sp_size, lr,
                              device, require=False):
    """Restore the exact optimizer shard, rejecting partial/incompatible state."""
    opt_dir = Path(resume_lora_dir) / "optimizer"
    meta_path = opt_dir / "META.json"
    shard = opt_dir / f"opt_rank{rank:03d}.pt"
    if not meta_path.exists() and not shard.exists():
        if require:
            raise RuntimeError(
                f"required optimizer checkpoint missing under {opt_dir}; "
                f"continuing would reset AdamW and invalidate the slope endpoint")
        return "missing(fresh AdamW)"
    if not meta_path.is_file() or not shard.is_file():
        raise RuntimeError(
            f"partial optimizer checkpoint under {opt_dir}: "
            f"meta={meta_path.is_file()} shard={shard.is_file()}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("world", -1)) != world or int(meta.get("sp_size", -1)) != sp_size:
        raise RuntimeError(
            f"optimizer topology mismatch: saved world/sp={meta.get('world')}/"
            f"{meta.get('sp_size')}, current={world}/{sp_size}")
    saved_lr = float(meta.get("lr", float("nan")))
    if not math.isclose(saved_lr, float(lr), rel_tol=1e-12, abs_tol=0.0):
        raise RuntimeError(
            f"optimizer LR mismatch: saved={saved_lr}, requested={lr}; "
            f"resume with the saved LR or start a new run")
    if meta.get("adapter") != Path(resume_lora_dir).name:
        raise RuntimeError(
            f"optimizer manifest adapter={meta.get('adapter')!r} does not match "
            f"resume dir {Path(resume_lora_dir).name!r}")
    payload = torch.load(shard, map_location=device, weights_only=False)
    if (int(payload.get("world", -1)) != world
            or int(payload.get("rank", -1)) != rank
            or int(payload.get("sp_size", -1)) != sp_size):
        raise RuntimeError(f"optimizer shard metadata mismatch in {shard}")
    # The manifest and the shards are separate files published by different ranks.
    # Topology agreement alone cannot detect a directory assembled from two
    # different saves (a hand-merged resume dir, a partial copy, an interrupted
    # sync), which would silently resume the wrong AdamW moments under a step
    # number that looks correct in the logs. Cross-check the fields both carry.
    shard_step = payload.get("step")
    meta_step = meta.get("step")
    if (shard_step is not None and meta_step is not None
            and int(shard_step) != int(meta_step)):
        raise RuntimeError(
            f"optimizer manifest/shard step mismatch under {opt_dir}: "
            f"META step={meta_step}, {shard.name} step={shard_step}; the "
            f"manifest and shards come from different saves")
    shard_lr = payload.get("lr")
    if shard_lr is not None and not math.isclose(
            float(shard_lr), float(lr), rel_tol=1e-12, abs_tol=0.0):
        raise RuntimeError(
            f"optimizer shard LR mismatch in {shard}: saved={shard_lr}, "
            f"requested={lr}")
    opt.load_state_dict(payload["state_dict"])
    # opt.load_state_dict() restores EVERY saved per-group hyperparameter, including
    # `foreach`/`fused` -- not just tensor state. Checkpoints saved before foreach=False
    # was added at construction (build_model_and_opt) carry foreach=None, and loading
    # them silently reverts a freshly-built foreach=False optimizer back to None. Since
    # None triggers torch's own auto-detect (which picks foreach=True for CUDA/DTensor
    # params), this reopened the exact DeviceMesh sharding-prop crash foreach=False was
    # meant to prevent -- confirmed live: a resume from this very step12 checkpoint
    # still hit torch._foreach_lerp_ / _multi_tensor_adam despite construction-time
    # foreach=False. Re-force it after every load_state_dict, unconditionally.
    for group in opt.param_groups:
        group["foreach"] = False
    return f"restored(step={meta.get('step')})"


def load_optimizer_checkpoint_distributed(
        opt, resume_lora_dir, rank, world, sp_size, lr, device,
        expected_step, require=False):
    """Load node-local state and make every rank agree on success or failure.

    A crash between shard publication and node-local META publication can
    leave one node complete and another partial. Hold local exceptions until
    after a global MIN vote so successful ranks never wait at a barrier after
    a peer has already exited.
    """
    status = None
    status_code = None
    local_error = None
    try:
        status = load_optimizer_checkpoint(
            opt, resume_lora_dir, rank, world, sp_size, lr, device,
            require=require)
        status_code = optimizer_status_code(status)
    except Exception as exc:
        local_error = exc

    ok = torch.tensor(
        0 if local_error is not None else 1,
        dtype=torch.uint8,
        device=device)
    if world > 1:
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not bool(ok.item()):
        if local_error is not None:
            detail = f"{type(local_error).__name__}: {local_error}"
        else:
            detail = "another rank reported an incomplete/incompatible checkpoint"
        raise RuntimeError(
            f"optimizer resume failed collectively: {detail}") from local_error
    assert status_code is not None
    status_fields = torch.tensor(
        status_code, dtype=torch.long, device=device)
    status_min = status_fields.clone()
    status_max = status_fields.clone()
    if world > 1:
        dist.all_reduce(status_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(status_max, op=dist.ReduceOp.MAX)
    try:
        return validate_optimizer_resume_consensus(
            int(status_min[0].item()),
            int(status_max[0].item()),
            int(status_min[1].item()),
            int(status_max[1].item()),
            require=require,
            expected_step=expected_step,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"optimizer resume failed collectively: {exc}") from exc


def global_max_T(local_T, device, world):
    """All-reduce-MAX the per-rank sequence length so every rank runs the SAME
    number of lm_head logprob chunks -> identical FSDP2 param all-gather counts
    -> no cross-rank deadlock on variable-length rollouts. Padding is masked
    with response_mask=0."""
    t = torch.tensor(int(local_T), device=device, dtype=torch.long)
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def right_pad_T(input_ids, attn, resp_mask, behavior_logp, scores, T_global, pad_id):
    """Right-pad [B, T] tensors to T_global (pad tokens masked everywhere)."""
    B, T = input_ids.shape
    if T >= T_global:
        return input_ids, attn, resp_mask, behavior_logp, scores
    pad = T_global - T
    dev = input_ids.device
    input_ids = torch.cat([input_ids, torch.full((B, pad), pad_id, dtype=torch.long, device=dev)], dim=1)
    attn = torch.cat([attn, torch.zeros((B, pad), dtype=attn.dtype, device=dev)], dim=1)
    resp_mask = torch.cat([resp_mask, torch.zeros((B, pad), dtype=resp_mask.dtype, device=dev)], dim=1)
    behavior_logp = torch.cat([behavior_logp, torch.zeros((B, pad), dtype=behavior_logp.dtype, device=dev)], dim=1)
    scores = torch.cat([scores, torch.zeros((B, pad), dtype=scores.dtype, device=dev)], dim=1)
    return input_ids, attn, resp_mask, behavior_logp, scores


# ---------------------------------------------------------------------------
# verl-native group-relative advantage over the full local batch
# ---------------------------------------------------------------------------
def filter_variance_groups(samples):
    """Drop prompt groups (keyed by uid) that cannot yield a group-relative signal:
    fewer than 2 distinct ROLLOUTS, or zero reward variance across those rollouts.

    Token replay expands one rollout into several segment rows, so the gate must count
    ROLLOUTS (dedup by rollout_id), NOT segment rows. A lone multi-segment rollout is
    still a singleton â€” its segments share one reward -> zero group variance -> a wasted
    zero-mean advantage; and verl's GRPO hands a true singleton the RAW uncentered reward
    = biased REINFORCE noise the group baseline exists to remove. Deterministic, so under
    SP all partner ranks (which hold the broadcast-identical samples) filter identically.
    NB: we skip such groups but do NOT resample (no oversampling loop) â€” a step with no
    variance anywhere is simply skipped. Surviving groups keep ALL their segment rows."""
    by_uid = defaultdict(list)
    for s in samples:
        by_uid[s["uid"]].append(s)
    out = []
    for grp in by_uid.values():
        rewards_by_rollout = {}
        for s in grp:
            rewards_by_rollout.setdefault(s["rollout_id"], float(s["reward"]))
        r = list(rewards_by_rollout.values())
        if len(r) < 2:
            continue
        if max(r) == min(r):
            continue
        out.extend(grp)
    return out


def compute_local_advantages(
    scores,
    resp_mask,
    index,
    rollout_index,
    norm_adv_by_std_in_grpo=False,
):
    """Compute one verl outcome advantage per rollout, then broadcast to segments.

    Token replay splits each multi-turn rollout into several causal segments, and every
    segment is emitted as its own [1, T] row carrying that rollout's (shared) reward. The
    group baseline is E[R | prompt] over the g ROLLOUTS, so grouping the segment ROWS
    directly by prompt uid would let a k-segment rollout vote k times in its prompt's
    group_mean â€” biasing every advantage whenever rollouts differ in turn count (e.g. a
    prompt with rewards [1, 0] and segment counts [1, 3] gets a baseline of 0.25, not 0.5).
    We therefore collapse the segment rows to one row per rollout, run verl's group_mean
    over rollouts, then scatter each rollout's scalar advantage back onto every response
    token of its segments. GRPO enables per-group std normalization; DAPO does not."""
    B = scores.shape[0]
    row_return = scores.sum(dim=-1)                    # [B]: reward at each row's last resp token
    order: dict = {}                                   # rollout id -> compact row index
    comp_return: list = []
    comp_index: list = []
    seg_to_comp = torch.empty(B, dtype=torch.long)
    for i in range(B):
        rid = str(rollout_index[i])
        j = order.get(rid)
        if j is None:
            j = len(comp_return)
            order[rid] = j
            comp_return.append(row_return[i])
            comp_index.append(str(index[i]))
        seg_to_comp[i] = j
    comp_scores = torch.stack(comp_return).unsqueeze(-1)          # [R, 1] one row per rollout
    adv_comp, _ = ca.compute_grpo_outcome_advantage(
        token_level_rewards=comp_scores, response_mask=torch.ones_like(comp_scores),
        index=np.array(comp_index),
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo)            # [R, 1]
    adv_scalar = adv_comp[:, 0][seg_to_comp.to(adv_comp.device)]     # [B]: each row's rollout adv
    return adv_scalar.unsqueeze(-1) * resp_mask                      # [B, T]


def microbatch_policy_loss(model, ids_1, attn_1, respmask_1,
                           behavior_logp_1, behavior_valid, adv_1, chunk,
                           clip_low, clip_high, clip_c, loss_agg_mode,
                           n_total_tokens,
                           rollout_total_tokens, n_total_rollouts,
                           sp_group=None, sp_rank=0, sp_size=1,
                           activation_offload=False):
    """Clipped policy loss for one token-replay segment [1, T].

    Process one trajectory per forward and accumulate gradients to bound the
    activation footprint. ``token-mean`` accumulates a
    global response-token mean. ``rollout-mean`` first token-averages all
    segments belonging to a rollout, then gives every rollout equal weight.

    Ulysses SP (sp_size>1): only the BODY forward runs sequence-parallel. We slice
    ids/attn to THIS rank's seq shard [1, T/sp], hand the model the GLOBAL position_ids
    of that shard (so the SP-patched layers' rotary is correct), run the body, then
    all-gather the seq-sharded hidden back to the FULL [1, T, D] across the SP group.
    The lm_head logprob + policy loss then run on the FULL sequence == identical to the
    no-SP path (correct at shard boundaries; lm_head is chunk-checkpointed so its peak
    stays bounded). Activation memory of the 64 token mixers is the 1/sp win.

    P1 fix: the PPO importance ratio's denominator is the vLLM BEHAVIOR logprob
    (behavior_logp_1, the sampled-token logprob captured at rollout), NOT
    logp.detach() from this same forward. Same-forward detach makes the ratio
    bit-identically 1.0, so clip-higher/clip_c never bind and DAPO degenerates to
    on-policy REINFORCE with no train/infer-mismatch correction. When a sample has
    no captured behavior trace (stub/dummy micro-batches, behavior_valid=False) we
    fall back to the proximal snapshot logp.detach() (ratio 1.0) so those paths stay
    unchanged. Returns (scalar_loss_for_backward, clipfrac, n_resp_tokens,
    behavior_ratio, behavior_proximal_diff)."""
    if sp_size > 1:
        T = ids_1.shape[1]
        assert T % sp_size == 0, f"T={T} not divisible by sp_size={sp_size}"
        chunk_T = T // sp_size
        sl = slice(sp_rank * chunk_T, (sp_rank + 1) * chunk_T)
        ids_sh = ids_1[:, sl]                                  # [1, T/sp]
        attn_sh = attn_1[:, sl]
        pos_ids = torch.arange(sp_rank * chunk_T, (sp_rank + 1) * chunk_T,
                               device=ids_1.device).unsqueeze(0)  # GLOBAL positions [1, T/sp]
        hidden_sh = _body_hidden(
            model,
            ids_sh,
            attn_sh,
            position_ids=pos_ids,
            activation_offload=activation_offload,
        )  # [1, T/sp, D]
        hidden = _all_gather_seq(hidden_sh, sp_group, sp_rank, sp_size)  # [1, T, D]
    else:
        hidden = _body_hidden(
            model,
            ids_1,
            attn_1,
            activation_offload=activation_offload,
        )  # [1, T, D]
    lm_head = _lm_head_of(model)
    logp = chunked_target_logprob(hidden, lm_head, ids_1, chunk)   # [1, T-1]
    # Ï€_current is gradient-bearing. Ï€_behavior came from vLLM's sampled-token
    # response and is the importance-ratio denominator. Ï€_proximal is retained
    # separately as the local pre-update snapshot for parity diagnostics. This
    # no-reference DAPO/GRPO path has KL=0, so Ï€_reference is not evaluated.
    proximal_logp = logp.detach()
    behavior_logp = behavior_logp_1[:, 1:] if behavior_valid else proximal_logp
    mask = respmask_1[:, 1:]
    adv_t = adv_1[:, 1:]

    cfg = ActorConfig(
        strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=1,
        ppo_mini_batch_size=1, ppo_epochs=1,
        clip_ratio=clip_low, clip_ratio_low=clip_low, clip_ratio_high=clip_high,
        clip_ratio_c=clip_c, loss_agg_mode="token-mean",
        policy_loss={"loss_mode": "vanilla"})
    loss_fn = ca.get_policy_loss_fn("vanilla")
    res = loss_fn(old_log_prob=behavior_logp, log_prob=logp, advantages=adv_t,
                  response_mask=mask, loss_agg_mode="token-mean", config=cfg)
    # verl 0.8.0 vanilla returns (pg_loss, pg_metrics_dict). pg_metrics has scalar
    # "actor/pg_clipfrac" etc. (NOT a positional scalar â€” that was the smoke crash).
    pg_loss = res[0] if isinstance(res, (tuple, list)) else res
    pg_metrics = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else {}
    clipfrac = float(pg_metrics.get("actor/pg_clipfrac", 0.0)) if isinstance(pg_metrics, dict) else 0.0
    # Vanilla token-mean divides by this segment's token count. Reweight that
    # scalar into either the global token mean or equal-rollout mean.
    this_tokens = float(mask.sum().clamp_min(1.0).item())
    loss_mb = pg_loss * policy_loss_weight(
        loss_agg_mode,
        segment_response_tokens=this_tokens,
        global_response_tokens=n_total_tokens,
        rollout_response_tokens=rollout_total_tokens,
        global_rollout_count=n_total_rollouts,
    )
    # Diagnostics that PROVE the fix is live: behavior_ratio = mean exp(Ï€_current -
    # Ï€_behavior) over action tokens (â‰¡1.0 only if behavior==proximal, i.e. the bug);
    # behavior_proximal_diff = mean |Ï€_behavior - Ï€_proximal| (the train/infer gap).
    behavior_ratio = (
        (torch.exp((logp.detach() - behavior_logp).clamp(max=math.log(10.0))) * mask)
        .sum() / mask.sum().clamp_min(1.0)
    )
    behavior_proximal_diff = (
        ((behavior_logp - proximal_logp).abs() * mask).sum()
        / mask.sum().clamp_min(1.0)
    )
    return (
        loss_mb, clipfrac, this_tokens,
        float(behavior_ratio), float(behavior_proximal_diff),
    )


def sp_allreduce_grads(model, sp_group):
    """Sum LoRA gradients across the Ulysses SP group, IN PLACE, before opt.step(). [fix #1]

    Under Ulysses SP each sequence is split across the sp_size partner ranks: every
    partner computes the SAME full-sequence loss (from the all-gathered hidden), but its
    backward only routes gradient through ITS sequence shard (the _AllGatherSeq backward
    slices grad to this rank's half). So each partner holds a PARTIAL sum of the true
    gradient; the full gradient is the sum over the SP group.

    FSDP2 reduce-scatters grads only over the dp_group mesh (build_fsdp2_lora), and the SP
    partners live in DIFFERENT dp_groups, so this cross-SP sum is NOT done automatically.
    We add it here. The partners hold the identical dp-shard slice of each param (both are
    dp-index r of their respective dp_groups), so all-reducing their local grad shards (SUM)
    gives both the full gradient AND keeps the two replicas bit-identical => they step in
    lockstep and never diverge (the invariant _AllGatherSeq's 'identical loss' premise needs).
    """
    for p in model.parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad
        local = g.to_local() if hasattr(g, "to_local") else g   # FSDP2 grads are DTensors
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=sp_group)


def trainable_parameter_digest(model) -> str:
    """Hash exact local bytes for every trainable parameter shard."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        local = (
            parameter.to_local()
            if hasattr(parameter, "to_local")
            else parameter
        )
        local = local.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(local.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(local.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(local.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument(
        "--prompt-schedule",
        default=None,
        help="Optional YAML/JSON absolute-step prompt schedule. It is validated "
             "against --max-steps, --n-prompts, GRPO topology, and the hydrated "
             "train pool; --start-step indexes it absolutely on resume.",
    )
    ap.add_argument("--tb-config", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--agent", default="think")
    ap.add_argument(
        "--max-agent-turns",
        type=int,
        default=0,
        help="Training-only cap for max_agent_sim_turns on hydrated cases "
             "(0 preserves each dataset value). Stricter dataset limits remain.")
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=14)   # divisible by world(14)
    ap.add_argument(
        "--prompt-repeats-per-task",
        type=int,
        default=1,
        help="GRPO-only virtual prompt groups per hydrated task. Each group "
             "runs fresh stochastic trajectories; values >1 are intended for "
             "controlled one/few-task overfit experiments. Default 1 preserves "
             "the normal without-replacement task pool.")
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--mcp-urls", nargs="+", default=None,
                    help="One or more MCP endpoint URLs to round-robin (load "
                         "balance) the rollouts across. Default: the single URL "
                         "from --tb-config (mcp_proxy.endpoint_url).")
    ap.add_argument("--sp-size", type=int, default=1,
                    help="Ulysses sequence-parallel size (1 = no SP, unchanged code "
                         "path). SP>1 partitions world into dp=world/sp DP groups x sp "
                         "SP ranks; the per-sequence activations of all 64 token mixers "
                         "(GDN + softmax) shard 1/sp via the Ulysses all-to-all, "
                         "unblocking 128k. Must divide world AND every head count "
                         "(softmax: 24 Q / 4 KV; GDN: 16 K / 48 V) -> sp in {1,2,4}.")
    ap.add_argument("--max-seq-len", type=int, default=65536)
    ap.add_argument(
        "--activation-offload",
        action="store_true",
        help="Save transformer-body autograd tensors on pinned CPU memory "
             "until backward. Intended for 128K SP2 updates; lm-head "
             "checkpointing remains GPU-resident.",
    )
    ap.add_argument("--logprob-chunk-size", type=int, default=2048)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-target-profile", choices=tuple(LORA_TARGET_PROFILES),
                    default=DEFAULT_TARGET_PROFILE,
                    help="Which Linear modules LoRA adapts. 'gdn_hybrid' (default) = "
                         "q/k/v/o + MLP + the Gated-DeltaNet input/output projections "
                         "(in_proj_*/out_proj), leaving conv1d/A_log/dt_bias/norm frozen. "
                         "'attention_mlp' = q/k/v/o + MLP only, which leaves every "
                         "Gated-DeltaNet (linear-attention) layer's token-mixing FROZEN.")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--algo", choices=("dapo", "grpo"), default="dapo",
                    help="Policy objective recipe (default: dapo). GRPO uses "
                         "std-normalized advantages; both variants default to "
                         "token-mean loss and intentionally use no reference "
                         "policy and KL=0.")
    ap.add_argument("--clip-low", type=float, default=None)
    ap.add_argument("--clip-high", type=float, default=None)
    ap.add_argument("--clip-c", type=float, default=None)
    ap.add_argument("--overlong-penalty", type=float, default=None)
    ap.add_argument(
        "--loss-agg-mode",
        choices=("token-mean", "rollout-mean"),
        default=None,
        help="Policy-loss aggregation. Both algorithms default to token-mean "
             "for long-horizon stability. rollout-mean is the original GRPO "
             "sequence objective adapted across token-replay segments and is "
             "available only with --algo grpo.")
    ap.add_argument("--reward-fn", default="binary_test_result")
    ap.add_argument("--rollout-timeout", type=float, default=1800.0)
    ap.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--lora-save-dir", default="checkpoints/training-run/lora")
    ap.add_argument("--tag", default="training-run")
    ap.add_argument("--seed", type=int, default=731)
    ap.add_argument("--metrics-out", default="checkpoints/training-run/metrics.jsonl")
    adapter_group = ap.add_mutually_exclusive_group()
    adapter_group.add_argument(
        "--resume-lora-dir", default=None,
        help="Resume LoRA weights from a saved vLLM adapter directory. "
             "Optimizer state is also restored when that directory contains an "
             "optimizer/ shard set written by a run with --save-optimizer-state "
             "at the SAME world size; otherwise AdamW starts fresh and the run "
             "is NOT slope-comparable across the boundary.")
    adapter_group.add_argument(
        "--init-lora-dir", default=None,
        help="Clean warm start from adapter weights at --start-step 0. Loads "
             "weights collectively on every rank but never restores optimizer "
             "state; mutually exclusive with --resume-lora-dir.")
    ap.add_argument("--save-optimizer-state", action="store_true", default=True,
                    help="Persist sharded AdamW state beside each saved adapter so a "
                         "resume continues the optimizer instead of resetting it.")
    ap.add_argument("--no-save-optimizer-state", dest="save_optimizer_state",
                    action="store_false",
                    help="Disable optimizer checkpointing (restores the old "
                         "adapter-only resume behaviour).")
    ap.add_argument("--require-optimizer-resume", action="store_true",
                    help="Fail fast if --resume-lora-dir has no usable optimizer "
                         "state. Use for any run whose endpoint is a reward slope.")
    ap.add_argument("--start-step", type=int, default=0,
                    help="First global step index to run. Use with --resume-lora-dir "
                         "so adapter names and metrics continue monotonically.")
    ap.add_argument(
        "--recovery-source-checkpoint-step",
        type=int,
        default=None,
        help="Cursor-selected source policy checkpoint step. Set 0 for explicit "
             "true-base recovery after a no-update step; otherwise it must match "
             "--resume-lora-dir and may be lower than --start-step.",
    )
    ap.add_argument(
        "--recovery-optimizer-checkpoint-step",
        type=int,
        default=None,
        help="Cursor-selected AdamW checkpoint step. Current durable checkpoints "
             "require it to match --recovery-source-checkpoint-step (or both 0).",
    )
    ap.add_argument(
        "--producer-source-commit",
        default=None,
        help="Pinned trainer source commit recorded in every durable metric.",
    )
    ap.add_argument(
        "--producer-source-archive-sha256",
        default=None,
        help="Pinned trainer source archive SHA-256 recorded in durable metrics.",
    )
    ap.add_argument(
        "--producer-metric-schema",
        default="training-metric/v1",
        help="Metric schema identity recorded in each durable row.",
    )
    ap.add_argument(
        "--producer-checkpoint-schema",
        default="training-checkpoint/v1",
        help="Checkpoint schema identity recorded in each durable row.",
    )
    ap.add_argument("--stub-rollout", action="store_true",
                    help="FAST machinery check: synthetic Bernoulli-reward trajectories "
                         "instead of real MCP rollouts (validates FSDP2 update + hot-swap).")
    ap.add_argument("--dynamic-sampling-max-rounds", type=int, default=None,
                    help="DAPO dynamic sampling: max EXTRA resample rounds beyond the initial "
                         "round (bounds rollout cost so a rank can never resample forever while "
                         "others wait). DAPO default 1 (1 initial + 1 retry); GRPO default 0. "
                         "raise for more informative groups at higher rollout cost, especially "
                         "with few prompts/rank.")
    # ---- wandb (rank-0 only; reads WANDB_API_KEY / WANDB_BASE_URL from env â€” never hardcode) ----
    ap.add_argument("--wandb", action="store_true", help="log train/val metrics to wandb")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-project", default="thinkingbox-training")
    ap.add_argument("--wandb-run-name", default=None, help="defaults to --tag")
    # ---- in-training validation (rank-0 rolls a held-out set every N steps) ----
    ap.add_argument("--eval-every", type=int, default=0,
                    help="run a held-out validation eval every N steps (0 = off). "
                         "Needs --eval-list. Slows the run (an extra rollout burst).")
    ap.add_argument("--eval-list", default=None,
                    help="YAML test-list for the held-out validation set.")
    ap.add_argument(
        "--eval-max-agent-turns",
        type=int,
        default=0,
        help="Validation-only max_agent_sim_turns cap (0 preserves dataset "
             "values). Set equal to --max-agent-turns for an apples-to-apples "
             "same-task overfit curve.")
    ap.add_argument(
        "--eval-timeout",
        type=float,
        default=1500.0,
        help="Wall-clock budget in seconds for one complete validation burst. "
             "Unfinished replicas count as failures in the stable denominator.")
    ap.add_argument(
        "--eval-metrics-out",
        default=None,
        help="Optional JSONL path for durable per-eval results. Requires "
             "--eval-list and --eval-every > 0.")
    args = ap.parse_args()
    try:
        resolve_algorithm_defaults(args)
        validate_training_args(args)
    except ValueError as exc:
        ap.error(str(exc))
    recovery_flags_supplied = (
        args.recovery_source_checkpoint_step is not None
        or args.recovery_optimizer_checkpoint_step is not None
    )
    if recovery_flags_supplied:
        if (
            args.recovery_source_checkpoint_step is None
            or args.recovery_optimizer_checkpoint_step is None
        ):
            ap.error(
                "--recovery-source-checkpoint-step and "
                "--recovery-optimizer-checkpoint-step must be supplied together"
            )
        try:
            source_checkpoint_step, optimizer_checkpoint_step = (
                resolve_recovery_resume(
                    next_schedule_step=args.start_step,
                    source_policy_checkpoint_step=
                        args.recovery_source_checkpoint_step,
                    optimizer_checkpoint_step=
                        args.recovery_optimizer_checkpoint_step,
                    has_resume_adapter=bool(args.resume_lora_dir),
                    require_optimizer_resume=args.require_optimizer_resume,
                )
            )
        except ValueError as exc:
            ap.error(str(exc))
    else:
        if args.start_step > 0:
            ap.error(
                "--start-step >0 requires explicit "
                "--recovery-source-checkpoint-step and "
                "--recovery-optimizer-checkpoint-step"
            )
        source_checkpoint_step = 0 if args.resume_lora_dir else None
        optimizer_checkpoint_step = 0 if args.resume_lora_dir else None
    if bool(args.producer_source_commit) != bool(args.producer_source_archive_sha256):
        ap.error(
            "--producer-source-commit and --producer-source-archive-sha256 "
            "must be supplied together"
        )
    if args.producer_source_commit and (
        len(args.producer_source_commit) != 40
        or any(char not in "0123456789abcdef" for char in args.producer_source_commit)
    ):
        ap.error("--producer-source-commit must be a lowercase 40-hex commit")
    if args.producer_source_archive_sha256 and (
        len(args.producer_source_archive_sha256) != 64
        or any(char not in "0123456789abcdef" for char in args.producer_source_archive_sha256)
    ):
        ap.error("--producer-source-archive-sha256 must be lowercase 64-hex")
    schema_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
    for option, value in (
        ("--producer-metric-schema", args.producer_metric_schema),
        ("--producer-checkpoint-schema", args.producer_checkpoint_schema),
    ):
        if not schema_pattern.fullmatch(value):
            ap.error(f"{option} has an invalid schema identity")
    if source_checkpoint_step not in (None, 0):
        expected_adapter_name = f"{args.tag}_step{source_checkpoint_step}"
        if Path(args.resume_lora_dir).name != expected_adapter_name:
            ap.error(
                "--resume-lora-dir name does not match "
                f"--recovery-source-checkpoint-step: expected {expected_adapter_name!r}"
            )
    args.recovery_source_checkpoint_step = source_checkpoint_step
    args.recovery_optimizer_checkpoint_step = optimizer_checkpoint_step
    if args.prompt_schedule and args.dynamic_sampling_max_rounds != 0:
        ap.error(
            "--prompt-schedule requires --dynamic-sampling-max-rounds 0 so "
            "each absolute step uses exactly its scheduled prompt groups"
        )
    algorithm_settings = effective_algorithm_settings(args)
    sp1_max_update_t = int(os.environ.get("SP1_MAX_UPDATE_T", "45000"))
    try:
        validate_sp1_update_limit(
            args.max_seq_len, args.sp_size, sp1_max_update_t)
    except ValueError as exc:
        ap.error(str(exc))

    pg_timeout_minutes = resolve_pg_timeout_minutes(
        args.rollout_timeout, args.dynamic_sampling_max_rounds)
    rank, world, local = setup_dist(
        pg_timeout_minutes=pg_timeout_minutes)
    device = torch.device(f"cuda:{local}")
    validate_vllm_writer_topology(rank, world, args.vllm_base_url)

    # ---- Ulysses SP process groups (sp_size=1 -> dp == world, all no-ops below) ----
    sp_size = args.sp_size
    if sp_size > 1:
        assert world % sp_size == 0, f"world={world} not divisible by sp_size={sp_size}"
        sp_group, sp_rank, dp_group, dp_rank, dp_world = build_process_groups(
            rank, world, sp_size, timeout_minutes=pg_timeout_minutes)
        log(rank, world, local,
            f"SP groups: sp_size={sp_size} sp_rank={sp_rank} dp_rank={dp_rank}/{dp_world}")
    else:
        sp_group, sp_rank = None, 0
        dp_group, dp_rank, dp_world = None, rank, world

    if args.n_prompts % dp_world != 0:
        if is_rank0(rank):
            log(rank, world, local,
                f"WARN: --n-prompts {args.n_prompts} not divisible by dp_world {dp_world}; "
                f"some DP groups idle. Prefer a multiple of {dp_world}.")

    log(rank, world, local,
        f"START verl-FSDP2 {args.algo.upper()} trainer | verl={verl.__version__} world={world} "
        f"max_seq_len={args.max_seq_len} g={args.g} n_prompts={args.n_prompts} "
        f"lora_r={args.lora_rank} | {torch.cuda.get_device_name(device)}")
    log(rank, world, local,
        f"effective_algorithm_settings={json.dumps(algorithm_settings, sort_keys=True)}")

    # ---- data (hydrate the 551 list; identical across ranks) ----
    # Patch AgentSession BEFORE any rollout: merges consecutive leading system
    # messages into one (Qwen3.x chat template 400s on a 2nd system msg), captures
    # the per-rollout tool schema, AND installs the exact-token trace-session capture
    # required by the P1 behavior-logprob fix (current_trace_session). Idempotent;
    # no-op under --stub-rollout (which never builds an AgentSession).
    from train import patches
    patches.apply()
    names = load_test_list(args.train_list)
    train_cases = hydrate(names, dataset_dir=args.dataset_dir, agent=args.agent, strict=True)
    n_turn_caps = cap_training_agent_turns(
        train_cases, args.max_agent_turns)
    by_name = build_prompt_group_pool(
        train_cases, args.prompt_repeats_per_task)
    prompt_schedule = None
    if args.prompt_schedule:
        schedule_error = None
        try:
            prompt_schedule = load_prompt_schedule(args.prompt_schedule)
            if (
                args.prompt_repeats_per_task == 1
                and len(by_name) != len(train_cases)
            ):
                raise ValueError(
                    "hydrated train pool has duplicate UIDs; cannot validate a "
                    "one-group-per-task schedule"
                )
            prompt_schedule.validate_for_training(
                by_name.keys(),
                max_steps=args.max_steps,
                n_prompts=args.n_prompts,
                g=args.g,
                dp_world=dp_world,
                algorithm=args.algo,
                prompt_repeats_per_task=args.prompt_repeats_per_task,
            )
        except Exception as exc:  # noqa: BLE001
            # Report malformed/local schedule failures collectively before any rank
            # enters a schedule-hash collective, avoiding a one-rank startup hang.
            schedule_error = f"{type(exc).__name__}: {exc}"
        if world > 1:
            schedule_validation = [
                {
                    "error": schedule_error,
                    "sha256": (
                        prompt_schedule.sha256
                        if prompt_schedule is not None else None
                    ),
                }
            ]
            gathered_schedule_validation = [None] * world
            dist.all_gather_object(
                gathered_schedule_validation, schedule_validation[0])
            invalid_ranks = [
                f"{index}: {entry['error']}"
                for index, entry in enumerate(gathered_schedule_validation)
                if entry["error"]
            ]
            if invalid_ranks:
                raise RuntimeError(
                    "invalid --prompt-schedule on one or more ranks: "
                    + "; ".join(invalid_ranks)
                )
            schedule_hashes = {
                entry["sha256"] for entry in gathered_schedule_validation
            }
            if len(schedule_hashes) != 1:
                raise RuntimeError(
                    "prompt schedule SHA-256 differs across ranks; refusing to "
                    "run divergent DP prompt selections"
                )
        elif schedule_error:
            raise RuntimeError(
                f"invalid --prompt-schedule {args.prompt_schedule!r}: {schedule_error}"
            )
        if prompt_schedule is None:
            raise RuntimeError(
                f"invalid --prompt-schedule {args.prompt_schedule!r}: "
                f"{schedule_error or 'unknown validation failure'}"
            )
        args.prompt_schedule_sha256 = prompt_schedule.sha256
        args.prompt_schedule_resolved = str(prompt_schedule.path)
        if is_rank0(rank):
            log(
                rank, world, local,
                f"prompt_schedule={prompt_schedule.path} "
                f"sha256={prompt_schedule.sha256} "
                f"absolute_steps={prompt_schedule.max_steps} "
                f"prompts_per_step={prompt_schedule.prompts_per_step}",
            )
    else:
        args.prompt_schedule_sha256 = None
        args.prompt_schedule_resolved = None
    log(rank, world, local,
        f"hydrated {len(train_cases)} tasks from {args.train_list}; "
        f"virtual_prompt_groups={len(by_name)} "
        f"repeats_per_task={args.prompt_repeats_per_task}")
    if args.max_agent_turns > 0:
        log(rank, world, local,
            f"training-only max_agent_sim_turns cap={args.max_agent_turns} "
            f"changed_cases={n_turn_caps}/{len(train_cases)}; eval cases unchanged")
    # held-out validation set (optional, --eval-list): hydrated identically on all ranks,
    # but only rank-0 rolls it (run_eval below). Empty -> validation disabled.
    eval_cases = []
    if args.eval_list:
        eval_cases = hydrate(load_test_list(args.eval_list), dataset_dir=args.dataset_dir,
                             agent=args.agent, strict=True)
        n_eval_turn_caps = cap_training_agent_turns(
            eval_cases, args.eval_max_agent_turns)
        log(rank, world, local,
            f"hydrated {len(eval_cases)} VAL tasks from {args.eval_list}; "
            f"max_agent_turns={args.eval_max_agent_turns or 'dataset'} "
            f"changed_cases={n_eval_turn_caps}/{len(eval_cases)}")
    require_materialized_input(args.tb_config)
    with open(args.tb_config) as f:
        tb_cfg = ConfigFile.model_validate(yaml.safe_load(f))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ---- model (FSDP2 + LoRA) ----
    # FSDP2 reshards across the DP group (dp_group); with sp_size=1 dp_group is None
    # -> shards over the full world (unchanged).
    model, opt = build_fsdp2_lora(args.model_name, args.lora_rank, args.lora_alpha,
                                  args.lr, rank, world, local, dp_group=dp_group,
                                  target_modules=resolve_lora_target_modules(
                                      args.lora_target_profile),
                                  resume_lora_dir=args.resume_lora_dir,
                                  init_lora_dir=args.init_lora_dir)
    # ---- SP init weight sync: make partner ranks bit-identical at init -------
    # PEFT's LoRA `A` matrix is kaiming_uniform_-initialized from torch's GLOBAL RNG, which is
    # NOT seeded per-rank, so ranks can build different initial LoRA weights.
    # Different partner weights violate the identical full-sequence loss
    # assumption and produce an incorrect gathered hidden state. Broadcast each
    # trainable param's local shard from the SP-pair root (src = rank - sp_rank) over sp_group so
    # both partners start identical; combined with fix #1's grad-sum they then stay bit-identical
    # forever. SP partners sit at the SAME dp_rank, so their FSDP local shards are broadcast-
    # compatible (same shape). No-op when sp_size=1 (no SP group).
    if sp_size > 1:
        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                # NB: use a fresh name (NOT `local`) â€” `local` is this rank's GPU index used
                # by log(); clobbering it would dump weight tensors into every later log line.
                lshard = p.to_local() if hasattr(p, "to_local") else p
                dist.broadcast(lshard, src=(rank - sp_rank), group=sp_group)
        log(rank, world, local,
            f"SP init-sync: broadcast LoRA weights from sp-root rank {rank - sp_rank} "
            f"-> sp partners identical at init [fix #9]")
    # ---- Ulysses SP monkey-patch (only sp_size>1): make every GDN + softmax decoder
    # layer sequence-parallel-aware. MUST run AFTER fully_shard so the patched forwards
    # see FSDP-managed params, and after the model is on-device. ----
    if sp_size > 1:
        n_gdn, n_softmax = patch_model_for_sp(model, sp_group, sp_rank, sp_size)
        log(rank, world, local,
            f"SP monkey-patch: {n_gdn} GDN (linear_attn) + {n_softmax} softmax "
            f"(self_attn) decoder layers patched Ulysses-aware (sp_size={sp_size})")
        # FAIL LOUD if layer-type detection misses (wrong attr name on a new HF rev):
        # patch_model_for_sp would silently patch 0 layers and every token mixer would run
        # UNPATCHED on a T/sp shard == silently wrong numerics. Refuse to train. [fix #6]
        assert n_gdn > 0 and n_softmax > 0, (
            f"SP patch matched {n_gdn} GDN + {n_softmax} softmax layers (expected >0 each); "
            f"layer-type detection failed -> would silently mistrain. Aborting.")

    # ---- Restore sharded AdamW state, if present ---------------------------
    initial_lora_dir = args.resume_lora_dir or args.init_lora_dir
    source_adapter = (
        str(Path(initial_lora_dir).resolve()) if initial_lora_dir else None)
    opt_resume_status = (
        "fresh_init_adapter" if args.init_lora_dir else "none")
    if args.resume_lora_dir:
        opt_resume_status = load_optimizer_checkpoint_distributed(
            opt, args.resume_lora_dir, rank, world, sp_size, args.lr, device,
            expected_step=args.recovery_optimizer_checkpoint_step,
            require=args.require_optimizer_resume)
    log(rank, world, local,
        f"optimizer_resume_status={opt_resume_status} "
        f"source_adapter={source_adapter or 'none'}")
    if world > 1:
        dist.barrier()

    # ---- LoRA hot-swap client (rank-0 talks to vLLM) ----
    lora_client = VLLMLoraClient(base_url=args.vllm_base_url, timeout=120.0)

    out_metrics = Path(args.metrics_out)
    eval_metrics_out = (
        Path(args.eval_metrics_out)
        if args.eval_metrics_out else None
    )
    if is_rank0(rank):
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        if eval_metrics_out is not None:
            eval_metrics_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- wandb (rank-0 only; key/host come from env WANDB_API_KEY/WANDB_BASE_URL) ----
    if is_rank0(rank) and args.wandb:
        wandb_log = WandbLogger.maybe_init(
            enabled=True, project=args.wandb_project, entity=args.wandb_entity,
            run_name=args.wandb_run_name or args.tag, run_id=None, weave_enabled=False,
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))})
    else:
        wandb_log = WandbLogger(enabled=False)

    # the adapter name the rollout vLLM should use this step (env read by RolloutRunner build)
    current_lora_name = (
        Path(initial_lora_dir).name if initial_lora_dir else None)
    current_lora_path = (
        Path(args.resume_lora_dir).resolve()
        if args.resume_lora_dir
        else (Path(args.init_lora_dir).resolve()
              if args.init_lora_dir else None))
    rollout_retry_count = 0
    rollout_max_retries = int(os.environ.get("ROLLOUT_STEP_MAX_RETRIES", "3"))
    require_clean_rollout_batch = (
        os.environ.get("REQUIRE_CLEAN_ROLLOUT_BATCH", "0") == "1")
    allow_timeout_censoring = (
        os.environ.get("ROLLOUT_ALLOW_TIMEOUT_CENSORING", "0") == "1")
    minimum_rollout_complete_groups = int(
        os.environ.get("MIN_ROLLOUT_COMPLETE_GROUPS", "2"))
    empty_update_retry_count = 0
    empty_update_max_retries = int(
        os.environ.get("MB_EMPTY_MAX_RETRIES", "3"))

    model.train()
    step = args.start_step
    while step < args.max_steps:
        t_step = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)

        # Collective vLLM-health gate. A rollout-endpoint crash
        # otherwise turns every rollout into all-sys_err -> 0 kept -> no-op update
        # while `step` still advances, so ONE vLLM outage silently burns the whole
        # step budget. Rank 0 probes the endpoint(s) and
        # BROADCASTS a healthy flag; if unhealthy the WHOLE world sleeps in lockstep
        # (FSDP-safe) and re-probes, so an outage PAUSES training instead of eating
        # steps (the watchdog restarts vLLM meanwhile). Bounded by VLLM_HEALTH_MAX_WAIT.
        if not args.stub_rollout:
            import urllib.request as _urlreq
            _health_urls = resolve_vllm_health_urls(args.vllm_base_url)
            _waited = 0
            _max_wait = int(os.environ.get("VLLM_HEALTH_MAX_WAIT", "1800"))
            while True:
                healthy = 1
                health_error = None
                if rank == 0:
                    for _u in _health_urls:
                        _u = _u.strip()
                        if not _u:
                            continue
                        try:
                            with _urlreq.urlopen(_u, timeout=5) as _r:
                                if getattr(_r, "status", 200) != 200:
                                    healthy = 0
                                    health_error = (
                                        f"health probe returned "
                                        f"{getattr(_r, 'status', 'unknown')}: {_u}")
                                    break
                        except Exception:
                            healthy = 0
                            health_error = f"health probe failed: {_u}"
                            break
                _h = torch.tensor([healthy], device=device)
                if world > 1:
                    dist.broadcast(_h, src=0)

                # Every configured vLLM writer verifies/reloads its own local
                # engine. A shallow /health check cannot detect a watchdog
                # restart that came back serving only the base model.
                if int(_h.item()) == 1 and current_lora_name:
                    adapter_healthy = 1
                    adapter_error = None
                    if _is_vllm_writer(rank):
                        try:
                            lora_client.ensure_adapter(
                                current_lora_name,
                                current_lora_path,
                                active_probe=True,
                            )
                        except Exception as exc:
                            adapter_healthy = 0
                            adapter_error = (
                                f"adapter {current_lora_name!r} not active: "
                                f"{type(exc).__name__}: {str(exc)[:300]}")
                            log(rank, world, local,
                                f"[step {step}] local vLLM adapter recovery "
                                f"failed: {adapter_error}")
                    _adapter_h = torch.tensor(
                        [adapter_healthy], device=device)
                    if world > 1:
                        dist.all_reduce(_adapter_h, op=dist.ReduceOp.MIN)
                    if int(_adapter_h.item()) == 0:
                        _h.zero_()
                        if rank == 0 and health_error is None:
                            health_error = (
                                "one or more node-local vLLMs failed adapter "
                                "recovery/active verification")
                if int(_h.item()) == 1:
                    break
                if rank == 0:
                    log(rank, world, local,
                        f"[step {step}] rollout endpoint UNHEALTHY -> pausing "
                        f"({_waited}s waited; {health_error}); "
                        f"watchdog should restart vLLM")
                time.sleep(15)
                _waited += 15
                if _waited >= _max_wait:
                    raise RuntimeError(
                        f"step {step} rollout endpoint remained unhealthy for "
                        f"{_waited}s; refusing to consume a training step")

        # ---- rollout phase (each DP group rolls its prompt stripe) ----
        # Stripe by DP coordinates (dp_rank/dp_world): the sp_size ranks of one SP pair
        # MUST process the SAME trajectories (they each hold half the sequence). Only
        # sp_rank 0 actually rolls (vLLM sampling is non-deterministic, so independent
        # rolls would diverge); it then broadcasts the tokenized `samples` to its SP
        # partner over sp_group. With sp_size=1 this is the original per-rank stripe.
        rng = random.Random(args.seed * 1_000_003 + step)
        scheduled_global_picks = (
            select_global_prompt_uids(
                by_name.keys(),
                rng,
                args.n_prompts,
                schedule=prompt_schedule,
                absolute_step=step,
            )
            if prompt_schedule is not None
            else None
        )
        if scheduled_global_picks is not None and is_rank0(rank):
            log(
                rank, world, local,
                f"[step {step}] scheduled {len(scheduled_global_picks)} "
                f"global prompt groups "
                f"(schedule_sha256={prompt_schedule.sha256})",
            )
        t_roll = time.monotonic()
        if sp_size > 1 and sp_rank != 0:
            # SP partner: skip the rollout, receive samples from sp_rank 0 below.
            samples, n_sys_err, local_picks = [], 0, []
            n_unfinished = n_scheduled = 0
            group_stats = {
                "complete_groups": 0,
                "informative_groups": 0,
                "incomplete_groups": 0,
                "incomplete_samples": 0,
                "timed_out_samples": 0,
                "valid_rollouts": 0,
                "valid_reward_sum": 0.0,
                "valid_reward_sq_sum": 0.0,
            }
        elif args.stub_rollout:
            samples, n_sys_err, local_picks = stub_local_stripe(
                by_name, rng, args.n_prompts, dp_world, dp_rank, args.g, tokenizer,
                global_picks=scheduled_global_picks)
            n_unfinished = 0
            n_scheduled = len(local_picks) * args.g
            group_stats = {
                "complete_groups": len(local_picks),
                "informative_groups": sum(
                    1 for uid in local_picks
                    if _group_has_variance([
                        float(sample["reward"])
                        for sample in samples
                        if sample["uid"] == uid
                    ])
                ),
                "incomplete_groups": 0,
                "incomplete_samples": 0,
                "timed_out_samples": 0,
                "valid_rollouts": len(local_picks) * args.g,
                "valid_reward_sum": sum(
                    float(sample["reward"]) for sample in samples),
                "valid_reward_sq_sum": sum(
                    float(sample["reward"]) ** 2 for sample in samples),
            }
        else:
            runner = build_runner(tb_cfg, current_lora_name, args.concurrency,
                                  mcp_urls=args.mcp_urls)
            # DAPO defaults to one bounded refill round for zero-variance groups.
            # GRPO defaults to zero extra rounds, so this same path performs one
            # initial rollout round without dynamic refill. Explicit CLI values
            # can override either recipe. The variance gate below remains a
            # safety net after the configured round budget.
            (
                samples,
                n_sys_err,
                local_picks,
                n_unfinished,
                n_scheduled,
                group_stats,
            ) = rollout_local_stripe_dynamic(
                runner, by_name, rng, args.n_prompts, dp_world, dp_rank, args.g,
                tokenizer, args.max_seq_len, args.reward_fn, args.overlong_penalty,
                args.rollout_timeout, args.dynamic_sampling_max_rounds,
                device, dp_group if sp_size > 1 else None, rank, world, local,
                complete_groups_only=(args.algo == "grpo"),
                filter_variance_groups_enabled=(args.algo == "dapo"),
                scheduled_global_picks=scheduled_global_picks)
        if sp_size > 1:
            # broadcast the rolled trajectories from sp_rank 0 to the whole SP group so
            # all sp ranks shard the IDENTICAL sequences. src = global rank of sp_rank 0
            # in this SP pair = (rank // sp_size) * sp_size = rank - sp_rank.
            payload = [
                samples,
                n_sys_err,
                local_picks,
                n_unfinished,
                n_scheduled,
                group_stats,
            ]
            dist.broadcast_object_list(payload, src=(rank - sp_rank), group=sp_group)
            (
                samples,
                n_sys_err,
                local_picks,
                n_unfinished,
                n_scheduled,
                group_stats,
            ) = payload
        roll_dt = time.monotonic() - t_roll
        # honest reward metric on raw rollouts (deduplicate token-replay segments by
        # rollout_id, so a multi-turn rollout counts ONCE â€” the natural reward unit).
        raw_rewards = {}
        for s in samples:
            raw_rewards.setdefault(s["rollout_id"], float(s["reward"]))
        if args.algo == "grpo":
            n_raw = int(group_stats["valid_rollouts"])
            rmean = float(group_stats["valid_reward_sum"]) / max(n_raw, 1)
            rvar = max(
                0.0,
                float(group_stats["valid_reward_sq_sum"]) / max(n_raw, 1)
                - rmean ** 2,
            )
        else:
            n_raw = len(raw_rewards)
            rmean = sum(raw_rewards.values()) / max(n_raw, 1)
            rvar = sum(
                (r - rmean) ** 2 for r in raw_rewards.values()
            ) / max(n_raw, 1)
        # DAPO filters/refills zero-variance groups. Canonical GRPO keeps every
        # complete group, including all-zero/all-one groups whose advantages are
        # zero, so they remain in token-mean normalization.
        if args.algo == "dapo":
            samples = filter_variance_groups(samples)
        n_local = len(samples)                                  # segment rows kept (micro-batch count)
        n_kept_roll = len({s["rollout_id"] for s in samples})   # distinct rollouts kept
        training_rows_by_group: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            training_rows_by_group[str(sample["uid"])].append(sample)
        _, training_group_stats = filter_complete_rollout_groups(
            training_rows_by_group,
            candidate_group_ids=list(training_rows_by_group),
            expected_g=args.g,
            drop_incomplete=False,
        )
        n_training_complete_groups = training_group_stats["complete_groups"]
        n_training_informative_groups = training_group_stats[
            "informative_groups"]
        n_training_partial_groups = training_group_stats["incomplete_groups"]
        count_values = [
            n_raw,
            n_sys_err,
            n_kept_roll,
            n_local,
            n_unfinished,
            n_scheduled,
            group_stats["complete_groups"],
            group_stats["informative_groups"],
            group_stats["incomplete_groups"],
            group_stats["incomplete_samples"],
            group_stats["timed_out_samples"],
            n_training_complete_groups,
            n_training_informative_groups,
            n_training_partial_groups,
        ]
        if sp_size > 1 and sp_rank != 0:
            count_values = [0] * len(count_values)
        rollout_counts = torch.tensor(
            count_values,
            dtype=torch.long,
            device=device,
        )
        if world > 1:
            dist.all_reduce(rollout_counts, op=dist.ReduceOp.SUM)
        (
            global_n_raw,
            global_sys_err,
            global_n_kept,
            global_n_segments,
            global_rollout_unfinished,
            global_rollout_scheduled,
            global_rollout_complete_groups,
            global_rollout_informative_groups,
            global_rollout_incomplete_groups,
            global_rollout_incomplete_samples,
            global_rollout_timed_out_samples,
            global_training_complete_groups,
            global_training_informative_groups,
            global_training_partial_groups,
        ) = (int(value) for value in rollout_counts.tolist())
        global_rollout_completed_frac = (
            1.0
            if global_rollout_scheduled == 0
            else (
                global_rollout_scheduled - global_rollout_unfinished
            ) / global_rollout_scheduled
        )
        log(rank, world, local,
            f"[step {step}] rollout: {n_raw} rollouts ({n_kept_roll} kept / {n_local} segments "
            f"after variance-gate) over {len(local_picks)} prompts (sys_err={n_sys_err}) "
            f"global_unfinished={global_rollout_unfinished}/{global_rollout_scheduled} "
            f"global_completed_frac={global_rollout_completed_frac:.3f} "
            f"complete_groups={global_rollout_complete_groups} "
            f"informative_groups={global_rollout_informative_groups} "
            f"incomplete_groups={global_rollout_incomplete_groups} "
            f"incomplete_samples={global_rollout_incomplete_samples} "
            f"timed_out_samples={global_rollout_timed_out_samples} "
            f"training_complete_groups={global_training_complete_groups} "
            f"training_informative_groups={global_training_informative_groups} "
            f"training_partial_groups={global_training_partial_groups} "
            f"reward_mean={rmean:.3f} var={rvar:.4f} ({roll_dt:.0f}s)")
        insufficient_complete_groups = not has_sufficient_complete_groups(
            args.algo, global_training_complete_groups)
        if require_clean_rollout_batch and allow_timeout_censoring:
            rollout_integrity_failed = (
                not is_timeout_censored_rollout_batch(
                    sys_errors=global_sys_err,
                    unfinished=global_rollout_unfinished,
                    incomplete_samples=global_rollout_incomplete_samples,
                    timed_out_samples=global_rollout_timed_out_samples,
                    complete_groups=global_training_complete_groups,
                    minimum_complete_groups=minimum_rollout_complete_groups,
                )
            )
        else:
            rollout_integrity_failed = (
                require_clean_rollout_batch
                and not is_clean_rollout_batch(
                    sys_errors=global_sys_err,
                    unfinished=global_rollout_unfinished,
                    incomplete_groups=global_rollout_incomplete_groups,
                    incomplete_samples=global_rollout_incomplete_samples,
                    timed_out_samples=global_rollout_timed_out_samples,
                )
            )
        if (global_n_raw == 0 or insufficient_complete_groups
                or rollout_integrity_failed):
            rollout_retry_count += 1
            if is_rank0(rank):
                if insufficient_complete_groups:
                    reason = (
                        f"only {global_training_complete_groups} complete GRPO "
                        "groups survived (minimum 2)")
                elif global_n_raw == 0:
                    reason = "zero valid rollouts globally"
                else:
                    reason = (
                        "unclean rollout batch "
                        f"(sys_err={global_sys_err}, "
                        f"unfinished={global_rollout_unfinished}, "
                        f"incomplete_groups={global_rollout_incomplete_groups}, "
                        f"incomplete_samples={global_rollout_incomplete_samples}, "
                        f"timed_out={global_rollout_timed_out_samples}, "
                        f"required_complete_groups="
                        f"{minimum_rollout_complete_groups})")
                log(rank, world, local,
                    f"!!! step {step} produced {reason} "
                    f"(sys_err={global_sys_err}); retrying the SAME step "
                    f"({rollout_retry_count}/{rollout_max_retries})")
            if rollout_retry_count > rollout_max_retries:
                raise RuntimeError(
                    f"step {step} produced insufficient or unclean rollout data on "
                    f"{rollout_retry_count} consecutive attempts; aborting without "
                    f"advancing the step")
            if world > 1:
                dist.barrier()
            time.sleep(5)
            continue
        rollout_retry_count = 0

        # ---- gradient update: MICRO-BATCHED (mbs=1) + collective lockstep ----
        # Process one trajectory per forward to bound activation memory while
        # accumulating LoRA gradients. Two collective-lockstep invariants
        # so the FSDP2 param all-gathers never desync across ranks:
        #   (1) every rank runs the SAME number of micro-batches = global-max local-B
        #       (short ranks run ZEROED dummy micro-batches), and
        #   (2) every micro-batch pads to the SAME global-max T (lm_head chunk parity).
        # Empty/dummy micro-batches multiply their loss by 0 -> collectives fire, no grad.
        did_update = False
        loss_val = adv_val = clipfrac = grad_norm = 0.0
        behavior_ratio = 1.0
        behavior_proximal_diff = 0.0
        local_resp_tokens = 0.0
        local_has_nonzero_advantage = False
        rollout_resp_tokens: list[float] = []
        # Cap BEFORE computing advantages, then recompute the group
        # baseline on exactly the rollouts that train. This is the semantic fix
        # for C6/B2/B5: no post-advantage row truncation and no singleton raw reward.
        n_rows_pre_cap = n_local
        n_rollouts_pre_cap = n_kept_roll
        n_groups_post_cap = n_rollouts_post_cap = 0
        n_complete_groups_post_cap = 0
        n_informative_groups_post_cap = 0
        n_partial_groups_post_cap = 0
        _mb_cap = int(os.environ.get("MB_CAP", "0"))
        _mb_tok = int(os.environ.get("MB_TOKEN_BUDGET", "0"))
        if n_local > 0:
            input_ids, attn, resp_mask, behavior_logp, behavior_valid, scores, index, rollout_index = \
                pack_local_batch(samples, pad_id, device)
            if _mb_cap > 0 or _mb_tok > 0:
                _keep = select_mb_budget_rows(
                    n_local,
                    attn.sum(dim=1).tolist(),
                    index,
                    rollout_index,
                    scores.sum(dim=-1).tolist(),
                    _mb_tok,
                    _mb_cap,
                    input_ids.device,
                    sp_size,
                    sp_rank,
                    sp_group,
                    rank,
                    allow_informative_pair=(args.algo != "grpo"),
                )
                if _keep is not None and int(_keep.numel()) < n_local:
                    _keep_np = _keep.detach().cpu().numpy()
                    input_ids = input_ids[_keep]
                    attn = attn[_keep]
                    resp_mask = resp_mask[_keep]
                    behavior_logp = behavior_logp[_keep]
                    behavior_valid = behavior_valid[_keep]
                    scores = scores[_keep]
                    index = index[_keep_np]
                    rollout_index = rollout_index[_keep_np]
                    n_local = int(_keep.numel())
            if n_local > 0:
                # Recompute advantages AFTER selection. A fallback positive/negative
                # rollout pair is therefore a valid informative G=2 group, not a
                # biased fragment of the original G=4 group.
                adv_full = compute_local_advantages(
                    scores,
                    resp_mask,
                    index,
                    rollout_index,
                    norm_adv_by_std_in_grpo=(args.algo == "grpo"),
                )
                adv_val = float(adv_full[:, 1:].abs().mean())
                local_has_nonzero_advantage = bool(
                    torch.count_nonzero(adv_full).item())
                local_resp_tokens = float(resp_mask.sum().item())
                rollout_resp_tokens = rollout_response_token_totals(
                    resp_mask.sum(dim=1).detach().cpu().tolist(),
                    rollout_index,
                )
                n_groups_post_cap = len({str(u) for u in index})
                n_rollouts_post_cap = len({str(r) for r in rollout_index})
                selected_rows_by_group: dict[str, list[dict]] = defaultdict(list)
                selected_rewards = scores.sum(dim=-1).detach().cpu().tolist()
                for uid, rollout_id, reward in zip(
                        index, rollout_index, selected_rewards):
                    selected_rows_by_group[str(uid)].append({
                        "uid": str(uid),
                        "rollout_id": str(rollout_id),
                        "reward": float(reward),
                    })
                _, selected_group_stats = filter_complete_rollout_groups(
                    selected_rows_by_group,
                    candidate_group_ids=list(selected_rows_by_group),
                    expected_g=args.g,
                    drop_incomplete=False,
                )
                n_complete_groups_post_cap = selected_group_stats[
                    "complete_groups"]
                n_informative_groups_post_cap = selected_group_stats[
                    "informative_groups"]
                n_partial_groups_post_cap = selected_group_stats[
                    "incomplete_groups"]
            else:
                adv_full = torch.zeros_like(scores)
            log(rank, world, local,
                f"[step {step}] update selection: {n_rows_pre_cap} -> {n_local} rows, "
                f"{n_rollouts_pre_cap} -> {n_rollouts_post_cap} rollouts, "
                f"groups={n_groups_post_cap} "
                f"(complete={n_complete_groups_post_cap}, "
                f"informative={n_informative_groups_post_cap}, "
                f"partial={n_partial_groups_post_cap}), row_cap={_mb_cap}, "
                f"token_budget={_mb_tok}")

        # Global conversion accounting. Under SP, partner ranks hold identical
        # samples, so count only SP roots before reducing across the world.
        _cap_values = [
            n_local,
            n_rollouts_post_cap,
            n_groups_post_cap,
            n_complete_groups_post_cap,
            n_informative_groups_post_cap,
            n_partial_groups_post_cap,
        ]
        if sp_size > 1 and sp_rank != 0:
            _cap_values = [0] * len(_cap_values)
        _cap_counts = torch.tensor(_cap_values, dtype=torch.long, device=device)
        if world > 1:
            dist.all_reduce(_cap_counts, op=dist.ReduceOp.SUM)
        (
            global_update_rows,
            global_update_rollouts,
            global_update_groups,
            global_update_complete_groups,
            global_update_informative_groups,
            global_update_partial_groups,
        ) = (int(v) for v in _cap_counts.tolist())
        insufficient_update_groups = not has_sufficient_complete_groups(
            args.algo, global_update_complete_groups)
        if global_update_rows == 0 or insufficient_update_groups:
            empty_update_retry_count += 1
            if is_rank0(rank):
                selection_reason = (
                    f"only {global_update_complete_groups} complete GRPO groups "
                    "selected globally (minimum 2)"
                    if insufficient_update_groups
                    else "zero update rows globally"
                )
                log(rank, world, local,
                    f"!!! step {step} selected {selection_reason} "
                    f"(kept_rollouts={global_n_kept}, "
                    f"segments_pre_cap={global_n_segments}, "
                    f"row_cap={_mb_cap}, token_budget={_mb_tok}); "
                    f"retrying the SAME step "
                    f"({empty_update_retry_count}/{empty_update_max_retries})")
            if empty_update_retry_count > empty_update_max_retries:
                raise RuntimeError(
                    f"step {step} selected insufficient complete update data on "
                    f"{empty_update_retry_count} consecutive attempts "
                    f"(kept_rollouts={global_n_kept}, "
                    f"segments_pre_cap={global_n_segments}, "
                    f"row_cap={_mb_cap}, token_budget={_mb_tok}); "
                    f"increase/disable the update budget or inspect the "
                    f"variance-gate output; aborting without advancing the step")
            if world > 1:
                dist.barrier()
            time.sleep(5)
            continue
        empty_update_retry_count = 0
        advantage_signal = torch.tensor(
            1 if local_has_nonzero_advantage else 0,
            dtype=torch.uint8,
            device=device,
        )
        if world > 1:
            dist.all_reduce(advantage_signal, op=dist.ReduceOp.MAX)
        global_has_nonzero_advantage = bool(advantage_signal.item())
        # token-mean uses the global response-token total below. rollout-mean
        # uses the already reduced global_update_rollouts plus each rollout's
        # local token total, because a rollout never crosses DP stripes. SP
        # duplicates are excluded from both global counts. In either case
        # loss_scale=dp_world exactly undoes FSDP2's DP gradient average. [fix #2]
        _gt = torch.tensor(local_resp_tokens, device=device)
        if (dp_group is not None) or world > 1:
            dist.all_reduce(_gt, op=dist.ReduceOp.SUM, group=dp_group)
        global_resp_tokens = float(_gt.clamp_min(1.0).item())
        loss_scale = float(dp_world)
        # global-max micro-batch count + global-max T (across ALL ranks) â€” lockstep
        n_mb_global = max(1, global_max_T(n_local, device, world))
        local_T = input_ids.shape[1] if n_local > 0 else 3
        T_global = global_max_T(local_T, device, world)
        # Row count controls update wall time, not peak memory: each
        # micro-batch is [1, T_global]. Fail before backward when SP1 exceeds
        # its explicitly configured sequence threshold.
        if (sp_size == 1 and sp1_max_update_t > 0
                and T_global > sp1_max_update_t):
            raise RuntimeError(
                f"T_global={T_global} exceeds "
                f"SP1_MAX_UPDATE_T={sp1_max_update_t}; "
                f"restart with --sp-size 2. Row/token caps cannot bound peak "
                f"memory because update micro-batches are [1, T_global].")
        # Ulysses SP needs every micro-batch's T divisible by sp_size (the seq is sharded
        # T/sp inside microbatch_policy_loss). Round the global-max T up to a multiple of
        # sp_size BEFORE padding so every rank (real + dummy mb) shards evenly. The extra
        # pad tokens are masked everywhere (response_mask=0). No-op when sp_size=1.
        if sp_size > 1 and T_global % sp_size != 0:
            T_global += sp_size - (T_global % sp_size)
        # pad the whole local batch (ids/attn/resp_mask/adv) to T_global ONCE
        if n_local > 0:
            input_ids, attn, resp_mask, behavior_logp, _ = right_pad_T(
                input_ids, attn, resp_mask, behavior_logp, scores, T_global, pad_id)
            if adv_full.shape[1] < T_global:
                adv_full = torch.cat(
                    [adv_full, torch.zeros((adv_full.shape[0], T_global - adv_full.shape[1]),
                                           dtype=adv_full.dtype, device=device)], dim=1)

        def _dummy_mb():
            ids1 = torch.full((1, T_global), pad_id, dtype=torch.long, device=device)
            at1 = torch.zeros((1, T_global), dtype=torch.long, device=device); at1[0, 0] = 1
            rm1 = torch.zeros((1, T_global), dtype=torch.float32, device=device); rm1[0, 1] = 1.0
            blp1 = torch.zeros((1, T_global), dtype=torch.float32, device=device)
            ad1 = torch.zeros((1, T_global), dtype=torch.float32, device=device)
            return ids1, at1, rm1, blp1, ad1

        opt.zero_grad(set_to_none=True)
        t_update = time.monotonic()        # wall-clock: start of the optimizer update
        clip_acc, tok_acc, loss_acc = 0.0, 0.0, 0.0
        br_acc, bd_acc = 0.0, 0.0
        try:
            for mb in range(n_mb_global):
                real_mb = (n_local > 0 and mb < n_local)
                if real_mb:
                    ids1, at1, rm1, ad1 = (input_ids[mb:mb + 1], attn[mb:mb + 1],
                                           resp_mask[mb:mb + 1], adv_full[mb:mb + 1])
                    blp1 = behavior_logp[mb:mb + 1]
                    bvalid = bool(behavior_valid[mb].item())
                else:
                    ids1, at1, rm1, blp1, ad1 = _dummy_mb()
                    bvalid = False
                parent_rollout_tokens = (
                    rollout_resp_tokens[mb] if real_mb else 1.0)
                loss_mb, cf, this_tok, br, bd = microbatch_policy_loss(
                    model, ids1, at1, rm1, blp1, bvalid, ad1, args.logprob_chunk_size,
                    args.clip_low, args.clip_high, args.clip_c,
                    args.loss_agg_mode,
                    global_resp_tokens, parent_rollout_tokens,
                    global_update_rollouts,
                    sp_group=sp_group, sp_rank=sp_rank, sp_size=sp_size,
                    activation_offload=args.activation_offload)
                loss_mb = loss_mb * loss_scale   # undo FSDP2's 1/dp_world grad average [fix #2]
                if not real_mb:
                    # zero the dummy contribution; nan_to_num guards 0*NaN=NaN if a degenerate
                    # all-pad forward ever produced a non-finite loss. [fix #8]
                    loss_mb = torch.nan_to_num(loss_mb) * 0.0
                loss_mb.backward()                # accumulate; FSDP2 reduce-scatters LoRA grads
                if real_mb:
                    loss_acc += float(loss_mb.detach())
                    clip_acc += cf * this_tok
                    tok_acc += this_tok
                    br_acc += br * this_tok
                    bd_acc += bd * this_tok
            # Ulysses SP: sum the partner ranks' (sequence-shard-partial) LoRA grads so both
            # replicas get the full-sequence gradient and stay bit-identical in lockstep. [fix #1]
            if sp_size > 1:
                sp_allreduce_grads(model, sp_group)
            # grad-norm signal (no-op clip at max=1e9 just reads the norm). Collective over
            # the FSDP mesh -> all ranks call it. After sp_allreduce the grad is the full
            # gradient (identical on SP partners), so the dp_group norm is the true norm.
            # A complete GRPO batch may legitimately have zero advantages (all
            # groups are all-zero/all-one). AdamW would still apply residual
            # momentum on zero gradients, so collectively skip opt.step and the
            # subsequent LoRA swap when no rank has nonzero advantage.
            if global_has_nonzero_advantage:
                # foreach=False: see the AdamW construction comment (build_model_and_opt)
                # for why the batched/foreach path is avoided for DTensor parameters here.
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1e9,
                    foreach=False))
            else:
                grad_norm = 0.0
            optimizer_stepped = step_optimizer_if_advantage(
                opt, global_has_nonzero_advantage)
            # ---- Exact SP weight-sync validation (optional) ----------------
            if os.environ.get("SP_SYNC_CHECK", "0") == "1" and sp_size > 1:
                local_digest = trainable_parameter_digest(model)
                digest_tensor = torch.tensor(
                    list(bytes.fromhex(local_digest)),
                    dtype=torch.uint8,
                    device=device,
                )
                gathered = [
                    torch.empty_like(digest_tensor) for _ in range(sp_size)
                ]
                dist.all_gather(gathered, digest_tensor, group=sp_group)
                sp_digests = [
                    bytes(item.cpu().tolist()).hex() for item in gathered
                ]
                if len(set(sp_digests)) != 1:
                    raise RuntimeError(
                        "sequence-parallel trainable parameters diverged: "
                        + ", ".join(sp_digests)
                    )
                log(rank, world, local,
                    f"SP_SYNC step={step} sha256={local_digest}")
            loss_val = loss_acc / max(loss_scale, 1.0)   # report unscaled loss magnitude
            clipfrac = clip_acc / max(tok_acc, 1.0)
            behavior_ratio = br_acc / max(tok_acc, 1.0)
            behavior_proximal_diff = bd_acc / max(tok_acc, 1.0)
            did_update = n_local > 0 and optimizer_stepped
        except torch.cuda.OutOfMemoryError as e:
            log(rank, world, local, f"!!! step {step} OOM in update: {str(e)[:300]}")
            raise

        # all-reduce did_update so every rank agrees whether to hot-swap
        upd_t = torch.tensor(1.0 if did_update else 0.0, device=device)
        if world > 1:
            dist.all_reduce(upd_t, op=dist.ReduceOp.MAX)
        any_update = upd_t.item() > 0

        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        log(rank, world, local,
            f"[step {step}] update: loss={loss_val:.5f} adv_abs_mean={adv_val:.4f} "
            f"clipfrac={clipfrac:.3f} behavior_ratio={behavior_ratio:.4f} "
            f"behavior_proximal_diff={behavior_proximal_diff:.4f} "
            f"n_mb={n_mb_global} T_global={T_global} "
            f"peak_vram={peak:.2f}GiB "
            f"nonzero_advantage={global_has_nonzero_advantage} "
            f"did_update={did_update}")

        # ---- LoRA hot-swap into vLLM (gather adapter + POST /load_lora_adapter) ----
        swap_ok = True
        if any_update:
            new_name = f"{args.tag}_step{step + 1}"
            # fsdp_sync loads new_name FIRST then unloads the one it replaces, so pass
            # current_lora_name (the adapter being superseded), NOT a 2-step-stale name. [fix #4]
            swap_ok = fsdp_sync_vllm_lora(model, lora_client, args.lora_save_dir,
                                          new_name, current_lora_name, rank, world)
            # only rank-0 POSTs, so broadcast its result (MIN) so all ranks agree. [fix #5]
            ok_t = torch.tensor(1.0 if swap_ok else 0.0, device=device)
            if world > 1:
                dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            swap_ok = ok_t.item() > 0
            if swap_ok:
                current_lora_name = new_name
                current_lora_path = (
                    Path(args.lora_save_dir) / new_name).resolve()
                # Persist every rank's FSDP2 optimizer shard atomically. Fail
                # closed: a run that loses AdamW moments is not slope-comparable.
                if args.save_optimizer_state:
                    save_optimizer_checkpoint(
                        opt,
                        Path(args.lora_save_dir) / new_name,
                        rank, world, sp_size, step + 1, args.lr,
                        barrier_fn=(dist.barrier if world > 1 else None))
            else:
                raise RuntimeError(
                    f"step {step} vLLM LoRA hot-swap to {new_name!r} failed; "
                    f"refusing to continue off-policy on {current_lora_name!r}")

        # ---- wall-clock step timing (rank 0) ----
        #   t_rollout = rollout phase (MCP + user-sim multi-turn generation)
        #   t_update  = optimizer update (fwd/bwd/step) + LoRA gather + vLLM hot-swap
        #   t_total   = whole step
        t_update_dt = time.monotonic() - t_update
        step_dt = time.monotonic() - t_step
        if is_rank0(rank):
            log(rank, world, local,
                f"STEP_TIMING step={step} t_rollout={roll_dt:.1f}s "
                f"t_update={t_update_dt:.1f}s t_total={step_dt:.1f}s")
        # aggregate reward for an honest global metric. Under SP both sp ranks of a pair
        # hold the SAME trajectories, so we reduce over the DP group (each unique stripe
        # counted once) instead of the world (which would double-count). The MEAN is in
        # fact invariant to the duplication either way; the DP reduce keeps the implied
        # totals honest. dp_group=None (sp_size=1) -> world reduce (unchanged).
        # Weight the reduced reward mean by n_raw (the pre-gate rollout count),
        # NOT n_local (POST-gate). `rmean` is the mean over the n_raw raw rollouts, so
        # rmean*n_raw == the raw reward SUM; weighting by n_local instead mis-scaled the global
        # mean whenever the variance gate dropped groups (i.e. almost every step). g_rmean is
        # now the true fraction of ALL rollouts that passed = the learning signal (n_rollouts /
        # n_kept below already report both counts, so the reported metric is self-consistent).
        red_group = dp_group if sp_size > 1 else None
        red_world = dp_world if sp_size > 1 else world
        # Adapter magnitude is measured every step by ALL ranks (the reduce
        # must not sit inside the rank-0 branch or it deadlocks). LoRA params are
        # FSDP2-sharded, so square locally and SUM across the DP group; SP partners
        # hold identical shards, so reducing over dp_group counts each shard once.
        # Exact ||dW||_F = (alpha/r)||BA|| can also be recomputed from saved
        # adapters.
        _sq = torch.zeros((), device=device, dtype=torch.float64)
        for _p in model.parameters():
            if _p.requires_grad:
                _d = _p.detach()
                _d = _d.to_local() if hasattr(_d, "to_local") else _d
                _sq += _d.to(torch.float64).pow(2).sum()
        if red_world > 1:
            dist.all_reduce(_sq, op=dist.ReduceOp.SUM, group=red_group)
        lora_param_norm = float(_sq.clamp_min(0).sqrt())
        if is_rank0(rank):
            rmean_t = torch.tensor([rmean * n_raw, float(n_raw)], device=device)
            if red_world > 1:
                dist.all_reduce(rmean_t, op=dist.ReduceOp.SUM, group=red_group)
            g_rmean = (rmean_t[0] / rmean_t[1].clamp_min(1)).item()
            # Keep the aggregate JSONL self-identifying: the external checkpoint
            # validator consumes the same durable stream across pod recreation.
            rec = {"schema": args.producer_metric_schema,
                   "run_id": args.tag,
                   "training_run_id": args.tag,
                   "tag": args.tag,
                   "producer_source_commit": args.producer_source_commit,
                   "producer_source_archive_sha256":
                       args.producer_source_archive_sha256,
                   "producer_run_id": args.tag,
                   "producer_prompt_schedule_sha256":
                       args.prompt_schedule_sha256,
                   "producer_metric_schema":
                       args.producer_metric_schema,
                   "producer_checkpoint_schema":
                       args.producer_checkpoint_schema,
                   "recovery_next_schedule_step": step + 1,
                   "recovery_source_policy_checkpoint_step":
                       args.recovery_source_checkpoint_step,
                   "recovery_optimizer_checkpoint_step":
                       args.recovery_optimizer_checkpoint_step,
                   "step": step, "algo": args.algo,
                   "max_agent_turns": args.max_agent_turns,
                   "n_prompts": args.n_prompts,
                   "prompt_schedule": args.prompt_schedule_resolved,
                   "prompt_schedule_sha256": args.prompt_schedule_sha256,
                   "prompt_schedule_absolute_step": (
                       step if prompt_schedule is not None else None),
                   "scheduled_prompt_groups": (
                       len(scheduled_global_picks)
                       if scheduled_global_picks is not None else None),
                   "prompt_repeats_per_task":
                       args.prompt_repeats_per_task,
                   "g": args.g,
                   "concurrency": args.concurrency,
                   "effective_clip_low": args.clip_low,
                   "effective_clip_high": args.clip_high,
                   "effective_clip_c": args.clip_c,
                   "effective_overlong_penalty": args.overlong_penalty,
                   "effective_dynamic_sampling_max_rounds":
                       args.dynamic_sampling_max_rounds,
                   "effective_rollout_timeout_s": args.rollout_timeout,
                   "activation_offload": args.activation_offload,
                   "require_clean_rollout_batch":
                       require_clean_rollout_batch,
                   "allow_timeout_censoring":
                       allow_timeout_censoring,
                   "minimum_rollout_complete_groups":
                       minimum_rollout_complete_groups,
                   "norm_adv_by_std_in_grpo": args.algo == "grpo",
                   "effective_loss_agg_mode": args.loss_agg_mode,
                   "variance_filter_enabled": args.algo == "dapo",
                   "has_nonzero_advantage":
                       global_has_nonzero_advantage,
                   "zero_advantage_batch":
                       not global_has_nonzero_advantage,
                   "kl_coef": 0.0,
                   "optimizer_resume_status": opt_resume_status,
                   "source_adapter": source_adapter,
                   "loss": round(loss_val, 5),
                   "adv_abs_mean": round(adv_val, 4), "clipfrac": round(clipfrac, 4),
                   "grad_norm": round(grad_norm, 4),
                   # P1 fix health: behavior_ratio = mean exp(Ï€_current-Ï€_behavior) over
                   # action tokens. The bug (old_logp=logp.detach) pins this at exactly
                   # 1.0; a live run now shows the real train/infer gap (ratio != 1.0).
                   "behavior_ratio": round(behavior_ratio, 4),
                   "behavior_proximal_diff": round(behavior_proximal_diff, 4),
                   "global_reward_mean": round(g_rmean, 4),
                   "reward_std": round(rvar ** 0.5, 4),
                   # variance-gate health: fraction of rolled ROLLOUTS that survived the
                   # zero-variance/singleton filter and produced gradient (local_traj is the
                   # token-replay segment-row count those rollouts expand into).
                   "n_rollouts": global_n_raw, "n_kept": global_n_kept,
                   "kept_frac": round(global_n_kept / max(global_n_raw, 1), 3),
                   "rollout_unfinished": global_rollout_unfinished,
                   "rollout_timed_out_samples":
                       global_rollout_timed_out_samples,
                   "rollout_scheduled_samples": global_rollout_scheduled,
                   "rollout_completed_frac": round(
                       global_rollout_completed_frac, 4),
                   "rollout_complete_groups":
                       global_rollout_complete_groups,
                   "rollout_informative_groups":
                       global_rollout_informative_groups,
                   "rollout_incomplete_groups":
                       global_rollout_incomplete_groups,
                   "rollout_incomplete_samples":
                       global_rollout_incomplete_samples,
                   "rollout_surviving_complete_groups":
                       global_training_complete_groups,
                   "rollout_surviving_informative_groups":
                       global_training_informative_groups,
                   "rollout_surviving_partial_groups":
                       global_training_partial_groups,
                   "training_complete_groups":
                       global_update_complete_groups,
                   "training_informative_groups":
                       global_update_informative_groups,
                   "training_partial_groups":
                       global_update_partial_groups,
                   "max_seq_len": int(T_global), "swap_ok": swap_ok,
                   "local_traj": n_local, "global_traj": global_n_segments,
                   # Signal-to-update conversion and policy displacement.
                   "rows_pre_cap": n_rows_pre_cap,
                   "local_update_rows": n_local,
                   "global_update_rows": global_update_rows,
                   "global_update_rollouts": global_update_rollouts,
                   "global_update_groups": global_update_groups,
                   "global_update_complete_groups":
                       global_update_complete_groups,
                   "global_update_informative_groups":
                       global_update_informative_groups,
                   "global_update_partial_groups":
                       global_update_partial_groups,
                   "rollout_conversion": round(
                       global_update_rollouts / max(global_n_kept, 1), 3),
                   "lora_param_norm": round(lora_param_norm, 4),
                   "sys_err": global_sys_err,
                   "peak_vram_gib": round(peak, 2), "step_s": round(step_dt, 1),
                   "rollout_s": round(roll_dt, 1), "update_s": round(t_update_dt, 1),
                   "did_update": bool(any_update),
                   "local_did_update": bool(did_update),
                   # A no-update step still has an explicit current policy so
                   # the durable consumer can carry it forward without
                   # fabricating a checkpoint.
                   "lora_name": current_lora_name or args.model_name}
            with open(out_metrics, "a") as f:
                f.write(json.dumps(rec) + "\n")
            wandb_log.log_train(step, rec)
            log(rank, world, local,
                f"=== STEP {step} DONE: algo={args.algo} loss={loss_val:.5f} "
                f"g_reward={g_rmean:.3f} "
                f"grad_norm={grad_norm:.3f} kept={n_kept_roll}/{n_raw} "
                f"peak={peak:.2f}GiB t_step={step_dt:.0f}s lora={current_lora_name} ===")
        else:
            # non-rank0 ranks still participate in the reward all-reduce above (SAME group
            # as the rank-0 branch, or the collective deadlocks). Use the same
            # n_raw weighting.
            rmean_t = torch.tensor([rmean * n_raw, float(n_raw)], device=device)
            if red_world > 1:
                dist.all_reduce(rmean_t, op=dist.ReduceOp.SUM, group=red_group)

        if world > 1:
            dist.barrier()

        # ---- periodic validation (opt-in via --eval-list + --eval-every): rank-0 rolls the
        #      held-out set against the current adapter; other ranks wait at the barrier.
        #      Gives a real generalization signal during training (logged to wandb eval/*). ----
        if eval_cases and current_lora_name and args.eval_every > 0 \
                and ((step + 1) % args.eval_every == 0):
            if is_rank0(rank):
                eval_row = run_eval(
                    step + 1,
                    tb_cfg,
                    eval_cases,
                    current_lora_name,
                    reward_fn=args.reward_fn,
                    timeout_s=args.eval_timeout,
                    mcp_urls=args.mcp_urls,
                )
                eval_row["max_agent_turns"] = (
                    args.eval_max_agent_turns
                )
                eval_row["timeout_s"] = args.eval_timeout
                if eval_metrics_out is not None:
                    with open(eval_metrics_out, "a") as f:
                        f.write(json.dumps(eval_row) + "\n")
                wandb_log.log_eval(step + 1, eval_row)
                log(rank, world, local,
                    f"VAL step={step + 1} pass={eval_row.get('pass_rate')} "
                    f"reward_mean={eval_row.get('reward_mean')} n={eval_row.get('n_cases')}")
            if world > 1:
                dist.barrier()
        step += 1

    if is_rank0(rank):
        log(rank, world, local, "=== TRAINING COMPLETE ===")
        wandb_log.finish()
    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
