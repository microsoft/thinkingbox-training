"""FSDP2 LoRA adapter gather and vLLM hot-swap.

Under FSDP2 (`torch.distributed.fsdp.fully_shard`) the LoRA params are DTensors
sharded across the train ranks. To save a peft adapter that vLLM can load via
/load_lora_adapter, we must materialize the FULL (unsharded) tensors. We follow
verl's own pattern (verl.utils.fsdp_utils.collect_lora_params, non-layered branch):

    sd = peft.utils.save_and_load.get_peft_model_state_dict(peft_model)   # DTensor values
    sd = {k: v.full_tensor() if hasattr(v, "full_tensor") else v ...}     # all-gather

`.full_tensor()` is a COLLECTIVE — EVERY train rank must call it. Local rank 0
on every node writes a resumable checkpoint; designated vLLM writers POST the
node-local copy to their rollout server. The adapter_config.json is static
across steps, and adapter_model.safetensors is rewritten each step with the
gathered weights. This reuses train/lora_sync.py::VLLMLoraClient for the HTTP
bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM

logger = logging.getLogger("train.fsdp_lora_sync")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_checksum_evidence(adapter_dir: str | Path) -> dict:
    """Return immutable size/SHA-256 evidence for a PEFT adapter directory."""
    root = Path(adapter_dir).resolve()
    files = {}
    combined = hashlib.sha256()
    total_bytes = 0
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"adapter evidence missing {path}")
        size = path.stat().st_size
        sha256 = _sha256_file(path)
        files[name] = {"bytes": size, "sha256": sha256}
        total_bytes += size
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(sha256))
    return {
        "path": str(root),
        "total_bytes": total_bytes,
        "sha256": combined.hexdigest(),
        "files": files,
    }


def _write_swap_evidence(adapter_dir: Path, evidence: dict) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    dst = adapter_dir / "swap_evidence.json"
    tmp = adapter_dir / "swap_evidence.json.tmp"
    tmp.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, dst)


def gather_lora_full_state_dict(peft_model) -> dict:
    """All-gather the FSDP2-sharded LoRA adapter into a full CPU state_dict.

    Collective: ALL ranks must call this (each .full_tensor() all-gathers a DTensor).
    Returns the peft-format adapter state_dict on every rank (small: ~80M params).
    """
    sd = get_peft_model_state_dict(peft_model)
    out = {}
    for k, v in sd.items():
        if hasattr(v, "full_tensor"):          # DTensor -> all-gather to full
            v = v.full_tensor()
        out[k] = v.detach().to(torch.bfloat16).cpu()
    return out


def _remap_lora_keys_for_vllm(sd: dict) -> tuple[dict, int]:
    """Rewrite text-model PEFT keys for multimodal-wrapper vLLM serving.

    A conditional-generation wrapper nests the text model under
    ``language_model.``, so its LoRA modules are named
    ``language_model.model.layers.N.<proj>``. Our peft adapter is built on the TEXT-ONLY
    ``Qwen3_5ForCausalLM``, so its keys are ``base_model.model.model.layers.N.<proj>.lora_*``.
    Inserting ``language_model.`` makes the parsed name match the served text
    modules, including packed Gated-DeltaNet projections. Set
    ``VLLM_LORA_MODULE_PREFIX=""`` only for a verified text-only serving path.
    """
    prefix = os.environ.get("VLLM_LORA_MODULE_PREFIX", "language_model.")
    if not prefix:
        return sd, 0
    old = "base_model.model.model."
    new = f"base_model.model.{prefix}model."
    out, n = {}, 0
    for k, v in sd.items():
        if k.startswith(old):
            k = new + k[len(old):]
            n += 1
        out[k] = v
    return out, n


def _remap_lora_keys_for_training(sd: dict) -> tuple[dict, int]:
    """Undo the multimodal vLLM namespace rewrite for the text-only trainer."""
    prefix = os.environ.get("VLLM_LORA_MODULE_PREFIX", "language_model.")
    if not prefix:
        return sd, 0
    old = f"base_model.model.{prefix}model."
    new = "base_model.model.model."
    out, n = {}, 0
    for k, v in sd.items():
        if k.startswith(old):
            k = new + k[len(old):]
            n += 1
        out[k] = v
    return out, n


def load_vllm_adapter_for_training(peft_model, adapter_dir: str | Path) -> int:
    """Restore a saved vLLM-formatted adapter into the text-only PEFT model.

    Call before ``fully_shard`` so PEFT can load ordinary, unsharded parameters.
    """
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"resume adapter config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"resume adapter weights not found: {weights_path}")

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    live_config = peft_model.peft_config["default"]
    compatibility = {
        "r": (int(saved_config.get("r", -1)), int(live_config.r)),
        "lora_alpha": (
            float(saved_config.get("lora_alpha", float("nan"))),
            float(live_config.lora_alpha),
        ),
        "target_modules": (
            set(saved_config.get("target_modules") or []),
            set(live_config.target_modules or []),
        ),
        "base_model_name_or_path": (
            str(saved_config.get("base_model_name_or_path") or ""),
            str(live_config.base_model_name_or_path or ""),
        ),
    }
    mismatches = {
        name: values for name, values in compatibility.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(
            f"resume adapter config mismatch for {adapter_dir}: {mismatches}")

    state = load_file(str(weights_path), device="cpu")
    state, n_remap = _remap_lora_keys_for_training(state)
    result = set_peft_model_state_dict(peft_model, state, adapter_name="default")
    unexpected = getattr(result, "unexpected_keys", []) or []
    missing = getattr(result, "missing_keys", []) or []
    if unexpected:
        raise RuntimeError(
            f"unexpected resume LoRA keys from {adapter_dir}: {unexpected[:5]} "
            f"({len(unexpected)} total)")
    lora_missing = [k for k in missing if "lora_A" in k or "lora_B" in k]
    if lora_missing:
        raise RuntimeError(
            f"resume LoRA keys missing after load: {lora_missing[:5]} "
            f"({len(lora_missing)} total)")
    if not state:
        raise RuntimeError(f"resume adapter is empty: {weights_path}")
    if not any(torch.count_nonzero(v).item() for k, v in state.items() if "lora_B" in k):
        raise RuntimeError(f"resume adapter has no nonzero LoRA-B weights: {weights_path}")
    logger.info(
        "restored %d LoRA tensors from %s (reversed %d vLLM namespace keys)",
        len(state), adapter_dir, n_remap)
    return len(state)


def write_adapter(peft_model, full_sd: dict, out_dir: str | Path) -> str:
    """rank-0 only: write adapter_config.json + adapter_model.safetensors.

    Produces the exact peft on-disk layout vLLM's /load_lora_adapter expects, with keys
    remapped into the served model's ``language_model.`` namespace (see
    ``_remap_lora_keys_for_vllm``) so the multimodal vLLM server APPLIES the adapter instead
    of silently dropping it. Returns the absolute path.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = peft_model.peft_config["default"]
    # adapter_model.safetensors — the gathered full LoRA weights, key-remapped for vLLM.
    full_sd, n_remap = _remap_lora_keys_for_vllm(full_sd)
    print(f"[vllm-remap] adapter {out.name}: remapped {n_remap}/{len(full_sd)} keys into "
          f"language_model.* namespace so vLLM applies them", flush=True)
    # Stage both files, then publish each with an atomic rename. This directory is
    # read concurrently by the Blob sync daemon and, on a later run, by a resuming
    # peer on ANOTHER node: a half-written multi-hundred-MB safetensors file is
    # indistinguishable from a complete one until the tensor load fails deep inside
    # a collective. Staging lives beside the target so the rename stays same-filesystem.
    need = ("adapter_config.json", "adapter_model.safetensors")
    with tempfile.TemporaryDirectory(
            dir=str(out.parent), prefix=f".{out.name}.staging.") as staging:
        stage = Path(staging)
        cfg.save_pretrained(str(stage))   # writes adapter_config.json only
        save_file(full_sd, str(stage / "adapter_model.safetensors"),
                  metadata={"format": "pt"})
        staged_missing = [name for name in need if not (stage / name).is_file()]
        if staged_missing:
            raise RuntimeError(
                f"FSDP2 adapter staging missing {staged_missing}; "
                f"got {sorted(p.name for p in stage.iterdir())}")
        for name in need:
            os.replace(stage / name, out / name)
    # sanity (mirrors train/lora_sync.save_adapter)
    have = {p.name for p in out.iterdir()}
    missing = set(need) - have
    if missing:
        raise RuntimeError(f"FSDP2 adapter write missing {missing}; got {sorted(have)}")
    return str(out)


def _is_vllm_writer(rank: int) -> bool:
    """Return whether this rank owns a vLLM adapter hot-swap.

    ``VLLM_WRITER_NODE=all`` designates local rank 0 on every trainer node.
    This is used when each node has its own rollout server: every designated
    writer hot-swaps that node's vLLM before the world barrier. Adapter
    checkpoint files are written separately by local rank 0 on every node.

    The existing 1/0 contract remains unchanged for one-vLLM multi-node runs.
    """
    mode = os.environ.get("VLLM_WRITER_NODE")
    if mode is None:
        return rank == 0
    if mode == "all":
        return int(os.environ.get("LOCAL_RANK", "0")) == 0
    if mode not in {"0", "1"}:
        raise ValueError(
            "VLLM_WRITER_NODE must be unset, '0', '1', or 'all'; "
            f"got {mode!r}")
    return mode == "1" and int(os.environ.get("LOCAL_RANK", "0")) == 0


def _is_checkpoint_writer(rank: int) -> bool:
    """Write a resumable adapter copy on local rank 0 of every node."""
    return int(os.environ.get("LOCAL_RANK", str(rank))) == 0


def _validate_vllm_writer_assignments(assignments: list[tuple]) -> None:
    """Require exactly one writer globally, or one per host in `all` mode."""
    if not assignments:
        raise RuntimeError("vLLM writer assignment list is empty")
    all_mode = any(mode == "all" for _, _, mode, _ in assignments)
    if all_mode:
        if any(mode != "all" for _, _, mode, _ in assignments):
            raise RuntimeError(
                "VLLM_WRITER_NODE=all must be set consistently on every rank")
        hosts = {host for host, _, _, _ in assignments}
        writer_hosts = [host for host, is_writer, _, _ in assignments
                        if is_writer]
        bad = {
            host: writer_hosts.count(host)
            for host in hosts
            if writer_hosts.count(host) != 1
        }
        if bad:
            raise RuntimeError(
                f"VLLM_WRITER_NODE=all requires one writer per host; got {bad}")
        nonlocal_urls = {
            host: base_url
            for host, is_writer, _, base_url in assignments
            if is_writer
            and urlparse(base_url).hostname not in {
                "127.0.0.1", "localhost", "::1"}
        }
        if nonlocal_urls:
            raise RuntimeError(
                "VLLM_WRITER_NODE=all requires each writer's "
                f"--vllm-base-url to be loopback-local; got {nonlocal_urls}")
    else:
        writers = sum(
            bool(is_writer) for _, is_writer, _, _ in assignments)
        if writers != 1:
            raise RuntimeError(
                f"single-vLLM topology requires exactly one writer; got {writers}")


def validate_vllm_writer_topology(
        rank: int, world_size: int, vllm_base_url: str) -> None:
    """Collectively validate writer coverage before the first rollout."""
    local = (
        socket.gethostname(),
        _is_vllm_writer(rank),
        os.environ.get("VLLM_WRITER_NODE"),
        vllm_base_url,
    )
    if world_size > 1:
        assignments = [None] * world_size
        dist.all_gather_object(assignments, local)
    else:
        assignments = [local]
    _validate_vllm_writer_assignments(assignments)


def fsdp_sync_vllm_lora(peft_model, lora_client, lora_save_dir: str,
                        lora_name: str, prev_lora_name, rank: int,
                        world_size: int) -> bool:
    """Gather the FSDP2 LoRA adapter, save it, hot-swap into vLLM. Returns ok.

    All ranks enter the gather collective. Local rank 0 on every node writes a
    resumable adapter checkpoint; only the designated vLLM writer (see
    VLLM_WRITER_NODE below) POSTs it, then unloads the previous adapter after
    the replacement becomes active.
    """
    t0 = time.monotonic()
    full_sd = gather_lora_full_state_dict(peft_model)   # COLLECTIVE (all ranks)
    # The adapter must be POSTed from a process co-located with the target
    # vLLM server, because vLLM's /load_lora_adapter reads a LOCAL filesystem path.
    # In a multi-node run the vLLM policy server lives on ONE node (node0), but under c10d
    # dynamic rendezvous global rank 0 can land on a DIFFERENT node whose local disk vLLM
    # cannot read -> every hot-swap fails (and the adapter never lands where the blob-sync
    # watcher looks). Designate the writer explicitly as local_rank 0 on the vLLM node via
    # VLLM_WRITER_NODE=1 (set that on the vLLM node's launch; VLLM_WRITER_NODE=0 elsewhere).
    # Set VLLM_WRITER_NODE=all when every trainer node owns a node-local rollout vLLM.
    # Falls back to global rank 0 when the env is unset (single-node runs, where rank 0 IS
    # the vLLM node). full_sd is identical on every rank (all-gather above), so any rank can
    # hot-swap it safely; this changes only WHICH rank does the POST, no collective is moved.
    is_writer = _is_vllm_writer(rank)
    is_checkpoint_writer = _is_checkpoint_writer(rank)
    out = (Path(lora_save_dir) / lora_name).resolve()
    path = None
    ok = True
    if is_checkpoint_writer:
        # Trainer filesystems are node-local in the supported pod topology.
        # Every node therefore needs its own adapter copy for a later resume,
        # even when only one node owns the rollout vLLM and performs the POST.
        try:
            path = write_adapter(peft_model, full_sd, out)
        except Exception as checkpoint_error:
            ok = False
            logger.error(
                "failed to write node-local adapter checkpoint %s (%s: %s)",
                out, type(checkpoint_error).__name__,
                str(checkpoint_error)[:300])
    if is_writer and ok:
        evidence = {
            "schema": "thinkingbox.lora_swap_evidence.v1",
            "policy_version": lora_name,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "status": "started",
            "vllm_base_url": lora_client.base_url,
        }
        loaded_new = False
        try:
            if path is None:
                # Defensive fallback for unusual rank mappings where the vLLM
                # writer is not local rank 0.
                path = write_adapter(peft_model, full_sd, out)
            before = adapter_checksum_evidence(path)
            evidence["adapter_before_load"] = before
            evidence["vllm_version_before"] = lora_client.server_version()
            # Serialize the swap against in-flight decode. Recorded here (not just
            # inside the client) so swap_evidence.json shows whether the engine was
            # actually quiescent when we mutated it. This MUST go through the
            # client's retrying drain: `wait_for_idle` surfaces the first failed
            # /metrics scrape as `metrics_unavailable`, so gating on it directly
            # turned one transient HTTP blip into a hard abort of every rank even
            # though the bounded retry budget inside `drain` would have cleared it.
            evidence["drain_before_load"] = lora_client.drain(
                f"load_lora_adapter({lora_name})")
            if not evidence["drain_before_load"].get("drained"):
                raise RuntimeError(
                    "refusing LoRA mutation without a confirmed idle vLLM: "
                    f"{evidence['drain_before_load']}")
            load_t0 = time.monotonic()
            lora_client.load_adapter(lora_name, path)
            loaded_new = True
            evidence["load_latency_s"] = round(time.monotonic() - load_t0, 6)
            evidence.update(
                lora_client.verify_adapter(
                    lora_name, path, active_probe=True))
            after = adapter_checksum_evidence(path)
            if after["sha256"] != before["sha256"]:
                raise RuntimeError(
                    f"adapter checksum changed while loading {lora_name!r}: "
                    f"{before['sha256']} -> {after['sha256']}")
            evidence["adapter_after_load"] = after
            evidence["vllm_version_after"] = lora_client.server_version()
            if evidence["vllm_version_after"] != evidence["vllm_version_before"]:
                raise RuntimeError(
                    "vLLM version changed during adapter swap: "
                    f"{evidence['vllm_version_before']} -> "
                    f"{evidence['vllm_version_after']}")
            if prev_lora_name and prev_lora_name != lora_name:
                try:
                    lora_client.unload_adapter(prev_lora_name)
                except Exception as e:  # best effort
                    logger.warning("unload prev adapter %r failed: %s", prev_lora_name, e)
            evidence["status"] = "verified"
            evidence["verified_at"] = datetime.now(timezone.utc).isoformat()
            evidence["total_latency_s"] = round(time.monotonic() - t0, 6)
            evidence["tensor_count"] = len(full_sd)
            _write_swap_evidence(out, evidence)
            logger.info("SWAP_EVIDENCE %s", json.dumps(evidence, sort_keys=True))
            logger.info("fsdp_sync_vllm_lora adapter=%s: gathered+loaded in %.1fs "
                        "(%d tensors)", lora_name, time.monotonic() - t0, len(full_sd))
        except Exception as e:
            ok = False
            evidence["status"] = "failed"
            evidence["failed_at"] = datetime.now(timezone.utc).isoformat()
            evidence["error_type"] = type(e).__name__
            evidence["error"] = str(e)[:1000]
            evidence["total_latency_s"] = round(time.monotonic() - t0, 6)
            if loaded_new:
                try:
                    lora_client.unload_adapter(lora_name)
                    evidence["failed_adapter_unloaded"] = True
                except Exception as unload_error:
                    evidence["failed_adapter_unloaded"] = False
                    evidence["unload_error"] = str(unload_error)[:1000]
            try:
                _write_swap_evidence(out, evidence)
            except Exception as evidence_error:
                logger.error(
                    "failed to persist swap failure evidence for %s (%s: %s)",
                    lora_name, type(evidence_error).__name__,
                    str(evidence_error)[:300])
            logger.error("SWAP_EVIDENCE %s", json.dumps(evidence, sort_keys=True))
            logger.error("fsdp_sync_vllm_lora FAILED (%s: %s); rollouts use stale adapter",
                         type(e).__name__, str(e)[:300])
    del full_sd
    if world_size > 1:
        dist.barrier()
    return ok


if __name__ == "__main__":
    # Single-GPU self-test: build LoRA, FSDP2-shard, gather, write, verify shapes.
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/fsdp_lora_selftest")
    args = ap.parse_args()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=dev)

    m = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    lc = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    task_type="CAUSAL_LM")
    pm = get_peft_model(m, lc)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    base = getattr(pm, "base_model").model
    inner = getattr(base, "model", base)
    for layer in getattr(inner, "layers", []):
        fully_shard(layer, mp_policy=mp)
    fully_shard(pm, mp_policy=mp)
    print("FSDP2 sharded; gathering LoRA...", flush=True)
    sd = gather_lora_full_state_dict(pm)
    print(f"gathered {len(sd)} lora tensors; sample keys:",
          list(sd.keys())[:3], flush=True)
    path = write_adapter(pm, sd, args.out)
    print(f"wrote adapter to {path}", flush=True)
    for f in os.listdir(path):
        print("  ", f, os.path.getsize(os.path.join(path, f)), "bytes")
    dist.destroy_process_group()
