"""LoRA target definitions and vLLM runtime adapter client."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Qwen3-style attention + MLP modules — matches the architecture used at
# inference. Update if the base model changes.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# --- Gated-DeltaNet (GDN / linear-attention) LoRA targets --------------------------
# The default profile above adapts only the softmax-attention + MLP Linears, so on a
# hybrid model it leaves the token-mixing of every linear-attention (GDN) layer FROZEN.
# This profile additionally adapts the GDN input/output PROJECTIONS. It is grounded in
# external prior art for LoRA on linear-attention / SSM layers (not model-specific
# guesswork):
#   * Axolotl's official Qwen3-Next QLoRA recipe targets the GDN projections
#     `linear_attn.in_proj_qkvz`, `linear_attn.in_proj_ba`, `linear_attn.out_proj`
#     (examples/qwen3-next/qwen3-next-80b-a3b-qlora.yaml).
#   * CVPR'25 "Parameter-Efficient Mamba Tuning via Projector-targeted ... Transformation"
#     finds the PROJECTORS (in_proj/out_proj), NOT the recurrent SSM core, dominate transfer
#     — so we LoRA the Linear projections and leave the state dynamics frozen.
#   * HarmoniqOS/ssm-aware-lora-finetuning warns that the vanilla q/k/v/o + MLP target list
#     SILENTLY skips SSM layers; the projections must be added explicitly.
# The recurrent-state parameters are NOT Linear (Conv1d / raw nn.Parameter / gated RMSNorm)
# and stay frozen — PEFT cannot wrap them and CVPR'25 says adapting them doesn't help:
GDN_FROZEN_MODULES: tuple[str, ...] = ("conv1d", "A_log", "dt_bias", "norm")
# GDN projection names. We include BOTH naming conventions so the profile is correct
# regardless of which base weights are loaded — PEFT only wraps modules that actually
# exist and q/k/v/o always match, so names absent from a given arch are harmless no-ops:
#   * stock HF Qwen3-Next keeps the input projection FUSED: in_proj_qkvz + in_proj_ba;
#   * our internal Qwen3.6-27B fork UN-FUSES it into split Linears: in_proj_qkv/z/a/b
#     (see train/sp_ulysses.py::gdn_sp_forward, which calls each split projection).
GDN_PROJECTION_MODULES: tuple[str, ...] = (
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",  # our fork (split)
    "in_proj_qkvz", "in_proj_ba",                          # stock HF Qwen3-Next (fused)
    "out_proj",                                            # GDN output (both)
)
GDN_HYBRID_TARGET_MODULES: tuple[str, ...] = (
    *DEFAULT_TARGET_MODULES, *GDN_PROJECTION_MODULES,
)

# Default to the GDN-hybrid profile: on our hybrid model the attention_mlp set
# leaves every Gated-DeltaNet layer's token-mixing frozen, so plain LoRA RL can
# only move a minority of the token mixers. gdn_hybrid is the intended default;
# opt back to attention-only with --lora-target-profile attention_mlp.
DEFAULT_TARGET_PROFILE = "gdn_hybrid"
LORA_TARGET_PROFILES: dict[str, tuple[str, ...]] = {
    "attention_mlp": DEFAULT_TARGET_MODULES,
    "gdn_hybrid": GDN_HYBRID_TARGET_MODULES,
}


def resolve_lora_target_modules(profile: str) -> tuple[str, ...]:
    """Map a --lora-target-profile name to its LoRA target module tuple.

    Fails closed on an unknown profile so a typo can never silently fall back to a
    different (e.g. GDN-frozen) target set mid-experiment.
    """
    try:
        return LORA_TARGET_PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown LoRA target profile {profile!r}; "
            f"choose from {sorted(LORA_TARGET_PROFILES)}"
        ) from None


@dataclass
class VLLMLoraClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 60.0
    # Drain in-flight generation before mutating the adapter registry. See
    # `wait_for_idle` for why this is the difference between a run that survives
    # overnight and one that wedges the engine. Disable only for a single-request
    # debug server where nothing else can be generating.
    drain_before_mutation: bool = True
    require_successful_drain: bool = True
    drain_timeout: float = 180.0
    drain_poll: float = 0.5
    drain_metrics_retries: int = 3

    # Gauges vLLM exports for queue occupancy. Matched exactly, so the sibling
    # series `vllm:num_requests_waiting_by_reason` is correctly ignored.
    _IDLE_GAUGES = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    )

    def in_flight(self) -> Optional[int]:
        """Running+waiting requests summed over engines, or None if unavailable.

        Returns None (rather than raising) when /metrics cannot be scraped, so a
        stats-disabled server degrades to the old un-drained behaviour instead of
        halting training.
        """
        try:
            r = requests.get(f"{self.base_url}/metrics", timeout=self.timeout)
            r.raise_for_status()
            body = r.text
        except Exception as exc:
            logger.warning("vLLM /metrics unavailable (%s: %s); cannot drain",
                           type(exc).__name__, str(exc)[:200])
            return None
        total = 0.0
        seen = False
        for line in body.splitlines():
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name not in self._IDLE_GAUGES:
                continue
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            seen = True
        if not seen:
            logger.warning("vLLM /metrics exposed no %s gauges; cannot drain",
                           " / ".join(self._IDLE_GAUGES))
            return None
        return int(total)

    def wait_for_idle(self, *, timeout: Optional[float] = None,
                      poll: Optional[float] = None) -> dict:
        """Block until vLLM reports zero in-flight requests; return evidence.

        vLLM adapter mutations and decode steps both execute tensor-parallel
        collectives. Draining the request queue serializes those operations so
        ranks cannot enter incompatible collectives concurrently.

        ``wait_for_idle`` reports failures as evidence. Registry mutations call it
        through ``_drain`` and fail closed by default: proceeding without confirmed
        idleness reintroduces the TP collective race this guard exists to prevent.
        Set ``require_successful_drain=False`` only for a controlled debug server.
        """
        timeout = self.drain_timeout if timeout is None else timeout
        poll = self.drain_poll if poll is None else poll
        t0 = time.monotonic()
        observed = self.in_flight()
        if observed is None:
            return {"drained": False, "reason": "metrics_unavailable",
                    "waited_s": 0.0}
        while observed > 0 and (time.monotonic() - t0) < timeout:
            time.sleep(poll)
            observed = self.in_flight()
            if observed is None:
                return {"drained": False, "reason": "metrics_unavailable",
                        "waited_s": round(time.monotonic() - t0, 3)}
        waited = round(time.monotonic() - t0, 3)
        if observed > 0:
            logger.warning(
                "vLLM still reported %d in-flight request(s) after %.1fs; "
                "proceeding with the LoRA mutation un-drained", observed, waited)
            return {"drained": False, "reason": "timeout",
                    "in_flight": observed, "waited_s": waited}
        return {"drained": True, "in_flight": 0, "waited_s": waited}

    def _drain(self, op: str) -> dict:
        if not self.drain_before_mutation:
            return {"drained": False, "reason": "disabled", "waited_s": 0.0}
        evidence = self.wait_for_idle()
        for _ in range(self.drain_metrics_retries):
            if evidence.get("reason") != "metrics_unavailable":
                break
            time.sleep(self.drain_poll)
            evidence = self.wait_for_idle()
        if self.require_successful_drain and not evidence.get("drained"):
            raise RuntimeError(
                f"refusing {op} without a confirmed idle vLLM: {evidence}")
        if evidence.get("waited_s", 0.0) > 0:
            logger.info("drained vLLM before %s: %s", op, evidence)
        return evidence

    def drain(self, op: str) -> dict:
        """Retrying drain for callers that record their own quiescence evidence.

        Any caller gating a registry mutation MUST use this rather than
        `wait_for_idle`. `wait_for_idle` reports the FIRST failed `/metrics`
        scrape as `metrics_unavailable` with no retry, and a single connection
        reset against a healthy engine is an ordinary transient over a multi-day
        run. Only this path spends the bounded `drain_metrics_retries` budget
        before failing closed, so one blip cannot abort a run that the very next
        scrape would have cleared.
        """
        return self._drain(op)

    def model_cards(self) -> list[dict]:
        """Return vLLM's raw model registry, including LoRA path metadata."""
        r = requests.get(f"{self.base_url}/v1/models", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    def list_models(self) -> list[str]:
        return [m["id"] for m in self.model_cards()]

    def server_version(self) -> str:
        r = requests.get(f"{self.base_url}/version", timeout=self.timeout)
        r.raise_for_status()
        version = r.json().get("version")
        if not version:
            raise RuntimeError(f"vLLM /version omitted version: {r.text}")
        return str(version)

    def load_adapter(self, lora_name: str, lora_path: str) -> None:
        """POST /v1/load_lora_adapter; raises on non-2xx.

        Drains in-flight generation first (see `wait_for_idle`). Deliberately NOT
        retried: a rejected load can leave a vLLM 0.24 worker in a state where the
        next load of the same name also 500s, so a blind retry converts one
        recoverable failure into a wedged registry. Callers fail closed and let the
        watchdog restart the server; the trainer's pre-rollout `ensure_adapter`
        gate then re-loads the adapter onto the fresh engine.
        """
        url = f"{self.base_url}/v1/load_lora_adapter"
        payload = {
            "lora_name": lora_name,
            "lora_path": str(Path(lora_path).resolve()),
            # A trainer may retry a step after the adapter was loaded but
            # before its checkpoint/metric committed. Replace that stale
            # same-name registry entry atomically instead of failing 400 or
            # accepting the stale weights through ensure_adapter.
            "load_inplace": True,
        }
        self._drain(f"load_lora_adapter({lora_name})")
        logger.info("vLLM load_lora_adapter: %s <- %s", lora_name, payload["lora_path"])
        r = requests.post(url, json=payload, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(
                f"load_lora_adapter failed [{r.status_code}]: {r.text}"
            )

    def unload_adapter(self, lora_name: str) -> None:
        url = f"{self.base_url}/v1/unload_lora_adapter"
        self._drain(f"unload_lora_adapter({lora_name})")
        logger.info("vLLM unload_lora_adapter: %s", lora_name)
        r = requests.post(url, json={"lora_name": lora_name}, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(
                f"unload_lora_adapter failed [{r.status_code}]: {r.text}"
            )

    def verify_adapter(
        self,
        lora_name: str,
        lora_path: str | os.PathLike,
        *,
        active_probe: bool = False,
    ) -> dict:
        """Verify registry identity and optionally execute one LoRA inference token."""
        cards = [card for card in self.model_cards() if card.get("id") == lora_name]
        if len(cards) != 1:
            raise RuntimeError(
                f"vLLM registry has {len(cards)} entries for adapter {lora_name!r}")
        card = cards[0]
        expected_root = str(Path(lora_path).resolve())
        served_root = str(Path(card.get("root", "")).resolve())
        if served_root != expected_root:
            raise RuntimeError(
                f"vLLM registry path mismatch for {lora_name!r}: "
                f"served={served_root!r} expected={expected_root!r}")

        evidence = {"served_model_card": card}
        if active_probe:
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": lora_name,
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "temperature": 0,
                    "max_tokens": 1,
                },
                timeout=self.timeout,
            )
            if not r.ok:
                raise RuntimeError(
                    f"active LoRA probe failed [{r.status_code}]: {r.text}")
            body = r.json()
            if body.get("model") != lora_name or not body.get("choices"):
                raise RuntimeError(
                    f"active LoRA probe did not execute {lora_name!r}: {body}")
            evidence["active_probe"] = {
                "model": body["model"],
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage"),
            }
        return evidence

    def ensure_adapter(
        self,
        lora_name: str,
        lora_path: str | os.PathLike,
        *,
        active_probe: bool = False,
    ) -> dict:
        """Load a missing adapter after a vLLM restart, then verify it exactly."""
        cards = [card for card in self.model_cards() if card.get("id") == lora_name]
        if not cards:
            self.load_adapter(lora_name, str(lora_path))
        return self.verify_adapter(
            lora_name, lora_path, active_probe=active_probe)
