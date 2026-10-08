# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Compatibility hooks required by the validated Qwen3.8 full-RLFT runtime."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def _init_on_device_preserving_hf_marker(
    device: torch.device, include_buffers: bool | None = None
):
    from accelerate.utils import parse_flag_from_env

    if include_buffers is None:
        include_buffers = parse_flag_from_env(
            "ACCELERATE_INIT_INCLUDE_BUFFERS", False
        )
    if include_buffers:
        with device:
            yield
        return

    original = nn.Module.register_parameter

    def register_parameter(module, name, parameter):
        original(module, name, parameter)
        if parameter is None:
            return
        current = module._parameters[name]
        parameter_type = type(current)
        kwargs = dict(current.__dict__)
        kwargs["requires_grad"] = parameter.requires_grad
        hf_initialized = kwargs.pop("_is_hf_initialized", None)
        module._parameters[name] = parameter_type(current.to(device), **kwargs)
        if hf_initialized is not None:
            module._parameters[name]._is_hf_initialized = hf_initialized

    try:
        nn.Module.register_parameter = register_parameter
        yield
    finally:
        nn.Module.register_parameter = original


def _has_multimodal_payload(multi_modal_inputs) -> bool:
    if not multi_modal_inputs:
        return False
    modality_keys = {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "input_features",
        "audio_values",
    }
    return any(
        key in modality_keys
        and value is not None
        and (not hasattr(value, "numel") or value.numel() > 0)
        for key, value in multi_modal_inputs.items()
    )


def _patch_verl_agent_loop(module) -> None:
    worker_class = module.AgentLoopWorker
    original = worker_class._compute_position_ids
    if getattr(original, "_thinkingbox_text_position_ids", False):
        return

    def compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs=None,
    ):
        if not _has_multimodal_payload(multi_modal_inputs):
            return module.compute_position_id_with_mask(attention_mask)
        return original(
            self,
            input_ids,
            attention_mask,
            multi_modal_inputs,
            mm_processor_kwargs,
        )

    compute_position_ids._thinkingbox_text_position_ids = True
    worker_class._compute_position_ids = compute_position_ids


class _VerlPatchLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch_verl_agent_loop(module)


class _VerlPatchFinder(importlib.abc.MetaPathFinder):
    _thinkingbox_verl_patch_finder = True

    def find_spec(self, fullname, path, target=None):
        if fullname != "verl.experimental.agent_loop.agent_loop":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _VerlPatchLoader(spec.loader)
        return spec


def _install_verl_patch_hook() -> None:
    module = sys.modules.get("verl.experimental.agent_loop.agent_loop")
    if module is not None:
        _patch_verl_agent_loop(module)
    if not any(
        getattr(finder, "_thinkingbox_verl_patch_finder", False)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _VerlPatchFinder())


def apply_full_runtime_compatibility() -> None:
    """Install idempotent process-local full-RLFT compatibility hooks."""

    import accelerate
    from accelerate import big_modeling

    if not getattr(big_modeling.init_on_device, "_thinkingbox_patched", False):
        _init_on_device_preserving_hf_marker._thinkingbox_patched = True
        big_modeling.init_on_device = _init_on_device_preserving_hf_marker
        accelerate.init_on_device = _init_on_device_preserving_hf_marker
    _install_verl_patch_hook()
