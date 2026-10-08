# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Activate the validated Qwen3.8 runtime hooks in drivers and Ray workers."""

try:
    from trainer.runtime_compat import apply_full_runtime_compatibility
except ModuleNotFoundError as error:
    if error.name not in {"accelerate", "torch"}:
        raise
else:
    apply_full_runtime_compatibility()
