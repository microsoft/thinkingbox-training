# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "patches" / "verl" / "v0.9.0" / "manifest.json"


def git_blob_id(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema") != "thinkingbox-training.verl-patches/v1":
        raise RuntimeError("unsupported Verl patch manifest schema")
    return manifest


def installed_source() -> Path:
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        raise RuntimeError("Verl is not installed")
    return Path(spec.origin).resolve().parent.parent


def verify_source(root: Path, manifest: dict[str, Any]) -> None:
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"installed Verl file is missing: {relative}")
        observed = git_blob_id(path)
        if observed != record["postimage_blob"]:
            raise RuntimeError(
                f"installed Verl file is not patched as expected: {relative}"
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Verify the installed Qwen3.8-compatible Verl source."
    )
    result.add_argument(
        "--source",
        help="Verify a source checkout instead of the installed package.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        manifest = load_manifest()
        if args.source:
            root = Path(args.source).expanduser().resolve()
            version = manifest["tag"].removeprefix("v")
        else:
            root = installed_source()
            version = importlib.metadata.version("verl")
            if version != manifest["tag"].removeprefix("v"):
                raise RuntimeError(
                    f"unexpected Verl version: expected "
                    f"{manifest['tag'].removeprefix('v')}, got {version}"
                )
        verify_source(root, manifest)
    except (OSError, RuntimeError, importlib.metadata.PackageNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "files": len(manifest["files"]),
                "root": str(root),
                "status": "verified",
                "version": version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
