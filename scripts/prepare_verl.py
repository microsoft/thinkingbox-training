from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = PROJECT_ROOT / "patches" / "verl" / "v0.9.0"
MANIFEST_PATH = PATCH_ROOT / "manifest.json"


def fail(message: str) -> RuntimeError:
    return RuntimeError(message)


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise fail(f"command failed ({' '.join(args)}): {detail}")
    return result


def git(
    repository: Path,
    *args: str,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(repository), *args], capture=capture)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_id(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema") != "thinkingbox-training.verl-patches/v1":
        raise fail("unsupported Verl patch manifest schema")
    if not manifest.get("commit") or not manifest.get("upstream"):
        raise fail("Verl patch manifest is missing upstream identity")
    return manifest


def verify_patch_files(manifest: dict[str, Any]) -> list[Path]:
    patches: list[Path] = []
    for record in manifest["patches"]:
        path = PATCH_ROOT / record["path"]
        if not path.is_file():
            raise fail(f"missing Verl patch: {record['path']}")
        observed = sha256(path)
        if observed != record["sha256"]:
            raise fail(
                f"Verl patch hash mismatch for {record['path']}: "
                f"expected {record['sha256']}, got {observed}"
            )
        patches.append(path)
    return patches


def clone_baseline(destination: Path, manifest: dict[str, Any]) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            manifest["upstream"],
            str(destination),
        ],
        capture=False,
    )
    git(destination, "fetch", "--depth", "1", "origin", manifest["commit"])
    git(destination, "checkout", "--detach", manifest["commit"])


def source_state(
    destination: Path,
    manifest: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    observed: dict[str, str] = {}
    for relative in manifest["files"]:
        path = destination / relative
        if not path.is_file():
            raise fail(f"expected Verl source file is missing: {relative}")
        observed[relative] = git_blob_id(path)

    if all(
        observed[path] == record["preimage_blob"]
        for path, record in manifest["files"].items()
    ):
        return "clean", observed
    if all(
        observed[path] == record["postimage_blob"]
        for path, record in manifest["files"].items()
    ):
        return "patched", observed
    raise fail(
        "Verl source is modified, partially patched, or incompatible with "
        "the pinned v0.9.0 baseline"
    )


def verify_repository_identity(
    destination: Path,
    manifest: dict[str, Any],
) -> None:
    if not (destination / ".git").exists():
        raise fail(f"destination is not a Git checkout: {destination}")
    head = git(destination, "rev-parse", "HEAD").stdout.strip()
    if head != manifest["commit"]:
        raise fail(f"unexpected Verl commit: expected {manifest['commit']}, got {head}")


def apply_patches(
    destination: Path,
    patches: list[Path],
) -> None:
    status = git(destination, "status", "--porcelain").stdout.strip()
    if status:
        raise fail("clean Verl baseline has unexpected worktree changes")
    for patch in patches:
        git(destination, "apply", "--check", str(patch))
        git(destination, "apply", str(patch))


def verify_final_tree(
    destination: Path,
    manifest: dict[str, Any],
) -> None:
    state, _ = source_state(destination, manifest)
    if state != "patched":
        raise fail("Verl patches did not produce the expected final source")

    expected_paths = set(manifest["files"])
    changed_paths = {
        line.strip()
        for line in git(destination, "diff", "--name-only").stdout.splitlines()
        if line.strip()
    }
    if changed_paths != expected_paths:
        raise fail(
            "patched Verl checkout contains unexpected changed files: "
            f"{sorted(changed_paths ^ expected_paths)}"
        )


def install(destination: Path, python: str) -> None:
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(destination),
        ],
        capture=False,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Prepare the pinned, patched Verl v0.9.0 source tree."
    )
    result.add_argument("--dest", default=".deps/verl")
    result.add_argument("--install", action="store_true")
    result.add_argument("--python", default=sys.executable)
    return result


def main() -> int:
    args = parser().parse_args()
    destination = Path(args.dest).expanduser().resolve()
    try:
        manifest = load_manifest()
        patches = verify_patch_files(manifest)
        clone_baseline(destination, manifest)
        verify_repository_identity(destination, manifest)
        state, _ = source_state(destination, manifest)
        if state == "clean":
            apply_patches(destination, patches)
        verify_final_tree(destination, manifest)
        if args.install:
            install(destination, args.python)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "commit": manifest["commit"],
                "destination": str(destination),
                "installed": bool(args.install),
                "patches": len(patches),
                "status": "ready",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
