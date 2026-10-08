# Verl v0.9.0 compatibility patches

This directory contains the minimal upstream delta required by the documented
Qwen3.8-27B full-parameter training runtime. It does not vendor Verl.

## Upstream baseline

- Repository: <https://github.com/verl-project/verl>
- Tag: `v0.9.0`
- Commit: `483b8a009ba3a97563edee3a19887e4862b8094a`

The patches must not be applied to another revision.

## Patch series

1. `0001-fused-lce-place-lm-head-on-hidden-device.patch`
   places an FSDP-offloaded LM-head weight on the hidden-state device before
   invoking fused linear cross-entropy.
2. `0002-fsdp-sync-gradients-every-microbatch.patch`
   reduce-scatters gradients after every microbatch instead of retaining full
   unsharded FP32 accumulated gradients.
3. `0003-qwen-ulysses-detect-actual-multimodal-input.patch`
   selects the multimodal path from actual batch inputs rather than the
   presence of `vision_config`, allowing text-only Qwen3.8 inputs to receive
   Ulysses sequence slicing.

## Apply and install

From the `thinkingbox-training` checkout:

```bash
python scripts/prepare_verl.py --dest .deps/verl --install
python scripts/verify_verl_install.py
```

`prepare_verl.py` verifies the upstream commit, patch hashes, source preimages,
forward applicability, and final source identities. It refuses modified,
partially patched, or incompatible source.

## Removal

Remove an individual patch only after an upstream Verl release contains an
equivalent fix and Qwen3.8 parity validation passes against that release.

## License

Verl is licensed under Apache License 2.0. These patch files are derivative
diffs against that upstream source and retain the original file headers and
upstream attribution. A copy of Verl's upstream license is included as
`UPSTREAM_LICENSE`.
