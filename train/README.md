# FSDP2/SP training package

`train.driver` is the only training entry point. It performs:

```text
scheduled prompt groups
  -> multi-turn MCP rollouts
  -> token replay with captured behavior logprobs
  -> binary or configured reward
  -> group-relative advantages
  -> token-mean clipped policy loss
  -> FSDP2/SP backward and AdamW
  -> PEFT adapter checkpoint
  -> vLLM adapter reload
```

## Modules

| Module | Responsibility |
|---|---|
| `driver.py` | Distributed lifecycle, rollout/update loop, optimizer state, metrics |
| `algorithm.py` | GRPO/DAPO defaults, validation, group filtering and weighting |
| `checkpoint_consistency.py` | Adapter/optimizer recovery consistency checks |
| `data_pipeline.py` | Explicit task-list loading and strict hydration |
| `eval_loop.py` | Optional bounded in-training evaluation |
| `token_replay.py` | Sampled-token and behavior-logprob reconstruction |
| `fsdp_lora_sync.py` | FSDP2 adapter gather, atomic save, and vLLM reload |
| `lora_sync.py` | LoRA target profiles and vLLM registry client |
| `patches.py` | Framework compatibility patches used during rollout |
| `prompt_schedule.py` | Absolute-step schedule validation and DP striping |
| `rewards.py` | Reward registry and optional termination shaping |
| `rollout.py` | Multi-turn agent/user/tool execution |
| `sp_ulysses.py` | Sequence-parallel collectives and Qwen mixer forwards |
| `tokenize_chat.py` | Chat rendering and assistant-token masks |
| `wandb_logger.py` | Optional rank-zero scalar logging |

## Distributed contract

- Every rank reads the same immutable prompt schedule.
- Prompt groups are striped by data-parallel rank and remain complete.
- Sequence-parallel partners hold identical parameter shards.
- SP gradients are summed before the optimizer update.
- FSDP2 reduces gradients over the data-parallel mesh only.
- Every rank executes the same number and padded length of microbatches.
- A checkpoint is resumable only with its complete optimizer shard set and
  matching world/SP topology.

The launch contract must explicitly provide node count, local rank count,
rendezvous identity, training devices, SP size, policy/MCP endpoints, model,
dataset, task list, and runtime configuration.

## Token replay

Training uses server-returned prompt and completion token IDs and captured
behavior logprobs. The local policy is teacher-forced on those exact tokens.
Response masks select only assistant action tokens. Missing or misaligned
trace fields fail closed rather than retokenizing a semantically similar
conversation.

## LoRA

The default `gdn_hybrid` profile adapts:

- softmax-attention Q/K/V/O projections;
- MLP gate/up/down projections; and
- Gated-DeltaNet input/output projections.

Recurrent state, convolution, decay, normalization, vision, and output-head
parameters remain frozen. Adapter target inventory and gradient reachability
must be checked against each supported model architecture.

## Sequence parallelism

`sp_ulysses.py` shards sequence activations while gathering token-mixer heads:

- GDN layers use the full-sequence recurrent kernel with head-sharded state.
- Full-attention layers apply global positions and full causal attention.
- The final hidden sequence is gathered before the output projection.

SP size must divide the distributed world and every relevant model head count.
At 128K/SP2, enable `--activation-offload` (or set
`ACTIVATION_OFFLOAD=1` in the strict launcher) so transformer-body autograd
tensors move to pinned CPU memory until backward.
Use a numerical SP1/SP2 oracle when adding a model or changing upstream model
kernels.

## Recovery

PEFT adapters are JSON plus safetensors and are independent of Python module
paths. Optimizer shards contain AdamW/Torch state and are restored only when:

- all expected rank shards exist;
- metadata agrees on step, learning rate, world size, and SP size;
- the adapter payload digest agrees across nodes; and
- the recovery cursor identifies the same policy and optimizer generation.

Adapter-free recovery is valid only for an explicit true-base cursor with
source and optimizer steps both equal to zero.
