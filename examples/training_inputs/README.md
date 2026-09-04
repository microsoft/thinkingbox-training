# Training input templates

Matching an existing experiment configuration requires immutable copies of:

1. training task list and hydrated runtime task list;
2. absolute-step prompt schedule and any behaviorally active overlay;
3. source-to-runtime UID mapping;
4. model, tokenizer, and chat-template files;
5. framework and dataset revisions;
6. rendered runtime configuration for the policy, MCP, user simulator, and
   judge;
7. resolved launch parameters, environment lock, and initial adapter; and
8. any checkpoint and optimizer state required for a resumed run.

Not required after the schedule and task inputs have been materialized:

- curation notebooks or generators;
- difficulty and balance reports;
- raw training or evaluation trajectories;
- pod snapshots, process receipts, and incident reports.
