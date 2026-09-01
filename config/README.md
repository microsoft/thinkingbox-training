# Runtime configuration

The checked-in files are safe examples. Angle-bracket values identify
deployment-specific settings.

Before a run:

1. copy the required files outside the worktree;
2. replace every angle-bracket identifier and adapt the runtime example;
3. pass the rendered path as `--tb-config` or `TB_CONFIG`.

Treat model, endpoint, prompt-rendering, sampling, context, and timeout
settings as experiment identity.
