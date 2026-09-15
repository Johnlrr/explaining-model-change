# Execution environments

## Local

Use local execution for static checks, unit tests, mocked integration tests, and cheap pipeline checks. The harness requires Python 3.10+, PyYAML, and an authenticated Kaggle CLI for remote commands.

## Kaggle free GPU

Treat Kaggle as a batch executor: stage a self-contained job, push a kernel version, poll its status, and fetch output and logs. Internet may be enabled only when the project rules allow it.

## Kaggle offline competition GPU

Keep Internet disabled. Run a preparation kernel with Internet enabled, write the
required dependencies or assets as kernel output, then attach that preparation
kernel to the offline job through `kernel_sources`. Run the offline job at smoke
scale before a full run. Existing Kaggle datasets may still be attached through
`dataset_sources`, but the harness does not automate Dataset upload or promotion.
Never put Kaggle credentials in a kernel.
