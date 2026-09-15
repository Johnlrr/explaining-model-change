# Execution environments

## Local

Use local execution for static checks, unit tests, mocked integration tests, and cheap pipeline checks. The harness requires Python 3.10+, PyYAML, and an authenticated Kaggle CLI for remote commands.

## Kaggle free GPU

Treat Kaggle as a batch executor: stage a self-contained job, push a kernel version, poll its status, and fetch output and logs. Internet may be enabled only when the project rules allow it.

## Kaggle offline competition GPU

Keep Internet disabled. Prepare dependencies or artifacts separately, publish them as a Kaggle Dataset from the local authenticated CLI, and attach that dataset as an input. Never put Kaggle credentials in a kernel.

## Future executors

Local and SSH executors may implement the same conceptual `submit`, `status`, `wait`, and `fetch` contract without changing the research loop.

