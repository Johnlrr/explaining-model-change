## Initialization prompt
Initialize this research branch from the repository specification.

Read `AGENTS.md`, `PROJECT_SETUP.md`, `docs/*`, and the research templates first.

Act as the primary Research Lead according to `AGENTS.md`.

Validate that the project specification is complete and internally consistent.
Initialize/update `research/*` as needed, verify environment readiness, reproduce
the baseline, and begin autonomous research according to the documented protocol.

Own scientific reasoning, including:

- literature and prior-work analysis;
- diagnostic interpretation;
- hypothesis formation and refinement;
- experiment design;
- contribution assessment;
- interpretation of experimental evidence;
- decisions about support, rejection, confirmation, and the next hypothesis.

When checking environment readiness, distinguish local development requirements
from remote experiment requirements.

Do not install or upgrade local Python packages merely because the full research
workload will need them on Kaggle.

Before installing any dependency locally:

- inspect the existing repository environment first;
- determine whether the dependency is actually required for local tests,
  validation, or repository tooling;
- if it is needed only by the Kaggle experiment runtime, leave the local
  environment unchanged and verify it through the Kaggle smoke path instead;
- prefer the repository's documented remote dependency/environment mechanism;
- do not mutate the local environment just to make remote experiment code
  executable locally.

A missing optional local dependency is not a blocker when the documented
experiment path runs remotely and local validation can still be performed.

Install or change local dependencies only when they are genuinely required for
local repository validation or explicitly required by the project specification.
Record any necessary environment change rather than making opportunistic upgrades.

When a bounded experiment is ready to execute, delegate implementation and
execution to the configured `research_executor`.

The Research Executor should handle, as appropriate:

- implementation under `src/`;
- experiment configuration;
- local tests and validation that are actually required;
- smoke and approved full runs;
- Kaggle execution and blocking wait;
- artifact/result retrieval and validation;
- reproducible execution evidence.

After delegated execution returns, independently review the evidence and make
the scientific interpretation in the primary Research Lead thread.

Do not let the executor invent or materially change the research direction.

Do not modify `docs/*` unless explicitly authorized.

Continue autonomously until a documented stop condition is reached or human
input is required.

## Continuation prompt
Continue the autonomous research from the current repository state.

Read `AGENTS.md`, `docs/*`, and `research/*` first.

Act as the primary Research Lead according to `AGENTS.md`.

Resume from the latest valid evidence, current research state, and active
hypothesis. Do not restart completed work unnecessarily.

Own scientific reasoning and research-direction decisions.

When checking environment readiness or recovering from an execution issue,
distinguish local development requirements from remote Kaggle runtime
requirements.

Do not install or upgrade local Python packages merely because the remote
experiment needs them.

Before changing the local environment:

- inspect the existing repository environment first;
- determine whether the dependency is actually required for local tests,
  validation, or repository tooling;
- if it is needed only by the Kaggle experiment runtime, keep the local
  environment unchanged and verify it through the Kaggle smoke path;
- prefer the repository's documented remote dependency/environment mechanism;
- do not mutate the local environment just to make remote experiment code run
  locally;
- do not perform opportunistic dependency upgrades unrelated to the current
  blocker.

A missing optional local dependency is not a blocker when the documented
experiment path is remote and the required local validation can still be
performed.

When the next approved experiment requires implementation or execution,
delegate the bounded work to the configured `research_executor`.

The executor should use local execution only for the validation actually needed
by the repository protocol, and should use the documented Kaggle path for the
remote experiment workload.

After the executor returns:

- validate the returned evidence;
- interpret the scientific result yourself;
- update the hypothesis and research state as appropriate;
- decide whether the result is supported, rejected, inconclusive, requires
  confirmation, or motivates a new hypothesis;
- delegate the next execution step only after its scientific purpose is defined.

Follow the documented experiment, diagnostic, confirmation, artifact, remote
execution, and final-verification rules.

Do not let the executor independently create the next scientific hypothesis or
change the approved experiment.

Do not modify `docs/*` unless explicitly authorized.

Continue until a documented stop condition is reached or human input is required.