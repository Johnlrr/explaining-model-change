# Research protocol

Use this loop:

```text
observe → hypothesize → predict → make one minimal change → smoke
        → full experiment → validate → compare → record → repeat
```

Before experimentation, read the current state and evaluation contract. State the hypothesis and predicted metric movement before implementing it. Change as little as practical so the result remains attributable.

A smoke run must use the real code path at reduced scale. For Python script kernels, pass generic runtime values through `parameters` in the executor YAML; the harness embeds them into the single submitted entrypoint as `HARNESS_PARAMETERS` and the JSON environment variable `RESEARCH_HARNESS_PARAMETERS`. Other files must be attached through explicit Kaggle inputs. A smoke passes only when remote execution succeeds, required files exist, result JSON parses, values are finite, and configured `json_checks` prove that the requested workload completed. A failed smoke never advances automatically to a full run.

Full runs use the same entrypoint and configuration shape with workload scale and timeout adjusted. Every conclusion must cite a run directory. After analysis, update the research state even when the result is negative or inconclusive.
