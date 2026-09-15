# Evaluation contract

Define these fields before substantive experiments:

- Primary metric, optimization direction, and exact computation.
- Secondary metrics and their purpose.
- Baseline and current-best run IDs.
- Dataset, split, preprocessing, seeds, and aggregation rules.
- Required outputs and validity criteria.
- Minimum meaningful improvement or uncertainty treatment.

Do not change evaluation data, metrics, or validity rules silently. Record an intentional change in `research/DECISIONS.md` and establish a new comparable baseline.

