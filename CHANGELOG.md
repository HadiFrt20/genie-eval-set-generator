# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`BREAKING:` prefix indicates a backwards-incompatible change.

## [Unreleased]

### Verification-driven fixes (post-release reviews + live A/B experiments)

- **Matcher v2** (`rows_match`): column-order/alias invariance is now ONE consistent column permutation
  across all rows — per-row cell-sorting let inconsistent cross-column value swaps falsely PASS, which
  inflated the lower bound; added a 0.1%-relative / 1e-9-absolute numeric-tolerance fallback so
  ROUND()/float-path differences between semantically identical SQL no longer score as misses.
  Cascade: exact → consistent-permutation → tolerant. 19 known-answer tests.
- **Question contract** (generation prompt rules 9–11): every question must name its result grain
  (rows vs entities), state exactly which quantities to return, and use literal date anchors (no
  NOW()/relative windows). Measured effect in live A/B: pairs surviving independent round-trip
  verification went from ~38% to 83–100%.
- **Scorecard run attribution**: notebook tags MLflow runs with the Jobs run-id; the app filters on it
  (untagged runs fall back to latest-in-experiment with a visible warning) — two runs sharing an
  experiment no longer show each other's results.
- **Contamination filter persistence**: new append-only `genie_eval_synthetic_log` table + per-space
  `replaceWhere` writes — previously each run's full-table overwrite erased the very history the filter read.
- Genie API errors are excluded from the concordance rate (not scored as misses); concordance ties across
  reruns count as fail (conservative); complete 50-row results are determinate (collectors fetch cap+1);
  scientific-notation and negative-zero values canonicalize; the diversity merge keys on `question_id`;
  the app follows SQL result chunks and backticks identifiers; README documents how the app actually starts.
- README: measured roadmap section — execution-verified answer keys (round-trip consensus, repair loop,
  `verified_fraction` with quarantine) grounded in the live experiments and the synthetic-SQL literature.

### BREAKING — honest-measurement overhaul (v0.1 strip → v0.2 rebuild)

An adversarial review of the methodology found that several inferential statistics were not valid as
built. This release **removes the invalid statistics and keeps only honest, descriptive measurement**,
then rebuilds the grading engine correctly. Scorecard metric keys and the app contract changed.

#### Removed (v0.1)
- The realism **z-score**, σ language, and good/expanded/borderline/bad **bands** — the z-score divided a
  cross-set quantity by the bootstrap SD of a within-set quantity, so it was not a standardised statistic.
- **Krippendorff's α** from the reliability gate — it scored a near-perfect system (one flaky item out of
  many) at α≈0, i.e. it punished the best case.
- The hardcoded **7 pp Wilson half-width gate** — mathematically unsatisfiable at the documented default
  sample size.
- **"Calibration" and "Discrimination"** as named scorecard axes — they were never actually computed in
  code (only bare per-group means were logged).

#### Changed
- Realism phase → **Diversity & leakage** (descriptive). Reports `mean_nn_cosine`, `nn_cosine_p10`,
  generated/reference self-similarity; the only verdict is a `leakage_risk` flag (near-verbatim to reference).
- Stability phase → **Reliability**. `rerun_agreement` (fraction of questions whose reruns are unanimous)
  replaces α; the Wilson CI is computed on **N independent questions**, not N×M correlated rerun cells.
  `gateable` now uses an achievable, configurable half-width target and prints the N needed to reach it.
- The headline metric is relabelled **row concordance (a lower bound)** — it is agreement between the
  system and LLM-authored expected SQL, not certified correctness.
- `grounded_literals` fails **visible** ("not evaluable") instead of silently passing when there is nothing
  to ground against; it is a deterministic heuristic, not an LLM judge. The answer-quality judge is labelled
  uncalibrated and only claims what it can see.
- App scorecard contract: `realism`→`diversity`, `stability`→`reliability`; per-bucket counts added; the
  "Gateable" tile distinguishes **not computed** from true/false; descriptive-only colours.

#### Rebuilt + hardened (v0.2, validated against a live workspace)
- **Same-engine grading**: `expected_sql` runs on the Genie space's own SQL warehouse (same engine the
  system uses), with a Spark fallback when warehouse access is denied. Row comparison is a **multiset**
  with cross-engine cell canonicalization, column-order invariant, and returns *indeterminate* (excluded
  from the rate) when a result is truncated at the row cap.
- **Result-chunk pagination**: statement results are paged via the chunk API — previously only the first
  chunk was read, which could silently truncate a result and corrupt the grade.
- **Robustness against real workspaces**: inaccessible registered tables are pruned with a warning (instead
  of crashing); identifiers are backtick-quoted (hyphenated / federated catalogs); the `DESCRIBE` parser no
  longer leaks metadata columns into the prompt; permission-denied baselines are scored *not-evaluable*
  rather than as misses; internal REST errors are surfaced (403/404) instead of silently swallowed.
- **Grounding**: difficulty is re-derived from SQL structure (not the generator's self-label); low-card
  numeric/date/boolean columns are grounded (not just strings); the contamination filter matches on a
  normalized form; shared-column-name join hints are inferred when no foreign keys are declared.
- **No hardcoded identifiers**: all defaults (gateway base, experiment path, prompt owner, app experiment
  default) derive from the live workspace/user at runtime.
- Added `tools/probe_workspace.py` (read-only workspace assumption probe), `tools/judge_calibration.py`
  (judge-vs-human Cohen's κ harness), and `tests/test_stats.py` (known-answer tests for every stat function).

## [0.1.0-rc2] - 2026-05-04

### Added
- **OBO auth** on the FastAPI backend: every Databricks REST call uses the caller's
  `X-Forwarded-Access-Token` header instead of a single service-principal token. Falls back to
  `DEV_USER_TOKEN` for local dev, then SP for app-internal calls. Cache keys are scoped per token.
- **UC runs-index table** at `<catalog>.genie_eval.genie_eval_runs_index`; the backend writes one row per
  submit and the Runs page reads it as the source of truth.
- **`databricks.yml` Databricks Asset Bundle** — one-command install
  (`databricks bundle deploy --var=output_catalog=<your_catalog>`): creates the UC schema, MLflow
  experiment, deploys the app, and syncs the notebook.
- Customization-hooks UI (`extra_prompt_rules`, `custom_prompt_uri`, pooled-pass guard overrides).

## [0.1.0-rc1] - 2026-05-04

First public-shape release candidate: React + FastAPI app (Spaces / Configure / Runs / Scorecard),
notebook customization hooks, Apache-2.0 `LICENSE`. Install path still manual.
