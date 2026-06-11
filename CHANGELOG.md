# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`BREAKING:` prefix indicates a backwards-incompatible change.

## [Unreleased]

### The trust gate ships — execution-verified answer keys (P0 + P1)

The roadmap's verification stage is now in the notebook (`verification_k`, default 5):

- **Cross-family solver panel:** k independent derivations (generator family at temps 0/0.4/0.8 +
  judge family at 0/0.5) re-write each pair's SQL from the question alone — under **context parity**
  (the panel sees the space's instructions + curated questions, the same declared semantics Genie
  has; verified necessary by live experiment). Executed results are compared with the same matcher
  that grades Genie.
- **Tiers:** GOLD (unanimous, both families) / VERIFIED (majority, both families) / QUARANTINE
  (kept + published with its tier, excluded from the Genie regression, surfaced as the human-review
  queue). One question-rewrite repair iteration runs before quarantining (5/7 recovery measured in
  the design experiments). `verified_fraction` is the new headline, labelled *execution-verified by
  two model families* — never "accuracy".
- **Empirical difficulty** (1 − panel pass rate: easy/medium/hard/suspect) ships as a free byproduct;
  `suspect` items (zero panel matches) are concentrated wrong-key candidates.
- **P0 deterministic guards** (sqlglot): typed lint rejects numeric aggregates over string/boolean
  columns and unparseable SQL; **literals-masked skeleton dedup** removes structural duplicates the
  question-embedding dedup can't see; column-coverage % is measured and logged.
- App scorecard: new "Trust gate" card + "Verified pairs" headline tile; concordance is computed on
  verified pairs only when the gate ran.
- 24 known-answer tests (tier logic, empirical difficulty, skeleton dedup key).


### Generation robustness (live context-parity experiments)

Three live experiments on instruction-bearing spaces surfaced four generation failure classes,
all now fixed:
- **Relative-time questions are rejected deterministically** — "this month" / CURRENT_DATE pairs
  are unverifiable and unstable by construction; curated few-shot kept re-introducing them despite
  prompt rules (measured twice), so a post-check enforces it.
- **Determinism double-execution** at validation: queries whose own re-execution differs (LIMIT
  without ORDER BY, sampling, drifting data) are flagged `sql_deterministic=false` and excluded
  from the Genie regression.
- **SELECT-based table pruning** — DESCRIBE can succeed where SELECT is denied (BROWSE-style
  grants); such tables previously generated questions whose every expected_sql failed.
- **Date-range introspection** — MIN/MAX per date/timestamp column now feeds the literal-date
  rule (date columns are never low-cardinality, so the generator had no dates to anchor to).
Measured outcome: on an instruction-rich space, 11/12 generated pairs were deterministic and
8/11 survived independent round-trip verification — and giving the verifier the space's declared
semantics (context parity) produced identical results, supporting the roadmap's verification
design on navigational-instruction spaces.


### Fixes from an independent zero-context review board

A panel of fresh-eyes reviewers and adversaries (no prior context, public repo only) audited the
project; every confirmed finding below is fixed in this release.

- **Cross-space data destruction path (critical):** the Genie-regression phase read the eval table
  WITHOUT filtering by space, so on a shared catalog it ran other spaces' questions against this
  space's Genie — and the scoped write's failure fallback then performed a full-table overwrite,
  destroying other spaces' history. The read is now space-scoped, and the write fallback is
  delete-this-space + append (never a full overwrite). The per-category summary is space-scoped too.
- **Deploy shipped the app without its UI (critical):** the built frontend is gitignored and DAB sync
  respects `.gitignore`, so `bundle deploy` uploaded an app whose UI was a "Frontend build not found"
  error. `databricks.yml` now explicitly syncs `app/server/static/**`.
- **Matcher false-PASS channels (critical):** (1) the 0.1% numeric tolerance could bridge distinct
  integer values from 999 upward (a COUNT off by one matched) — tolerance now applies only to
  fractional values, integer cells must match exactly; (2) results wider than 7 columns silently fell
  back to per-row sorted-cell matching, re-admitting the cross-column-swap false PASS the changelog
  previously claimed eliminated — wide results that don't match in column order are now NOT-EVALUABLE
  (the earlier claim was over-broad and is hereby corrected); (3) two empty result sets counted as
  agreement — now NOT-EVALUABLE.
- **Leakage flag blind to partial leakage:** flag now fires on the FRACTION of near-verbatim questions
  (≥10% with NN cosine ≥ 0.97) instead of the mean, which hid 30%-verbatim sets; `leakage_fraction`
  is logged alongside.
- Robustness: generator items are normalized before use (a single malformed LLM item could crash the
  run); truncated generator JSON raises a diagnostic error instead of a bare parse crash; only
  SELECT/WITH statements are executed (an LLM-emitted DML/DDL statement would have run with the
  runner's grants); Genie statement state is checked (FAILED/EXPIRED results no longer score as
  misses); `pooled_pass_rate` is not logged when zero units were measured; `CREATE SCHEMA` is
  backticked and blank catalog/schema fail fast.
- App/deploy: the bundle's `--var=output_catalog` now actually reaches the form (`/api/health`
  exposes it; the Configure page prefers it over the space's data catalog); the questions-per-table
  form cap (80) no longer blocks the documented 100+ gating config (now 300); the bundled MLflow
  experiment matches the notebook's default path; the workspace probe's deep row-match check was
  silently disabled by a stale import (now fixed, and import failures are loud); the judge-calibration
  harness marks single-class human labels "not calibratable" instead of awarding κ=1.0.
- README: positions the tool relative to Genie's built-in Benchmarks feature (complement, not
  replacement; fleet-scale audience named); matcher semantics described precisely (integer-exact
  tolerance, wide-result and empty-result carve-outs); realism explicitly listed as not-yet-measured;
  gating-run cost/time and space-history side effects disclosed; roadmap A/B numbers annotated with
  their provenance and reproducibility plan.

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
