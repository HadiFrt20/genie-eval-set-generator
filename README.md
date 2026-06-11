# Genie Eval Set Generator

> **Experimental** — a field-built tool, no SLA, expect rough edges. **Nothing leaves your Databricks
> workspace**: every model, embedding, Genie, and SQL call runs inside your own tenant. No telemetry.

## The problem

[Databricks Genie](https://www.databricks.com/product/business-intelligence/ai-bi-genie) lets people ask
questions of their data in plain English — they type a question, Genie writes the SQL and gives them the
answer. It's great when it works. But once you've set up a Genie space, a hard question follows:

> **Is it actually answering correctly — and how would you even know?**

To answer that honestly you need an *evaluation set*: a list of realistic questions, each paired with the
known-correct answer, that you run Genie against and grade. Writing one by hand is slow and thankless — and
two things make it genuinely hard:

- **Are the questions realistic?** A test full of questions no real user would ask tells you nothing useful.
- **Is the answer key actually right?** If your "correct" answers are wrong, a passing grade is meaningless.

So most Genie spaces ship with no real evaluation at all, and you find out they're wrong when a user complains.

## What this tool does

Point it at a Genie space and it does the tedious part for you:

1. **Reads the space** the way a careful analyst would — its tables, real sample values from the columns,
   the operator's curated example questions, and the questions users have actually asked.
2. **Writes a batch of realistic questions** a business user of *that* space would plausibly ask — varied in
   shape (totals, rankings, trends, ratios) and grounded in the space's real vocabulary and values.
3. **Works out the expected answer** for each (the SQL and the rows it returns) and checks that the SQL runs.
4. **Asks Genie the same questions** and compares its answers to the expected ones.
5. **Hands you a scored eval set** — a reusable Delta table — plus a report card on how Genie did.

The result is a reusable test set (a "regression set"): re-run it after any change to your Genie space (a
new instruction, a new table, a model upgrade) and see whether quality moved up or down.

## The approach — and why the scores are deliberately cautious

The entire value of an eval tool is that you can *trust its numbers*. A score that looks rigorous but isn't
is worse than no score, because it misleads you exactly when you're about to act on it. So this tool is
conservative on purpose, and tells you plainly what each number **is** and **is not**:

- **The headline "did Genie match?" figure — what the tool calls *concordance* — is a lower bound, not a
  verdict.** The expected answers are themselves machine-generated, so a mismatch can mean the *answer key*
  was wrong, not Genie. It's reported as agreement, not as truth.
- **Statistics are only used where they fit.** A pass rate comes with a confidence interval computed on the
  right unit (the questions), with the sample size shown — not a number dressed up to look more certain than
  it is.
- **Rules of thumb and AI opinions are labelled as such.** "Is this question clear?" is an *uncalibrated LLM
  judge*; the grounding check is a *heuristic*. Neither is presented as a measurement.
- **"Diversity" is described, not graded.** It tells you whether the generated questions are genuinely varied
  or near-duplicates (of each other, or of the examples the generator was shown) — without pretending one
  number is a pass/fail.

In short: it automates the tedious 80% — generating realistic questions, computing expected answers, running
the regression — and is upfront about the 20% a human still needs to eyeball.

> *Earlier iterations leaned on heavier statistics (a "realism" z-score, a reliability gate) that didn't hold
> up under scrutiny, so they were removed in favour of the plainer, conservative reads above. Re-earning that
> rigor properly — independently-verified answer keys, calibrated judges — is the active roadmap (below).*

### Roadmap: from "lower bound" to execution-verified answer keys

Live A/B experiments on real Genie spaces (single- and multi-table) shaped where this goes next:

- **Round-trip verification works.** Having an *independent model family* re-derive SQL from the question
  alone and comparing executed results caught answer keys that were provably wrong (e.g. row-grain
  aggregates answering entity-grain questions) — errors nothing in the original pipeline could detect.
- **Question precision is most of the battle.** With explicit result-grain / returned-quantities / literal-date
  rules in the generation prompt (now shipped), 83–100% of generated pairs survived independent
  verification, vs ~38% without them.
- **A one-iteration repair loop** (rewrite the question to pin down the query's semantics, re-verify)
  recovered most remaining failures.

The next release builds this in as a **verification stage**: k independent SQL derivations across two model
families, execution-consensus clustering, and a per-pair certificate — `verified_fraction` becomes the
headline metric (labelled *execution-verified by two model families*, deliberately never "accuracy"), with
unverified pairs quarantined for human review instead of silently shipped. Grounding:
[GAZP](https://arxiv.org/abs/2009.07396), [OmniSQL](https://arxiv.org/abs/2503.02240),
[CHASE-SQL](https://arxiv.org/abs/2410.01943), [FLEX](https://arxiv.org/abs/2409.19014).

---

## How the report card reads

You don't need any of these terms to *run* the tool — this is just how to read the report once you have one:

| Section | What it answers | How to read it |
|---|---|---|
| **Validity** | Does the generated SQL run on your data and return rows? | A direct check — closer to 1.0 is better. |
| **Grounding** | Do the questions' filter values come from real column values (not invented)? | A heuristic flag; "not evaluable" when there's nothing to check against. |
| **Quality** | Are the questions clear, and does each SQL actually answer its question? | *Uncalibrated LLM-judge* opinions — directional, not gospel. |
| **Diversity & leakage** | Are the questions varied, or near-copies of the examples? | Descriptive numbers + one `leakage` flag (too close to the reference). |
| **Concordance** | How often did Genie's answer match the expected one? | A **lower bound** — see the caution above. Shown per difficulty (how complex the SQL is) and category (totals/rankings/trends/ratios), with counts. |
| **Reliability** | If you re-ask the same questions, do you get the same result? | Agreement across reruns + a confidence interval on the rate (needs `stability_runs ≥ 2`). |

## Quickstart

```bash
git clone https://github.com/HadiFrt20/genie-eval-set-generator
cd genie-eval-set-generator

databricks auth login --host https://<your-workspace>.cloud.databricks.com
(cd app/web && npm install && npm run build)        # build the app frontend
databricks bundle deploy --var=output_catalog=<your_catalog>
```

After deploy, grant the app's service principal `CAN_USE` on the serving endpoints it uses (the question
generator and LLM judge always; the embedding endpoint only if you keep the diversity check) and a SQL
warehouse, plus `USE CATALOG` + `MODIFY` on your catalog (`output_catalog` above is the same catalog the
notebook calls `uc_catalog`). Then start the app and get its URL:

```bash
databricks bundle run genie_eval_app          # starts the app
databricks apps list                          # shows the app's URL (or find it under Compute → Apps)
```

Prefer no app? Upload `genie_eval_set_generator.py` and run it with widget values — see
[`examples/`](examples/) for sample Jobs-API payloads, and [`app/README.md`](app/README.md) for local dev.

## Key parameters

| Widget | Default | Purpose |
|---|---|---|
| `genie_space_id` | (required) | The Genie space to evaluate (32-char hex from its URL) |
| `genie_tables` | (auto) | Tables to cover; blank = the space's registered tables |
| `uc_catalog` / `uc_schema` | `main` / `genie_eval` | Where the eval tables + prompt registry live (the bundle sets the catalog via `--var=output_catalog`) |
| `questions_per_table` | `14` | 14 = a fast preview; 100+ **per table** for numbers solid enough to gate a release on (total questions = this × #tables) |
| `stability_runs` | `1` | Re-ask each question this many times (≥2 to measure reliability; 3+ for a stable number) |
| `embedding_endpoint` | `…bge-large-en` | Powers the diversity/leakage check (blank = skip it) |
| `generator_endpoint` / `judge_endpoint` | `…llama-4-maverick` / `…claude-sonnet-4-5` | Question generator + LLM judge |

Blank `mlflow_experiment_path` and `ai_gateway_base_url` default to a per-tool experiment under your home
folder and `<workspace-host>/serving-endpoints`. There are a few more sampling/customization knobs (history
depth, low-cardinality cap, prompt overrides) — they're documented on each widget.

## Outputs

- **`{catalog}.{schema}.genie_eval_set`** — the curated items: question, expected SQL, expected rows,
  validity flags, and difficulty. (The per-question LLM-judge scores live in the eval-set-quality MLflow
  run, not in this table.)
- **`{catalog}.{schema}.genie_eval_runs`** — Genie's response per question and the concordance result.
- **MLflow runs** for eval-set quality, diversity/leakage, reliability, and the Genie regression.

## Good to know (limits)

1. **Concordance is a lower bound, not correctness** — the answer key is machine-generated and, when you
   lack access to the space's own warehouse, graded on a different engine. Matching is forgiving where it
   should be (one consistent column reordering across all rows; 0.1% numeric tolerance for ROUND()/float
   differences) and strict where it must be (duplicates preserved; inconsistent cross-column swaps fail).
2. **The LLM judges aren't calibrated against human labels** — read their scores as opinions.
3. **Diversity needs a reference** — it runs only with an embedding endpoint and enough example/history
   questions; sparse spaces correctly skip it.
4. **Small N is noisy** — per-metric scores swing run-to-run below ~100 questions; raise
   `questions_per_table` (and `stability_runs`) for numbers you'd gate a release on.
5. **Permissions** — the runner needs read access to the space's tables, `CAN_USE` on the serving endpoints,
   and (for same-engine grading) the space's SQL warehouse. Tables it can't read are skipped with a warning.

## Repo layout

```
genie_eval_set_generator.py   the notebook engine
app/                          React + FastAPI Databricks App (backend: app/server, frontend: app/web)
tools/probe_workspace.py      read-only check that the tool's assumptions hold on YOUR workspace
tools/judge_calibration.py    judge-vs-human agreement (Cohen's κ) harness, CSV-driven
tests/test_stats.py           known-answer unit tests for every statistical function
examples/                     sample Jobs-API submit payloads
databricks.yml, resources/    Databricks Asset Bundle (one-command deploy)
```

## License

[Apache 2.0](LICENSE).
