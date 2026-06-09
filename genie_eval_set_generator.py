# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Eval Set Generator
# MAGIC
# MAGIC Generate a synthetic NL→SQL evaluation set for a Genie space, **validate and judge the eval
# MAGIC set itself**, and publish it to UC as a reusable regression asset.
# MAGIC
# MAGIC **Scope.** This notebook generates and scores a synthetic eval set, **then runs it against
# MAGIC Genie** for a row-concordance regression (read it as a lower bound — see Phase 8). It produces
# MAGIC a curated `(question, expected_sql, expected_rows, quality_scores)` Delta table plus a Genie
# MAGIC regression table that downstream systems (Agent Framework evals, benchmark harnesses, CI jobs)
# MAGIC can consume.
# MAGIC
# MAGIC **Stack**
# MAGIC - Databricks Genie SDK → `w.genie.get_space` for grounding context only
# MAGIC - OpenAI client against Databricks serving-endpoints for eval-question synthesis
# MAGIC - **UC MLflow Prompt Registry** for versioned question-generation prompt (`@candidate` / `@production`)
# MAGIC - **Grounded generation**: space description + sample column values fed to the generator
# MAGIC - `spark.sql` to execute every candidate `expected_sql` and capture `expected_rows`
# MAGIC - `mlflow.genai.evaluate` + `Guidelines` + custom `@scorer` for eval-item quality (clarity,
# MAGIC   SQL-answers-question alignment, grounded-literals, safety)
# MAGIC - **`mlflow.genai.optimize_prompts`** (GEPA) to refine the generator prompt against these scorers
# MAGIC
# MAGIC **Phases**
# MAGIC 0. Introspect the Genie space + sample low-cardinality column values
# MAGIC 1. Register the question-generation prompt in UC
# MAGIC 2. Generate the grounded eval set
# MAGIC 3. Validate each `expected_sql` (executes? rows returned? capture `expected_rows`)
# MAGIC 4. Score eval-set quality (clarity, SQL-answers-question, grounded literals, safety)
# MAGIC 5. Publish curated eval set to UC
# MAGIC 6. **Run the curated set against Genie** — deterministic result-match + LLM judge on NL answer
# MAGIC 7. (Optional) Optimize the generator prompt with GEPA against the same scorers
# MAGIC
# MAGIC LLM-calling functions are wrapped with `@mlflow.trace` so the MLflow UI links each trace
# MAGIC back to the prompt version that drove it.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.5.0" "openai>=1.50.0" "databricks-sdk>=0.40.0" "gepa>=0.0.26"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parameters

# COMMAND ----------

dbutils.widgets.text("mlflow_experiment_path", "", "MLflow experiment path (blank = /Users/<you>/genie_eval_set_generator)")
dbutils.widgets.text("generator_endpoint", "databricks-llama-4-maverick", "Question-generation endpoint")
dbutils.widgets.text("judge_endpoint", "databricks-claude-sonnet-4-5", "LLM-judge endpoint")
dbutils.widgets.text("ai_gateway_base_url", "", "Serving-endpoints base URL (blank = <workspace-host>/serving-endpoints)")
dbutils.widgets.text("genie_space_id", "", "Genie space ID (required)")
dbutils.widgets.text("genie_tables", "", "(Optional) Comma-separated UC tables — blank = auto-extract all tables registered on the space")
dbutils.widgets.text("max_conversation_sample", "50", "Cap on historical conversations to mine for style exemplars")
dbutils.widgets.text("uc_catalog", "main", "UC catalog (for eval tables AND prompt registry)")
dbutils.widgets.text("uc_schema", "genie_eval", "UC schema (for eval tables AND prompt registry)")
dbutils.widgets.text("questions_per_table", "14", "Questions per table (14 = fast preview, NOT gateable; 100+ for gating; total N = this × #tables)")
dbutils.widgets.text("min_hard_per_table", "5", "Minimum 'hard' questions per table (enforced in the prompt)")
dbutils.widgets.text("max_distinct_values_per_column", "30", "Low-cardinality column sampling cap (SELECT DISTINCT LIMIT)")
dbutils.widgets.text("extra_space_instructions", "", "(Optional) extra space instructions to inject — paste Genie space Instructions text here")
dbutils.widgets.text("embedding_endpoint", "databricks-bge-large-en", "Embedding endpoint for realism scorer (blank = skip realism)")
dbutils.widgets.text("stability_runs", "1", "Genie rerun count M for reliability (1 = skip; ≥2 to compute it, 3+ for a stable number)")
dbutils.widgets.dropdown("include_historical_in_context", "false", ["false", "true"], "Inject historical_qa into SPACE_CONTEXT (off = test realism without anchor)")
dbutils.widgets.dropdown("run_prompt_optimization", "false", ["false", "true"], "Run mlflow.genai.optimize_prompts (GEPA, optional)")

# --- Customization hooks (v0.1.0). All optional. ---
# Hook 1: append-only addendum to question-gen prompt.
dbutils.widgets.text("extra_prompt_rules", "", "(Optional) extra rules appended to the question-gen prompt (free text)")
# Hook 5: full prompt override via MLflow Prompt Registry URI. Must define the same template
# variables: {{n}}, {{min_hard}}, {{table}}, {{schema}}, {{space_context}}, {{column_samples}}.
# Example: prompts:/<your.uc.catalog>.<schema>.your_prompt@production
dbutils.widgets.text("custom_prompt_uri", "", "(Optional) MLflow Prompt Registry URI (prompts:/...) overriding the default question-gen prompt")
# Hook 6: stability gate floor / ceiling. Wider band = more permissive gateable=True.
dbutils.widgets.text("custom_min_pooled_pass", "0.10", "Minimum pooled concordance for gateable=True (degenerate-eval guard)")
dbutils.widgets.text("custom_max_pooled_pass", "0.90", "Maximum pooled concordance for gateable=True (memorizing-eval guard)")
# Hook 7 (v0.1): achievable half-width target for the gate. The old hardcoded 7pp was UNSATISFIABLE
# at the advertised N=14/M=3 (half-width floor there ~9.3pp).
dbutils.widgets.text("gate_max_half_width_pp", "10.0", "Max Wilson 95% half-width (pp) for gateable=True (achievable target)")

# COMMAND ----------

import os
import json
import time
from datetime import timedelta
from functools import reduce
from openai import OpenAI
from databricks.sdk import WorkspaceClient
import mlflow
import pandas as pd
from pyspark.sql import functions as F

MLFLOW_EXPERIMENT   = dbutils.widgets.get("mlflow_experiment_path")
GENERATOR_ENDPOINT  = dbutils.widgets.get("generator_endpoint")
JUDGE_ENDPOINT      = dbutils.widgets.get("judge_endpoint")
AI_GATEWAY_BASE_URL = dbutils.widgets.get("ai_gateway_base_url")
GENIE_SPACE_ID      = dbutils.widgets.get("genie_space_id").strip()
GENIE_TABLES        = [t.strip() for t in dbutils.widgets.get("genie_tables").split(",") if t.strip()]
MAX_CONVERSATIONS   = int(dbutils.widgets.get("max_conversation_sample") or "50")
CATALOG             = dbutils.widgets.get("uc_catalog")
SCHEMA              = dbutils.widgets.get("uc_schema")
QUESTIONS_PER_TABLE = int(dbutils.widgets.get("questions_per_table"))
MIN_HARD_PER_TABLE  = int(dbutils.widgets.get("min_hard_per_table"))
MAX_DISTINCT        = int(dbutils.widgets.get("max_distinct_values_per_column"))
assert MIN_HARD_PER_TABLE <= QUESTIONS_PER_TABLE, "min_hard_per_table cannot exceed questions_per_table"
EXTRA_INSTRUCTIONS  = dbutils.widgets.get("extra_space_instructions").strip()
EMBEDDING_ENDPOINT  = dbutils.widgets.get("embedding_endpoint").strip()
STABILITY_RUNS      = int(dbutils.widgets.get("stability_runs") or "1")
INCLUDE_HISTORICAL  = dbutils.widgets.get("include_historical_in_context") == "true"
assert STABILITY_RUNS >= 1, "stability_runs must be >= 1"
RUN_OPTIMIZATION    = dbutils.widgets.get("run_prompt_optimization") == "true"

# v0.1.0 hooks — see the design notes
EXTRA_PROMPT_RULES  = dbutils.widgets.get("extra_prompt_rules").strip()
CUSTOM_PROMPT_URI   = dbutils.widgets.get("custom_prompt_uri").strip()
MIN_POOLED_PASS_OVERRIDE = float(dbutils.widgets.get("custom_min_pooled_pass") or "0.10")
MAX_POOLED_PASS_OVERRIDE = float(dbutils.widgets.get("custom_max_pooled_pass") or "0.90")
assert 0.0 <= MIN_POOLED_PASS_OVERRIDE < MAX_POOLED_PASS_OVERRIDE <= 1.0, \
    "custom_min_pooled_pass must be < custom_max_pooled_pass and both in [0,1]"
GATE_MAX_HALF_WIDTH_PP = float(dbutils.widgets.get("gate_max_half_width_pp") or "10.0")
assert 0.0 < GATE_MAX_HALF_WIDTH_PP <= 50.0, "gate_max_half_width_pp must be in (0, 50]"

assert GENIE_SPACE_ID, "Set the genie_space_id widget before running"
# genie_tables can be blank — in that case Phase 0 auto-populates from the space's table_identifiers.

EVAL_SET_TABLE    = f"{CATALOG}.{SCHEMA}.genie_eval_set"
JUDGE_MODEL_URI   = f"databricks:/{JUDGE_ENDPOINT}"
PROMPT_NAME       = f"{CATALOG}.{SCHEMA}.genie_eval_question_gen"

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

w = WorkspaceClient()
try:
    CURRENT_USER = w.current_user.me().user_name
except Exception:
    CURRENT_USER = "unknown"

# Derive defaults from the live workspace instead of hardcoding a user/workspace:
#   - gateway base = <workspace-host>/serving-endpoints (the FMAPI chat base — NOT /mlflow/v1)
#   - experiment   = the runner's home folder
if not AI_GATEWAY_BASE_URL:
    AI_GATEWAY_BASE_URL = f"{(w.config.host or '').rstrip('/')}/serving-endpoints"
if not MLFLOW_EXPERIMENT:
    MLFLOW_EXPERIMENT = f"/Users/{CURRENT_USER}/genie_eval_set_generator"

openai_client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=AI_GATEWAY_BASE_URL)

# UC Prompt Registry — prompts are first-class UC objects (catalog.schema.name + aliases)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"Genie space:           {GENIE_SPACE_ID}")
print(f"Tables:                {GENIE_TABLES}")
print(f"Eval set table:        {EVAL_SET_TABLE}")
print(f"UC Prompt Registry:    {PROMPT_NAME}")
print(f"MLflow experiment:     {MLFLOW_EXPERIMENT}")
print(f"Run optimization:      {RUN_OPTIMIZATION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Phase 0 — Introspect the Genie space + sample column values
# MAGIC
# MAGIC Ground the question generator so it stops inventing values. For each target table we:
# MAGIC - pull the Genie space title + description (so questions sound like real users of that space)
# MAGIC - run `DESCRIBE TABLE` for the column + type list
# MAGIC - sample distinct values for low-cardinality string columns (fixes the `season='summer'` vs `'SS'` class of miss)

# COMMAND ----------

import urllib.request, urllib.error
# (confirmed via live workspace validation): derive the data-rooms/Genie REST host from the
# WORKSPACE client, NOT by string-splitting the AI-gateway URL. The shipped default gateway URL has
# no "/serving-endpoints" substring, so the old rsplit was a no-op that left a wrong-tenant
# ".../mlflow/v1" host → data-rooms GETs returned 400 Invalid Token / SPA HTML, and _get_json
# swallowed it to {} (silent degraded run). The gateway URL still feeds ONLY the OpenAI client.
space_host = (w.config.host or "").rstrip("/")


def _get_json(path: str, timeout: int = 15) -> dict:
    """Internal GET helper. Returns {} on error (callers degrade) but LOGS the HTTP status so a
    403/permission or 404/wrong-id is distinguishable from genuinely-empty data — history, FK, and
    diversity were silently masked otherwise (caught live)."""
    try:
        req = urllib.request.Request(
            f"{space_host}{path}",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        kind = {401: "auth", 403: "permission", 404: "not-found"}.get(e.code, "http")
        print(f"  [_get_json] {path} → HTTP {e.code} ({kind}) — empty result; downstream will degrade")
        return {}
    except Exception as e:
        print(f"  [_get_json] {path} → {type(e).__name__}: {e} — empty result")
        return {}


# --- 1. Pull space-level config (title, description, capabilities, all registered tables) ---
space_sdk = w.genie.get_space(GENIE_SPACE_ID)
space_title = getattr(space_sdk, "title", "<unknown>")
space_description = getattr(space_sdk, "description", "") or ""

dr = _get_json(f"/api/2.0/data-rooms/{GENIE_SPACE_ID}")
space_instructions      = (dr.get("suggestion_description") or "").strip()
space_tables_registered = list(dr.get("table_identifiers") or [])
space_warehouse_id      = dr.get("warehouse_id")

print(f"Space: {space_title}")
print(f"  description: {space_description}")
print(f"  warehouse : {space_warehouse_id}")
print(f"  tables registered : {space_tables_registered}")
print(f"  instructions len  : {len(space_instructions)}")

# --- 2. Curated questions (operator-blessed samples) ---
# Page through next_page_token so spaces with >200 curated aren't silently truncated (L2.1).
curated_questions = []
_cq_tok = None
for _ in range(50):  # hard cap: 50 pages × 200 = 10k, far beyond any real space
    _cq_path = f"/api/2.0/data-rooms/{GENIE_SPACE_ID}/curated-questions?page_size=200"
    if _cq_tok:
        _cq_path += f"&page_token={_cq_tok}"
    cq = _get_json(_cq_path)
    curated_questions.extend(cq.get("curated_questions") or cq.get("questions") or [])
    _cq_tok = cq.get("next_page_token") or cq.get("next_page")
    if not _cq_tok:
        break
print(f"  curated questions : {len(curated_questions)}")

# --- 3. Historical conversations — mine real user questions + Genie's generated SQL ---
# CONTAMINATION GUARD: every notebook run sends synthetic Qs to Genie which then appear in
# subsequent /conversations queries. Read prior eval-set table (if it exists) and exclude
# any historical message that matches a previously-generated synthetic question.
def _norm_q(s: str) -> str:
    """Normalize a question for contamination matching: lowercase, collapse whitespace, strip
    trailing punctuation. Catches casing/whitespace/punctuation drift that exact-match misses
Embedding-based paraphrase exclusion is a further v0.2 step."""
    return " ".join((s or "").strip().lower().split()).rstrip("?.! ")


prior_synthetic = set()
try:
    rows = spark.sql(
        f"SELECT DISTINCT question FROM {EVAL_SET_TABLE} WHERE genie_space_id = '{GENIE_SPACE_ID}'"
    ).collect()
    prior_synthetic = {(r[0] or "").strip() for r in rows if r[0]}
except Exception:
    pass  # table doesn't exist on first run; nothing to exclude
prior_synthetic_norm = {_norm_q(x) for x in prior_synthetic}
print(f"  prior synthetic Qs (excluded from history mining): {len(prior_synthetic)}")

historical_qa: list[dict] = []  # {question, genie_sql, genie_description, status, created_ts}
convs_resp = _get_json(f"/api/2.0/data-rooms/{GENIE_SPACE_ID}/conversations?page_size={MAX_CONVERSATIONS}")
for c in (convs_resp.get("conversations") or [])[:MAX_CONVERSATIONS]:
    cid = c.get("conversation_id") or c.get("id")
    if not cid:
        continue
    m = _get_json(f"/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{cid}/messages")
    for msg in (m.get("messages") or []):
        # Mine terminal states that still carry content + SQL. QUERY_RESULT_EXPIRED keeps the
        # question text and the attachment query even though the result rows have aged out (L2.3).
        if msg.get("status") not in ("COMPLETED", "FAILED", "QUERY_RESULT_EXPIRED"):
            continue
        content = (msg.get("content") or "").strip()
        if content in prior_synthetic or _norm_q(content) in prior_synthetic_norm:
            continue  # contamination filter (exact + normalized match)
        # first attachment with a query is the main Genie SQL
        genie_sql, genie_desc = "", ""
        for a in (msg.get("attachments") or []):
            q = a.get("query") or {}
            if q.get("query"):
                genie_sql = q["query"]
                genie_desc = q.get("description") or ""
                break
        historical_qa.append({
            "question": content,
            "genie_sql": genie_sql,
            "genie_description": genie_desc,
            "status": msg.get("status"),
            "created_ts": msg.get("created_timestamp"),
        })
print(f"  historical Q&A    : {len(historical_qa)} (COMPLETED={sum(1 for x in historical_qa if x['status']=='COMPLETED')}, FAILED={sum(1 for x in historical_qa if x['status']=='FAILED')})")

# --- 4. Resolve which tables to eval: widget override > all registered tables ---
if GENIE_TABLES:
    print(f"  using widget table list ({len(GENIE_TABLES)} tables)")
else:
    GENIE_TABLES = list(space_tables_registered)
    print(f"  widget blank → using all {len(GENIE_TABLES)} registered tables")
assert GENIE_TABLES, "No tables resolved — space has no registered tables AND genie_tables widget is blank"

# --- 5. Per-table UC metadata: comments, row counts, low-card samples, FKs ---
def _bt(fqn: str) -> str:
    """Backtick each part of a 3-part FQN so hyphenated / Lakebase-federated catalog names
    (e.g. test-cesar-lakebase-federated.public.t) don't raise INVALID_IDENTIFIER 42602 (caught live)."""
    parts = fqn.split(".")
    return ".".join(f"`{p.strip('`')}`" for p in parts) if len(parts) == 3 else fqn


def describe_table_extended(fqn: str) -> dict:
    """Pulls schema + column comments + table comment via DESCRIBE TABLE EXTENDED."""
    try:
        rows = spark.sql(f"DESCRIBE TABLE EXTENDED {_bt(fqn)}").collect()
    except Exception as e:
        return {"schema_lines": [], "table_comment": "", "error": str(e)}
    schema_lines, table_comment = [], ""
    in_col_section = True
    for r in rows:
        name, typ, cmt = (r["col_name"] or "").strip(), (r["data_type"] or "").strip(), (r.get("comment") or "" if hasattr(r,"get") else getattr(r,"comment","") or "")
        if not name or name.startswith("#"):
            # The column section is ALWAYS emitted first, so the FIRST '#'-prefixed header ends it.
            # Allowlisting only '# Detailed'/'# Partitioning' leaked '# Metadata Columns' → a phantom
            # `_metadata` struct column into the LLM grounding context (caught live on SAP MVs).
            if name.startswith("#"):
                in_col_section = False
            continue
        if in_col_section:
            desc = f"  - {name}: {typ}" + (f"  -- {cmt}" if cmt else "")
            schema_lines.append(desc)
        elif name.lower() == "comment":
            table_comment = typ  # DESCRIBE EXTENDED puts the table comment in data_type col
    return {"schema_lines": schema_lines, "table_comment": table_comment}


def sample_low_card_columns(fqn: str, cap: int, n_rows=None) -> str:
    # Empty / inaccessible table → nothing to ground against (L2.2). n_rows: 0 = empty, -1 = denied.
    if n_rows is not None and n_rows <= 0:
        return "  (table empty or inaccessible — no values to ground against)"
    cols = spark.sql(f"DESCRIBE TABLE {_bt(fqn)}").collect()
    # L2.6: ground categorical-ish columns, not just strings — booleans, dates, and integer/decimal
    # columns (years, codes, statuses like census_year) too. The cardinality guard below cheaply
    # excludes high-card ids/measures via LIMIT. Skip float/double/timestamp/complex types.
    _skip = ("struct", "map", "array", "double", "float", "timestamp", "binary", "variant", "interval")
    _ok = ("string", "boolean", "date", "tinyint", "smallint", "int", "bigint", "decimal", "long")
    cand = []
    for r in cols:
        nm, ty = r["col_name"], (r["data_type"] or "").lower()
        if not nm or nm.startswith("#"):
            continue
        if any(s in ty for s in _skip):
            continue
        if any(t in ty for t in _ok):
            cand.append(nm)
    lines = []
    for c in cand:
        try:
            distinct = spark.sql(
                f"SELECT DISTINCT `{c}` AS v FROM {_bt(fqn)} WHERE `{c}` IS NOT NULL LIMIT {cap + 1}"
            ).collect()
        except Exception:
            continue
        if 0 < len(distinct) <= cap:
            vals = sorted({str(r["v"]) for r in distinct})
            lines.append(f"  - {c}: {vals}")
    return "\n".join(lines) if lines else "  (no low-cardinality columns)"


def row_count(fqn: str) -> int:
    try:
        return spark.sql(f"SELECT COUNT(*) c FROM {_bt(fqn)}").collect()[0]["c"]
    except Exception:
        return -1


def foreign_keys_for_table(fqn: str) -> list[dict]:
    """Returns [{fk_col, ref_table, ref_col}] from INFORMATION_SCHEMA.
    Requires FK constraints declared on the table — silent if absent."""
    try:
        cat, sch, tbl = fqn.split(".", 2)
    except ValueError:
        return []
    sql = f"""
    SELECT kcu.column_name AS fk_col,
           ccu.table_catalog || '.' || ccu.table_schema || '.' || ccu.table_name AS ref_table,
           ccu.column_name AS ref_col
    FROM `{cat}`.information_schema.table_constraints tc
    JOIN `{cat}`.information_schema.key_column_usage kcu
      ON tc.constraint_catalog = kcu.constraint_catalog
     AND tc.constraint_schema  = kcu.constraint_schema
     AND tc.constraint_name    = kcu.constraint_name
    JOIN `{cat}`.information_schema.referential_constraints rc
      ON tc.constraint_catalog = rc.constraint_catalog
     AND tc.constraint_schema  = rc.constraint_schema
     AND tc.constraint_name    = rc.constraint_name
    JOIN `{cat}`.information_schema.constraint_column_usage ccu
      ON rc.unique_constraint_catalog = ccu.constraint_catalog
     AND rc.unique_constraint_schema  = ccu.constraint_schema
     AND rc.unique_constraint_name    = ccu.constraint_name
    WHERE tc.table_schema = '{sch}' AND tc.table_name = '{tbl}' AND tc.constraint_type = 'FOREIGN KEY'
    """
    try:
        return [r.asDict() for r in spark.sql(sql).collect()]
    except Exception:
        return []


table_contexts = {}
all_fks: list[tuple] = []  # (from_table, from_col, to_table, to_col)
for t in GENIE_TABLES:
    print(f"\nIntrospecting {t}...")
    meta = describe_table_extended(t)
    if meta.get("error"):
        print(f"  ERROR: {meta['error']}")
        continue
    schema = "\n".join(meta["schema_lines"])
    rc = row_count(t)
    col_samples = sample_low_card_columns(t, MAX_DISTINCT, rc)
    fks = foreign_keys_for_table(t)
    for fk in fks:
        all_fks.append((t, fk["fk_col"], fk["ref_table"], fk["ref_col"]))
    table_contexts[t] = {
        "schema": schema,
        "col_samples": col_samples,
        "table_comment": meta["table_comment"],
        "row_count": rc,
        "foreign_keys": fks,
    }
    print(f"  row_count : {rc}")
    print(f"  comment   : {meta['table_comment'][:160]}")
    print(f"  schema+comments:\n{schema}")
    print(f"  col samples:\n{col_samples}")
    if fks:
        print(f"  foreign keys: {fks}")

# Prune tables the runner couldn't introspect (USE CATALOG/USE SCHEMA denied, invalid identifier).
# The data-rooms API returns ALL registered tables regardless of the caller's UC grants, so on a
# shared workspace many are inaccessible. WITHOUT this prune, generation (which iterates the full
# GENIE_TABLES) hits `table_contexts[t]` → KeyError and crashes the whole run (confirmed live: 3
# of 12 probed spaces were zero-survivor wipeouts). Prune to what resolved, warn, fail loudly if empty.
_dropped = [t for t in GENIE_TABLES if t not in table_contexts]
if _dropped:
    print(f"\nWARNING: {len(_dropped)}/{len(GENIE_TABLES)} registered tables were inaccessible and "
          f"have been dropped (eval set will not cover them): {_dropped}")
GENIE_TABLES = [t for t in GENIE_TABLES if t in table_contexts]
assert GENIE_TABLES, (
    "All registered tables are inaccessible to the runner (USE CATALOG / USE SCHEMA denied, or "
    "invalid identifiers). Grant the notebook's principal access to the Genie space's tables, "
    "or set the genie_tables widget to tables you can read."
)

# --- 6. Assemble SPACE_CONTEXT block fed to the generator ---
def _fmt_curated(items: list[dict], cap: int = 20) -> str:
    out = []
    for q in items[:cap]:
        t = q.get("question_text") or q.get("text") or q.get("question") or ""
        if t:
            out.append(f"  - {t}")
    return "\n".join(out)


def _fmt_historical(items: list[dict], cap: int = 5) -> str:
    """Show the NL question + one-line SQL summary for each, highlight FAILED as gaps."""
    out = []
    # Prioritize FAILED (known gaps) then COMPLETED (style)
    failed = [x for x in items if x["status"] == "FAILED"][:cap]
    completed = [x for x in items if x["status"] == "COMPLETED"][:cap - len(failed)]
    for x in failed:
        out.append(f"  - [KNOWN GAP] {x['question'][:200]}")
    for x in completed:
        q = x["question"][:200]
        sql_preview = " ".join((x["genie_sql"] or "").split())[:160]
        out.append(f"  - {q}  -- genie answered with: {sql_preview}")
    return "\n".join(out)


def _fmt_fks(fks: list[tuple]) -> str:
    if not fks:
        return ""
    lines = ["Join paths declared in UC foreign keys (use these for cross-table questions):"]
    for from_t, from_c, to_t, to_c in fks:
        lines.append(f"  - {from_t}.{from_c} → {to_t}.{to_c}")
    return "\n".join(lines)


def _infer_shared_join_columns(contexts: dict) -> str:
    """L2.5: when no UC foreign keys are declared (the common case — only ~1/17 spaces had any),
    infer candidate join columns by shared column NAME across tables, so cross-table questions have
    *some* grounding. These are heuristic, not authoritative — labelled as such for the generator."""
    import re as _re
    col_to_tables = {}
    for tbl, ctx in contexts.items():
        for line in (ctx.get("schema") or "").splitlines():
            m = _re.match(r"\s*-\s*([A-Za-z0-9_]+):", line)
            if m:
                col_to_tables.setdefault(m.group(1).lower(), set()).add(tbl)
    shared = {c: sorted(ts) for c, ts in col_to_tables.items()
              if len(ts) >= 2 and c not in ("id", "name", "date", "value", "type", "status")}
    if not shared:
        return ""
    lines = ["Candidate join columns INFERRED from shared column names (NOT declared FKs — verify "
             "before relying on them; prefer columns that look like keys):"]
    for c, ts in list(shared.items())[:15]:
        lines.append(f"  - `{c}` appears in: {', '.join(ts)}")
    return "\n".join(lines)


_sections = [f"Space title: {space_title}",
             f"Space description: {space_description}"]
if space_instructions:
    _sections.append(f"Space instructions / capabilities / limitations:\n{space_instructions}")
if space_tables_registered:
    _sections.append(f"Tables registered on this space: {space_tables_registered}")
if curated_questions and len(GENIE_TABLES) > 1:
    # Multi-table spaces benefit from few-shot curated examples — they unlock domain
    # vocabulary the generator can't infer from schema alone (observed: curated few-shot
    # lifts sql_executes and sql_answers on multi-table spaces).
    _sections.append(
        "GOLD CURATED QUESTIONS (operator-blessed examples — these define the style, domain "
        "vocabulary, and business framing this space's users actually employ). EMULATE this "
        "style: same business terminology (metric names, time framing, entity types, segmentation "
        "dimensions). Do NOT copy verbatim — generate NEW questions in the SAME STYLE that explore "
        "DIFFERENT filter combinations, time windows, or aggregation shapes.\n"
        + _fmt_curated(curated_questions)
    )
elif curated_questions:
    # Single-table mode: skip few-shot in prompt — too many examples mode-collapse the
    # chunked generator (observed: clarity drops on single-table spaces). Curated still feeds
    # the realism reference, just not the generator prompt.
    print(f"  {len(curated_questions)} curated questions kept for realism reference but NOT injected into generator prompt (single-table mode avoids collapse)")
if historical_qa and INCLUDE_HISTORICAL:
    _sections.append(
        "Real user questions from this space's history — use ONLY to learn the vocabulary, "
        "metric names, and entity types this space's users care about. "
        "DO NOT copy or paraphrase these questions verbatim — your generated questions must "
        "explore NEW shapes, NEW filter combinations, NEW time windows that these examples did NOT cover. "
        "[KNOWN GAP] markers = Genie failed on this shape; generate similar-shape questions to stress-test it.\n"
        + _fmt_historical(historical_qa)
    )
elif historical_qa:
    print(f"  (historical_qa kept for realism scorer; NOT injected into SPACE_CONTEXT — include_historical_in_context=false)")
if all_fks:
    _sections.append(_fmt_fks(all_fks))
elif len(GENIE_TABLES) > 1:
    # No declared FKs (the common case) — fall back to shared-column-name join hints (L2.5).
    _inferred = _infer_shared_join_columns(table_contexts)
    if _inferred:
        _sections.append(_inferred)
if EXTRA_INSTRUCTIONS:
    _sections.append(f"Extra instructions (operator-provided):\n{EXTRA_INSTRUCTIONS}")

SPACE_CONTEXT = "\n\n".join(_sections)

# If the grounding context is essentially empty, tell the operator WHY — distinguish a
# genuinely sparse space from a broken extraction (permission/URL). _get_json already logged any
# HTTP errors above; this is the summary banner so a low-quality run isn't misread as "weak generator".
if not space_instructions and not curated_questions and not historical_qa:
    print("\n" + "!" * 60)
    print("WARNING: grounding context is nearly empty — no space instructions, no curated questions,")
    print("and no mineable history. If you expected content, check the _get_json log lines above for")
    print("HTTP 403 (permission) / 404 (wrong space id). Otherwise this space is simply sparse and the")
    print("generator will infer from table schema + samples alone (lower clarity is expected).")
    print("!" * 60)

# Single-table mode: when only 1 table is in scope, the generator tends to produce
# repetitive questions because there's no cross-table structure to differentiate. Inject
# explicit shape-diversity guidance to break the pattern. (Single-table spaces are where mode
# collapse is worst — clarity can crater without this guidance.)
if len(GENIE_TABLES) == 1:
    SPACE_CONTEXT += (
        "\n\nSINGLE-TABLE MODE — only one target table is in scope. The default rules tend to "
        "produce repetitive questions when there are no joins to vary shape. To avoid collapse, "
        "REQUIRE the generated set to vary along ALL of these axes (no two questions should be "
        "identical along all four):\n"
        "  - Time grain: include daily, weekly, monthly, and YoY/MoM granularity\n"
        "  - Filter dimension: alternate which categorical column drives the WHERE clause\n"
        "  - Aggregation depth: mix flat aggregates with windowed (LAG/LEAD/RANK) and "
        "self-join cohort/funnel shapes\n"
        "  - Ranking metric: top-N by different metrics (revenue, count, price, ratio, density)\n"
        "Two questions whose only difference is a column name in the WHERE clause should NOT both appear."
    )

print("\n" + "=" * 60)
print("SPACE_CONTEXT (what the generator sees):")
print("=" * 60)
print(SPACE_CONTEXT[:3000] + ("..." if len(SPACE_CONTEXT) > 3000 else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register the question-generation prompt in UC Prompt Registry
# MAGIC
# MAGIC Versioned under `{catalog}.{schema}.genie_eval_question_gen` with `@candidate` → `@production` aliases.
# MAGIC Template variables: `n`, `table`, `schema`, `space_context`, `column_samples`.

# COMMAND ----------

QUESTION_GEN_PROMPT = """You are building a rigorous evaluation set for a Databricks Genie (text-to-SQL) space.
Your job is to stress-test Genie — produce questions hard enough to find real gaps, not just confirm basics work.

Context about this Genie space (use its vocabulary, metrics, and typical user framing).
If the context lists "Capabilities:", bias questions toward stress-testing those exact capabilities.
If it lists "Limitations:" or "Cannot...", DO NOT generate questions that fall in those limitations —
stay inside scope, but pick the HARDEST in-scope shapes.
{{space_context}}

Target table: {{table}}

Schema:
{{schema}}

Actual values observed in low-cardinality columns (ALWAYS use these literal values in WHERE clauses — do not invent codes):
{{column_samples}}

Produce {{n}} realistic, VARIED natural-language questions a real business user would ask. Each paired with:
  - question: the NL question, phrased in the business vocabulary of the space (not schema column names verbatim).
  - expected_sql: standard Databricks SQL, using {{table}} exactly as given, with grounded literals.
  - expected_shape: short description of the result shape (e.g. "single scalar", "2 columns x 5 rows", "time series by month").
  - category: one of {aggregation, filter, time_series, ranking, distinct_count, ratio_or_delta, windowed, multi_condition, cohort, funnel}
  - difficulty: one of {easy, medium, hard}
  - why_non_trivial: ONE sentence on why this question is not trivial (hint at SQL complexity or business insight).

Difficulty definitions — be strict:
  - easy = single aggregation over one filter. No grouping beyond one dimension. Single-digit line count SQL.
  - medium = single-level aggregation + grouping + ORDER BY/LIMIT, or a straightforward time-bucketed count.
  - hard = REQUIRES AT LEAST ONE of these SQL features:
       * a named CTE (WITH clause), or two+ subqueries
       * a window function: LAG, LEAD, RANK, DENSE_RANK, NTILE, ROW_NUMBER, or percentile_approx()
       * a self-join across time periods (period-over-period lift, cohort retention)
       * a correlated subquery or EXISTS/NOT EXISTS
       * a HAVING clause that references a different aggregate than the SELECT
       * conditional aggregation with multiple CASE WHEN ... THEN ... ELSE branches inside SUM/COUNT
       * APPROX_COUNT_DISTINCT on a filtered cohort combined with overall
       * a pivot or lateral explode
    Each hard question must have at least one of the above bullets in its expected_sql.

Distribution constraints across the {{n}} questions — enforce exactly:
  - AT MOST 1 plain scalar COUNT or SUM question (i.e. at most 1 "easy"-difficulty item).
  - AT LEAST {{min_hard}} questions MUST be difficulty="hard" by the strict definition above.
  - AT LEAST 1 time_series question using date_trunc or a windowed aggregation.
  - AT LEAST 1 ratio_or_delta question (A/B, WoW, MoM, percent-of-total, or YoY delta).
  - AT LEAST 1 multi_condition question combining 2+ AND'd predicates.
  - AT LEAST 1 ranking question with a TOP-N and an explicit tiebreaker or HAVING.
  - If the table looks event-like (has timestamps + entity IDs + event labels), at least 1 funnel or cohort question.

Rules:
  1. Every string WHERE/HAVING literal must come from the observed column values. Never invent codes.
  2. Any numeric threshold, window, or ordering MUST be stated in the question itself — no ambiguous "recently", "high", "top" without a number.
  3. Questions must be answerable from {{table}} standalone (self-joins allowed).
  4. Do NOT produce trivial questions whose answer is equivalent to DESCRIBE TABLE or SHOW COLUMNS.
  5. Use business vocabulary from space_context, not raw column names, in the NL question.
  6. The hard questions should be the kind a seasoned data analyst would need 5+ minutes to write and debug.
  7. Do NOT copy or paraphrase any of the historical user questions shown in the context — those are vocabulary
     references, not templates. Each generated question must materially differ from every historical example
     in at least one of: filter conditions, time window, grouping granularity, ranking dimension, or output shape.
  8. If the context shows "GOLD CURATED QUESTIONS", treat them as the gold standard for this space's style and
     domain vocabulary. Your generated questions MUST use the same business terminology (metric names, entity
     names, time framing, segmentation dimensions) as those curated examples. A question that uses generic
     analyst language ("count of records", "average value") instead of the operator's domain language
     ("attainment by program", "policyholder retention by region") is a FAILURE. Match the style, explore new shapes.

Return ONLY a JSON object with a single key `questions` whose value is an array of objects with EXACTLY these fields: question, expected_sql, expected_shape, category, difficulty, why_non_trivial.
Do NOT include markdown fences or commentary.
"""

# Hook 1: append operator-supplied extra rules after rule #8.
if EXTRA_PROMPT_RULES:
    QUESTION_GEN_PROMPT = QUESTION_GEN_PROMPT.replace(
        "Return ONLY a JSON object",
        f"  9. {EXTRA_PROMPT_RULES.strip()}\n\nReturn ONLY a JSON object",
    )
    print(f"  hook 1 (extra_prompt_rules): appended {len(EXTRA_PROMPT_RULES)} chars to prompt")

pv = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=QUESTION_GEN_PROMPT,
    commit_message="Grounded v2: space context + low-card column samples",
    tags={
        "ai_function": "genie_eval_gen",
        "project": "genie_eval_set_generator",
        "owner": CURRENT_USER,
    },
)
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="candidate", version=pv.version)
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=pv.version)
print(f"Registered {PROMPT_NAME} v{pv.version} — @candidate + @production")

# Hook 5 validation: if a custom prompt URI is set, load it now and verify it formats
# cleanly with the same template variables we pass at runtime. Fail fast on missing/extra
# variables so the error surfaces before we waste time on extraction + generation.
if CUSTOM_PROMPT_URI:
    print(f"\nHook 5: validating custom prompt at {CUSTOM_PROMPT_URI}...")
    try:
        _custom_pv = mlflow.genai.load_prompt(CUSTOM_PROMPT_URI)
    except Exception as e:
        raise RuntimeError(
            f"custom_prompt_uri='{CUSTOM_PROMPT_URI}' could not be loaded: {e}. "
            f"Register your prompt first with mlflow.genai.register_prompt() and ensure the alias exists."
        )
    _required_vars = {"n", "min_hard", "table", "schema", "space_context", "column_samples"}
    try:
        _custom_pv.format(
            n="3", min_hard="1", table="cat.sch.tbl", schema="(probe)",
            space_context="(probe)", column_samples="(probe)",
        )
    except Exception as e:
        raise RuntimeError(
            f"custom_prompt_uri='{CUSTOM_PROMPT_URI}' template format() failed: {e}. "
            f"Required template variables: {sorted(_required_vars)}. "
            f"Each must appear as {{{{var}}}} (Jinja-style) in your template."
        )
    print(f"  ✓ custom prompt validates against required template variables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate the grounded eval set
# MAGIC
# MAGIC Loads `{PROMPT_NAME}@production`, renders with per-table context, calls the generator via AI Gateway.

# COMMAND ----------

# Shape focuses for chunked generation. Each chunk asks the generator to bias toward a
# specific SQL pattern, breaking the mode collapse that happens when a single call has to
# enforce many distribution constraints simultaneously.
SHAPE_FOCUSES = [
    ("aggregation_filter",  "Bias this batch toward aggregations + multi-condition filters. "
                            "Most questions should be 'sum/avg/count of X grouped by Y where Z'."),
    ("ranking_topn",        "Bias this batch toward TOP-N rankings with explicit tiebreakers and HAVING clauses. "
                            "Vary the ranking metric (revenue, count, price, ratio, density)."),
    ("window_cohort",       "Bias this batch toward window functions (LAG, LEAD, RANK, ROW_NUMBER, PERCENTILE) "
                            "and self-joins for cohort/funnel/period-over-period analysis."),
    ("ratio_delta_time",    "Bias this batch toward ratio_or_delta and time_series questions: WoW, MoM, YoY, "
                            "percent-of-total, conditional aggregation with multiple CASE WHEN branches."),
]

@mlflow.trace
def generate_questions_chunk(table_fqn: str, n: int, min_hard: int, focus_label: str, focus_clause: str,
                              prompt_alias: str = "production") -> list[dict]:
    """One generation call with a shape-focus extension appended.

    Hook 5: when CUSTOM_PROMPT_URI is set, load that URI instead of the default.
    GEPA optimization (Phase 7) still operates on the registered default; the override
    is run-time only.
    """
    ctx = table_contexts[table_fqn]
    prompt_uri = CUSTOM_PROMPT_URI or f"prompts:/{PROMPT_NAME}@{prompt_alias}"
    pv_loaded = mlflow.genai.load_prompt(prompt_uri)
    rendered = pv_loaded.format(
        n=str(n),
        min_hard=str(min_hard),
        table=table_fqn,
        schema=ctx["schema"],
        space_context=SPACE_CONTEXT + f"\n\nSHAPE FOCUS for this batch ({focus_label}):\n{focus_clause}",
        column_samples=ctx["col_samples"],
    )
    gen_temp = 0.85 if len(GENIE_TABLES) == 1 else 0.7
    resp = openai_client.chat.completions.create(
        model=GENERATOR_ENDPOINT,
        messages=[{"role": "user", "content": rendered}],
        max_tokens=8192,
        temperature=gen_temp,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if content is None:
        finish = getattr(resp.choices[0], "finish_reason", None)
        raise RuntimeError(
            f"Generator returned None content for {table_fqn}/{focus_label} "
            f"(finish_reason={finish}). Likely max_tokens overflow at n={n}."
        )
    parsed = json.loads(content)
    questions = parsed.get("questions", [])
    for q in questions:
        q["table"] = table_fqn
        q["_shape_focus"] = focus_label
    return questions


def generate_questions(table_fqn: str, n: int, min_hard: int, prompt_alias: str = "production") -> list[dict]:
    """Chunked generation: split n into ≤4 batches with rotating shape focuses to break mode collapse.
    Falls back to a single call when n is small enough that chunking adds no diversity benefit."""
    if n <= 12:
        # Small enough — no chunk benefit. Use the first focus to keep the same code path.
        focus_label, focus_clause = SHAPE_FOCUSES[0]
        return generate_questions_chunk(table_fqn, n, min_hard, focus_label, focus_clause, prompt_alias)
    n_chunks  = min(4, max(2, n // 10))
    per_chunk = -(-n // n_chunks)            # ceil divide
    hard_per_chunk = max(1, min_hard // n_chunks)
    out = []
    for i in range(n_chunks):
        focus_label, focus_clause = SHAPE_FOCUSES[i % len(SHAPE_FOCUSES)]
        chunk_n = min(per_chunk, n - len(out))
        if chunk_n <= 0:
            break
        print(f"  chunk {i+1}/{n_chunks} → {chunk_n} questions, focus={focus_label}")
        out.extend(generate_questions_chunk(table_fqn, chunk_n, hard_per_chunk, focus_label, focus_clause, prompt_alias))
    return out


all_questions = []
for t in GENIE_TABLES:
    print(f"Generating {QUESTIONS_PER_TABLE} questions for {t} ({MIN_HARD_PER_TABLE} hard min)...")
    qs = generate_questions(t, QUESTIONS_PER_TABLE, MIN_HARD_PER_TABLE)
    print(f"  got {len(qs)}")
    all_questions.extend(qs)

# Self-dedup: embed all generated, drop those whose nearest-neighbor cosine to a kept Q
# exceeds 0.92. Reduces within-set redundancy (lowers gen_self_similarity in phase 6.5).
def _selfdedup(items: list[dict], threshold: float = 0.92) -> list[dict]:
    if not EMBEDDING_ENDPOINT or len(items) < 2:
        return items
    try:
        import numpy as _np
        out_emb = []
        for i in range(0, len(items), 32):
            r = openai_client.embeddings.create(
                model=EMBEDDING_ENDPOINT,
                input=[q["question"] for q in items[i:i+32]],
            )
            out_emb.extend([d.embedding for d in r.data])
        E = _np.asarray(out_emb, dtype=_np.float32)
        E /= (_np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        kept_idx = []
        for i in range(len(items)):
            if not kept_idx:
                kept_idx.append(i)
                continue
            sims = E[kept_idx] @ E[i]
            if sims.max() < threshold:
                kept_idx.append(i)
        kept = [items[i] for i in kept_idx]
        dropped = len(items) - len(kept)
        if dropped:
            print(f"Self-dedup: dropped {dropped}/{len(items)} near-duplicates (cos ≥ {threshold})")
        return kept
    except Exception as e:
        print(f"Self-dedup: skipping due to {e}")
        return items

all_questions = _selfdedup(all_questions)

import uuid as _uuid
eval_pdf = pd.DataFrame(all_questions)
eval_pdf["question_id"] = [str(_uuid.uuid4()) for _ in range(len(eval_pdf))]
print(f"Generated {len(eval_pdf)} candidate eval items (will be validated + scored before publish).")
display(spark.createDataFrame(eval_pdf))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validate every `expected_sql`
# MAGIC
# MAGIC Execute each generated `expected_sql` against UC. Capture `expected_rows` (up to 50) so the
# MAGIC eval item is self-contained — any downstream harness can diff against this without re-executing.
# MAGIC Records that fail to parse or return zero rows are kept but flagged, so the scoring phase can score them.

# COMMAND ----------

def execute_expected(sql: str):
    try:
        rows = [list(r) for r in spark.sql(sql).limit(50).collect()]
        return rows, None
    except Exception as e:
        return None, str(e)


# v0.2: re-derive difficulty from SQL STRUCTURE, not the generator's self-label, so the
# per-difficulty profile is not circular. The generator's own `difficulty` is kept for reference;
# the profiles downstream group on `difficulty_structural`.
HARD_SQL_TOKENS = (" WITH ", " OVER (", " OVER(", "LAG(", "LEAD(", "RANK(", "DENSE_RANK(", "NTILE(",
                   "ROW_NUMBER(", "PERCENTILE", "APPROX_COUNT_DISTINCT", " EXISTS", " HAVING ",
                   "CASE WHEN", " INTERSECT ", " QUALIFY ")


def structural_difficulty(sql: str) -> str:
    s = f" {(sql or '').upper()} "
    if any(tok in s for tok in HARD_SQL_TOKENS):
        return "hard"
    if " JOIN " in s or "GROUP BY" in s or ("ORDER BY" in s and " LIMIT " in s):
        return "medium"
    return "easy"


validated = []
for _, r in eval_pdf.iterrows():
    rows, err = execute_expected(r["expected_sql"])
    n_rows = len(rows) if rows is not None else 0
    struct_diff = structural_difficulty(r["expected_sql"])
    validated.append({
        **r.to_dict(),
        "expected_rows_json": json.dumps(rows, default=str) if rows is not None else "",
        "n_result_rows": n_rows,
        "sql_error": err or "",
        "sql_executes": err is None,
        "nonempty_result": n_rows > 0,
        "difficulty_structural": struct_diff,
        "difficulty_label_matches": (str(r.get("difficulty", "")).lower() == struct_diff),
    })

validated_pdf = pd.DataFrame(validated)
print(f"SQL executes:      {validated_pdf['sql_executes'].sum()} / {len(validated_pdf)}")
print(f"Nonempty results:  {validated_pdf['nonempty_result'].sum()} / {len(validated_pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Score eval-set quality
# MAGIC
# MAGIC Per-item scorers (applied to the `(question, expected_sql)` pair — *not* to Genie):
# MAGIC - **`question_clarity`** (LLM judge): question must be unambiguous given the schema.
# MAGIC - **`sql_answers_question`** (LLM judge): SQL must semantically answer the NL question.
# MAGIC - **`grounded_literals`** (deterministic): every WHERE literal must appear in the sampled column values.
# MAGIC - **`sql_valid`** (deterministic): expected_sql executed and returned rows.
# MAGIC - **`safety`** (built-in): no unsafe content.

# COMMAND ----------

from mlflow.genai.scorers import Safety, Guidelines, scorer
from mlflow.entities import Feedback
import re

question_clarity = Guidelines(
    name="question_clarity",
    guidelines=(
        "The `inputs.question` must be unambiguous: a human reading just the question and the "
        "table schema in `inputs.schema` should know exactly what is being asked, including any "
        "aggregation, grouping, filter, or ordering. Penalize vague terms ('recently', 'some', "
        "'high', 'top') unless the question itself defines them with a specific threshold, "
        "timeframe, or numeric limit."
    ),
    model=JUDGE_MODEL_URI,
)

sql_answers_question = Guidelines(
    name="sql_answers_question",
    guidelines=(
        "The `outputs` SQL must answer `inputs.question` over the table described in "
        "`inputs.schema`. Verify: aggregation type (count vs sum vs distinct vs avg), grouping "
        "granularity, filter conditions, ordering, and limit all match the question's intent. "
        "If the SQL would return a different result than what the question asks for, this fails."
    ),
    model=JUDGE_MODEL_URI,
)


@scorer
def grounded_literals(inputs, outputs) -> Feedback:
    allowed = set()
    for line in (inputs.get("column_samples") or "").splitlines():
        allowed.update(re.findall(r"'([^']+)'", line))
        allowed.update(re.findall(r'"([^"]+)"', line))
    sql_literals = set(re.findall(r"'([^']+)'", outputs or ""))
    invented = {
        v for v in sql_literals - allowed
        if not re.match(r"^\d{4}-\d{2}-\d{2}$|^\d+(\.\d+)?$", v)
    }
    # Fail VISIBLE, not open: when there is nothing to ground against, this is "not
    # evaluable" — value=None is excluded from the aggregate rather than silently scoring a pass.
    if not allowed:
        return Feedback(value=None, rationale="not evaluable — no column samples to ground against")
    if invented:
        return Feedback(value=False, rationale=f"Ungrounded literals: {sorted(invented)}")
    return Feedback(value=True, rationale="All string literals grounded in column_samples")


@scorer
def sql_valid(expectations) -> Feedback:
    err = expectations.get("sql_error") or ""
    n = int(expectations.get("n_result_rows") or 0)
    if err:
        return Feedback(value=False, rationale=f"SQL error: {err[:200]}")
    if n == 0:
        return Feedback(value=False, rationale="SQL executes but returned 0 rows — likely bad filter")
    return Feedback(value=True, rationale=f"{n} rows returned")


judge_df = pd.DataFrame([
    {
        "inputs": {
            "question": r["question"],
            "schema": table_contexts[r["table"]]["schema"],
            "column_samples": table_contexts[r["table"]]["col_samples"],
        },
        "outputs": r["expected_sql"] or "",
        "expectations": {
            "sql_error": r["sql_error"] or "",
            "n_result_rows": int(r["n_result_rows"]),
        },
    }
    for _, r in validated_pdf.iterrows()
])

with mlflow.start_run(run_name="eval_set_quality") as run:
    mlflow.log_param("genie_space_id", GENIE_SPACE_ID)
    mlflow.log_param("n_eval_items", len(validated_pdf))
    mlflow.log_metric("fraction_sql_executes", float(validated_pdf["sql_executes"].mean()))
    mlflow.log_metric("fraction_nonempty", float(validated_pdf["nonempty_result"].mean()))
    eval_result = mlflow.genai.evaluate(
        data=judge_df,
        scorers=[Safety(), question_clarity, sql_answers_question, grounded_literals, sql_valid],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.5 Diversity & leakage (descriptive — NOT a verdict)
# MAGIC
# MAGIC Embeds each generated question and each **reference question** (historical_qa + curated_questions
# MAGIC merged into one operator-validated corpus) and reports **descriptive numbers only** — no z-score,
# MAGIC no σ, no good/bad bands. The SME reads them:
# MAGIC
# MAGIC - **`mean_nn_cosine`** (P@H): mean nearest-neighbour cosine of each generated question to the
# MAGIC   reference corpus. ≈1.0 ⇒ the synthetic set is near-verbatim copies of existing questions;
# MAGIC   lower ⇒ more novel phrasing.
# MAGIC - **`nn_cosine_p10`**: 10th-percentile NN cosine — surfaces off-vocabulary outliers in the lower tail.
# MAGIC - **`gen_self_similarity`**: mean within-generated NN cosine — intra-set redundancy (lower = more diverse).
# MAGIC - **`ref_self_similarity`**: mean within-reference NN cosine — descriptive context for how tight the
# MAGIC   reference itself is. This is NOT a baseline for a hypothesis test; it is shown for comparison only.
# MAGIC
# MAGIC The **one** verdict here is a leakage guard:
# MAGIC - **`leakage_risk = (mean_nn_cosine ≥ 0.97)`** — the synthetic set is near-verbatim copies of
# MAGIC   reference questions, so it cannot catch real Genie mistakes. Everything else is for the SME to read.
# MAGIC
# MAGIC Requires `embedding_endpoint` and ≥ `MIN_REF_CORPUS` reference questions; skips with an explicit reason otherwise.
# MAGIC
# MAGIC > **v0.1 note.** This phase previously computed a bootstrap "realism z-score" with
# MAGIC > good/expanded/borderline/bad bands. That statistic was removed: it divided a cross-set quantity
# MAGIC > (P@H) by the bootstrap SD of a within-set quantity, so it was not a standardised statistic and the
# MAGIC > σ bands were arbitrary cut points. A *proper* null is a v0.2 item.

# COMMAND ----------

import numpy as np

# Build reference corpus = historical_qa + curated_questions (deduped). Curated Qs are
# operator-blessed, history is real users — both belong in the realism reference for narrow spaces.
historical_questions_for_realism = [x["question"] for x in historical_qa if x.get("question")]
_curated_texts = [
    (q.get("question_text") or q.get("text") or q.get("question") or "")
    for q in curated_questions
]
reference_corpus = list(historical_questions_for_realism)
_seen = set(reference_corpus)
for t in _curated_texts:
    t = (t or "").strip()
    if t and t not in _seen:
        reference_corpus.append(t)
        _seen.add(t)

n_history = len(historical_questions_for_realism)
n_curated = len([t for t in _curated_texts if (t or "").strip()])
n_ref     = len(reference_corpus)
MIN_REF_CORPUS = 10
# NOTE: the diversity reference is whatever real history + operator-curated questions exist for
# the space. We do NOT augment it with LLM paraphrases (v20 tried; paraphrases cluster ~0.95 with
# their seeds and distort every downstream number). Sparse-history spaces simply yield fewer
# reference Qs; the descriptive numbers below carry n_ref so the SME can weigh them accordingly.

if not EMBEDDING_ENDPOINT:
    print("Diversity: skipped — embedding_endpoint widget is blank.")
elif n_ref < MIN_REF_CORPUS:
    print(f"Diversity: skipped — reference corpus too small ({n_ref} = {n_history} history + {n_curated} curated; need ≥{MIN_REF_CORPUS}).")
else:
    def _embed_batch(texts, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            r = openai_client.embeddings.create(model=EMBEDDING_ENDPOINT, input=texts[i:i + batch])
            out.extend([d.embedding for d in r.data])
        v = np.asarray(out, dtype=np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v

    generated_questions_for_realism = validated_pdf["question"].tolist()
    try:
        G = _embed_batch(generated_questions_for_realism)
        H = _embed_batch(reference_corpus)  # reference corpus (history + curated)
    except Exception as e:
        print(f"Diversity: embedding call failed — {e}. Skipping.")
        G = H = None

    if G is not None:
        # mean_nn_cosine (P@H): how close each generated Q sits to its nearest reference Q.
        sim_GH = G @ H.T  # (N_gen, N_ref)
        nn_idx = sim_GH.argmax(axis=1)
        nn_sims = sim_GH.max(axis=1)
        mean_nn_cosine = float(nn_sims.mean())
        nn_cosine_p10 = float(np.percentile(nn_sims, 10))

        # gen_self_similarity: within-generated NN cosine (intra-set redundancy; lower = more diverse).
        if len(G) > 1:
            sim_GG = G @ G.T - np.eye(len(G), dtype=np.float32)
            gen_self_sim = float(sim_GG.max(axis=1).mean())
        else:
            gen_self_sim = 0.0

        # ref_self_similarity: within-reference NN cosine — DESCRIPTIVE context only (how tight the
        # reference itself is). NOT a baseline / standard error for any hypothesis test.
        if len(H) > 1:
            sim_HH = H @ H.T - np.eye(len(H), dtype=np.float32)
            ref_self_sim = float(sim_HH.max(axis=1).mean())
        else:
            ref_self_sim = 0.0

        # The one verdict: near-verbatim copies of reference questions = leakage (eval set can't
        # catch real Genie mistakes). Everything else is descriptive, for the SME to read.
        leakage_risk = bool(mean_nn_cosine >= 0.97)

        print("Diversity & leakage (descriptive):")
        print(f"  mean_nn_cosine (P@H)  : {mean_nn_cosine:.3f}  (≈1.0 = near-verbatim copies of reference)")
        print(f"  nn_cosine_p10         : {nn_cosine_p10:.3f}  (low tail = off-vocabulary outliers)")
        print(f"  gen_self_similarity   : {gen_self_sim:.3f}  (lower = more diverse generated set)")
        print(f"  ref_self_similarity   : {ref_self_sim:.3f}  (context: how tight the reference is, n_ref={n_ref})")
        print(f"  leakage_risk          : {leakage_risk}  (mean_nn_cosine ≥ 0.97)")

        per_q_realism_pdf = pd.DataFrame({
            "question": generated_questions_for_realism,
            "nearest_reference": [reference_corpus[int(i)] for i in nn_idx],
            "nn_sim": nn_sims.tolist(),
        }).sort_values("nn_sim").reset_index(drop=True)
        validated_pdf = validated_pdf.merge(
            per_q_realism_pdf[["question", "nn_sim"]].rename(columns={"nn_sim": "diversity_nn_sim"}),
            on="question", how="left",
        )

        # Run name kept as "eval_set_realism" for back-compat with the app's scorecard lookup.
        with mlflow.start_run(run_name="eval_set_realism"):
            mlflow.log_param("genie_space_id", GENIE_SPACE_ID)
            mlflow.log_param("embedding_endpoint", EMBEDDING_ENDPOINT)
            mlflow.log_param("n_generated", len(generated_questions_for_realism))
            mlflow.log_param("n_reference", n_ref)
            mlflow.log_param("n_history", n_history)
            mlflow.log_param("n_curated", n_curated)
            mlflow.log_param("leakage_risk", str(leakage_risk))
            mlflow.log_metric("mean_nn_cosine", mean_nn_cosine)
            mlflow.log_metric("nn_cosine_p10", nn_cosine_p10)
            mlflow.log_metric("gen_self_similarity", gen_self_sim)
            mlflow.log_metric("ref_self_similarity", ref_self_sim)
            mlflow.log_table(per_q_realism_pdf, "diversity_per_question.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Publish curated eval set to UC
# MAGIC
# MAGIC The Delta table is the hand-off to downstream Genie benchmarks / Agent Framework evals.

# COMMAND ----------

final_sdf = (
    spark.createDataFrame(validated_pdf)
    .withColumn("genie_space_id", F.lit(GENIE_SPACE_ID))
)
final_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(EVAL_SET_TABLE)
print(f"Wrote {final_sdf.count()} validated eval items to {EVAL_SET_TABLE}")
display(final_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Run the eval set against Genie (regression)
# MAGIC
# MAGIC Now that we have a curated set, *use* it — call Genie for every question, capture
# MAGIC its SQL + NL answer + executed rows, and score two ways:
# MAGIC - **Row concordance (lower bound)**: do Genie's rows match `expected_rows`?
# MAGIC - **`answer_quality`** (uncalibrated LLM judge): does Genie's NL answer address the question?
# MAGIC
# MAGIC > **Read the concordance number as a LOWER BOUND, not a correctness score:**
# MAGIC > (1) `expected_sql` is LLM-generated and only
# MAGIC > checked for executability — never independently verified to answer the question — so a
# MAGIC > "fail" can mean Genie was right and the generator wrong (inter-model agreement, not truth);
# MAGIC > (2) `expected_rows` are captured on the notebook's Spark cluster while Genie runs on the
# MAGIC > space's SQL warehouse, so cross-engine float/decimal/timestamp formatting can fail a correct
# MAGIC > answer. Same-engine execution + independent ground-truth verification are v0.2 items.
# MAGIC
# MAGIC Results land in `{CATALOG}.{SCHEMA}.genie_eval_runs`. Re-run this notebook after any
# MAGIC Genie space change and diff the MLflow `genie_regression` runs to get a quality delta.

# COMMAND ----------

EVAL_RUNS_TABLE = f"{CATALOG}.{SCHEMA}.genie_eval_runs"


def _collect_statement_rows(statement_id, first_result, row_cap: int = 50):
    """Page through statement-result chunks until we have > row_cap rows, so the cap-indeterminate
    guard in rows_match sees the TRUE result size — chunk 0 can be a small fraction of the result
    (confirmed live: 36 of 70 rows). Returns up to row_cap+1 rows."""
    rows, res = [], first_result
    while res is not None:
        rows.extend(list(r) for r in (getattr(res, "data_array", None) or []))
        nxt = getattr(res, "next_chunk_index", None)
        if len(rows) > row_cap or nxt is None:
            break
        try:
            res = w.statement_execution.get_statement_result_chunk_n(statement_id, nxt)
        except Exception:
            break
    return rows[: row_cap + 1]


def extract_genie_response(msg):
    """Return (genie_sql, genie_answer, statement_id) from a GenieMessage."""
    genie_sql = None
    genie_answer = None
    statement_id = None
    att_statement_id = None
    for att in (msg.attachments or []):
        if getattr(att, "query", None):
            genie_sql = att.query.query
            genie_answer = genie_answer or getattr(att.query, "description", None)
            # Capture the attachment's own statement_id as a fallback (L2.4) — some messages
            # carry it on the attachment query rather than on msg.query_result.
            att_statement_id = att_statement_id or getattr(att.query, "statement_id", None)
        if getattr(att, "text", None):
            genie_answer = att.text.content
    qr = getattr(msg, "query_result", None)
    if qr is not None:
        statement_id = getattr(qr, "statement_id", None)
    return genie_sql, genie_answer, (statement_id or att_statement_id)


@mlflow.trace
def run_one_genie(question: str) -> dict:
    """Traced: each Genie call shows up as a trace in the MLflow experiment UI."""
    t0 = time.time()
    try:
        msg = w.genie.start_conversation_and_wait(
            space_id=GENIE_SPACE_ID,
            content=question,
        )
        g_sql, g_ans, sid = extract_genie_response(msg)
        rows = None
        if sid:
            try:
                stmt = w.statement_execution.get_statement(sid)
                rows = _collect_statement_rows(sid, stmt.result, row_cap=50)
            except Exception as e:
                rows = None
        return {
            "genie_sql": g_sql or "",
            "genie_answer": g_ans or "",
            "genie_rows_json": json.dumps(rows, default=str) if rows is not None else "",
            "genie_elapsed_s": round(time.time() - t0, 2),
            "genie_error": "",
        }
    except Exception as e:
        return {
            "genie_sql": "",
            "genie_answer": "",
            "genie_rows_json": "",
            "genie_elapsed_s": round(time.time() - t0, 2),
            "genie_error": str(e)[:500],
        }


from collections import Counter as _Counter
from decimal import Decimal as _Decimal


def _parse_rows(s: str):
    """Parse a stored rows-JSON string to a list of row-lists, or None if blank/invalid."""
    if not s:
        return None
    try:
        rows = json.loads(s)
    except Exception:
        return None
    return rows or []


def _canon_cell(v) -> str:
    """Canonicalize a cell so cross-engine formatting (1 vs 1.0, Decimal, None, bool) doesn't
    cause a false mismatch. The warehouse returns every value as a string in data_array. Uses
    Decimal (not float) so big integers keep full precision, and preserves codes with leading
    zeros (zips/phones/IDs) instead of coercing them to numbers (caught live)."""
    if v is None:
        return "\x00NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, _Decimal):
        return format(v.normalize(), "f")
    if isinstance(v, int):
        return str(v)                              # exact — no float precision loss
    if isinstance(v, float):
        return format(_Decimal(str(v)).normalize(), "f")
    s = str(v).strip()
    core = s.lstrip("+-")
    # coerce genuinely-numeric strings, but keep leading-zero codes (e.g. "01234") verbatim
    if core and core.replace(".", "", 1).isdigit() and not (
        len(core) > 1 and core[0] == "0" and not core.startswith("0.")
    ):
        try:
            return format(_Decimal(s).normalize(), "f")
        except Exception:
            return s
    return s


def _canon_row(row) -> tuple:
    """Column-order- and alias-invariant: sort the canonical cell values within the row.
    Two valid answers that select the same values in a different column order (or with
    different aliases) still match. Mildly lenient (permutes values within a row) — acceptable
    for a regression signal, and far safer than position-exact string compare."""
    return tuple(sorted(_canon_cell(v) for v in (row or [])))


def rows_match(expected_rows, genie_rows, row_cap: int = 50):
    """Multiset row comparison. Returns True / False / None.

    - multiset (Counter), NOT a set — duplicate rows are preserved, so a wrong-cardinality
      answer can't pass by silent dedup.
    - None = NOT-EVALUABLE: excluded from the rate (neither pass nor fail). Two cases:
        (a) no expected baseline (expected_sql failed / permission-denied) — scoring this as a
            Genie miss falsely deflated concordance on perm-gated tables (caught live); and
        (b) a side hit the row cap, so truncation can't be trusted to reflect full equality."""
    if expected_rows is None:
        return None   # no baseline captured → not-evaluable, NOT a Genie miss
    if genie_rows is None:
        return False  # we have a baseline but Genie returned nothing → real miss
    if len(expected_rows) >= row_cap or len(genie_rows) >= row_cap:
        return None
    return _Counter(_canon_row(r) for r in expected_rows) == _Counter(_canon_row(r) for r in genie_rows)


# Run questions in parallel — Genie calls are I/O-bound and one can take 30-60s.
# When STABILITY_RUNS > 1, each question is run M times so we can compute test-retest
# reliability (rerun agreement + Wilson CI on N questions). Run 0 is treated as the canonical
# pass for downstream tables; the others feed only the reliability phase (8.5).
from concurrent.futures import ThreadPoolExecutor

eval_pub_pdf = spark.table(EVAL_SET_TABLE).toPandas()
N_questions = len(eval_pub_pdf)
print(f"Running {N_questions} eval questions × {STABILITY_RUNS} rerun(s) against Genie space {GENIE_SPACE_ID}...")

all_rerun_results: list[list[dict]] = []
for run_idx in range(STABILITY_RUNS):
    print(f"  rerun {run_idx + 1}/{STABILITY_RUNS}...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        rerun = list(ex.map(run_one_genie, eval_pub_pdf["question"].tolist()))
    all_rerun_results.append(rerun)

# Run 0 = canonical pass that feeds the runs_pdf publish + per-category metrics.
genie_results = all_rerun_results[0]
runs_pdf = eval_pub_pdf.copy()
for k in genie_results[0].keys():
    runs_pdf[k] = [r[k] for r in genie_results]

# v0.2: execute each expected_sql on the SAME engine Genie uses — the space SQL
# warehouse — so the row comparison is engine-consistent (no Spark-cluster vs DBSQL formatting
# false-mismatches). Fall back to the Spark-captured expected_rows_json (a lower bound) per
# question when the warehouse is unavailable or the statement fails.
def execute_on_warehouse(sql, warehouse_id, row_cap=50):
    if not warehouse_id or not sql:
        return None, "no warehouse_id or empty sql"
    try:
        # on_wait_timeout must be the ENUM, not the string "CANCEL" — passing a str raises
        # `'str' object has no attribute 'value'` (caught empirically during live validation).
        from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout as _OWT
        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id, statement=sql,
            wait_timeout="50s", on_wait_timeout=_OWT.CANCEL,
        )
        state = getattr(getattr(getattr(resp, "status", None), "state", None), "value", None)
        if state != "SUCCEEDED":
            return None, f"warehouse state={state}"
        rows = _collect_statement_rows(resp.statement_id, getattr(resp, "result", None), row_cap=row_cap)
        return rows, None
    except Exception as e:
        return None, str(e)[:300]


expected_rows_engine, expected_engine_kind = [], []
_exp_sql_list = eval_pub_pdf["expected_sql"].tolist()
_exp_json_list = eval_pub_pdf["expected_rows_json"].tolist()
if space_warehouse_id:
    with ThreadPoolExecutor(max_workers=4) as ex:
        _wh = list(ex.map(lambda s: execute_on_warehouse(s, space_warehouse_id), _exp_sql_list))
    for i, (rows, err) in enumerate(_wh):
        if rows is not None:
            expected_rows_engine.append(rows); expected_engine_kind.append("warehouse")
        else:
            expected_rows_engine.append(_parse_rows(_exp_json_list[i])); expected_engine_kind.append("spark_fallback")
else:
    for i in range(N_questions):
        expected_rows_engine.append(_parse_rows(_exp_json_list[i])); expected_engine_kind.append("spark_fallback")
n_same_engine = sum(1 for k in expected_engine_kind if k == "warehouse")
print(f"Expected rows: {n_same_engine}/{N_questions} executed on the space warehouse (same engine as Genie); "
      f"{N_questions - n_same_engine} fell back to Spark (lower bound).")

# Row concordance: multiset, engine-consistent, cap-aware (True / False / None=indeterminate).
genie_rows_parsed = [_parse_rows(s) for s in runs_pdf["genie_rows_json"].tolist()]
match_results = [rows_match(expected_rows_engine[i], genie_rows_parsed[i]) for i in range(len(runs_pdf))]
runs_pdf["result_match"] = [m is True for m in match_results]
runs_pdf["match_indeterminate"] = [m is None for m in match_results]
runs_pdf["expected_engine"] = expected_engine_kind

_determinate = [m for m in match_results if m is not None]
n_indeterminate = sum(1 for m in match_results if m is None)
pass_rate = (sum(1 for m in _determinate if m) / len(_determinate)) if _determinate else 0.0
n_errors = int((runs_pdf["genie_error"] != "").sum())
print(f"Row concordance vs synthetic SQL (run 0): {pass_rate:.1%} over {len(_determinate)} determinate "
      f"| indeterminate (cap hit): {n_indeterminate} | Genie errors: {n_errors}/{len(runs_pdf)}")

# Uncalibrated LLM judge — answer quality. NOT validated against human labels; read its
# mean as an opinion, not a measurement. It claims only what it can SEE (question + Genie's answer
# text) — it has no access to the underlying data, so it judges specificity/on-topic-ness, not
# numerical correctness (that is the separate row-concordance lower bound).
answer_quality = Guidelines(
    name="answer_quality",
    guidelines=(
        "The `outputs` text must directly and specifically answer `inputs.question`: it should give "
        "a concrete figure, ranking, or named entity rather than restating the question or "
        "describing what it *would* do. Penalize generic restatements and unwarranted hedging "
        "('it seems', 'roughly') when the question asks for a specific value. NOTE: you can see "
        "only the question and Genie's answer text — judge specificity and on-topic-ness, NOT "
        "numerical correctness (that is verified separately by row concordance)."
    ),
    model=JUDGE_MODEL_URI,
)

judge_df = pd.DataFrame([
    {
        "inputs": {"question": r["question"]},
        "outputs": r["genie_answer"] or "",
        "expectations": {"expected_sql": r["expected_sql"] or ""},
    }
    for _, r in runs_pdf.iterrows()
    if not r["genie_error"]
])

with mlflow.start_run(run_name="genie_regression"):
    mlflow.log_param("genie_space_id", GENIE_SPACE_ID)
    mlflow.log_param("eval_set_table", EVAL_SET_TABLE)
    # Document what the headline number is and is NOT.
    mlflow.log_param(
        "pass_rate_deterministic_meaning",
        "row concordance vs synthetic expected_sql (multiset, cap-aware); same-engine when the "
        "space warehouse is available, else Spark lower bound; inter-model agreement, NOT "
        "certified correctness. Computed over DETERMINATE rows (cap-hit rows excluded).",
    )
    mlflow.log_metric("n_questions", len(runs_pdf))
    mlflow.log_metric("n_genie_errors", n_errors)
    mlflow.log_metric("n_indeterminate", n_indeterminate)
    mlflow.log_metric("n_same_engine_expected", n_same_engine)
    # Key kept as pass_rate_deterministic for back-compat. Now over determinate rows only.
    mlflow.log_metric("pass_rate_deterministic", pass_rate)
    # Per-bucket means are DESCRIPTIVE PROFILES, not "calibration"/"discrimination" axes.
    # Computed over determinate rows; per-bucket COUNTS logged so the scorecard can flag n<5.
    det_pdf = runs_pdf[~runs_pdf["match_indeterminate"]]
    for cat, grp in det_pdf.groupby("category"):
        if len(grp):
            mlflow.log_metric(f"pass_rate_{cat}", float(grp["result_match"].mean()))
            mlflow.log_metric(f"count_{cat}", int(len(grp)))
    # Group on STRUCTURAL difficulty (v0.2) when present; fall back to the self-label otherwise.
    diff_col = "difficulty_structural" if "difficulty_structural" in det_pdf.columns else "difficulty"
    diff_profile = {}
    for diff, grp in det_pdf.groupby(diff_col):
        if len(grp):
            rate = float(grp["result_match"].mean())
            mlflow.log_metric(f"pass_rate_{diff}", rate)
            mlflow.log_metric(f"count_{diff}", int(len(grp)))
            if len(grp) >= 5:                       # min-n guard before trusting the bucket
                diff_profile[diff] = rate
    # v0.2: monotonicity check (easy ≥ medium ≥ hard) over min-n-guarded buckets only.
    _present = [d for d in ("easy", "medium", "hard") if d in diff_profile]
    if len(_present) >= 2:
        _rates = [diff_profile[d] for d in _present]
        _violations = sum(1 for a, b in zip(_rates, _rates[1:]) if a + 1e-9 < b)
        mlflow.log_metric("difficulty_monotonicity_violations", _violations)
        mlflow.log_param("difficulty_monotonic", str(_violations == 0))
    if "difficulty_label_matches" in runs_pdf.columns:
        mlflow.log_metric("difficulty_label_match_rate", float(runs_pdf["difficulty_label_matches"].mean()))
    if len(judge_df):
        mlflow.genai.evaluate(data=judge_df, scorers=[Safety(), answer_quality])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.5 Reliability — is the row-concordance signal reliable across reruns?
# MAGIC
# MAGIC Computed only when `stability_runs >= 2`. Two SEPARATE questions, two separate numbers:
# MAGIC - **Rerun agreement** = fraction of questions whose M reruns are unanimous (all-pass or
# MAGIC   all-fail). This is the test-retest reliability of Genie on this set. *(Replaces a
# MAGIC   chance-corrected coefficient that was removed — it scored a near-perfect system (one flaky
# MAGIC   item out of many) at ~0, i.e. it punished the best case.)*
# MAGIC - **Wilson 95% CI** on the question-level concordance rate, computed on **N independent
# MAGIC   questions** — NOT N×M correlated rerun cells, which would inflate n by ~M and understate
# MAGIC   uncertainty. This bounds the precision of the rate.
# MAGIC
# MAGIC **`gateable`** = `rerun_agreement ≥ 0.90` AND `half-width ≤ gate_max_half_width_pp` AND pooled
# MAGIC concordance in `[min, max]` (degenerate-eval guard). The half-width target is a widget
# MAGIC (default **10pp**) because the old hardcoded **7pp** gate was UNSATISFIABLE at the advertised
# MAGIC N=14/M=3 config — the half-width floor there is ~9.3pp (verified). When not gateable,
# MAGIC the notebook prints the **N you need** to reach the target at the observed concordance, so the
# MAGIC verdict is actionable instead of a dead red ✗.

# COMMAND ----------

def _wilson_half(p: float, n: int, zc: float = 1.96) -> float:
    """Wilson 95% half-width for a proportion p over n INDEPENDENT trials."""
    if n <= 0:
        return 0.0
    denom = 1.0 + zc * zc / n
    return float(zc * np.sqrt(p * (1.0 - p) / n + zc * zc / (4.0 * n * n)) / denom)


def _n_for_half(p: float, target: float, n_floor: int = 2, zc: float = 1.96) -> int:
    """Smallest n whose Wilson half-width ≤ target at proportion p (search, capped)."""
    n = max(n_floor, 2)
    while _wilson_half(p, n) > target and n < 100000:
        n += 1
    return n


if STABILITY_RUNS >= 2:
    # pass_matrix[q, r] ∈ {0.0, 1.0, NaN}. NaN = Genie error OR indeterminate (cap hit) — both
    # excluded from agreement + CI. Uses the same engine-consistent expected rows as run 0.
    pass_matrix = np.full((N_questions, STABILITY_RUNS), np.nan, dtype=float)
    for r_idx, results in enumerate(all_rerun_results):
        for q_idx, res in enumerate(results):
            if res["genie_error"]:
                continue
            m = rows_match(expected_rows_engine[q_idx], _parse_rows(res["genie_rows_json"]))
            if m is None:
                continue  # indeterminate (row cap) — leave as NaN
            pass_matrix[q_idx, r_idx] = 1.0 if m else 0.0

    # --- Per-question reliability (replaces Krippendorff α) ---
    # A question is "unanimous" if all its non-error reruns agree (all-pass or all-fail).
    # rerun_agreement = fraction of judgeable questions (≥2 valid reruns) that are unanimous.
    # Also build the per-question canonical pass (mean over valid reruns) as the CI unit.
    unanimous = 0
    judgeable = 0
    q_pass: list[float] = []      # one entry per question with ≥1 valid rerun
    for q in range(N_questions):
        vals = pass_matrix[q][~np.isnan(pass_matrix[q])]
        if len(vals) >= 1:
            q_pass.append(float(vals.mean()))
        if len(vals) >= 2:
            judgeable += 1
            if vals.min() == vals.max():
                unanimous += 1
    rerun_agreement = (unanimous / judgeable) if judgeable else float("nan")

    # --- Question-level concordance + Wilson CI on N INDEPENDENT units ---
    # Unit = question (NOT N×M correlated cells). A question "passes" if its mean rerun pass ≥ 0.5.
    n_units = len(q_pass)
    passes = sum(1 for p in q_pass if p >= 0.5)
    pooled_p = (passes / n_units) if n_units else 0.0

    z = 1.96
    half = _wilson_half(pooled_p, n_units)
    center = ((pooled_p + z * z / (2 * n_units)) / (1.0 + z * z / n_units)) if n_units else 0.0
    ci_low, ci_high = max(0.0, center - half), min(1.0, center + half)
    half_pp = float(half * 100)

    target_half = GATE_MAX_HALF_WIDTH_PP / 100.0
    n_needed = _n_for_half(pooled_p, target_half, n_floor=n_units)

    # Degenerate-eval guard: a near-all-pass or near-all-fail set clears any half-width gate
    # without saying anything useful, so require pooled concordance in [min, max].
    pass_rate_in_range = MIN_POOLED_PASS_OVERRIDE <= pooled_p <= MAX_POOLED_PASS_OVERRIDE
    agreement_ok = (not np.isnan(rerun_agreement)) and rerun_agreement >= 0.90
    gateable = bool(agreement_ok and half <= target_half and pass_rate_in_range)

    print(f"Reliability (M={STABILITY_RUNS}, N={N_questions}, judgeable units={n_units}):")
    if not np.isnan(rerun_agreement):
        print(f"  Rerun agreement      : {rerun_agreement:.3f}  ({unanimous}/{judgeable} questions unanimous across reruns)")
    else:
        print("  Rerun agreement      : NaN (need ≥2 valid reruns per question)")
    print(f"  Question concordance : {pooled_p:.3f}  ({'in range' if pass_rate_in_range else f'OUT of [{MIN_POOLED_PASS_OVERRIDE},{MAX_POOLED_PASS_OVERRIDE}] — gate disabled'})")
    print(f"  Wilson 95% CI (n={n_units}): [{ci_low:.3f}, {ci_high:.3f}]  half-width {half_pp:.1f}pp  (target ≤ {GATE_MAX_HALF_WIDTH_PP:.1f}pp)")
    if half > target_half:
        print(f"  → to reach ≤{GATE_MAX_HALF_WIDTH_PP:.1f}pp at this concordance you need ~{n_needed} questions (raise questions_per_table).")
    print(f"  Gateable             : {gateable}  (need agreement≥0.90 AND half-width≤{GATE_MAX_HALF_WIDTH_PP:.1f}pp AND {MIN_POOLED_PASS_OVERRIDE}≤concordance≤{MAX_POOLED_PASS_OVERRIDE})")

    with mlflow.start_run(run_name="eval_set_stability"):
        mlflow.log_param("genie_space_id", GENIE_SPACE_ID)
        mlflow.log_param("M_reruns", STABILITY_RUNS)
        mlflow.log_param("N_questions", N_questions)
        mlflow.log_metric("n_units", n_units)
        mlflow.log_metric("pooled_pass_rate", pooled_p)
        if not np.isnan(rerun_agreement):
            mlflow.log_metric("rerun_agreement", rerun_agreement)
        mlflow.log_metric("ci95_low", ci_low)
        mlflow.log_metric("ci95_high", ci_high)
        mlflow.log_metric("ci95_half_width_pp", half_pp)
        mlflow.log_metric("gate_max_half_width_pp", GATE_MAX_HALF_WIDTH_PP)
        mlflow.log_metric("n_needed_for_target", n_needed)
        mlflow.log_param("gateable", gateable)

        per_q_stability_pdf = pd.DataFrame({
            "question_id": eval_pub_pdf["question_id"].tolist(),
            "question": eval_pub_pdf["question"].tolist(),
            **{f"pass_run_{i}": pass_matrix[:, i].tolist() for i in range(STABILITY_RUNS)},
            "modal_pass": np.nanmean(pass_matrix, axis=1).tolist(),
        })
        mlflow.log_table(per_q_stability_pdf, "stability_per_question.json")
else:
    print(f"Reliability: skipped — stability_runs={STABILITY_RUNS} (set ≥2 to compute it).")

# Persist the regression run
runs_sdf = spark.createDataFrame(runs_pdf)
runs_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(EVAL_RUNS_TABLE)
print(f"Wrote {runs_sdf.count()} Genie regression rows to {EVAL_RUNS_TABLE}")
display(runs_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. (Optional) Optimize the question-generation prompt (GEPA)
# MAGIC
# MAGIC `mlflow.genai.optimize_prompts` uses DSPy/GEPA to search for a better version of the registered
# MAGIC prompt against a training set + scorers. Scorers used here:
# MAGIC
# MAGIC - **`sql_parses`**: every `expected_sql` returned must execute without error on the target table.
# MAGIC - **`grounded_values`**: every WHERE literal must appear in the column_samples we gave the generator.
# MAGIC - **`json_shape`**: output must be a JSON object with a `questions` array whose items have the required fields.
# MAGIC
# MAGIC Output: a new version of `{PROMPT_NAME}`. You can then manually promote it from `@candidate` to `@production`.

# COMMAND ----------

if RUN_OPTIMIZATION:
    from mlflow.genai.optimize import GepaPromptOptimizer

    # Scorers return floats in [0, 1] so GEPA has a proper gradient to reflect on.
    # Boolean scorers kill GEPA when baseline already passes — the reflection model
    # sees no loss signal and never proposes an improved candidate.

    REQUIRED_KEYS = {"question", "expected_sql", "expected_shape", "category",
                     "difficulty", "why_non_trivial"}

    @scorer
    def json_shape(outputs: str) -> Feedback:
        """Fraction of items that have all 6 required keys."""
        try:
            parsed = json.loads(outputs)
            qs = parsed.get("questions", [])
        except Exception as e:
            return Feedback(value=0.0, rationale=f"JSON parse error: {e}")
        if not isinstance(qs, list) or not qs:
            return Feedback(value=0.0, rationale="no questions")
        ok = sum(1 for q in qs if set(q.keys()) >= REQUIRED_KEYS)
        frac = ok / len(qs)
        return Feedback(value=frac, rationale=f"{ok}/{len(qs)} items have all required keys")

    @scorer
    def sql_parses(outputs: str) -> Feedback:
        """Fraction of items whose expected_sql executes on Spark."""
        try:
            parsed = json.loads(outputs)
            qs = parsed.get("questions", [])
        except Exception as e:
            return Feedback(value=0.0, rationale=f"JSON parse error: {e}")
        if not qs:
            return Feedback(value=0.0, rationale="no questions")
        ok_count = 0
        for q in qs:
            try:
                spark.sql(q["expected_sql"]).limit(1).collect()
                ok_count += 1
            except Exception:
                pass
        frac = ok_count / len(qs)
        return Feedback(value=frac, rationale=f"{ok_count}/{len(qs)} SQL executed")

    @scorer
    def grounded_values(outputs: str, inputs: dict) -> Feedback:
        """Fraction of items whose expected_sql uses only literals from column_samples."""
        try:
            parsed = json.loads(outputs)
            qs = parsed.get("questions", [])
        except Exception as e:
            return Feedback(value=0.0, rationale=f"JSON parse error: {e}")
        if not qs:
            return Feedback(value=0.0, rationale="no questions")
        import re
        literals_allowed = set()
        for line in (inputs.get("column_samples") or "").splitlines():
            m = re.findall(r"'([^']+)'", line) + re.findall(r'"([^"]+)"', line)
            literals_allowed.update(m)
        if not literals_allowed:
            return Feedback(value=1.0, rationale="no column samples to ground against")
        good = 0
        for q in qs:
            sql = q.get("expected_sql") or ""
            sql_literals = set(re.findall(r"'([^']+)'", sql))
            invented = sql_literals - literals_allowed - {"", " "}
            invented = {v for v in invented if not re.match(r"^\d{4}-\d{2}-\d{2}$|^\d+(\.\d+)?$", v)}
            if not invented:
                good += 1
        frac = good / len(qs)
        return Feedback(value=frac, rationale=f"{good}/{len(qs)} items use only grounded literals")

    # Train data: one example per target table × two n-values for variety
    train_data = []
    for t in GENIE_TABLES:
        for n in (3, 5):
            train_data.append({
                "inputs": {
                    "n": str(n),
                    "min_hard": str(max(1, n // 2)),
                    "table": t,
                    "schema": table_contexts[t]["schema"],
                    "space_context": SPACE_CONTEXT,
                    "column_samples": table_contexts[t]["col_samples"],
                }
            })

    token_local = DATABRICKS_TOKEN
    base_url_local = AI_GATEWAY_BASE_URL
    model_local = GENERATOR_ENDPOINT

    @mlflow.trace
    def predict_fn(n, min_hard, table, schema, space_context, column_samples) -> str:
        pv_cand = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@candidate")
        rendered = pv_cand.format(
            n=n, min_hard=min_hard, table=table, schema=schema,
            space_context=space_context, column_samples=column_samples,
        )
        local_client = OpenAI(api_key=token_local, base_url=base_url_local)
        resp = local_client.chat.completions.create(
            model=model_local,
            messages=[{"role": "user", "content": rendered}],
            max_tokens=2048,
            temperature=0.3,  # lower for optimization stability
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    HARD_TOKENS = (" WITH ", " OVER (", "LAG(", "LEAD(", "RANK(", "DENSE_RANK(", "NTILE(",
                   "ROW_NUMBER(", "PERCENTILE_APPROX", "APPROX_COUNT_DISTINCT", "EXISTS",
                   "HAVING ", "CASE WHEN")

    @scorer
    def hard_distribution(outputs: str, inputs: dict) -> Feedback:
        """min(hard_ok/target, 1.0) where hard_ok = items with difficulty=hard AND
        expected_sql contains at least one hard-SQL token. Float gradient so GEPA
        can reflect on "fewer hard than asked" vs "nailed the distribution"."""
        try:
            parsed = json.loads(outputs)
            qs = parsed.get("questions", [])
        except Exception as e:
            return Feedback(value=0.0, rationale=f"JSON parse error: {e}")
        target = max(1, int(inputs.get("min_hard") or 1))
        hard_ok = 0
        for q in qs:
            if (q.get("difficulty") or "").lower() != "hard":
                continue
            sql_upper = (q.get("expected_sql") or "").upper()
            if any(tok in sql_upper for tok in HARD_TOKENS):
                hard_ok += 1
        frac = min(hard_ok / target, 1.0)
        return Feedback(value=frac, rationale=f"{hard_ok}/{target} items are strict-hard")

    print(f"Running optimize_prompts on {PROMPT_NAME} with {len(train_data)} training examples...")
    mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=train_data,
        prompt_uris=[f"prompts:/{PROMPT_NAME}@candidate"],
        optimizer=GepaPromptOptimizer(reflection_model=JUDGE_MODEL_URI),
        scorers=[json_shape, sql_parses, grounded_values, hard_distribution],
    )
    latest = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@candidate")
    print(f"Optimization complete. Latest version: v{latest.version}")
    print("Review in the MLflow Prompt Registry, then promote with:")
    print(f"  mlflow.genai.set_prompt_alias(name='{PROMPT_NAME}', alias='production', version=<N>)")
else:
    print("Skipping GEPA optimization — set run_prompt_optimization widget to 'true' to enable.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary report

# COMMAND ----------

final_sdf = spark.table(EVAL_SET_TABLE)
summary = (
    final_sdf.groupBy("category")
    .agg(
        F.count("*").alias("n"),
        F.round(F.avg(F.col("sql_executes").cast("int")), 3).alias("pct_sql_executes"),
        F.round(F.avg(F.col("nonempty_result").cast("int")), 3).alias("pct_nonempty"),
    )
    .orderBy("category")
)

print("Per-category eval-set health:")
display(summary)

print(f"Eval set:       {EVAL_SET_TABLE}")
print(f"Eval runs:      {EVAL_RUNS_TABLE}")
print(f"Prompt:         {PROMPT_NAME} (@candidate / @production)")
print(f"MLflow runs:    {MLFLOW_EXPERIMENT}")
print(f"  - eval_set_quality: per-item scorer distribution on the curated set")
print(f"  - genie_regression: Genie pass rate + NL-answer quality on the same set")
