"""FastAPI backend for the Genie Eval Generator app.

Auth model:
  * On Databricks Apps, every request carries ``X-Forwarded-Access-Token`` — the
    calling user's OAuth token. We thread it through ``DatabricksClient`` so all
    REST calls run on-behalf-of the user.
  * App-internal calls (``/api/health``) and the SP-fallback path use the service
    principal token resolved at startup.
  * Local dev: set ``DEV_USER_TOKEN`` env var to forge an OBO token for any
    request that doesn't carry the header. See app/README.md.

The React build is mounted as static files at ``/``, so the same uvicorn process
serves both API and UI.
"""
from __future__ import annotations

import json as _json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.databricks_client import (
    AuthError,
    DatabricksClient,
    get_host,
    get_sp_token,
    get_notebook_path,
)


app = FastAPI(title="Genie Eval Generator", version="0.1.0")


# ---------------------------------------------------------------------------
# Auth dependency: resolve the per-request token (OBO → DEV_USER_TOKEN → SP)
# ---------------------------------------------------------------------------
OBO_HEADER = "x-forwarded-access-token"


def get_user_token(request: Request) -> str:
    """Return the per-request token to authenticate Databricks REST calls.

    Priority (matches Databricks Apps behavior):
      1. ``X-Forwarded-Access-Token`` header — present in production
      2. ``DEV_USER_TOKEN`` env var — forge for local dev
      3. SP token — last-resort fallback so the app keeps working when neither
         is set (single-user dev with `databricks auth login`)
    """
    header_token = request.headers.get(OBO_HEADER)
    if header_token:
        return header_token
    dev_token = os.environ.get("DEV_USER_TOKEN", "").strip()
    if dev_token:
        return dev_token
    return get_sp_token()


def get_client(token: str = Depends(get_user_token)) -> DatabricksClient:
    """FastAPI dependency: a per-request DatabricksClient bound to the caller's token."""
    return DatabricksClient(token=token)


# ---------------------------------------------------------------------------
# Lightweight in-process caches (TTL) for hot-path Genie endpoints
#
# IMPORTANT: cache keys are scoped by token-prefix so different OBO callers do
# not share cached results. Without this, user A's first /api/spaces fetch
# would return cached entries to user B even if B has different permissions.
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[float, Any]] = {}


def _token_scope(token: str) -> str:
    # Use first 8 + last 4 chars of the token as a stable, low-entropy scope key.
    if not token:
        return "anon"
    return f"{token[:8]}..{token[-4:]}"


def _cached(key: str, ttl: float, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    sp_token = get_sp_token()
    return {
        "host": get_host(),
        "authed": bool(get_host() and sp_token),
        "notebook_path": get_notebook_path(),
        "obo_dev_fallback": bool(os.environ.get("DEV_USER_TOKEN")),
        # Set by the DAB (resources/app.yml) from --var=output_catalog; the form uses it as the
        # uc_catalog default so eval tables land in the catalog the bundle prepared.
        "default_output_catalog": os.environ.get("DEFAULT_OUTPUT_CATALOG", ""),
    }


@app.get("/api/me")
def me(c: DatabricksClient = Depends(get_client)) -> dict:
    """OBO-resolved calling user — lets the UI default the experiment path to the caller's home
    folder instead of a hardcoded email. Best-effort: degrades to empty user_name."""
    try:
        who = c.whoami()
        return {"user_name": who.get("userName") or ""}
    except Exception:
        return {"user_name": ""}


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------
@app.get("/api/spaces")
def list_spaces(c: DatabricksClient = Depends(get_client)) -> list[dict]:
    try:
        scope = _token_scope(c.token)
        spaces = _cached(f"{scope}:spaces", 60.0, c.list_spaces)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    out = []
    for s in spaces:
        rid = s.get("id") or s.get("space_id") or s.get("data_room_id")
        out.append(
            {
                "space_id": rid,
                "title": s.get("title") or "(untitled)",
                "description": s.get("description") or "",
                "n_tables": len(s.get("table_identifiers") or []),
                "warehouse_id": s.get("warehouse_id") or "",
                "table_identifiers": s.get("table_identifiers") or [],
            }
        )
    return out


@app.get("/api/spaces/{space_id}")
def get_space(space_id: str, c: DatabricksClient = Depends(get_client)) -> dict:
    try:
        scope = _token_scope(c.token)
        d = _cached(f"{scope}:space:{space_id}", 120.0, lambda: c.get_space_detail(space_id))
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    # Propagate upstream errors with their status — surface 404 / 403 to the client
    # instead of silently returning a 200 wrapping {_error, _status}.
    if isinstance(d, dict) and d.get("_error"):
        status = d.get("_status") or 502
        raise HTTPException(status_code=status, detail=d["_error"])
    return d


@app.get("/api/spaces/{space_id}/curated-count")
def curated_count(space_id: str, c: DatabricksClient = Depends(get_client)) -> dict:
    try:
        scope = _token_scope(c.token)
        n = _cached(f"{scope}:curated:{space_id}", 120.0, lambda: c.get_curated_count(space_id))
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {"count": n}


@app.get("/api/spaces/{space_id}/conversation-count")
def conversation_count(space_id: str, c: DatabricksClient = Depends(get_client)) -> dict:
    try:
        scope = _token_scope(c.token)
        n = _cached(f"{scope}:conv:{space_id}", 60.0, lambda: c.get_conversation_count(space_id))
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {"count": n}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
class SubmitRunBody(BaseModel):
    run_name: str = Field(..., min_length=1)
    uc_catalog: str
    uc_schema: str
    genie_space_id: str
    genie_tables: str = ""
    mlflow_experiment_path: str
    generator_endpoint: str
    judge_endpoint: str
    ai_gateway_base_url: str
    questions_per_table: str = "14"
    min_hard_per_table: str = "5"
    max_distinct_values_per_column: str = "30"
    extra_space_instructions: str = ""
    embedding_endpoint: str
    stability_runs: str = "1"
    include_historical_in_context: str = "false"
    run_prompt_optimization: str = "false"
    # v0.1.0 hooks
    extra_prompt_rules: str = ""
    custom_prompt_uri: str = ""
    custom_min_pooled_pass: str = "0.10"
    custom_max_pooled_pass: str = "0.90"

    def widget_params(self) -> dict[str, str]:
        return {
            "uc_catalog": self.uc_catalog,
            "uc_schema": self.uc_schema,
            "genie_space_id": self.genie_space_id,
            "genie_tables": self.genie_tables,
            "mlflow_experiment_path": self.mlflow_experiment_path,
            "generator_endpoint": self.generator_endpoint,
            "judge_endpoint": self.judge_endpoint,
            "ai_gateway_base_url": self.ai_gateway_base_url,
            "questions_per_table": str(self.questions_per_table),
            "min_hard_per_table": str(self.min_hard_per_table),
            "max_distinct_values_per_column": str(self.max_distinct_values_per_column),
            "extra_space_instructions": self.extra_space_instructions,
            "embedding_endpoint": self.embedding_endpoint,
            "stability_runs": str(self.stability_runs),
            "include_historical_in_context": self.include_historical_in_context,
            "run_prompt_optimization": self.run_prompt_optimization,
            "extra_prompt_rules": self.extra_prompt_rules,
            "custom_prompt_uri": self.custom_prompt_uri,
            "custom_min_pooled_pass": self.custom_min_pooled_pass,
            "custom_max_pooled_pass": self.custom_max_pooled_pass,
        }


@app.post("/api/runs")
def submit_run(body: SubmitRunBody, c: DatabricksClient = Depends(get_client)) -> dict:
    try:
        # Resolve calling user's identity (best-effort — populates submitted_by).
        submitted_by = ""
        try:
            me = c.whoami()
            submitted_by = me.get("userName") or (
                (me.get("emails") or [{}])[0].get("value", "")
            )
        except Exception:
            pass
        res = c.submit_run(body.run_name, body.widget_params())
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    if res.get("_error"):
        raise HTTPException(status_code=502, detail=res["_error"])

    submitted_at = datetime.now(timezone.utc).isoformat()
    run_id = res.get("run_id")
    params_json = _json.dumps(body.widget_params())

    # Best-effort write to the runs-index table. Failure is logged but does not
    # block the submit response (the run is real on Databricks even if our index
    # write failed — we'd rather return success and lose the index row than fail
    # a working run).
    index_status = "skipped"
    try:
        wh = c.pick_warehouse()
        if wh and run_id:
            c.ensure_runs_index_table(body.uc_catalog, wh["id"])
            d = c.write_runs_index_row(
                catalog=body.uc_catalog,
                warehouse_id=wh["id"],
                run_id=int(run_id),
                run_name=body.run_name,
                space_id=body.genie_space_id,
                experiment_path=body.mlflow_experiment_path,
                submitted_at=submitted_at,
                submitted_by=submitted_by,
                params_json=params_json,
            )
            state = d.get("status", {}).get("state")
            if state == "SUCCEEDED":
                index_status = "written"
            else:
                err = (d.get("status", {}).get("error") or {}).get("message", "")
                index_status = f"failed: {state} {err[:200]}"
                print(f"WARN: runs-index write returned {state}: {err[:300]}", file=sys.stderr)
    except Exception as e:
        index_status = f"failed: {e}"
        print(f"WARN: runs-index write raised: {e}", file=sys.stderr)

    return {
        "run_id": run_id,
        "run_name": body.run_name,
        "experiment_path": body.mlflow_experiment_path,
        "submitted_at": submitted_at,
        "submitted_by": submitted_by,
        "params": body.widget_params(),
        "index_status": index_status,
    }


@app.get("/api/eval-runs-index")
def eval_runs_index(
    catalog: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=500),
    c: DatabricksClient = Depends(get_client),
) -> dict:
    """Read the persistent runs-index table for the given catalog.

    Replaces in-session zustand as the source of truth for "what eval-runs have
    been submitted from this install". OBO scoping: UC grants on the index table
    determine what each user sees.
    """
    try:
        if not catalog.replace("_", "").replace("-", "").isalnum():
            raise HTTPException(status_code=400, detail=f"Invalid catalog: {catalog}")
        wh = c.pick_warehouse()
        if not wh:
            raise HTTPException(status_code=503, detail="No SQL warehouse available")
        # Idempotent ensure — table may not exist on a fresh install.
        c.ensure_runs_index_table(catalog, wh["id"])
        rows = c.read_runs_index(catalog, wh["id"], limit)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {
        "ok": True,
        "warehouse_id": wh["id"],
        "table": c.runs_index_fqn(catalog),
        "rows": rows,
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: int, c: DatabricksClient = Depends(get_client)) -> dict:
    try:
        d = c.get_run(run_id)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    if d.get("_error"):
        raise HTTPException(status_code=502, detail=d["_error"])
    state = d.get("state", {}) or {}
    return {
        "run_id": d.get("run_id"),
        "run_name": d.get("run_name", ""),
        "life_cycle_state": state.get("life_cycle_state", ""),
        "result_state": state.get("result_state", "") or "",
        "state_message": state.get("state_message", "") or "",
        "start_time": d.get("start_time"),
        "end_time": d.get("end_time"),
        "run_page_url": d.get("run_page_url", ""),
    }


@app.get("/api/runs")
def list_runs(
    limit: int = Query(25, ge=1, le=100),
    c: DatabricksClient = Depends(get_client),
) -> list[dict]:
    try:
        runs = c.list_recent_runs(limit=limit)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    out = []
    for r in runs:
        s = r.get("state", {}) or {}
        out.append(
            {
                "run_id": r.get("run_id"),
                "run_name": r.get("run_name", ""),
                "life_cycle_state": s.get("life_cycle_state", ""),
                "result_state": s.get("result_state", "") or "",
                "start_time": r.get("start_time"),
                "end_time": r.get("end_time"),
                "run_page_url": r.get("run_page_url", ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def _shape_scorecard(bag: dict[str, dict]) -> dict:
    realism_metrics = bag.get("eval_set_realism", {}).get("metrics", {})
    realism_params = bag.get("eval_set_realism", {}).get("params", {})
    stability_metrics = bag.get("eval_set_stability", {}).get("metrics", {})
    stability_params = bag.get("eval_set_stability", {}).get("params", {})
    quality_metrics = bag.get("eval_set_quality", {}).get("metrics", {})
    verification_metrics = bag.get("eval_set_verification", {}).get("metrics", {})
    regression_metrics = bag.get("genie_regression", {}).get("metrics", {})
    regression_params = bag.get("genie_regression", {}).get("params", {})

    by_difficulty: dict[str, float] = {}
    by_category: dict[str, float] = {}
    by_difficulty_n: dict[str, float] = {}
    by_category_n: dict[str, float] = {}
    pass_rate_deterministic = regression_metrics.get("pass_rate_deterministic")
    diff_keys = {"easy", "medium", "hard"}
    for k, v in regression_metrics.items():
        if k.startswith("pass_rate_"):
            bucket = k.replace("pass_rate_", "")
            if bucket == "deterministic":
                continue
            (by_difficulty if bucket in diff_keys else by_category)[bucket] = v
        elif k.startswith("count_"):
            bucket = k.replace("count_", "")
            (by_difficulty_n if bucket in diff_keys else by_category_n)[bucket] = v

    # NOTE the new contract (v0.1): "diversity" (descriptive, was "realism") and "reliability"
    # (was "stability"). Old keys are read as fallbacks so pre-v0.1 runs still render.
    return {
        "diversity": {
            "mean_nn_cosine": realism_metrics.get("mean_nn_cosine", realism_metrics.get("p_at_h")),
            "nn_cosine_p10": realism_metrics.get("nn_cosine_p10", realism_metrics.get("p_at_h_p10")),
            "gen_self_similarity": realism_metrics.get("gen_self_similarity", realism_metrics.get("self_sim")),
            "ref_self_similarity": realism_metrics.get("ref_self_similarity", realism_metrics.get("historical_self_sim")),
            "leakage_risk": realism_params.get("leakage_risk"),
            "computed": "eval_set_realism" in bag,
            "n_history": realism_params.get("n_history"),
            "n_curated": realism_params.get("n_curated"),
            "n_reference": realism_params.get("n_reference"),
        },
        "reliability": {
            "rerun_agreement": stability_metrics.get("rerun_agreement"),
            "half_pp": stability_metrics.get("ci95_half_width_pp"),
            "pooled": stability_metrics.get("pooled_pass_rate"),
            "gateable": stability_params.get("gateable"),
            "computed": "eval_set_stability" in bag,
            "m_reruns": stability_params.get("M_reruns"),
            "n_questions": stability_params.get("N_questions"),
            "n_units": stability_metrics.get("n_units"),
            "n_needed_for_target": stability_metrics.get("n_needed_for_target"),
            "gate_max_half_width_pp": stability_metrics.get("gate_max_half_width_pp"),
        },
        # Trust gate: execution-verified by two model families — certificates, never "accuracy".
        "verification": {
            "computed": "eval_set_verification" in bag,
            "verified_fraction": verification_metrics.get("verified_fraction"),
            "gold_count": verification_metrics.get("gold_count"),
            "verified_count": verification_metrics.get("verified_count"),
            "quarantine_count": verification_metrics.get("quarantine_count"),
            "repaired_count": verification_metrics.get("repaired_count"),
            "mean_panel_agreement": verification_metrics.get("mean_panel_agreement"),
            "n_gated": verification_metrics.get("n_gated"),
        },
        "quality": {
            "sql_executes": quality_metrics.get("fraction_sql_executes"),
            "nonempty": quality_metrics.get("fraction_nonempty"),
            "clarity": quality_metrics.get("question_clarity/mean"),
            "sql_answers": quality_metrics.get("sql_answers_question/mean"),
            "grounded": quality_metrics.get("grounded_literals/mean"),
        },
        "regression": {
            "pass_rate_deterministic": pass_rate_deterministic,
            "by_difficulty": by_difficulty,
            "by_category": by_category,
            "by_difficulty_n": by_difficulty_n,
            "by_category_n": by_category_n,
            "n_questions": regression_metrics.get("n_questions"),
            "n_indeterminate": regression_metrics.get("n_indeterminate"),
            "n_same_engine": regression_metrics.get("n_same_engine_expected"),
            "difficulty_monotonic": regression_params.get("difficulty_monotonic"),
            "difficulty_label_match_rate": regression_metrics.get("difficulty_label_match_rate"),
        },
    }


@app.get("/api/scorecard/{run_id}")
def scorecard(
    run_id: int,
    experiment_path: str = Query(...),
    c: DatabricksClient = Depends(get_client),
) -> dict:
    try:
        run = c.get_run(run_id)
        state = run.get("state", {}) or {}
        life_cycle_state = state.get("life_cycle_state", "")
        result_state = state.get("result_state", "") or ""

        exp_id = c.get_experiment_id(experiment_path)
        if not exp_id:
            return {
                "ready": False,
                "reason": f"MLflow experiment not found at {experiment_path}",
                "life_cycle_state": life_cycle_state,
                "result_state": result_state,
                "scorecard": None,
            }
        # Attribute scorecard runs to THIS job run via the tag the notebook sets. Without the
        # filter, two runs sharing an experiment would silently show each other's results.
        runs = c.search_runs(
            exp_id, max_results=12, filter_string=f"tags.job_run_id = '{run_id}'"
        )
        attribution = "exact"
        if not runs:
            # Fallback for runs from notebook versions that didn't tag (or interactive runs):
            # latest runs in the experiment — surfaced as unattributed so the UI can warn.
            runs = c.search_runs(exp_id, max_results=12)
            attribution = "latest"
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))

    bag: dict[str, dict] = {}
    expected = {"eval_set_realism", "eval_set_stability", "genie_regression", "eval_set_quality", "eval_set_verification"}
    for r in runs:
        rn = r.get("info", {}).get("run_name", "")
        if rn in expected and rn not in bag:
            metrics = {
                m["key"]: float(m["value"])
                for m in r.get("data", {}).get("metrics", [])
            }
            params = {
                p["key"]: p["value"] for p in r.get("data", {}).get("params", [])
            }
            bag[rn] = {"metrics": metrics, "params": params}

    if not bag:
        return {
            "ready": False,
            "reason": "No scorecard runs found in MLflow yet",
            "life_cycle_state": life_cycle_state,
            "result_state": result_state,
            "scorecard": None,
        }

    return {
        "ready": True,
        "life_cycle_state": life_cycle_state,
        "result_state": result_state,
        "available_sections": sorted(bag.keys()),
        "attribution": attribution,
        "scorecard": _shape_scorecard(bag),
    }


# ---------------------------------------------------------------------------
# Eval set rows (read from UC via SQL Statements API)
# ---------------------------------------------------------------------------
@app.get("/api/eval-set")
def eval_set(
    catalog: str = Query(...),
    schema: str = Query(...),
    limit: int = Query(200, ge=1, le=2000),
    c: DatabricksClient = Depends(get_client),
) -> dict:
    try:
        wh = c.pick_warehouse()
        if not wh:
            raise HTTPException(status_code=503, detail="No SQL warehouse available")
        for v in (catalog, schema):
            if not v.replace("_", "").replace("-", "").isalnum():
                raise HTTPException(status_code=400, detail=f"Invalid identifier: {v}")
        table = DatabricksClient.bt(catalog, schema, "genie_eval_set")
        stmt = (
            "SELECT question, expected_sql, category, difficulty, "
            f"sql_executes, n_result_rows FROM {table} LIMIT {int(limit)}"
        )
        d = c.sql_run(wh["id"], stmt)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))

    state = d.get("status", {}).get("state")
    if state != "SUCCEEDED":
        err = (d.get("status", {}).get("error") or {}).get("message", "")
        return {
            "ok": False,
            "error": err or f"SQL query state={state}",
            "warehouse_id": wh["id"],
            "rows": [],
            "columns": [],
        }
    cols = [c0["name"] for c0 in d.get("manifest", {}).get("schema", {}).get("columns", [])]
    data = c.sql_rows(d)[: int(limit)]
    rows = [dict(zip(cols, row)) for row in data]
    return {
        "ok": True,
        "warehouse_id": wh["id"],
        "warehouse_name": wh.get("name", ""),
        "columns": cols,
        "rows": rows,
        "table": f"{catalog}.{schema}.genie_eval_set",
    }


# ---------------------------------------------------------------------------
# Static frontend mount (production build at server/static/)
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/api/{rest_of_path:path}")
def _api_404(rest_of_path: str):
    return JSONResponse({"detail": f"Unknown API path: /api/{rest_of_path}"}, status_code=404)


if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def _spa_root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{full_path:path}")
    def _spa_fallback(full_path: str):
        candidate = STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    def _no_build_root() -> dict:
        return {
            "message": (
                "Frontend build not found at server/static/. Run `cd app/web && npm "
                "run build` to populate it, or use the Vite dev server at :5173 with "
                "the built-in proxy."
            ),
            "api_docs": "/docs",
        }
