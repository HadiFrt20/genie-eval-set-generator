"""Thin Databricks REST client.

Auth model (v0.1.0):
  - Per-request token (preferred): each request is authed as the calling user via
    ``X-Forwarded-Access-Token`` injected by Databricks Apps. The FastAPI dependency
    ``get_user_token`` (in main.py) extracts it and threads it through.
  - Service-principal fallback: when no per-request token is present (health checks,
    local-dev convenience, embedding calls from inside the notebook), we fall back
    to the SP token resolved from env vars or the local CLI.

For local dev, callers can set ``DEV_USER_TOKEN`` in the environment to forge an
OBO header. The dependency in main.py reads that as a fallback when no real header
is present, so a single uvicorn process behaves "as the user" without needing a
proxy in front. See ``app/README.md`` for the recipe.

Public surface:
  * ``DatabricksClient(token)`` — construct a client bound to one token (per-request)
  * ``sp_client`` — module-level singleton authed with the SP token (fallback)
  * ``get_sp_token()`` / ``get_host()`` — for callers that just need the values
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

import requests


class AuthError(RuntimeError):
    """Raised when host/token cannot be resolved."""


# ---------------------------------------------------------------------------
# Host + SP-token resolution (process-level, cached after first read)
# ---------------------------------------------------------------------------
_SP_HOST: str | None = None
_SP_TOKEN: str | None = None


def _normalize_host(host: str) -> str:
    host = host.rstrip("/")
    if host and not host.startswith("http"):
        host = f"https://{host}"
    return host


def _resolve_sp_host_token() -> tuple[str, str]:
    """Resolve the SP-level (host, token).

    Order:
      1. ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` env vars (Apps inject these)
      2. Local CLI fallback via ``databricks auth describe`` + ``databricks auth token``
    """
    host = _normalize_host(os.environ.get("DATABRICKS_HOST", ""))
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if host and token:
        return host, token

    try:
        cfg_raw = subprocess.check_output(
            ["databricks", "auth", "describe", "-o", "json"],
            stderr=subprocess.DEVNULL,
        ).decode()
        cfg = json.loads(cfg_raw)
        host = host or _normalize_host((cfg.get("details", {}).get("host", "") or ""))
        if not token and host:
            tok_raw = subprocess.check_output(
                ["databricks", "auth", "token", "--host", host],
                stderr=subprocess.DEVNULL,
            ).decode()
            token = json.loads(tok_raw)["access_token"]
    except Exception:
        pass
    return host, token


def get_host() -> str:
    global _SP_HOST, _SP_TOKEN
    if _SP_HOST is None:
        _SP_HOST, _SP_TOKEN = _resolve_sp_host_token()
    return _SP_HOST or ""


def get_sp_token() -> str:
    global _SP_HOST, _SP_TOKEN
    if _SP_TOKEN is None:
        _SP_HOST, _SP_TOKEN = _resolve_sp_host_token()
    return _SP_TOKEN or ""


def refresh_auth() -> None:
    """Re-resolve host/token (useful when local CLI session was renewed)."""
    global _SP_HOST, _SP_TOKEN
    _SP_HOST, _SP_TOKEN = _resolve_sp_host_token()


def get_notebook_path() -> str:
    # NOTEBOOK_PATH is injected by the DAB at deploy; the fallback is a neutral placeholder.
    return os.environ.get(
        "NOTEBOOK_PATH",
        "/Workspace/Shared/genie_eval_set_generator/genie_eval_set_generator",
    )


# ---------------------------------------------------------------------------
# Client — one per request (or per-call when SP fallback)
# ---------------------------------------------------------------------------
class DatabricksClient:
    """Per-request Databricks client. Each instance is authed with one token.

    Construct via:
      * ``DatabricksClient(user_token)`` from a FastAPI dependency for OBO requests
      * ``DatabricksClient.sp()`` for SP-fallback (health, app-internal)
    """

    def __init__(self, token: str, host: str | None = None) -> None:
        self.token = token
        self.host = host or get_host()
        self.notebook_path = get_notebook_path()

    @classmethod
    def sp(cls) -> "DatabricksClient":
        return cls(token=get_sp_token())

    def is_authed(self) -> bool:
        return bool(self.host and self.token)

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        if not self.host:
            raise AuthError("DATABRICKS_HOST not configured")
        return f"{self.host}{path}"

    def get(self, path: str, params: dict | None = None, timeout: int = 30) -> dict:
        if not self.is_authed():
            raise AuthError("Databricks credentials missing")
        r = requests.get(
            self._url(path), headers=self._headers(), params=params or {}, timeout=timeout
        )
        if r.status_code >= 400:
            return {"_error": f"HTTP {r.status_code}: {r.text[:500]}", "_status": r.status_code}
        try:
            return r.json()
        except Exception as e:
            return {"_error": f"JSON parse: {e}"}

    def post(self, path: str, body: dict, timeout: int = 60) -> dict:
        if not self.is_authed():
            raise AuthError("Databricks credentials missing")
        r = requests.post(
            self._url(path),
            headers=self._headers(),
            data=json.dumps(body),
            timeout=timeout,
        )
        if r.status_code >= 400:
            return {"_error": f"HTTP {r.status_code}: {r.text[:500]}", "_status": r.status_code}
        try:
            return r.json()
        except Exception as e:
            return {"_error": f"JSON parse: {e}"}

    # ------------------------------------------------------------------
    # Genie API
    # ------------------------------------------------------------------
    def list_spaces(self) -> list[dict]:
        d = self.get("/api/2.0/data-rooms", params={"page_size": 100})
        return d.get("data_rooms") or []

    def get_space_detail(self, space_id: str) -> dict:
        return self.get(f"/api/2.0/data-rooms/{space_id}")

    def get_curated_count(self, space_id: str) -> int:
        d = self.get(
            f"/api/2.0/data-rooms/{space_id}/curated-questions",
            params={"page_size": 200},
        )
        items = d.get("curated_questions") or d.get("questions") or []
        return len(items)

    def get_conversation_count(self, space_id: str) -> int:
        d = self.get(
            f"/api/2.0/data-rooms/{space_id}/conversations",
            params={"page_size": 100},
        )
        return len(d.get("conversations") or [])

    def whoami(self) -> dict:
        """Resolve the calling identity. Used by /api/runs to populate `submitted_by`."""
        return self.get("/api/2.0/preview/scim/v2/Me")

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    def submit_run(self, run_name: str, params: dict[str, str]) -> dict:
        body = {
            "run_name": run_name,
            "tasks": [
                {
                    "task_key": "eval",
                    "notebook_task": {
                        "notebook_path": self.notebook_path,
                        "source": "WORKSPACE",
                        "base_parameters": params,
                    },
                    "environment_key": "env",
                }
            ],
            "environments": [
                {
                    "environment_key": "env",
                    "spec": {"client": "3", "environment_version": "3"},
                }
            ],
        }
        return self.post("/api/2.2/jobs/runs/submit", body)

    def get_run(self, run_id: int) -> dict:
        return self.get("/api/2.2/jobs/runs/get", params={"run_id": run_id})

    def list_recent_runs(self, limit: int = 25) -> list[dict]:
        d = self.get(
            "/api/2.2/jobs/runs/list",
            params={"limit": limit, "active_only": "false"},
        )
        return d.get("runs") or []

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    def get_experiment_id(self, path: str) -> str | None:
        d = self.get(
            "/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": path},
        )
        return (d.get("experiment") or {}).get("experiment_id")

    def search_runs(
        self, experiment_id: str, max_results: int = 12, filter_string: str = ""
    ) -> list[dict]:
        body: dict = {
            "experiment_ids": [experiment_id],
            "max_results": max_results,
            "order_by": ["attributes.start_time DESC"],
        }
        if filter_string:
            body["filter"] = filter_string
        d = self.post("/api/2.0/mlflow/runs/search", body)
        return d.get("runs") or []

    # ------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------
    def list_warehouses(self) -> list[dict]:
        return self.get("/api/2.0/sql/warehouses").get("warehouses") or []

    def pick_warehouse(self) -> dict | None:
        warehouses = self.list_warehouses()
        running_serverless = next(
            (
                w
                for w in warehouses
                if w.get("state") == "RUNNING" and w.get("enable_serverless_compute")
            ),
            None,
        )
        if running_serverless:
            return running_serverless
        running_any = next((w for w in warehouses if w.get("state") == "RUNNING"), None)
        if running_any:
            return running_any
        return warehouses[0] if warehouses else None

    def sql_run(
        self,
        warehouse_id: str,
        statement: str,
        parameters: list[dict] | None = None,
        timeout: int = 60,
    ) -> dict:
        body: dict[str, Any] = {
            "warehouse_id": warehouse_id,
            "statement": statement,
            "wait_timeout": "30s",
            "on_wait_timeout": "CONTINUE",
        }
        if parameters:
            body["parameters"] = parameters
        d = self.post("/api/2.0/sql/statements", body)
        sid = d.get("statement_id")
        if not sid:
            return d
        deadline = time.time() + timeout
        while (
            d.get("status", {}).get("state") in ("PENDING", "RUNNING")
            and time.time() < deadline
        ):
            time.sleep(2)
            d = self.get(f"/api/2.0/sql/statements/{sid}")
        return d

    def sql_rows(self, d: dict) -> list[list]:
        """All data rows of a finished statement, following result CHUNKS — chunk 0 alone can be
        a fraction of the result, which silently truncated reads before."""
        res = d.get("result") or {}
        rows = list(res.get("data_array") or [])
        sid = d.get("statement_id")
        nxt = res.get("next_chunk_index")
        guard = 0
        while sid and nxt is not None and guard < 1000:
            ch = self.get(f"/api/2.0/sql/statements/{sid}/result/chunks/{int(nxt)}")
            rows.extend(ch.get("data_array") or [])
            nxt = ch.get("next_chunk_index")
            guard += 1
        return rows

    @staticmethod
    def bt(*parts: str) -> str:
        """Backtick-quote identifier parts — hyphenated / federated catalog names pass the
        identifier validator but raise INVALID_IDENTIFIER when interpolated unquoted."""
        return ".".join(f"`{p.strip('`')}`" for p in parts)

    # ------------------------------------------------------------------
    # Runs index — UC metadata table that survives across browsers / sessions.
    # Schema lives at `<catalog>.genie_eval.genie_eval_runs_index`. The DAB declares
    # the schema; we re-create both schema and table at runtime (idempotent) so the
    # backend works even pre-bundle-deploy.
    # ------------------------------------------------------------------
    RUNS_INDEX_SCHEMA = "genie_eval"
    RUNS_INDEX_TABLE = "genie_eval_runs_index"

    def runs_index_fqn(self, catalog: str) -> str:
        return self.bt(catalog, self.RUNS_INDEX_SCHEMA, self.RUNS_INDEX_TABLE)

    def ensure_runs_index_table(self, catalog: str, warehouse_id: str) -> dict:
        """Idempotent: create schema + table if missing. Returns the second statement's response."""
        self.sql_run(
            warehouse_id,
            f"CREATE SCHEMA IF NOT EXISTS {self.bt(catalog, self.RUNS_INDEX_SCHEMA)}",
        )
        return self.sql_run(
            warehouse_id,
            f"""
            CREATE TABLE IF NOT EXISTS {self.runs_index_fqn(catalog)} (
              run_id          BIGINT,
              run_name        STRING,
              space_id        STRING,
              experiment_path STRING,
              submitted_at    TIMESTAMP,
              submitted_by    STRING,
              params_json     STRING
            )
            """,
        )

    def write_runs_index_row(
        self,
        catalog: str,
        warehouse_id: str,
        run_id: int,
        run_name: str,
        space_id: str,
        experiment_path: str,
        submitted_at: str,  # ISO8601 string
        submitted_by: str,
        params_json: str,
    ) -> dict:
        """Insert one row into the runs-index table.

        Uses parameterized INSERT so user-supplied strings cannot inject SQL.
        ``submitted_at`` must be a timestamp-parseable string (ISO8601 works).
        """
        stmt = f"""
            INSERT INTO {self.runs_index_fqn(catalog)}
              (run_id, run_name, space_id, experiment_path, submitted_at, submitted_by, params_json)
            VALUES (:run_id, :run_name, :space_id, :exp, CAST(:submitted_at AS TIMESTAMP), :submitted_by, :params_json)
        """
        params = [
            {"name": "run_id",       "value": str(int(run_id)),  "type": "BIGINT"},
            {"name": "run_name",     "value": run_name,          "type": "STRING"},
            {"name": "space_id",     "value": space_id,          "type": "STRING"},
            {"name": "exp",          "value": experiment_path,   "type": "STRING"},
            {"name": "submitted_at", "value": submitted_at,      "type": "STRING"},
            {"name": "submitted_by", "value": submitted_by,      "type": "STRING"},
            {"name": "params_json",  "value": params_json,       "type": "STRING"},
        ]
        return self.sql_run(warehouse_id, stmt, parameters=params)

    def read_runs_index(self, catalog: str, warehouse_id: str, limit: int = 50) -> list[dict]:
        stmt = f"""
            SELECT run_id, run_name, space_id, experiment_path,
                   CAST(submitted_at AS STRING) AS submitted_at,
                   submitted_by, params_json
            FROM {self.runs_index_fqn(catalog)}
            ORDER BY submitted_at DESC
            LIMIT {int(limit)}
        """
        d = self.sql_run(warehouse_id, stmt)
        if d.get("status", {}).get("state") != "SUCCEEDED":
            return []
        cols = [c["name"] for c in d.get("manifest", {}).get("schema", {}).get("columns", [])]
        rows = self.sql_rows(d)
        return [dict(zip(cols, row)) for row in rows]
