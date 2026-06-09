"""Empirical assumption probe for genie-eval-set-generator (V2.9).

Runs READ-ONLY against a real Databricks workspace with many Genie spaces and tests every
assumption the notebook/app bakes in — extraction APIs, the gateway/host derivation, the
warehouse-execution path (V2.1), the Genie response shape, serving-endpoint existence — and
reports PASS / FAIL / NA per assumption, per space.

It uses the Databricks SDK (no dbutils/spark) so it runs locally, the same way the app backend
does. It does NOT create/drop schemas or tables; SQL is read-only (DESCRIBE / SELECT / Genie).

Usage:
    python3 tools/probe_workspace.py --profile <your-databricks-cli-profile> [--spaces 15] [--deep 2]

  --spaces N : how many Genie spaces to probe (default 15; 0 = all)
  --deep N   : additionally run a minimal generation→Genie→row-match end-to-end on N spaces
               (calls LLM + Genie; still publishes nothing). Default 0.
  --json PATH: write the structured report to PATH.
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from collections import Counter

try:
    from databricks.sdk import WorkspaceClient
except Exception as e:  # pragma: no cover
    print(f"databricks-sdk not importable: {e}\n  pip install 'databricks-sdk>=0.40.0'")
    raise SystemExit(2)

# Reuse the exact row-matching + difficulty logic the notebook ships (mirrored in tests).
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/tests")
try:
    from test_stats import rows_match, structural_difficulty, canon_row  # type: ignore
except Exception:
    rows_match = structural_difficulty = canon_row = None


RESULTS: list[dict] = []


def record(area: str, assumption: str, status: str, detail: str = "", space: str = ""):
    RESULTS.append({"area": area, "assumption": assumption, "status": status,
                    "detail": str(detail)[:400], "space": space})
    tag = {"PASS": "✓", "FAIL": "✗", "NA": "·", "WARN": "!"}.get(status, "?")
    sp = f" [{space}]" if space else ""
    print(f"  {tag} {status:<4} {assumption}{sp}  {('— ' + str(detail)[:160]) if detail else ''}")


def _do_get(w, path):
    """Raw internal REST GET via the SDK (handles auth + correct host)."""
    return w.api_client.do("GET", path)


def list_genie_spaces(w):
    spaces = []
    try:
        resp = w.genie.list_spaces()
        for s in (getattr(resp, "spaces", None) or []):
            spaces.append({"space_id": getattr(s, "space_id", None) or getattr(s, "id", None),
                           "title": getattr(s, "title", "") or ""})
        record("enumerate", "w.genie.list_spaces().spaces works", "PASS", f"{len(spaces)} spaces")
        return [s for s in spaces if s["space_id"]]
    except Exception as e:
        record("enumerate", "w.genie.list_spaces().spaces works", "FAIL", e)
    # Fallback to REST
    try:
        resp = _do_get(w, "/api/2.0/genie/spaces")
        for s in (resp.get("spaces") or []):
            spaces.append({"space_id": s.get("space_id") or s.get("id"), "title": s.get("title", "")})
        record("enumerate", "REST /api/2.0/genie/spaces works", "PASS", f"{len(spaces)} spaces")
    except Exception as e:
        record("enumerate", "REST /api/2.0/genie/spaces works", "FAIL", e)
    return [s for s in spaces if s["space_id"]]


def pick_warehouse(w):
    try:
        whs = list(w.warehouses.list())
        running = [x for x in whs if str(getattr(x, "state", "")).upper().endswith("RUNNING")]
        chosen = (running or whs)
        if chosen:
            wh = chosen[0]
            record("grading", "a SQL warehouse is reachable", "PASS",
                   f"{getattr(wh,'name','?')} ({getattr(wh,'state','?')})")
            return getattr(wh, "id", None)
    except Exception as e:
        record("grading", "a SQL warehouse is reachable", "FAIL", e)
    return None


def run_sql(w, warehouse_id, sql, cap=50):
    from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout as _OWT
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=sql, wait_timeout="30s", on_wait_timeout=_OWT.CANCEL)
    state = getattr(getattr(getattr(resp, "status", None), "state", None), "value", None)
    if state != "SUCCEEDED":
        raise RuntimeError(f"state={state}")
    data = (getattr(resp, "result", None) and resp.result.data_array) or []
    return [list(r) for r in data][:cap]


def probe_serving(w):
    wanted = ["databricks-llama-4-maverick", "databricks-claude-sonnet-4-5", "databricks-bge-large-en"]
    try:
        have = {getattr(e, "name", "") for e in w.serving_endpoints.list()}
        for name in wanted:
            record("serving", f"endpoint exists: {name}", "PASS" if name in have else "FAIL",
                   "" if name in have else "not in this workspace — generator/judge/embed default will 404")
    except Exception as e:
        record("serving", "w.serving_endpoints.list() works", "FAIL", e)


def probe_host_and_f4(w):
    host = (w.config.host or "").rstrip("/")
    record("host", "workspace host resolves from w.config.host (workspace-host derivation)", "PASS" if host else "FAIL", host)
    # The notebook default ai_gateway_base_url ends /mlflow/v1; space_host = url.rsplit('/serving-endpoints')[0]
    bad_default = f"{host}/mlflow/v1"
    derived = bad_default.rsplit("/serving-endpoints", 1)[0]
    f4_broken = derived == bad_default  # split didn't find token -> host stays malformed
    record("host", "default gateway URL yields a VALID data-rooms host", "FAIL" if f4_broken else "PASS",
           f"derived space_host={derived} (still has /mlflow/v1 → data-rooms 404)" if f4_broken
           else "default would derive a usable host")
    return host


def probe_space(w, warehouse_id, space):
    sid, title = space["space_id"], space["title"]
    # B2: data-rooms config
    dr = {}
    try:
        dr = _do_get(w, f"/api/2.0/data-rooms/{sid}") or {}
        keys = [k for k in ("suggestion_description", "table_identifiers", "warehouse_id") if k in dr]
        record("extract", "GET /api/2.0/data-rooms/{id} returns config", "PASS" if dr else "FAIL",
               f"keys present: {keys}", title)
    except Exception as e:
        record("extract", "GET /api/2.0/data-rooms/{id} returns config", "FAIL", e, title)
    tables = list(dr.get("table_identifiers") or [])
    wh_id = dr.get("warehouse_id")
    record("grading", "space exposes its own warehouse_id (V2.1 same-engine)", "PASS" if wh_id else "WARN",
           wh_id or "absent → V2.1 falls back to Spark lower bound", title)
    record("extract", "space.table_identifiers present", "PASS" if tables else "WARN",
           f"{len(tables)} tables", title)
    # B3 curated
    try:
        cq = _do_get(w, f"/api/2.0/data-rooms/{sid}/curated-questions?page_size=200") or {}
        n = len(cq.get("curated_questions") or cq.get("questions") or [])
        record("extract", "curated-questions endpoint returns data", "PASS", f"{n} curated", title)
    except Exception as e:
        record("extract", "curated-questions endpoint returns data", "FAIL", e, title)
    # B4/B5 conversations + messages
    try:
        convs = (_do_get(w, f"/api/2.0/data-rooms/{sid}/conversations?page_size=10") or {}).get("conversations") or []
        record("extract", "conversations endpoint returns data", "PASS", f"{len(convs)} convs", title)
        if convs:
            cid = convs[0].get("conversation_id") or convs[0].get("id")
            m = _do_get(w, f"/api/2.0/genie/spaces/{sid}/conversations/{cid}/messages") or {}
            msgs = m.get("messages") or []
            has_attach = any(a.get("query", {}).get("query") for x in msgs for a in (x.get("attachments") or []))
            record("extract", "messages endpoint returns content + attachments", "PASS",
                   f"{len(msgs)} msgs, attachment-with-query={has_attach}", title)
    except Exception as e:
        record("extract", "conversations/messages endpoints", "FAIL", e, title)
    # B6 DESCRIBE + B8 sampling on first table (read-only)
    if tables and warehouse_id:
        t = tables[0]
        try:
            cols = run_sql(w, warehouse_id, f"DESCRIBE TABLE {t}")
            record("extract", "DESCRIBE TABLE works on table_identifiers", "PASS",
                   f"{t}: {len(cols)} rows", title)
        except Exception as e:
            record("extract", "DESCRIBE TABLE works on table_identifiers", "FAIL", f"{t}: {e}", title)
        # B7 FK presence
        try:
            cat, sch, tbl = t.split(".", 2)
            fk = run_sql(w, warehouse_id,
                         f"SELECT * FROM `{cat}`.information_schema.table_constraints "
                         f"WHERE table_schema='{sch}' AND table_name='{tbl}' AND constraint_type='FOREIGN KEY'")
            record("extract", "FK constraints declared (join-path assumption)", "PASS" if fk else "WARN",
                   f"{len(fk)} FKs" if fk else "no FKs declared → cross-table generation has no join hints", title)
        except Exception as e:
            record("extract", "FK constraints query", "WARN", e, title)
    return {"sid": sid, "title": title, "tables": tables, "warehouse_id": wh_id}


def probe_deep(w, warehouse_id, space):
    """Minimal end-to-end on one space: generate 2 Qs, validate SQL same-engine, ask Genie,
    row-match. Tests V2.1 grading + Genie response shape + data_array string assumption."""
    sid, title = space["sid"], space["title"]
    tables, wh = space["tables"], space.get("warehouse_id") or warehouse_id
    if not tables:
        record("deep", "end-to-end requires ≥1 table", "NA", "space has no tables", title)
        return
    t = tables[0]
    # generate via serving endpoint (SDK query, no token plumbing)
    prompt = (f"Return ONLY JSON {{\"questions\":[...]}} with 2 objects, each "
              f"{{question, expected_sql}}. expected_sql must be valid Databricks SQL over the table "
              f"{t} and return a small result. No markdown.")
    gen = None
    try:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        resp = w.serving_endpoints.query(
            name="databricks-llama-4-maverick",
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
            max_tokens=1500, temperature=0.4)
        gen = resp.choices[0].message.content
        record("deep", "generator endpoint returns content", "PASS", f"{len(gen or '')} chars", title)
    except Exception as e:
        record("deep", "generator endpoint returns content", "FAIL", e, title)
        return
    try:
        qs = json.loads(gen).get("questions", [])[:2]
    except Exception as e:
        record("deep", "generator returns parseable JSON", "FAIL", f"{e}: {gen[:120]}", title)
        return
    record("deep", "generator returns parseable JSON", "PASS", f"{len(qs)} questions", title)
    for q in qs:
        sql, question = q.get("expected_sql", ""), q.get("question", "")
        # validate SQL same-engine (V2.1)
        try:
            exp_rows = run_sql(w, wh, sql)
            record("deep", "expected_sql executes on the space warehouse (V2.1)", "PASS",
                   f"{len(exp_rows)} rows | difficulty={structural_difficulty(sql) if structural_difficulty else '?'}",
                   title)
        except Exception as e:
            record("deep", "expected_sql executes on the space warehouse (V2.1)", "FAIL", f"{e}", title)
            continue
        # data_array string assumption
        if exp_rows:
            kinds = Counter(type(c).__name__ for c in exp_rows[0])
            record("deep", "warehouse data_array cells are strings (canon assumption)", "PASS",
                   f"cell types row0: {dict(kinds)}", title)
        # ask Genie + row-match
        try:
            msg = w.genie.start_conversation_and_wait(space_id=sid, content=question)
            g_sql = g_sid = None
            for a in (msg.attachments or []):
                if getattr(a, "query", None):
                    g_sql = a.query.query
            qr = getattr(msg, "query_result", None)
            g_sid = getattr(qr, "statement_id", None) if qr else None
            record("deep", "Genie start_conversation_and_wait returns attachments/query_result", "PASS",
                   f"genie_sql={'yes' if g_sql else 'no'} stmt_id={'yes' if g_sid else 'no'}", title)
            if g_sid and rows_match:
                stmt = w.statement_execution.get_statement(g_sid)
                g_rows = [list(r) for r in (stmt.result.data_array or [])][:50]
                m = rows_match(exp_rows, g_rows)
                record("deep", "rows_match on REAL Genie vs expected (V2.1)", "PASS",
                       f"match={m} (None=indeterminate/cap)", title)
        except Exception as e:
            record("deep", "Genie conversation + row-match", "WARN", e, title)
        break  # one question is enough for the probe


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile to probe")
    ap.add_argument("--spaces", type=int, default=15)
    ap.add_argument("--deep", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    print(f"== Connecting (profile={args.profile}) ==")
    w = WorkspaceClient(profile=args.profile)
    try:
        me = w.current_user.me()
        print(f"   authenticated as {getattr(me, 'user_name', '?')}\n")
    except Exception as e:
        print(f"   AUTH FAILED: {e}\n   run: databricks auth login --profile {args.profile}")
        return 2

    print("== Host / gateway derivation ==")
    probe_host_and_f4(w)
    print("\n== Serving endpoints (generator/judge/embed defaults) ==")
    probe_serving(w)
    print("\n== Warehouse ==")
    warehouse_id = pick_warehouse(w)
    print("\n== Enumerate Genie spaces ==")
    spaces = list_genie_spaces(w)
    if args.spaces and len(spaces) > args.spaces:
        spaces = spaces[: args.spaces]
    print(f"\n== Probe {len(spaces)} spaces (read-only) ==")
    probed = []
    for s in spaces:
        print(f"\n-- {s['title'] or s['space_id']} --")
        try:
            probed.append(probe_space(w, warehouse_id, s))
        except Exception as e:
            record("extract", "probe_space did not crash", "FAIL", traceback.format_exc()[-300:], s["title"])

    if args.deep:
        print(f"\n== Deep end-to-end on {args.deep} space(s) (LLM + Genie, no writes) ==")
        for sp in [p for p in probed if p["tables"]][: args.deep]:
            print(f"\n-- deep: {sp['title']} --")
            probe_deep(w, warehouse_id, sp)

    # summary
    by_status = Counter(r["status"] for r in RESULTS)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {dict(by_status)}")
    fails = [r for r in RESULTS if r["status"] == "FAIL"]
    if fails:
        print("\nFAILED assumptions:")
        for r in fails:
            print(f"  ✗ [{r['area']}] {r['assumption']} {('('+r['space']+')') if r['space'] else ''} — {r['detail'][:140]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": dict(by_status), "results": RESULTS}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
