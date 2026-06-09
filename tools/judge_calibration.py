"""Judge-vs-human calibration harness.

The LLM judges (`question_clarity`, `sql_answers_question`, `answer_quality`) are uncalibrated —
their means are opinions, not measurements, until checked against human labels. This tool closes
that gap: hand-label a sample of items, drop them in a CSV, and this computes judge↔human agreement
(Cohen's κ + raw agreement + a confusion count). The gate policy: do NOT trust a judge
to gate anything until its κ clears a stated floor.

CSV format (one row per labelled item), header required:
    item_id,judge,judge_label,human_label
  - judge        : which judge (question_clarity | sql_answers_question | answer_quality)
  - judge_label  : the judge's binary verdict — 1/0, true/false, pass/fail, yes/no
  - human_label  : your binary verdict, same encoding

Usage:
    python3 tools/judge_calibration.py labels.csv
    python3 tools/judge_calibration.py labels.csv --floor 0.6

Exit code is non-zero if any judge with ≥10 labelled items falls below the κ floor (default 0.6),
so this can run in CI once labels exist.
"""
from __future__ import annotations
import csv
import sys
from collections import Counter, defaultdict

_TRUE = {"1", "true", "pass", "yes", "y", "t"}
_FALSE = {"0", "false", "fail", "no", "n", "f"}


def to_binary(s: str):
    v = (s or "").strip().lower()
    if v in _TRUE:
        return 1
    if v in _FALSE:
        return 0
    raise ValueError(f"unrecognised label {s!r} (use 1/0, pass/fail, true/false, yes/no)")


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's κ for two equal-length label lists. 1.0 = perfect, 0 = chance, <0 = worse."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    cats = set(a) | set(b)
    pe = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in cats)
    if pe >= 1.0:               # both raters constant + identical → perfect agreement
        return 1.0
    return (po - pe) / (1.0 - pe)


def raw_agreement(a: list, b: list) -> float:
    return (sum(1 for x, y in zip(a, b) if x == y) / len(a)) if a else float("nan")


def calibrate(rows: list[dict]) -> dict:
    """rows: dicts with judge, judge_label, human_label. Returns per-judge stats."""
    by_judge: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        by_judge[r["judge"]].append((to_binary(r["judge_label"]), to_binary(r["human_label"])))
    out = {}
    for judge, pairs in sorted(by_judge.items()):
        j = [p[0] for p in pairs]
        h = [p[1] for p in pairs]
        out[judge] = {
            "n": len(pairs),
            "kappa": cohen_kappa(j, h),
            "raw_agreement": raw_agreement(j, h),
            "judge_pos_rate": sum(j) / len(j),
            "human_pos_rate": sum(h) / len(h),
        }
    return out


def _main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    path = argv[0]
    floor = 0.6
    if "--floor" in argv:
        floor = float(argv[argv.index("--floor") + 1])
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no rows in CSV")
        return 1
    stats = calibrate(rows)
    print(f"Judge calibration ({len(rows)} labels, κ floor = {floor}):\n")
    print(f"{'judge':<24} {'n':>4} {'kappa':>7} {'agree':>7} {'judge+':>7} {'human+':>7}  verdict")
    failed = 0
    for judge, s in stats.items():
        gated = s["n"] >= 10
        ok = (not gated) or s["kappa"] >= floor
        if gated and not ok:
            failed += 1
        verdict = "—" if not gated else ("OK" if ok else f"BELOW κ {floor}")
        print(f"{judge:<24} {s['n']:>4} {s['kappa']:>7.3f} {s['raw_agreement']:>7.3f} "
              f"{s['judge_pos_rate']:>7.2f} {s['human_pos_rate']:>7.2f}  {verdict}")
    print(f"\n{'PASS' if not failed else 'FAIL'} — {failed} judge(s) below the κ floor "
          f"(judges with <10 labels are not gated yet).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
