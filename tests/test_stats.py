"""Known-answer tests for the statistical functions that survived the v0.1 strip.

These are the *reference implementations* the notebook mirrors (the notebook uses numpy and
can't be imported outside a Databricks runtime). v0.2 item V2.8 unifies them into one shared
module imported by both. Until then, keep these in lock-step with `genie_eval_set_generator.py`
phase 8.5 (`_wilson_half`, `_n_for_half`, rerun-agreement) and 6.5 (leakage flag).

Run:  python3 tests/test_stats.py   (no pytest needed; exits non-zero on failure)
"""
import math


# --- reference implementations (mirror the notebook) ---------------------------------------

def wilson_half(p: float, n: int, zc: float = 1.96) -> float:
    """Wilson 95% half-width for proportion p over n INDEPENDENT trials."""
    if n <= 0:
        return 0.0
    denom = 1.0 + zc * zc / n
    return zc * math.sqrt(p * (1.0 - p) / n + zc * zc / (4.0 * n * n)) / denom


def n_for_half(p: float, target: float, n_floor: int = 2, zc: float = 1.96) -> int:
    n = max(n_floor, 2)
    while wilson_half(p, n, zc) > target and n < 100000:
        n += 1
    return n


def rerun_agreement(matrix):
    """matrix: list of per-question rerun outcomes, each a list of {0,1,None(error)}.
    Returns (agreement, unanimous, judgeable). A question is judgeable with >=2 valid reruns;
    unanimous if all valid reruns agree."""
    unanimous = judgeable = 0
    for row in matrix:
        vals = [v for v in row if v is not None]
        if len(vals) >= 2:
            judgeable += 1
            if min(vals) == max(vals):
                unanimous += 1
    agreement = (unanimous / judgeable) if judgeable else float("nan")
    return agreement, unanimous, judgeable


def question_concordance(matrix):
    """Question-level concordance: a question 'passes' if its mean valid-rerun pass >= 0.5.
    Unit of analysis = question with >=1 valid rerun (NOT N*M cells)."""
    q_pass = []
    for row in matrix:
        vals = [v for v in row if v is not None]
        if vals:
            q_pass.append(sum(vals) / len(vals))
    n_units = len(q_pass)
    passes = sum(1 for p in q_pass if p > 0.5)  # even split = fail (conservative lower bound)
    return (passes / n_units if n_units else 0.0), n_units


def leakage_flag(mean_nn_cosine: float) -> bool:
    return mean_nn_cosine >= 0.97


# --- v0.2 reference impls (mirror notebook rows_match / structural_difficulty / _norm_q) ------
from collections import Counter
from decimal import Decimal


import re as _re

_NUM_STR_RE = _re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?$")


def dec_fmt(d: Decimal) -> str:
    if d == 0:
        d = Decimal(0)                  # normalize -0 / "-0.0" to 0
    return format(d.normalize(), "f")


def canon_cell(v) -> str:
    if v is None:
        return "\x00NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Decimal):
        return dec_fmt(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return dec_fmt(Decimal(str(v)))
    s = str(v).strip()
    core = s.lstrip("+-")
    leading_zero_code = len(core) > 1 and core[0] == "0" and not core.startswith("0.")
    if _NUM_STR_RE.match(s) and not leading_zero_code:
        try:
            return dec_fmt(Decimal(s))
        except Exception:
            return s
    return s


def cells_close(a, b):
    if a == b:
        return True
    try:
        da, db = Decimal(a), Decimal(b)
    except Exception:
        return False
    diff = abs(da - db)
    if diff <= Decimal("1e-9"):
        return True
    scale = max(abs(da), abs(db))
    return (diff / scale) <= Decimal("0.001")


def match_with_permutation(E, G, tolerant):
    from itertools import permutations
    ncols = len(E[0])
    if any(len(r) != ncols for r in E) or any(len(r) != ncols for r in G):
        return False
    if ncols > 7:
        return Counter(tuple(sorted(r)) for r in E) == Counter(tuple(sorted(r)) for r in G)
    ce = Counter(E)
    for p in permutations(range(ncols)):
        Gp = [tuple(g[i] for i in p) for g in G]
        if not tolerant:
            if ce == Counter(Gp):
                return True
        else:
            used = [False] * len(Gp)
            ok = True
            for e in E:
                hit = False
                for j, g in enumerate(Gp):
                    if not used[j] and all(cells_close(x, y) for x, y in zip(e, g)):
                        used[j] = True
                        hit = True
                        break
                if not hit:
                    ok = False
                    break
            if ok:
                return True
    return False


def rows_match(expected_rows, genie_rows, row_cap=50):
    if expected_rows is None:
        return None
    if genie_rows is None:
        return False
    if len(expected_rows) > row_cap or len(genie_rows) > row_cap:
        return None
    E = [tuple(canon_cell(v) for v in (r or [])) for r in expected_rows]
    G = [tuple(canon_cell(v) for v in (r or [])) for r in genie_rows]
    if len(E) != len(G):
        return False
    if not E:
        return True
    if Counter(E) == Counter(G):
        return True
    if match_with_permutation(E, G, tolerant=False):
        return True
    return match_with_permutation(E, G, tolerant=True)


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


def norm_q(s: str) -> str:
    return " ".join((s or "").strip().lower().split()).rstrip("?.! ")


# --- tests -----------------------------------------------------------------------------------

def _approx(a, b, tol=1e-3):
    return abs(a - b) <= tol


def test_wilson_half_n14_m3():
    """The headline from the internal review: at N=14*M=3=42 the 7pp gate is unsatisfiable."""
    n = 14 * 3
    assert _approx(wilson_half(0.5, n), 0.144743, 2e-5), wilson_half(0.5, n)
    floor = min(wilson_half(p / 1000, n) for p in range(100, 901))
    assert _approx(floor, 0.09314, 1e-4), floor
    assert floor > 0.07, "the 7pp gate must be unreachable at N=14/M=3 — that's the whole point"


def test_wilson_half_monotone_in_n():
    assert wilson_half(0.5, 50) > wilson_half(0.5, 200)
    assert wilson_half(0.5, 0) == 0.0


def test_n_for_half_target_10pp():
    n = n_for_half(0.5, 0.10)
    assert wilson_half(0.5, n) <= 0.10
    assert wilson_half(0.5, n - 1) > 0.10
    assert 90 <= n <= 100, n  # ~96 at p=0.5


def test_rerun_agreement():
    # 13 unanimous-pass + 1 flaky (the near-perfect-but-one-flaky case): agreement should be 13/14, NOT 0.
    matrix = [[1, 1, 1]] * 13 + [[1, 0, 1]]
    agr, unan, judge = rerun_agreement(matrix)
    assert (unan, judge) == (13, 14)
    assert _approx(agr, 13 / 14), agr
    # all unanimous -> 1.0
    assert _approx(rerun_agreement([[1, 1], [0, 0]])[0], 1.0)
    # questions with <2 valid reruns are not judgeable
    agr2, _, judge2 = rerun_agreement([[1, None], [1, 0]])
    assert judge2 == 1 and _approx(agr2, 0.0)


def test_question_concordance_tie_is_fail():
    # M=2 even split (1 pass, 1 fail) -> mean 0.5 -> conservative: NOT a pass
    conc, n_units = question_concordance([[1, 0]])
    assert n_units == 1 and conc == 0.0


def test_question_concordance_is_n_units_not_cells():
    # 13/14 pass at question level; n_units must be 14, not 42.
    matrix = [[1, 1, 1]] * 13 + [[0, 0, 1]]  # last question majority-fails
    conc, n_units = question_concordance(matrix)
    assert n_units == 14, n_units
    assert _approx(conc, 13 / 14), conc


def test_leakage_flag():
    assert leakage_flag(0.98) is True
    assert leakage_flag(0.97) is True
    assert leakage_flag(0.9699) is False
    assert leakage_flag(0.6) is False


def test_rows_match_basic_and_canon():
    assert rows_match([[1, 2]], [[1, 2]]) is True
    # cross-engine numeric formatting: 1 vs "1.0" vs 1.00 must match
    assert rows_match([[1, "abc"]], [["1.0", "abc"]]) is True
    assert rows_match([[Decimal("1.00")]], [[1.0]]) is True
    # different values -> no match
    assert rows_match([[1]], [[2]]) is False


def test_rows_match_multiset_not_set():
    # duplicate rows preserved: 2 identical rows != 1 row (the set-based dedup bug)
    assert rows_match([[1], [1]], [[1]]) is False
    assert rows_match([[1], [1]], [[1], [1]]) is True


def test_rows_match_column_order_invariant():
    # same values, columns reordered/aliased -> still match (consistent permutation)
    assert rows_match([[1, 2]], [[2, 1]]) is True
    assert rows_match([[1, "a"], [2, "b"]], [["a", 1], ["b", 2]]) is True


def test_rows_match_inconsistent_swap_fails():
    # cross-column value swap in ONE row must NOT match (the per-row-sort false-PASS bug):
    # no single column permutation maps both rows.
    assert rows_match([[1, 2], [3, 4]], [[2, 1], [3, 4]]) is False


def test_rows_match_numeric_tolerance():
    # ROUND()/float-path differences within 0.1% relative -> match (caught live: 16.67 vs 16.666..)
    assert rows_match([["16.67"]], [["16.666666666"]]) is True
    assert rows_match([["33.333333"]], [["33.33"]]) is True
    # genuinely different values -> no match
    assert rows_match([["33.33"]], [["34.0"]]) is False
    # tolerance must not bridge distinct integers
    assert rows_match([["1000"]], [["1002"]]) is False


def test_rows_match_cap_indeterminate():
    # collectors fetch cap+1 rows: len == cap is a COMPLETE result (determinate);
    # len > cap means truncation -> None (indeterminate)
    exact = [[i] for i in range(50)]
    assert rows_match(exact, exact) is True
    truncated = [[i] for i in range(51)]
    assert rows_match(truncated, truncated) is None
    assert rows_match(truncated, [[0]]) is None


def test_rows_match_none_semantics():
    # no expected baseline (expected_sql failed/denied) -> not-evaluable (None), NOT a miss
    assert rows_match(None, [[1]]) is None
    # we DO have a baseline but Genie returned nothing -> real miss (False)
    assert rows_match([[1]], None) is False


def test_canon_cell_scientific_and_negative_zero():
    # scientific-notation strings unify with their plain forms (warehouse vs Spark fallback)
    assert canon_cell("1e3") == canon_cell(1000) == "1000"
    assert canon_cell("1.5E2") == canon_cell(150)
    # negative zero unifies with zero across all representations
    assert canon_cell(-0.0) == canon_cell(0) == canon_cell("-0") == canon_cell("0.0")


def test_canon_cell_precision_and_codes():
    # 1 vs 1.0 vs "1.00" vs Decimal all unify
    assert canon_cell(1) == canon_cell(1.0) == canon_cell("1.00") == canon_cell(Decimal("1.000"))
    # big integers keep full precision (float .10g would have mangled this)
    assert canon_cell("123456789012345") == "123456789012345"
    assert canon_cell(123456789012345) == "123456789012345"
    # leading-zero codes (zips/phones/IDs) are preserved, not coerced to numbers
    assert canon_cell("01234") == "01234"
    assert canon_cell("007") == "007"
    # genuine decimals still normalize
    assert canon_cell("1.50") == "1.5"
    assert canon_cell("0.5") == "0.5"


def test_structural_difficulty():
    assert structural_difficulty("SELECT count(*) FROM t WHERE a = 'x'") == "easy"
    assert structural_difficulty("SELECT a, sum(b) FROM t GROUP BY a") == "medium"
    assert structural_difficulty("SELECT * FROM a JOIN b ON a.id = b.id") == "medium"
    assert structural_difficulty("WITH c AS (SELECT 1) SELECT * FROM c") == "hard"
    assert structural_difficulty("SELECT RANK() OVER (ORDER BY x) FROM t") == "hard"
    assert structural_difficulty("SELECT a FROM t HAVING count(*) > 1") == "hard"


def test_norm_q():
    assert norm_q("  Show ME the top 5? ") == "show me the top 5"
    assert norm_q("Revenue by region.") == "revenue by region"
    assert norm_q("a   b\tc") == "a b c"


def test_cohen_kappa():
    import os
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from judge_calibration import cohen_kappa, to_binary
    assert _approx(cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)     # perfect agreement
    assert _approx(cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]), -1.0)    # perfect disagreement
    assert cohen_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0            # constant + identical
    assert to_binary("PASS") == 1 and to_binary("fail") == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
