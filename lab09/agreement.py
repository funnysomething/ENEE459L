
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    """Summarize finite, nonnegative durations in milliseconds."""
    values = [_finite(v, f"ms[{i}]") for i, v in enumerate(ms)]
    if not values:
        raise AgreementError("no latency samples.")
    if any(v < 0 for v in values):
        raise AgreementError("latency samples must be nonnegative.")
    report = {
        "n": len(values),
        "mean": sum(values) / len(values),
        "worst": max(values),
    }
    report.update({f"p{q}": _percentile(values, q)
                   for q in LATENCY_PERCENTILES})
    report["jitter"] = report["p95"] - report["p50"]
    return report


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    """Compare durations; a negative saving means the other backend is slower."""
    base = _finite(base_ms, "base_ms")
    other = _finite(other_ms, "other_ms")
    if base < 0 or other <= 0:
        raise AgreementError("base_ms must be nonnegative and other_ms positive.")
    factor = base / other
    return {"factor": factor, "faster": factor > 1, "saved_ms": base - other}


def _paired_values(a: Sequence[float], b: Sequence[float],
                   where: str) -> tuple[list[float], list[float]]:
    if len(a) != len(b) or not len(a):
        raise AgreementError(f"{where} must have equal, nonzero lengths.")
    return ([_finite(v, f"{where}.a[{i}]") for i, v in enumerate(a)],
            [_finite(v, f"{where}.b[{i}]") for i, v in enumerate(b)])


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Find the largest absolute deviation, choosing the first tied index."""
    av, bv = _paired_values(a, b, "outputs")
    deviations = [abs(x - y) for x, y in zip(av, bv)]
    index = max(range(len(av)), key=deviations.__getitem__)
    return {"max_abs": deviations[index], "index": index,
            "a": av[index], "b": bv[index], "n": len(av)}


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    """Measure relative error against b and disclose denominator clamping."""
    floor = _finite(floor, "floor")
    if floor <= 0:
        raise AgreementError("floor must be strictly positive.")
    av, bv = _paired_values(a, b, "outputs")
    magnitudes = [abs(v) for v in bv]
    deviations = [abs(x - y) / max(magnitude, floor)
                  for x, y, magnitude in zip(av, bv, magnitudes)]
    index = max(range(len(av)), key=deviations.__getitem__)
    return {"max_rel": deviations[index], "index": index,
            "floored": sum(v < floor for v in magnitudes),
            "denom_min": min(magnitudes), "n": len(av)}


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Compare paired argmaxes, breaking score ties by lowest class index."""
    if len(rows_a) != len(rows_b) or not len(rows_a):
        raise AgreementError("score batches must have equal, nonzero lengths.")
    disagreements = []
    for i, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        av, bv = _paired_values(row_a, row_b, f"rows[{i}]")
        top_a = max(range(len(av)), key=av.__getitem__)
        top_b = max(range(len(bv)), key=bv.__getitem__)
        if top_a != top_b:
            disagreements.append(i)
    n = len(rows_a)
    disagree = len(disagreements)
    return {"n": n, "agree": n - disagree, "disagree": disagree,
            "rate": (n - disagree) / n,
            "first_disagreement": disagreements[0] if disagree else None}


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    """Certify margins against a per-score absolute error bound, tol.

    The winner can fall by tol while the runner-up rises by tol. A margin
    must therefore exceed 2 * tol to guarantee the same decision; equality
    can produce a tie and is conservatively counted as unsafe.
    """
    tol = _finite(tol, "tol")
    if tol < 0:
        raise AgreementError("tol must be nonnegative.")
    if not len(rows):
        raise AgreementError("no score rows.")
    margins = []
    for i, row in enumerate(rows):
        if len(row) < 2:
            raise AgreementError(f"rows[{i}] needs at least two classes.")
        scores = sorted((_finite(v, f"rows[{i}][{j}]")
                         for j, v in enumerate(row)), reverse=True)
        margins.append(scores[0] - scores[1])
    argmin = min(range(len(margins)), key=margins.__getitem__)
    below_tol = sum(margin <= 2 * tol for margin in margins)
    return {"n": len(margins), "min": margins[argmin],
            "median": _percentile(margins, 50),
            "mean": sum(margins) / len(margins), "argmin": argmin,
            "below_tol": below_tol, "safe": below_tol == 0}

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()
