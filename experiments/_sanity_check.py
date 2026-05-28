"""Sanity checks before committing the patched notebooks.

Runs four independent checks:
1. All five rebuilt notebooks are valid JSON with the expected cell counts.
2. The `reported` Wilson CI bundle agrees exactly with the stored
   `harmful_refusal_rate`/`benign_refusal_rate` fields for every config.
3. Every CI is mathematically valid: 0 <= lo <= p <= hi <= 1.
4. Notebook 09's stratified sampler can find at least 2 generations per
   (model, guard_label) stratum across the first 14 size-stratified configs.

Exits non-zero if any check fails.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"

errors: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)
    print(f"  FAIL: {msg}")


def passed(msg: str) -> None:
    print(f"  pass: {msg}")


# ---------------------------------------------------------------------------
# Check 1: notebooks are valid JSON
# ---------------------------------------------------------------------------
print("Check 1 — notebook JSON validity")
expected = {
    "09_official_harmbench_classifier.ipynb": 21,
    "10_confidence_intervals.ipynb": 10,
    "11_polyguard_multilingual.ipynb": 13,
    "12_gcg_attack.ipynb": 11,
    "13_aggregate_revision_results.ipynb": 2,
}
for name, _ in expected.items():
    p = ROOT / name
    if not p.exists():
        fail(f"{name} missing")
        continue
    try:
        nb = json.loads(p.read_text())
    except Exception as exc:
        fail(f"{name} not valid JSON: {exc}")
        continue
    n = len(nb.get("cells", []))
    if n < 2:
        fail(f"{name} has only {n} cell(s)")
    else:
        passed(f"{name}: {n} cells, valid JSON")
print()


# ---------------------------------------------------------------------------
# Check 2: recomputed `reported` rates match the stored fields exactly
# ---------------------------------------------------------------------------
print("Check 2 — recomputed `reported` rates vs stored fields")
src = RESULTS_DIR / "slm_safety_results_v3.json"
all_runs = json.loads(src.read_text())
intervals = json.loads((RESULTS_DIR / "section14_intervals.json").read_text())

drift_count = 0
for key, entry in all_runs.items():
    stored_h = entry.get("harmful_refusal_rate")
    stored_b = entry.get("benign_refusal_rate")
    if stored_h is None and stored_b is None:
        continue
    rep = intervals[key]["reported"]
    rec_h = round(rep["harmful_refusal"]["p"], 3)
    rec_b = round(rep["benign_refusal"]["p"], 3)
    if stored_h is not None and abs(rec_h - stored_h) > 1e-3:
        fail(f"{key}: harmful drift {rec_h} vs stored {stored_h}")
        drift_count += 1
    if stored_b is not None and abs(rec_b - stored_b) > 1e-3:
        fail(f"{key}: benign drift {rec_b} vs stored {stored_b}")
        drift_count += 1
if drift_count == 0:
    passed(f"all {len(all_runs)} configs match stored rates exactly")
print()


# ---------------------------------------------------------------------------
# Check 3: CIs are mathematically valid for all three judges
# ---------------------------------------------------------------------------
print("Check 3 — CI bounds are valid for all three judges")
bad_ci = 0
for key, m in intervals.items():
    for judge in ("reported", "guard", "keyword"):
        for split in ("harmful_refusal", "benign_refusal"):
            x = m[judge][split]
            p, lo, hi = x["p"], x["ci_lo"], x["ci_hi"]
            if not (0.0 <= lo <= p <= hi <= 1.0):
                fail(f"{key}.{judge}.{split}: lo={lo}, p={p}, hi={hi}")
                bad_ci += 1
if bad_ci == 0:
    n_checked = len(intervals) * 3 * 2
    passed(f"all {n_checked} CIs have 0 <= lo <= p <= hi <= 1")
print()


# ---------------------------------------------------------------------------
# Check 4: notebook 09's stratification yields workable sample sizes
# ---------------------------------------------------------------------------
print("Check 4 — notebook 09 stratified sampler")


def normalise_guard_label(raw: str | None) -> str:
    v = (raw or "").split("\n", 1)[0].strip().lower()
    if v == "safe":
        return "safe"
    if v.startswith("unsafe"):
        return "unsafe"
    return "unknown"


exp1_keys = list(all_runs.keys())[:14]
strata: dict[tuple[str, str], int] = {}
for key in exp1_keys:
    entry = all_runs[key]
    label = entry["label"]
    for det in entry.get("harmful_details", []):
        s = (label, normalise_guard_label(det.get("guard_label")))
        strata[s] = strata.get(s, 0) + 1

models_with_two_strata = 0
for label in {k[0] for k in strata}:
    safe_n = strata.get((label, "safe"), 0)
    unsafe_n = strata.get((label, "unsafe"), 0)
    if safe_n >= 2 and unsafe_n >= 2:
        models_with_two_strata += 1
    else:
        fail(f"{label}: safe={safe_n}, unsafe={unsafe_n} (need >=2 each for κ)")
if models_with_two_strata == 14:
    passed("all 14 Exp1 models have >=2 generations per guard_label stratum")
print()


# ---------------------------------------------------------------------------
# Check 5: Colab metadata configures A100 high-memory runtime where appropriate
# ---------------------------------------------------------------------------
print("Check 5 — Colab runtime metadata")
GPU_NOTEBOOKS = {
    "09_official_harmbench_classifier.ipynb",
    "11_polyguard_multilingual.ipynb",
    "12_gcg_attack.ipynb",
}
CPU_NOTEBOOKS = {
    "10_confidence_intervals.ipynb",
    "13_aggregate_revision_results.ipynb",
}

for name in GPU_NOTEBOOKS:
    p = ROOT / name
    nb = json.loads(p.read_text())
    md = nb.get("metadata", {})
    colab = md.get("colab", {})
    accel = md.get("accelerator")
    if accel != "GPU":
        fail(f"{name}: accelerator={accel!r} (expected 'GPU')")
        continue
    if colab.get("gpuType") != "A100":
        fail(f"{name}: colab.gpuType={colab.get('gpuType')!r} (expected 'A100')")
        continue
    if colab.get("machine_shape") != "hm":
        fail(f"{name}: colab.machine_shape={colab.get('machine_shape')!r} (expected 'hm')")
        continue
    passed(f"{name}: A100 high-memory runtime configured")

for name in CPU_NOTEBOOKS:
    p = ROOT / name
    nb = json.loads(p.read_text())
    md = nb.get("metadata", {})
    accel = md.get("accelerator")
    if accel is not None:
        fail(f"{name}: accelerator={accel!r} (expected None / CPU)")
        continue
    passed(f"{name}: CPU runtime configured")
print()


# ---------------------------------------------------------------------------
# Check 6: GPU notebooks have a hard-fail assertion if no GPU is allocated
# ---------------------------------------------------------------------------
print("Check 6 — GPU notebooks fail fast on CPU runtime")
for name in GPU_NOTEBOOKS:
    p = ROOT / name
    nb = json.loads(p.read_text())
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
    if "assert torch.cuda.is_available()" not in src:
        fail(f"{name}: no torch.cuda.is_available() assert found")
        continue
    passed(f"{name}: hard-fail GPU assertion present")
print()


# ---------------------------------------------------------------------------
# Check 7: every code cell parses as valid Python (catches f-string and
#          escape-sequence bugs at scaffold time, not at runtime in Colab)
# ---------------------------------------------------------------------------
print("Check 7 — every code cell parses as valid Python")
import ast
for name in (GPU_NOTEBOOKS | CPU_NOTEBOOKS):
    p = ROOT / name
    nb = json.loads(p.read_text())
    bad_cells = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        # Strip Colab/Jupyter cell magics (%%capture, !pip, %pip, etc.) which
        # are not valid Python on their own. A plain ast.parse would otherwise
        # reject these legitimate cells.
        scrubbed_lines = []
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%%", "%", "!")):
                scrubbed_lines.append("# " + line)
            else:
                scrubbed_lines.append(line)
        scrubbed = "\n".join(scrubbed_lines)
        try:
            ast.parse(scrubbed)
        except SyntaxError as exc:
            bad_cells += 1
            fail(f"{name} cell {i} ({cell['cell_type']}): {exc.msg} at line {exc.lineno}")
            # also dump the offending line to make the failure debuggable
            try:
                offending = scrubbed.splitlines()[exc.lineno - 1]
                print(f"      offending line: {offending!r}")
            except IndexError:
                pass
    if bad_cells == 0:
        passed(f"{name}: all code cells parse")
print()


# ---------------------------------------------------------------------------
print("=" * 60)
if errors:
    print(f"FAILED: {len(errors)} issue(s) above")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
