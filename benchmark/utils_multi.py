# utils_multi.py
# --------------------------------------------------------------------
# Expand a YAML that may contain scalars *or* lists into a list of
# scenario-specific run-configs.  After expansion we:
#   • remove parameters irrelevant to each scenario
#   • derive model_name from hf_model if missing
#   • drop duplicate dicts created by list values that were later zapped
#   • sort so the same (backend, hf_model) are consecutive
# --------------------------------------------------------------------
import json, os, yaml
from itertools import product
from pathlib import Path
from typing import Dict, Any, List

# ───────────────────────── helpers
def _derive_name(path: str) -> str:
    return os.path.basename(path.rstrip("/"))

def _zap(run: Dict[str, Any], *keys):
    for k in keys:
        run.pop(k, None)

def _sanitize(run: Dict[str, Any]) -> Dict[str, Any]:
    sc = run["scenario"]

    if sc == "server":
        _zap(run, "batch_size", "samples")

    elif sc == "batch":
        _zap(run, "run_time", "concurrent_users", "requests_per_user_per_min")

    elif sc == "single":
        _zap(run, "batch_size", "run_time",
                  "concurrent_users", "requests_per_user_per_min")

    elif sc == "long_context":
        _zap(run, "batch_size", "run_time",
                  "concurrent_users", "requests_per_user_per_min")

    # ensure model_name
    if not run.get("model_name"):
        run["model_name"] = _derive_name(run["hf_model"])

    return run
# ───────────────────────── main loader
def load_multi_cfg(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 1) broadcast scalars → 1-element lists
    lists = {k: (v if isinstance(v, list) else [v]) for k, v in raw.items()}

    # 2) Cartesian product
    keys   = list(lists)
    combos = [dict(zip(keys, vals)) for vals in product(*lists.values())]

    # 3) sanitise
    combos = [_sanitize(c) for c in combos]

    # 4) remove duplicates that arose after sanitising
    seen, uniq = set(), []
    for c in combos:
        sig = json.dumps(c, sort_keys=True)
        if sig not in seen:
            uniq.append(c)
            seen.add(sig)
    combos = uniq

    # 5) stable ordering so we reuse engines
    combos.sort(key=lambda d: (d["backend"], d["hf_model"]))
    return combos

# ───────────────────────── smoke-test
if __name__ == "__main__":
    cfg_path = Path("config.yaml")        # adjust as needed
    for i, combo in enumerate(load_multi_cfg(cfg_path), 1):
        print(f"{i:03d}. {combo}")