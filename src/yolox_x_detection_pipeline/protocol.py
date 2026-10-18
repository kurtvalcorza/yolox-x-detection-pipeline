"""The tutorial's pre-registered selection and fresh-test rules (docs/yolox-x-preregistration.md, 2026-10-18).

`PREREGISTERED_SETTINGS` lists every setting the pre-registration fixes. A run is `preregistered` only when all
of them hold, the validation grid is complete and ran with them, the model in hand is the selected system, and the
generated fresh pool matches its committed digest; only such a run scores the fresh test. Every other run is
exploratory: validation and the labelled exposed development test only, written to `*_exploratory_<n>` files. A
runtime cannot see other runtimes, so `RECORDED_EVALUATION` carries the hosted run that is the evaluation of record
(docs/release-verification.md). Once it is set, every pre-registered scoring, including the first one in a new
runtime, is a reproduction: it writes `*_reproduction_<n>` files, never the main files, and is compared with the
record (`compare_with_record`). With no record, a repeat in the same runtime is the reproduction.

Shared design for the YOLOX tutorials (rows 116 and 117): only `VALIDATION_POOL`, `FRESH_TEST_POOL` and the
digests differ between the two repositories.
"""

# ruff: noqa: E501, I001  -- I001: the two single-line package imports below must stay separate (standalone generator)
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# One name list per line: the standalone generator strips single-line package imports.
from .pipeline import COCO_IOU_THRESHOLDS, EVAL_DETECTION_THRESHOLD, EVAL_NMS_THRESHOLD, MAX_EVAL_DETECTIONS
from .pipeline import ap_from_matches, prepared_matches
from .pipeline import DETERMINISM

# ---- pools (drawn by samples.sign_dataset; digests computed under the pinned Pillow 11.3.0) ------------------------
VALIDATION_POOL = {"n": 40, "seed": 1171}
# Amendment 2: fresh test 3. Seed 1170 (fresh test 2) is now the exposed test scored by the round-2 record.
FRESH_TEST_POOL = {"n": 200, "seed": 1172}
ROUND2_FRESH_TEST_POOL = {"n": 200, "seed": 1170}
TRAINING_DIGEST = "60b0aaeea62cf9bb0498af03c1b99b43c644a4a09fee5e196e4d69dc9aef94bc"
EXPOSED_DEV_TEST_DIGEST = "8729a7c00ce71282f406a899e88bc63b713795932240257de6545d82cb34eef8"
VALIDATION_DIGEST = "e98aec93282e64b15b982d102e6dfb0594f321906399086ac62cb71ee08df02b"
FRESH_TEST_DIGEST = "098fa0dc920a166ff2ec3393bd3e175aa88e3a864525bb5862827a39e5ba7b36"
ROUND2_FRESH_TEST_DIGEST = "1f5ae3d9561b81b12e2e3122231ef91a75d6659bf0cf8734d82c9dba4f52a2d7"

# ---- grid and selection --------------------------------------------------------------------------------------------
GRID: tuple[dict[str, Any], ...] = (
    {"name": "C0", "epochs": 6, "learning_rate": 1e-3},
    {"name": "C1", "epochs": 10, "learning_rate": 1e-3},
    {"name": "C2", "epochs": 8, "learning_rate": 2e-3},
)
DEFAULT_CONFIG = "C0"
# What a grid-off (CPU) run trains: the selection of the amendment-2 evaluation of record (C1; hosted T4, 057d2d8),
# the same choice as the round-2 record (35dc06f); C0 before both. A grid-off run is exploratory either way.
RECORDED_SELECTION = "C1"
TIE_ORDER = ("C0", "C2", "C1")  # exact ties go to the smaller config (fewer epochs)
TIE_TOLERANCE = 1e-9
MIN_AP_GAIN = 0.03  # a challenger must beat C0's validation AP@[.50:.95] by this much
MAX_AP50_DROP = 0.02  # ... and lose at most this much validation AP50
KEEP_BASELINE_GAIN = 0.03  # the adapted model replaces the pretrained baseline only above this validation AP gain
BASELINE_SYSTEM = "baseline"

# ---- reading rule --------------------------------------------------------------------------------------------------
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261018
IMPROVED_MIN_GAIN = 0.03

PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": "sample",
    "n_images": 40,
    "dataset_seed": 0,
    "holdout": 0.25,
    "split_seed": 0,
    "training_digest": TRAINING_DIGEST,
    "validation_pool": dict(VALIDATION_POOL),
    "validation_digest": VALIDATION_DIGEST,
    "fresh_test_pool": dict(FRESH_TEST_POOL),
    "fresh_test_digest": FRESH_TEST_DIGEST,
    "batch_size": 2,
    "adapt_seed": 0,
    "freeze_backbone": True,
    "eval_threshold": EVAL_DETECTION_THRESHOLD,
    "eval_nms_threshold": EVAL_NMS_THRESHOLD,
    "max_detections": MAX_EVAL_DETECTIONS,
    "iou_thresholds": [float(t) for t in COCO_IOU_THRESHOLDS],
    "grid": [dict(c) for c in GRID],
    "run_grid": True,
    "min_ap_gain": MIN_AP_GAIN,
    "max_ap50_drop": MAX_AP50_DROP,
    "keep_baseline_gain": KEEP_BASELINE_GAIN,
    "device_class": "t4",  # amendment 1: the decision is taken on a hosted T4 only
    "determinism": dict(DETERMINISM),  # amendment 2
}

# Round 2's hosted T4 evaluation of record on fresh test 2, kept as history (docs/release-verification.md, "Hosted T4
# evaluation of record (2026-10-18)"); values as printed by Sections 8 and 9 of that run (4 decimals).
ROUND2_RECORDED_EVALUATION: dict[str, Any] = {
    "label": "pre-registered evaluation of record",
    "date": "2026-10-18",
    "runner": "Colab CLI, fresh hosted T4 session",
    "device_class": "t4",
    "commit": "35dc06f07ad914834eb56cca5495f1fa0ee38906",
    "notebook_blob": "a90023b2b9bb43db2055da1e60197b87aade7f40",
    "evidence": "docs/execution-evidence/2026-10-18-35dc06f/",
    "selected_config": "C1",
    "selected_system": "C1",
    "validation_ap": {"baseline": 0.0378, "C0": 0.8510, "C1": 0.9237, "C2": 0.8533},
    "fresh_test": {
        "n_images": 200,
        "n_references": 413,
        "selected_ap": 0.9165,
        "baseline_ap": 0.0445,
        "ap_difference": 0.872,
        "ci95": [0.8555, 0.8887],
        "verdict": "improved",
    },
    "fresh_pool": dict(ROUND2_FRESH_TEST_POOL),
    "status": "history: the round-2 record on fresh test 2 (amendment 2 made that pool an exposed test)",
}

# Amendment 2: the evaluation of record is the first hosted T4 run that scores fresh test 3 with deterministic
# training (docs/release-verification.md, "Amendment-2 evaluation of record: hosted T4 (2026-10-18, commit 057d2d8)");
# values as printed by Sections 8 and 9 of that run (4 decimals). Its grid hash matched both masked determinism
# checks. Every later identical pre-registered scoring, in any runtime, is a reproduction compared with it. The
# round-2 record above stays as history and is never overwritten.
RECORDED_EVALUATION: dict[str, Any] | None = {
    "label": "pre-registered evaluation of record",
    "date": "2026-10-18",
    "runner": "Colab CLI, fresh hosted T4 session",
    "device_class": "t4",
    "commit": "057d2d8ff3bc30ede1e67f74d610274e02986513",
    "notebook_blob": "6f13a083a88d4bc9fe67957b8d16f00e0fc27ada",
    "evidence": "docs/execution-evidence/2026-10-18-057d2d8/",
    "selected_config": "C1",
    "selected_system": "C1",
    "validation_ap": {"baseline": 0.0378, "C0": 0.8510, "C1": 0.9237, "C2": 0.8533},
    "grid_hash": "16768b30e5930007674d365a4b6a00d71b26927cc431bbebf96235988e38e544",
    "fresh_test": {
        "n_images": 200,
        "n_references": 377,
        "selected_ap": 0.914,
        "baseline_ap": 0.0365,
        "ap_difference": 0.8775,
        "ci95": [0.8603, 0.8944],
        "verdict": "improved",
    },
    "fresh_pool": dict(FRESH_TEST_POOL),
    "status": "the evaluation of record on fresh test 3 (amendment 2)",
}

TEST_LOOK_PREREGISTERED = "pre-registered evaluation (this runtime)"
TEST_LOOK_REPRODUCTION = "reproduction"
TEST_LOOK_EXPLORATORY = "not scored (exploratory)"
TEST_LOOK_MASKED = "not scored (fresh test masked for a development check)"


def device_class(torch_module: Any) -> tuple[str, str]:
    """(class, name) of the device this runtime trains on: 't4', 'cuda-other' or 'cpu' (amendment 1)."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return "cpu", "cpu"
    try:
        name = str(cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - an unnamed CUDA device is simply not a T4
        name = "unknown CUDA device"
    return ("t4" if "T4" in name.upper() else "cuda-other"), name


def record_digest(record: Mapping[str, Any]) -> str:
    """sha256 over one record's decoded pixels (mode, size, RGB bytes)."""
    image = record["image"]
    digest = hashlib.sha256(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def layout_key(record: Mapping[str, Any]) -> str:
    """The record's box layout (labels and boxes rounded to 0.1 px), for the layout-disjointness check."""
    parts = [f"{label}@{','.join(f'{v:.1f}' for v in box)}" for box, label in zip(record["boxes"], record["labels"], strict=True)]
    return ";".join(sorted(parts))


def pool_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """sha256 over the ordered per-record pixel digests and box layouts."""
    lines = [f"{i}:{record_digest(r)}:{layout_key(r)}" for i, r in enumerate(records)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def preregistration_differences(current: Mapping[str, Any]) -> list[str]:
    """Every pre-registered setting whose current value differs (a missing setting differs)."""
    missing = object()
    return [key for key, value in PREREGISTERED_SETTINGS.items() if current.get(key, missing) != value]


def config_record(
    config: Mapping[str, Any],
    run: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    *,
    digests: Mapping[str, str],
    fingerprint: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """One grid row, validation only: the settings it ran with and its validation scores."""
    row: dict[str, Any] = {"name": config["name"], "config": dict(config), "digests": dict(digests)}
    if error is not None or run is None or validation is None:
        row.update({"status": "failed", "error": error or "no result"})
        return row
    row.update(
        {
            "status": "complete",
            "settings": {k: run[k] for k in ("epochs", "learning_rate", "batch_size", "seed", "freeze_backbone")},
            "val_ap": float(validation["ap"]),
            "val_ap50": float(validation["ap50"]),
            "final_total_loss": float(run["history"][-1]["total_loss"]) if run.get("history") else None,
            "fingerprint": fingerprint,
            "loss_curve": [float(h["total_loss"]) for h in run.get("history") or []],
        }
    )
    return row


def grid_hash(grid_rows: Mapping[str, Mapping[str, Any]], selection: Mapping[str, Any] | None) -> str:
    """sha256 over what two deterministic runs must reproduce exactly (amendment 2's masked determinism check):
    each row's status, validation AP and AP50, loss curve and weight fingerprint, and the selection."""
    rows = {
        name: {k: row.get(k) for k in ("status", "val_ap", "val_ap50", "loss_curve", "fingerprint")}
        for name, row in sorted(grid_rows.items())
    }
    chosen = None if selection is None else {k: selection.get(k) for k in ("selected_config", "selected_system")}
    text = json.dumps({"rows": rows, "selection": chosen}, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def grid_check(
    grid: Mapping[str, Mapping[str, Any]] | None,
    settings: Mapping[str, Any],
    digests: Mapping[str, str],
) -> dict[str, Any]:
    """Is the grid complete, run with the current fixed settings, on the current training and validation data?"""
    problems: list[str] = []
    grid = grid or {}
    for config in GRID:
        row = grid.get(config["name"])
        if row is None:
            problems.append(f"{config['name']}: not run")
            continue
        if row.get("status") != "complete":
            problems.append(f"{config['name']}: {row.get('status')} ({row.get('error')})")
            continue
        expected = {
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "batch_size": settings.get("batch_size"),
            "seed": settings.get("adapt_seed"),
            "freeze_backbone": settings.get("freeze_backbone"),
        }
        if row.get("settings") != expected:
            problems.append(f"{config['name']}: ran with {row.get('settings')}, the pre-registered settings imply {expected}")
        if dict(row.get("digests", {})) != dict(digests):
            problems.append(f"{config['name']}: ran on different training or validation data")
    extra = sorted(set(grid) - {c["name"] for c in GRID})
    if extra:
        problems.append(f"configs outside the pre-registered grid: {extra}")
    return {"complete": not problems, "problems": problems}


def select_config(grid: Mapping[str, Mapping[str, Any]], baseline_val_ap: float) -> dict[str, Any]:
    """The pre-registered choice (validation only): C0 unless a challenger qualifies; then baseline vs adapted."""
    done = {name: row for name, row in grid.items() if row.get("status") == "complete"}
    if not done:
        raise ValueError("no grid config completed")
    base_name = DEFAULT_CONFIG if DEFAULT_CONFIG in done else min(done, key=TIE_ORDER.index)
    base = done[base_name]
    gains = {
        name: row["val_ap"] - base["val_ap"]
        for name, row in done.items()
        if name != base_name
        and row["val_ap"] - base["val_ap"] >= MIN_AP_GAIN - TIE_TOLERANCE
        and base["val_ap50"] - row["val_ap50"] <= MAX_AP50_DROP + TIE_TOLERANCE
    }
    if gains:
        best = max(gains.values())
        tied = [n for n, g in gains.items() if best - g <= TIE_TOLERANCE]
        config = min(tied, key=TIE_ORDER.index)
        reason = (
            f"{config} beat {base_name} by {gains[config]:.4f} validation AP (margin {MIN_AP_GAIN}) with AP50 at most "
            f"{MAX_AP50_DROP} lower"
        )
    else:
        config = base_name
        reason = f"no challenger beat {base_name} by {MIN_AP_GAIN} validation AP with AP50 at most {MAX_AP50_DROP} lower"
    adapted_gain = done[config]["val_ap"] - baseline_val_ap
    keeps_adapted = adapted_gain >= KEEP_BASELINE_GAIN - TIE_TOLERANCE
    return {
        "selected_config": config,
        "selected_system": config if keeps_adapted else BASELINE_SYSTEM,
        "reason": reason,
        "adapted_gain_over_baseline": adapted_gain,
        "keep_baseline_rule": (
            f"{config} beats the pretrained baseline by {adapted_gain:.4f} validation AP "
            f"({'>=' if keeps_adapted else '<'} {KEEP_BASELINE_GAIN}): "
            + ("the adapted model is the selected system" if keeps_adapted else "the baseline is kept (a negative result)")
        ),
        "scores": {n: {"val_ap": r["val_ap"], "val_ap50": r["val_ap50"]} for n, r in done.items()},
        "fingerprint": done[config].get("fingerprint"),
    }


def paired_bootstrap(
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired image-level bootstrap of AP@[.50:.95] (selected - baseline), 95 % percentile interval.

    `selected` and `baseline` are `match_detections` tables over the same images; each resample draws images with
    replacement and recomputes both APs with the same multiplicities, so an image moves with all its boxes."""
    n = selected["n_images"]
    if baseline["n_images"] != n or selected["references"] != baseline["references"]:
        raise ValueError("the two systems must be scored on the same images in the same order")
    prep_sel, prep_base = prepared_matches(selected), prepared_matches(baseline)
    point_sel = ap_from_matches(selected, prepared=prep_sel)
    point_base = ap_from_matches(baseline, prepared=prep_base)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        weights = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        diffs[b] = ap_from_matches(selected, weights, prepared=prep_sel)["ap"] - ap_from_matches(baseline, weights, prepared=prep_base)["ap"]
    low, high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "n_images": n,
        "resampling_unit": "image",
        "n_boot": n_boot,
        "seed": seed,
        "selected_ap": point_sel["ap"],
        "baseline_ap": point_base["ap"],
        "primary_ap_difference": point_sel["ap"] - point_base["ap"],
        "primary_ap_ci95": [low, high],
        "ci_half_width": (high - low) / 2,
        "bootstrap_se": float(diffs.std(ddof=1)),
        "secondary_ap50_difference": point_sel["ap50"] - point_base["ap50"],
    }


def reading_verdict(boot: Mapping[str, Any] | None, *, same_model: bool) -> str:
    """The pre-registered verdict, from the primary AP difference and its CI only."""
    if same_model:
        return "same model (adaptation not selected)"
    low, high = boot["primary_ap_ci95"]
    if low > 0 and boot["primary_ap_difference"] >= IMPROVED_MIN_GAIN:
        return "improved"
    if high < 0:
        return "worse"
    return "no detectable difference"


def scoring_mode(
    *, differences: Sequence[str], grid: Mapping[str, Any], state_matches: bool, fresh_digest_ok: bool | None, masked: bool
) -> str:
    """'preregistered' only when every check holds; 'masked' for a development check; else 'exploratory'.

    `fresh_digest_ok` is None when the fresh pool was not generated (masked, or an earlier check already failed)."""
    if differences or not grid.get("complete") or not state_matches:
        return "exploratory"
    if masked:
        return "masked"
    return "preregistered" if fresh_digest_ok else "exploratory"


def compare_with_record(
    recorded: Mapping[str, Any] | None, fresh: Mapping[str, Any] | None, selection: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Does this pre-registered scoring match the evaluation of record? Compared at the record's printed precision.

    None when there is no record or nothing was scored. Display only: it changes no setting and no label."""
    if recorded is None or fresh is None:
        return None
    boot = fresh.get("bootstrap") or {}
    this_run = {
        "selected_config": selection.get("selected_config"),
        "selected_system": selection.get("selected_system"),
        "selected_ap": round(float(fresh["selected"]["ap"]), 4),
        "baseline_ap": round(float(fresh["baseline"]["ap"]), 4),
        "ap_difference": round(float(boot["primary_ap_difference"]), 4) if boot else None,
        "ci95": [round(float(v), 4) for v in boot["primary_ap_ci95"]] if boot else None,
        "verdict": fresh.get("verdict"),
    }
    record = {
        "selected_config": recorded["selected_config"],
        "selected_system": recorded["selected_system"],
        **{k: recorded["fresh_test"][k] for k in ("selected_ap", "baseline_ap", "ap_difference", "ci95", "verdict")},
    }
    differing = [k for k in record if record[k] != this_run[k]]
    return {
        "record": {"date": recorded["date"], "commit": recorded["commit"], "notebook_blob": recorded["notebook_blob"]},
        "matches": not differing,
        "differing": differing,
        "fields": {k: {"record": record[k], "this_run": this_run[k]} for k in record},
    }


def matches_to_json(matches: Mapping[str, Any]) -> dict[str, Any]:
    """A `match_detections` table as plain JSON (rows in their original order), so the bootstrap can be re-run."""
    return {
        "thresholds": [float(t) for t in matches["thresholds"]],
        "class_names": list(matches["class_names"]),
        "n_images": int(matches["n_images"]),
        "references": {k: [int(x) for x in v] for k, v in matches["references"].items()},
        "rows": [
            {"threshold": float(t), "class": c, "scores": [float(x) for x in s], "hits": [bool(x) for x in h], "image_index": [int(x) for x in o]}
            for (t, c), (s, h, o) in matches["rows"].items()
        ],
    }


def matches_from_json(data: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of `matches_to_json`: a table `paired_bootstrap` accepts."""
    return {
        "thresholds": [float(t) for t in data["thresholds"]],
        "class_names": list(data["class_names"]),
        "n_images": int(data["n_images"]),
        "references": {k: list(v) for k, v in data["references"].items()},
        "rows": {
            (float(r["threshold"]), r["class"]): (list(r["scores"]), list(r["hits"]), list(r["image_index"])) for r in data["rows"]
        },
    }


def per_image_summary(selected: Mapping[str, Any], baseline: Mapping[str, Any], *, iou: float = 0.5) -> list[dict[str, Any]]:
    """Per fresh-test image: reference boxes, and each system's detections and true positives at IoU `iou`.

    Display only; the bootstrap itself is re-run from the saved match tables (`matches_to_json`)."""
    counts = {name: [[0, 0] for _ in range(selected["n_images"])] for name in ("selected", "baseline")}
    for name, table in (("selected", selected), ("baseline", baseline)):
        for (threshold, _cls), (_scores, hits, owners) in table["rows"].items():
            if abs(threshold - iou) > 1e-9:
                continue
            for hit, owner in zip(hits, owners, strict=True):
                counts[name][owner][0] += 1
                counts[name][owner][1] += int(bool(hit))
    key = f"true_positives_iou{iou:g}"
    return [
        {
            "image": i,
            "references": sum(v[i] for v in selected["references"].values()),
            **{name: {"detections": counts[name][i][0], key: counts[name][i][1]} for name in counts},
        }
        for i in range(selected["n_images"])
    ]


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write via a temporary file and os.replace: a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def read_json_or_none(path: str | Path) -> Any:
    """The parsed file, or None when it is missing, empty or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class LookLedger:
    """Counts fresh-test scorings in this runtime, crash-safely.

    A look is recorded only by `record()`, after the report is safely on disk. Whether the pre-registered slot is
    spent is decided from the outputs as well as the ledger (amendment 1): an existing main report file (readable or
    not), any reproduction report, or a ledger entry means every later scoring is a reproduction. Only an absent main
    report with no other evidence re-opens the slot; an interrupted atomic write leaves no main report, so it is not a
    look. A corrupt, empty, negative or non-finite ledger value reads as absent and never crashes a run. Once an
    evaluation of record exists (`RECORDED_EVALUATION`), `plan` never returns the main slot.
    """

    def __init__(self, path: str | Path, outputs_dir: str | Path = "outputs", stem: str = "yolox_x_detection") -> None:
        self.path = Path(path)
        self.outputs = Path(outputs_dir)
        self.stem = stem

    def _load(self) -> dict[str, int]:
        data = read_json_or_none(self.path)
        if not isinstance(data, dict):
            return {}
        return {
            k: int(v)
            for k, v in data.items()
            if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
        }

    def _save(self, counts: Mapping[str, int]) -> None:
        atomic_write_text(self.path, json.dumps(dict(counts)))

    @property
    def main_report(self) -> Path:
        return self.outputs / f"{self.stem}_evaluation_report.json"

    def _highest(self, kind: str) -> int:
        pattern = re.compile(rf"^{re.escape(self.stem)}_evaluation_report_{kind}_(\d+)\.json$")
        found = [
            int(m.group(1))
            for f in (self.outputs.glob("*.json") if self.outputs.is_dir() else [])
            if (m := pattern.match(f.name))
        ]
        return max(found, default=0)

    def plan(self, mode: str, recorded: Mapping[str, Any] | None = None) -> tuple[str, str]:
        """(test_look, file suffix) for a scoring of kind `mode`. Records nothing.

        `recorded` is `RECORDED_EVALUATION`: once an evaluation of record exists, a pre-registered scoring is always a
        reproduction, including the first one in a new runtime, and never writes the main report."""
        if mode not in ("preregistered", "exploratory", "masked"):
            raise ValueError(f"unknown scoring mode {mode!r}")
        counts = self._load()
        if mode == "preregistered" and recorded is not None:
            n = max(self._highest("reproduction"), counts.get("reproductions", 0)) + 1
            return TEST_LOOK_REPRODUCTION, f"_reproduction_{n}"
        if mode == "preregistered":
            # Amendment 1 (F3): an existing main report spends the slot even when unreadable, and so does any
            # reproduction report; only an absent main report with no completed look in the ledger opens it.
            if self.main_report.exists() or self._highest("reproduction") > 0 or counts.get("preregistered_completed", 0) > 0:
                n = max(self._highest("reproduction"), counts.get("reproductions", 0)) + 1
                return TEST_LOOK_REPRODUCTION, f"_reproduction_{n}"
            return TEST_LOOK_PREREGISTERED, ""
        n = max(self._highest(mode), counts.get(mode, 0)) + 1
        return (TEST_LOOK_MASKED if mode == "masked" else TEST_LOOK_EXPLORATORY), f"_{mode}_{n}"

    def record(self, mode: str, suffix: str) -> None:
        """Count a completed scoring; call it only after its report is on disk."""
        counts = self._load()
        if mode == "preregistered":
            if suffix == "":
                counts["preregistered_completed"] = counts.get("preregistered_completed", 0) + 1
            else:
                counts["reproductions"] = max(counts.get("reproductions", 0), int(suffix.rsplit("_", 1)[1]))
        else:
            counts[mode] = max(counts.get(mode, 0), int(suffix.rsplit("_", 1)[1]))
        self._save(counts)

    def completed(self) -> int:
        counts = self._load()
        return counts.get("preregistered_completed", 0) + counts.get("reproductions", 0)
