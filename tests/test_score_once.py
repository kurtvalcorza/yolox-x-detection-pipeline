"""Score-once, grid and reading-rule tests for the 2026-10-18 review fixes (row 117 R117-M1, R117-M2, R117-m2, R117-m6,
R117-m1; shared YOLOX design for rows 116 and 117).

The generated notebook's own Sections 6-9 are executed with a stand-in detector built on the `_StubPipe` of
tests/test_review_fixes.py: no weights and no torch call. The stand-in's average precision is computed by the real
`average_precision` on stand-in detections. None of this is model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib
import itertools
import json
import os
import subprocess
import types
from pathlib import Path

import pytest

from test_review_fixes import NOTEBOOK, PACKAGE, PIPELINE_CLASS, _code, _namespace, _StubPipe

pipeline_module = importlib.import_module(f"{PACKAGE}.pipeline")
samples_module = importlib.import_module(f"{PACKAGE}.samples")
protocol = importlib.import_module(f"{PACKAGE}.protocol")
ROOT = Path(__file__).resolve().parents[1]
STEM = NOTEBOOK.name.replace("_finetune_colab.ipynb", "")
NEEDLES = {
    6: "validation_records = sign_dataset(",
    7: "BASELINE_FINGERPRINT = ",
    8: "grid_rows, grid_models = {}, {}",
    9: "ledger = LookLedger(",
}


class _GridStub(_StubPipe):
    """`_StubPipe` plus a state fingerprint, per-config behaviour and real AP on stand-in detections."""

    fail_epochs: set[int] = set()
    adapted_quality = 1.0  # 1.0: boxes exact; 0.0: the adapted model is no better than the baseline

    def __init__(self, class_names=("a",), adapted=False) -> None:
        super().__init__(class_names, adapted)
        self.trained_with: tuple = ()

    def finetune(self, records, **kwargs):
        if kwargs.get("epochs") in type(self).fail_epochs:
            raise RuntimeError("CUDA out of memory (stand-in)")
        out = super().finetune(records, **kwargs)
        self.trained_with += (kwargs.get("epochs"), kwargs.get("learning_rate"))
        out["history"] = [{"epoch": 1, "total_loss": 2.0}, {"epoch": 2, "total_loss": 1.0}]
        return out

    def state_fingerprint(self):
        return hashlib.sha256(f"{self.adapted}:{self.trained_with}:{self.finetune_calls}".encode()).hexdigest()

    def _detections(self, record):
        good = self.adapted and type(self).adapted_quality > 0.5
        dets = []
        for box, label in zip(record["boxes"], record["labels"], strict=True):
            shift = 0.0 if good else (box[2] - box[0]) * 0.6
            dets.append({"label": label, "score": 0.9 if good else 0.4, "box": [box[0] + shift, box[1], box[2] + shift, box[3]]})
        return dets

    def evaluate(self, records, keep_matches=False, **_):
        predictions = [self._detections(r) for r in records]
        out = pipeline_module.average_precision(predictions, records, self.class_names)
        out["implementation"] = "stand-in"
        if keep_matches:
            out["matches"] = pipeline_module.match_detections(predictions, records, self.class_names)
        return out


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell(notebook: dict, section: int) -> str:
    (source,) = [s for s in _code(notebook) if NEEDLES[section] in s]
    return source


@pytest.fixture
def harness(notebook, tmp_path, monkeypatch):
    _GridStub.fail_epochs, _GridStub.adapted_quality = set(), 1.0
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    ns = _namespace(tmp_path, monkeypatch)
    ns.update({k: v for k, v in vars(protocol).items() if not k.startswith("__")})
    ns[PIPELINE_CLASS] = _GridStub
    ns["paired_bootstrap"] = functools.partial(protocol.paired_bootstrap, n_boot=200)
    # The record-less path; the record tests below set it.
    ns["RECORDED_EVALUATION"] = None
    ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda index=0: "Tesla T4"))
    calls: list[int] = []
    real_sign = samples_module.sign_dataset

    def sign_dataset(n, *, seed=0, **kw):
        calls.append(seed)
        return real_sign(n, seed=seed, **kw)

    ns["sign_dataset"] = sign_dataset
    ns["_seeds"] = calls

    def run(sections=(6, 7, 8, 9), **edits):
        for section in sections:
            source = _cell(notebook, section)
            for old, new in edits.items():
                if old in source:
                    source = source.replace(old, new)
            exec(source, ns)  # noqa: S102 - the notebook's own cells
        return ns

    return run, ns, tmp_path / "outputs"


def _report(tmp_path: Path, suffix: str = "") -> dict:
    return json.loads((tmp_path / f"{STEM}_evaluation_report{suffix}.json").read_text(encoding="utf-8"))


# ---- the pre-registered path, and its reproduction ------------------------------------------------------------------


def test_default_gpu_run_scores_the_fresh_test_once_then_labels_a_rerun_a_reproduction(harness) -> None:
    run, ns, tmp = harness
    run()
    first = _report(tmp)
    assert first["test_look"] == protocol.TEST_LOOK_PREREGISTERED and first["preregistered"] is True
    assert first["settings_differing"] == [] and first["grid"]["complete"] is True
    assert first["fresh_test"]["verdict"] == "improved" and first["fresh_test"]["bootstrap"]["n_images"] == 200
    assert protocol.FRESH_TEST_POOL["seed"] in ns["_seeds"]
    before = (tmp / f"{STEM}_evaluation_report.json").read_bytes()
    run(sections=(9,))
    again = _report(tmp, "_reproduction_1")
    assert again["test_look"] == "reproduction" and again["fresh_test"] is not None
    assert (tmp / f"{STEM}_evaluation_report.json").read_bytes() == before
    assert {p.name for p in (tmp / "grid" / "run_1").iterdir()} == {"C0.json", "C1.json", "C2.json"}


@pytest.mark.parametrize(
    ("edit", "differing"),
    [
        ({"BATCH_SIZE = 2": "BATCH_SIZE = 1"}, "batch_size"),
        ({"\nSEED = 0  #": "\nSEED = 1  #"}, "split_seed"),
        ({"N_IMAGES = 40": "N_IMAGES = 36"}, "n_images"),
        ({"DATASET_SEED = 0": "DATASET_SEED = 2"}, "dataset_seed"),
        ({"FREEZE_BACKBONE = True": "FREEZE_BACKBONE = False"}, "freeze_backbone"),
        ({"HOLDOUT = 0.25": "HOLDOUT = 0.3"}, "holdout"),
    ],
)
def test_a_changed_setting_is_exploratory_and_never_scores_the_fresh_test(harness, edit, differing) -> None:
    run, ns, tmp = harness
    run(**edit)
    report = _report(tmp, "_exploratory_1")
    assert report["test_look"] == protocol.TEST_LOOK_EXPLORATORY and report["fresh_test"] is None
    assert differing in report["settings_differing"] and report["preregistered"] is False
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"]
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()


def test_every_fixed_setting_is_in_the_preregistered_check() -> None:
    for key in ("training_digest", "validation_digest", "fresh_test_digest", "batch_size", "adapt_seed", "freeze_backbone",
                "grid", "run_grid", "eval_threshold", "max_detections", "min_ap_gain", "keep_baseline_gain"):
        changed = dict(protocol.PREREGISTERED_SETTINGS, **{key: "changed"})
        assert protocol.preregistration_differences(changed) == [key]
    assert protocol.preregistration_differences({}) == list(protocol.PREREGISTERED_SETTINGS)


def test_round2_record_is_unchanged_history() -> None:
    """The 2026-10-18 Colab CLI T4 run (docs/execution-evidence/2026-10-18-35dc06f/), values as Sections 8-9 printed them."""
    record = protocol.ROUND2_RECORDED_EVALUATION
    assert record is not None and record["date"] == "2026-10-18"
    assert record["device_class"] == protocol.PREREGISTERED_SETTINGS["device_class"] == "t4"
    assert record["commit"].startswith("35dc06f") and record["notebook_blob"].startswith("a90023b2")
    assert record["selected_config"] == record["selected_system"] == "C1"
    assert record["validation_ap"] == {"baseline": 0.0378, "C0": 0.8510, "C1": 0.9237, "C2": 0.8533}
    assert record["fresh_test"] == {
        "n_images": 200, "n_references": 413, "selected_ap": 0.9165, "baseline_ap": 0.0445,
        "ap_difference": 0.872, "ci95": [0.8555, 0.8887], "verdict": "improved",
    }
    assert (ROOT / record["evidence"]).is_dir()


def test_cpu_default_grid_off_trains_the_recorded_selection_and_is_exploratory(harness) -> None:
    run, ns, tmp = harness
    ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    run()
    report = _report(tmp, "_exploratory_1")
    assert report["fresh_test"] is None and "run_grid" in report["settings_differing"]
    assert report["test_look"] == protocol.TEST_LOOK_EXPLORATORY and report["preregistered"] is False
    # The recorded selection follows the hosted record (35dc06f selected C1): ten epochs at 1e-3.
    assert list(ns["grid_rows"]) == [protocol.RECORDED_SELECTION] == ["C1"]
    assert protocol.ROUND2_RECORDED_EVALUATION["selected_config"] == protocol.RECORDED_SELECTION
    assert ns["grid_rows"]["C1"]["settings"]["epochs"] == 10 and ns["grid_rows"]["C1"]["settings"]["learning_rate"] == 1e-3


def test_a_partial_grid_is_exploratory_and_not_a_look(harness) -> None:
    run, ns, tmp = harness
    _GridStub.fail_epochs = {8}
    run()
    report = _report(tmp, "_exploratory_1")
    assert report["grid"]["complete"] is False and any("C2: failed" in p for p in report["grid"]["problems"])
    assert report["fresh_test"] is None and report["grid_rows"]["C2"]["status"] == "failed"
    assert protocol.LookLedger(tmp / "work" / "look_ledger.json", tmp, STEM).completed() == 0


def test_a_changed_model_state_is_exploratory(harness) -> None:
    run, ns, tmp = harness
    run(sections=(6, 7, 8))
    ns["adapter"].finetune([], epochs=1)
    run(sections=(9,))
    assert _report(tmp, "_exploratory_1")["fresh_test"] is None
    assert "the model in hand is not the selected system" in _report(tmp, "_exploratory_1")["reasons"]


@pytest.mark.parametrize("failure", ["raise", "digest"])
def test_a_failed_generation_or_digest_check_is_a_friendly_refusal_and_not_a_look(harness, capsys, failure) -> None:
    run, ns, tmp = harness
    real_sign, real_digest = ns["sign_dataset"], ns["pool_digest"]
    if failure == "raise":
        ns["sign_dataset"] = lambda n, *, seed=0, **kw: (_ for _ in ()).throw(ValueError("stand-in")) if seed == protocol.FRESH_TEST_POOL["seed"] else real_sign(n, seed=seed, **kw)
    else:
        ns["pool_digest"] = lambda records: "0" * 64 if len(records) == protocol.FRESH_TEST_POOL["n"] else real_digest(records)
    run()
    out = capsys.readouterr().out
    assert "does not match its committed digest" in out and "Traceback" not in out
    report = _report(tmp, "_exploratory_1")
    assert report["fresh_test"] is None and "the fresh pool failed its digest check" in report["reasons"]
    ns["sign_dataset"], ns["pool_digest"] = real_sign, real_digest
    run(sections=(9,))
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED


def test_an_interrupted_report_write_is_not_a_look(harness) -> None:
    run, ns, tmp = harness
    real = ns["atomic_write_json"]

    def interrupted(path, payload):
        if "evaluation_report" in Path(path).name:
            raise KeyboardInterrupt
        return real(path, payload)

    ns["atomic_write_json"] = interrupted
    with pytest.raises(KeyboardInterrupt):
        run()
    assert not list(tmp.glob(f"{STEM}_evaluation_report*"))
    ns["atomic_write_json"] = real
    run(sections=(9,))
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED


@pytest.mark.parametrize("ledger_text", ["", "{not json", "[1, 2]", '{"preregistered_completed": "x"}'])
def test_a_corrupt_or_empty_ledger_never_crashes_or_relabels(harness, ledger_text) -> None:
    run, ns, tmp = harness
    (tmp / "work").mkdir(parents=True)
    (tmp / "work" / "look_ledger.json").write_text(ledger_text, encoding="utf-8")
    run()
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED
    (tmp / "work" / "look_ledger.json").write_text(ledger_text, encoding="utf-8")
    run(sections=(9,))
    assert _report(tmp, "_reproduction_1")["test_look"] == "reproduction"


def test_an_existing_main_report_spends_the_slot_even_when_unreadable(harness) -> None:
    """Amendment 1 (F3): an existing main report, readable or not, spends the pre-registered slot; nothing relabels."""
    run, ns, tmp = harness
    tmp.mkdir()
    (tmp / f"{STEM}_evaluation_report.json").write_text("", encoding="utf-8")
    run()
    assert _report(tmp, "_reproduction_1")["test_look"] == protocol.TEST_LOOK_REPRODUCTION
    assert (tmp / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8") == ""


def test_a_masked_run_never_generates_the_fresh_test(harness, monkeypatch) -> None:
    run, ns, tmp = harness
    monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    run()
    report = _report(tmp, "_masked_1")
    assert report["test_look"] == protocol.TEST_LOOK_MASKED and report["masked"] is True
    assert report["preregistered"] is False and report["preregistered_settings_hold"] is True
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"] and report["fresh_test"] is None


def test_the_keep_baseline_rule_selects_the_baseline_and_reports_same_model(harness) -> None:
    run, ns, tmp = harness
    _GridStub.adapted_quality = 0.0
    run()
    report = _report(tmp)
    assert report["selection"]["selected_system"] == protocol.BASELINE_SYSTEM
    assert report["fresh_test"]["verdict"] == "same model (adaptation not selected)" and report["fresh_test"]["bootstrap"] is None


def test_rerunning_the_finetune_cell_trains_fresh_models(harness) -> None:
    run, ns, tmp = harness
    run(sections=(6, 7, 8))
    first = ns["adapter"]
    run(sections=(8,), **{"FREEZE_BACKBONE = True": "FREEZE_BACKBONE = False"})
    assert ns["adapter"] is not first and first.finetune_calls == 1 and ns["adapter"].finetune_calls == 1
    assert ns["run"]["freeze_backbone"] is False and isinstance(ns["FINETUNE_SECONDS"], float)


# ---- selection, reading rule, tie rule ------------------------------------------------------------------------------


def _rows(**scores):
    return {n: {"status": "complete", "val_ap": ap, "val_ap50": ap50, "fingerprint": n} for n, (ap, ap50) in scores.items()}


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"C0": (0.50, 0.90), "C1": (0.529, 0.95), "C2": (0.52, 0.95)}, "C0"),  # below the 0.03 margin
        ({"C0": (0.50, 0.90), "C1": (0.54, 0.87), "C2": (0.52, 0.95)}, "C0"),  # AP50 more than 0.02 lower
        ({"C0": (0.50, 0.90), "C1": (0.54, 0.89), "C2": (0.52, 0.95)}, "C1"),
        ({"C0": (0.50, 0.90), "C1": (0.54, 0.90), "C2": (0.56, 0.90)}, "C2"),  # both qualify: the larger gain
        ({"C0": (0.50, 0.90), "C1": (0.55, 0.90), "C2": (0.55, 0.90)}, "C2"),  # tie: the smaller config
    ],
)
def test_selection_rule(scores, expected) -> None:
    assert protocol.select_config(_rows(**scores), baseline_val_ap=0.1)["selected_config"] == expected


def test_keep_baseline_rule_margin() -> None:
    rows = _rows(C0=(0.30, 0.9), C1=(0.30, 0.9), C2=(0.30, 0.9))
    assert protocol.select_config(rows, baseline_val_ap=0.28)["selected_system"] == protocol.BASELINE_SYSTEM
    assert protocol.select_config(rows, baseline_val_ap=0.27)["selected_system"] == "C0"


@pytest.mark.parametrize(
    ("ci", "diff", "verdict"),
    [([0.01, 0.2], 0.1, "improved"), ([0.001, 0.05], 0.02, "no detectable difference"), ([-0.2, -0.01], -0.1, "worse"), ([-0.1, 0.1], 0.0, "no detectable difference")],
)
def test_reading_verdict(ci, diff, verdict) -> None:
    assert protocol.reading_verdict({"primary_ap_ci95": ci, "primary_ap_difference": diff}, same_model=False) == verdict


def test_constant_scores_give_one_ap_whatever_the_image_order() -> None:
    """R117-m1: 2 hits and 2 false positives, all at score 0.5, used to give AP50 0.2525-0.505 by image order."""
    records = [{"boxes": [[0, 0, 10, 10]], "labels": ["a"]} for _ in range(4)]
    dets = [[{"label": "a", "score": 0.5, "box": [0, 0, 10, 10]}], [{"label": "a", "score": 0.5, "box": [0, 0, 10, 10]}],
            [{"label": "a", "score": 0.5, "box": [50, 50, 60, 60]}], [{"label": "a", "score": 0.5, "box": [50, 50, 60, 60]}]]
    values = {round(pipeline_module.average_precision([dets[i] for i in order], [records[i] for i in order], ["a"])["ap50"], 9)
              for order in itertools.permutations(range(4))}
    assert len(values) == 1
    result = pipeline_module.average_precision(dets, records, ["a"])
    assert result["tied_score_groups"] == 1 and "order-independent" in result["tie_rule"]


def test_distinct_scores_keep_the_per_detection_curve_and_the_bootstrap_reweights_images() -> None:
    records = [{"boxes": [[0, 0, 10, 10]], "labels": ["a"]} for _ in range(3)]
    dets = [[{"label": "a", "score": 0.9, "box": [0, 0, 10, 10]}], [{"label": "a", "score": 0.8, "box": [40, 40, 50, 50]}],
            [{"label": "a", "score": 0.7, "box": [0, 0, 10, 10]}]]
    result = pipeline_module.average_precision(dets, records, ["a"], iou_thresholds=(0.5,))
    assert result["ap50"] == pytest.approx((1.0 * 34 + (2 / 3) * 33) / 101)  # precision 1 up to recall 1/3, then 2/3
    matches = pipeline_module.match_detections(dets, records, ["a"], iou_thresholds=(0.5,))
    assert pipeline_module.ap_from_matches(matches, [1, 0, 1])["ap50"] == pytest.approx(1.0)
    boot = protocol.paired_bootstrap(matches, matches, n_boot=50)
    assert boot["primary_ap_difference"] == 0.0 and boot["primary_ap_ci95"] == [0.0, 0.0]


# ---- pins, atomic writes, provenance --------------------------------------------------------------------------------


def test_pool_digests_match_the_pins_and_the_pools_are_disjoint() -> None:
    seed0 = samples_module.sign_dataset(40, seed=0)
    train, exposed = pipeline_module.split_records(seed0, holdout=0.25, seed=0)
    val = samples_module.sign_dataset(protocol.VALIDATION_POOL["n"], seed=protocol.VALIDATION_POOL["seed"])
    fresh = samples_module.sign_dataset(protocol.FRESH_TEST_POOL["n"], seed=protocol.FRESH_TEST_POOL["seed"])
    assert protocol.pool_digest(train) == protocol.TRAINING_DIGEST
    assert protocol.pool_digest(exposed) == protocol.EXPOSED_DEV_TEST_DIGEST
    assert protocol.pool_digest(val) == protocol.VALIDATION_DIGEST
    assert protocol.pool_digest(fresh) == protocol.FRESH_TEST_DIGEST
    pools = [seed0, val, fresh, samples_module.sign_dataset(3, seed=99)]
    for a, b in itertools.combinations(pools, 2):
        assert not {protocol.record_digest(r) for r in a} & {protocol.record_digest(r) for r in b}
        assert not {protocol.layout_key(r) for r in a} & {protocol.layout_key(r) for r in b}


def test_atomic_write_replaces_whole_files_only(tmp_path: Path) -> None:
    target = tmp_path / "r.json"
    protocol.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"]
    assert protocol.read_json_or_none(tmp_path / "missing.json") is None


def test_notebook_stamp_names_a_commit_that_contains_the_carried_modules(notebook: dict) -> None:
    generated = notebook["metadata"]["dimer"]["generated_from"]
    revision = generated["revision"]
    for rel in generated["modules"]:
        try:
            committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{rel}"])
        except (OSError, subprocess.CalledProcessError):
            if os.environ.get("CI"):
                pytest.fail("the stamped commit is not in this CI checkout: provenance cannot be checked (fetch-depth 0?)")
            pytest.skip("the stamped commit is not in this clone (shallow checkout)")
        assert committed.decode("utf-8").replace("\r\n", "\n") == (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n"), rel
    assert notebook["metadata"]["dimer"]["notebook_kind"] == "TUTORIAL"


# ---- row 116 peer review, mirrored for row 117 (F1, F2, F3, F4, F7, F10) -----------------------------------------


@pytest.mark.parametrize("device", ["cpu", "Tesla L4", "NVIDIA A100-SXM4-40GB"])
def test_F1_a_non_t4_run_with_the_grid_on_is_exploratory_and_never_generates_the_fresh_test(harness, device) -> None:
    run, ns, tmp = harness
    if device == "cpu":
        ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    else:
        ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda index=0: device))
    run(**{'RUN_GRID = "auto"': 'RUN_GRID = "on"'})
    report = _report(tmp, "_exploratory_1")
    assert report["fresh_test"] is None and report["test_look"] == protocol.TEST_LOOK_EXPLORATORY
    assert "device_class" in report["settings_differing"] and report["device"]["name"] == device
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"]
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()


@pytest.mark.parametrize("ledger_text", ['{"preregistered_completed": NaN}', '{"preregistered_completed": Infinity}', '{"exploratory": -Infinity}'])
def test_F2_a_non_finite_ledger_never_crashes(harness, ledger_text) -> None:
    run, ns, tmp = harness
    (tmp / "work").mkdir(parents=True)
    (tmp / "work" / "look_ledger.json").write_text(ledger_text, encoding="utf-8")
    run()
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED


def test_F3_a_corrupt_report_and_ledger_after_a_look_never_relabel_the_next_look(harness) -> None:
    run, ns, tmp = harness
    run()
    (tmp / f"{STEM}_evaluation_report.json").write_text("", encoding="utf-8")
    (tmp / "work" / "look_ledger.json").write_text("{bad", encoding="utf-8")
    run(sections=(9,))
    assert _report(tmp, "_reproduction_1")["test_look"] == protocol.TEST_LOOK_REPRODUCTION
    assert (tmp / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8") == ""


def test_F4_an_exploratory_rerun_never_overwrites_the_grid_rows_of_an_earlier_run(harness) -> None:
    run, ns, tmp = harness
    run()
    first = json.loads((tmp / "grid" / "run_1" / "C0.json").read_text(encoding="utf-8"))
    run(sections=(8, 9), **{"FREEZE_BACKBONE = True": "FREEZE_BACKBONE = False"})
    assert json.loads((tmp / "grid" / "run_1" / "C0.json").read_text(encoding="utf-8")) == first
    assert json.loads((tmp / "grid" / "run_2" / "C0.json").read_text(encoding="utf-8"))["settings"]["freeze_backbone"] is False


class _Interrupting(_GridStub):
    interrupt_epochs: set = set()

    def finetune(self, records, **kwargs):
        if kwargs.get("epochs") in type(self).interrupt_epochs:
            raise KeyboardInterrupt
        return super().finetune(records, **kwargs)


def test_F10_an_interrupted_section_8_rerun_leaves_nothing_for_section_9_to_score(harness) -> None:
    run, ns, tmp = harness
    ns[PIPELINE_CLASS] = _Interrupting
    _Interrupting.interrupt_epochs = set()
    run(**{"FREEZE_BACKBONE = True": "FREEZE_BACKBONE = False"})
    assert _report(tmp, "_exploratory_1")["fresh_test"] is None
    _Interrupting.interrupt_epochs = {10}
    with pytest.raises(KeyboardInterrupt):
        run(sections=(8,))
    _Interrupting.interrupt_epochs = set()
    assert ns["selection"] is None and ns["adapter"] is None and ns["grid_status"]["complete"] is False
    with pytest.raises(RuntimeError, match="Section 8 did not finish"):
        run(sections=(9,))
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"]


def test_F10_section_9_recomputes_the_grid_check_instead_of_trusting_section_8(harness) -> None:
    run, ns, tmp = harness
    run(sections=(6, 7, 8))
    ns["grid_rows"].pop("C2")  # a stale summary would still say complete
    run(sections=(9,))
    report = _report(tmp, "_exploratory_1")
    assert report["fresh_test"] is None and any("C2" in p for p in report["grid"]["problems"])


def test_F7_a_kept_baseline_is_the_system_shown_and_exported(harness, capsys) -> None:
    run, ns, tmp = harness
    _GridStub.adapted_quality = 0.0
    run()
    assert ns["selected_pipe"] is ns["baseline_pipe"]
    assert "Sections 10-11 show and export the baseline" in capsys.readouterr().out


# ---- with the evaluation of record set (2026-10-18) -----------------------------------------------------------------


def test_with_the_record_the_first_run_in_a_new_runtime_is_a_reproduction_and_never_writes_the_main_files(harness, capsys) -> None:
    run, ns, tmp = harness
    ns["RECORDED_EVALUATION"] = protocol.ROUND2_RECORDED_EVALUATION
    run()
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()
    assert not (tmp / f"{STEM}_fresh_test_per_image.json").exists()
    first = _report(tmp, "_reproduction_1")
    assert first["test_look"] == protocol.TEST_LOOK_REPRODUCTION == "reproduction" and first["preregistered"] is True
    assert first["fresh_test"] is not None and first["recorded_evaluation"] == protocol.ROUND2_RECORDED_EVALUATION
    comparison = first["record_comparison"]
    assert comparison["record"]["commit"].startswith("35dc06f") and comparison["matches"] is False
    assert "selected_ap" in comparison["differing"]  # the stand-in detector is not the record
    out = capsys.readouterr().out
    assert "reproduction of the pre-registered evaluation of 2026-10-18 (commit 35dc06f): differs from the record in" in out
    assert (tmp / f"{STEM}_fresh_test_per_image_reproduction_1.json").is_file()
    run(sections=(9,))
    assert _report(tmp, "_reproduction_2")["test_look"] == "reproduction"
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()


def test_with_the_record_a_changed_setting_is_still_exploratory(harness) -> None:
    run, ns, tmp = harness
    ns["RECORDED_EVALUATION"] = protocol.ROUND2_RECORDED_EVALUATION
    run(**{"BATCH_SIZE = 2": "BATCH_SIZE = 1"})
    report = _report(tmp, "_exploratory_1")
    assert report["test_look"] == protocol.TEST_LOOK_EXPLORATORY and report["fresh_test"] is None
    assert "record_comparison" not in report and not list(tmp.glob(f"{STEM}_evaluation_report_reproduction_*.json"))


def test_ledger_plan_with_a_record_never_returns_the_main_slot(tmp_path) -> None:
    ledger = protocol.LookLedger(tmp_path / "work" / "ledger.json", tmp_path, STEM)
    assert ledger.plan("preregistered") == (protocol.TEST_LOOK_PREREGISTERED, "")
    assert ledger.plan("preregistered", protocol.ROUND2_RECORDED_EVALUATION) == ("reproduction", "_reproduction_1")
    ledger.record("preregistered", "_reproduction_1")
    assert ledger.plan("preregistered", protocol.ROUND2_RECORDED_EVALUATION) == ("reproduction", "_reproduction_2")
    assert ledger.plan("exploratory", protocol.ROUND2_RECORDED_EVALUATION) == (protocol.TEST_LOOK_EXPLORATORY, "_exploratory_1")


def test_compare_with_record_reports_a_match_and_each_differing_field() -> None:
    record = protocol.ROUND2_RECORDED_EVALUATION
    fresh = {
        "selected": {"ap": 0.91651}, "baseline": {"ap": 0.04449}, "verdict": "improved",
        "bootstrap": {"primary_ap_difference": 0.87198, "primary_ap_ci95": [0.85551, 0.88871]},
    }
    selection = {"selected_config": "C1", "selected_system": "C1"}
    same = protocol.compare_with_record(record, fresh, selection)
    assert same["matches"] is True and same["differing"] == []
    other = protocol.compare_with_record(record, dict(fresh, selected={"ap": 0.85}), {"selected_config": "C0", "selected_system": "C0"})
    assert other["matches"] is False and other["differing"] == ["selected_config", "selected_system", "selected_ap"]
    assert protocol.compare_with_record(None, fresh, selection) is None
    assert protocol.compare_with_record(record, None, selection) is None


def test_the_saved_per_image_tables_rerun_the_bootstrap_exactly(harness) -> None:
    run, ns, tmp = harness
    run()
    report = _report(tmp)
    saved = json.loads((tmp / f"{STEM}_fresh_test_per_image.json").read_text(encoding="utf-8"))
    assert report["fresh_test"]["per_image_file"].endswith(f"{STEM}_fresh_test_per_image.json")
    rows = saved["per_image"]
    assert len(rows) == 200 and sum(r["references"] for r in rows) == report["fresh_test"]["selected"]["n_references"]
    assert saved["bootstrap"] == {"n_boot": 200, "seed": protocol.BOOTSTRAP_SEED, "resampling_unit": "image"}
    selected = protocol.matches_from_json(saved["matches"]["selected"])
    baseline = protocol.matches_from_json(saved["matches"]["baseline"])
    again = protocol.paired_bootstrap(selected, baseline, n_boot=200)
    assert again["primary_ap_ci95"] == report["fresh_test"]["bootstrap"]["primary_ap_ci95"]
    assert again["primary_ap_difference"] == report["fresh_test"]["bootstrap"]["primary_ap_difference"]


# ---- amendment 2 (round 3): deterministic training and fresh test 3 ------------------------------------------------


def test_amendment2_fresh_test_3_is_pinned_and_the_round2_pool_is_kept_as_history() -> None:
    assert protocol.FRESH_TEST_POOL == {"n": 200, "seed": 1172}
    assert protocol.FRESH_TEST_DIGEST == "098fa0dc920a166ff2ec3393bd3e175aa88e3a864525bb5862827a39e5ba7b36"
    assert protocol.pool_digest(samples_module.sign_dataset(200, seed=1172)) == protocol.FRESH_TEST_DIGEST
    assert protocol.ROUND2_FRESH_TEST_POOL == {"n": 200, "seed": 1170}
    assert protocol.ROUND2_FRESH_TEST_DIGEST == "1f5ae3d9561b81b12e2e3122231ef91a75d6659bf0cf8734d82c9dba4f52a2d7"
    assert protocol.PREREGISTERED_SETTINGS["fresh_test_digest"] == protocol.FRESH_TEST_DIGEST


def test_amendment2_no_record_yet_and_the_round2_record_is_unchanged() -> None:
    assert protocol.RECORDED_EVALUATION is None
    assert protocol.ROUND2_RECORDED_EVALUATION["commit"].startswith("35dc06f")
    assert protocol.ROUND2_RECORDED_EVALUATION["fresh_test"]["ap_difference"] == 0.872
    assert protocol.RECORDED_SELECTION == "C1"  # the CPU default stays the round-2 selection until the new record


def test_amendment2_determinism_is_a_preregistered_setting_and_is_set_before_training() -> None:
    import inspect

    import torch

    assert protocol.PREREGISTERED_SETTINGS["determinism"] == pipeline_module.DETERMINISM
    assert pipeline_module.DETERMINISM["warn_only"] is False
    source = inspect.getsource(pipeline_module.YoloxXDetectionPipeline.finetune)
    assert source.index("enable_determinism(seed)") < source.index("torch.optim.SGD")
    before = torch.are_deterministic_algorithms_enabled()
    try:
        state = pipeline_module.enable_determinism(0)
        assert state == pipeline_module.DETERMINISM == pipeline_module.determinism_state()
        assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
    finally:
        torch.use_deterministic_algorithms(before)


def test_amendment2_first_fresh_test_3_scoring_is_the_preregistered_evaluation(harness, capsys) -> None:
    run, ns, tmp = harness
    run()
    report = _report(tmp)
    assert report["test_look"] == protocol.TEST_LOOK_PREREGISTERED == "pre-registered evaluation (this runtime)"
    assert report["fresh_test"]["pool"] == {"n": 200, "seed": 1172} and 1172 in ns["_seeds"] and 1170 not in ns["_seeds"]
    assert report["determinism"] == pipeline_module.DETERMINISM and report["device"]["class"] == "t4"
    assert report["grid_hash"] == protocol.grid_hash(ns["grid_rows"], ns["selection"])
    assert f"grid hash (two deterministic runs must print the same value): {report['grid_hash']}" in capsys.readouterr().out


def test_amendment2_masked_check_never_generates_fresh_test_3_and_prints_the_same_grid_hash(harness, monkeypatch, capsys) -> None:
    run, ns, tmp = harness
    monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    run()
    first = _report(tmp, "_masked_1")
    run(sections=(8, 9))
    second = _report(tmp, "_masked_2")
    assert first["fresh_test"] is None and 1172 not in ns["_seeds"]
    assert first["grid_hash"] == second["grid_hash"] is not None


def test_amendment2_changed_determinism_flags_are_exploratory(harness, monkeypatch) -> None:
    run, ns, tmp = harness
    monkeypatch.setitem(ns, "enable_determinism", lambda seed: dict(pipeline_module.DETERMINISM, warn_only=True))
    run()
    report = _report(tmp, "_exploratory_1")
    assert "determinism" in report["settings_differing"] and report["fresh_test"] is None
