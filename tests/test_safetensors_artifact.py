"""The exported adapter is a safetensors state dict plus a plain-JSON sidecar, never a pickle.

Covers the round trip (identical tensors and identical predictions), the digest check that runs before
the weights are opened, the refusal of `.pt` paths in both directions, and the sidecar being pure JSON.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from PIL import Image  # noqa: E402

from yolox_x_detection_pipeline import pipeline as pipeline_module  # noqa: E402
from yolox_x_detection_pipeline.pipeline import (  # noqa: E402
    ARTIFACT_FILE,
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    YoloxXDetectionPipeline,
)

CLASSES = ("stop-sign", "yield-sign", "speed-limit-sign")


@pytest.fixture(scope="module")
def adapted_pipe() -> YoloxXDetectionPipeline:
    torch.manual_seed(0)
    model = YoloxXDetectionPipeline._build(len(CLASSES)).eval()
    return YoloxXDetectionPipeline(
        model=model,
        device="cpu",
        class_names=CLASSES,
        source="test",
        base_state_digest="0" * 64,
        adapted=True,
    )


@pytest.fixture(scope="module")
def exported(adapted_pipe, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifact")
    descriptor = adapted_pipe.save_artifact(out / ARTIFACT_FILE, notes="unit test")
    return out, descriptor


def _copy_artifact(src_dir, dst_dir):
    for name in (ARTIFACT_FILE, ARTIFACT_FILE.replace(".safetensors", ".json")):
        (dst_dir / name).write_bytes((src_dir / name).read_bytes())
    return dst_dir / ARTIFACT_FILE


def test_artifact_file_is_safetensors() -> None:
    assert ARTIFACT_FILE.endswith(".safetensors")
    assert ARTIFACT_FORMAT.endswith("/2")


def test_save_writes_safetensors_and_json_sidecar_and_no_pickle(exported) -> None:
    out, descriptor = exported
    names = sorted(p.name for p in out.iterdir())
    sidecar_name = ARTIFACT_FILE.replace(".safetensors", ".json")
    assert names == sorted([ARTIFACT_FILE, sidecar_name])
    assert not any(p.suffix in {".pt", ".pth", ".bin", ".ckpt", ".partial"} for p in out.iterdir())
    # A safetensors file starts with an 8-byte little-endian header length followed by a JSON header;
    # a torch.save zip archive starts with "PK" and a raw pickle with 0x80.
    raw = (out / ARTIFACT_FILE).read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_len])
    assert header["__metadata__"]["format"] == ARTIFACT_FORMAT
    assert raw[:2] != b"PK" and raw[0] != 0x80
    assert descriptor["path"].endswith(ARTIFACT_FILE)
    assert descriptor["sidecar"].endswith(sidecar_name)


def test_sidecar_is_pure_json_with_the_weights_digest(exported) -> None:
    out, descriptor = exported
    sidecar_path = out / ARTIFACT_FILE.replace(".safetensors", ".json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))

    def plain(value) -> bool:
        if isinstance(value, dict):
            return all(isinstance(k, str) and plain(v) for k, v in value.items())
        if isinstance(value, list):
            return all(plain(v) for v in value)
        return value is None or isinstance(value, (str, int, float, bool))

    assert plain(payload)
    # Round-tripping through json gives the same object: nothing in it needed a custom encoder.
    assert json.loads(json.dumps(payload)) == payload
    assert payload["format"] == ARTIFACT_FORMAT
    assert payload["model_id"] == MODEL_ID and payload["model_revision"] == MODEL_REVISION
    assert payload["model_key"] == MODEL_KEY
    assert payload["class_names"] == list(CLASSES)
    assert payload["weights_file"] == ARTIFACT_FILE
    assert payload["weights_sha256"] == descriptor["sha256"]
    assert payload["weights_sha256"] == pipeline_module._sha256(out / ARTIFACT_FILE)
    assert "state_dict" not in payload


def test_round_trip_gives_identical_tensors_and_predictions(adapted_pipe, exported) -> None:
    out, _ = exported
    reloaded = YoloxXDetectionPipeline.load_artifact(out / ARTIFACT_FILE, device="cpu")
    assert reloaded.class_names == CLASSES and reloaded.adapted
    before, after = adapted_pipe.model.state_dict(), reloaded.model.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert before[name].dtype == after[name].dtype, name
        assert torch.equal(before[name].cpu(), after[name].cpu()), name

    generator = torch.Generator().manual_seed(1)
    batch = torch.rand(1, 3, 128, 128, generator=generator)
    with torch.no_grad():
        assert torch.equal(adapted_pipe.model(batch), reloaded.model(batch))
    image = Image.new("RGB", (96, 80), (120, 40, 200))
    assert adapted_pipe.detect(image, threshold=0.01) == reloaded.detect(image, threshold=0.01)


def test_digest_mismatch_is_refused_before_the_weights_are_opened(exported, tmp_path, monkeypatch) -> None:
    out, _ = exported
    weights = _copy_artifact(out, tmp_path)
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))

    import safetensors.torch

    def never(*_args, **_kwargs):
        raise AssertionError("safetensors.torch.load_file reached despite a digest mismatch")

    monkeypatch.setattr(safetensors.torch, "load_file", never)
    monkeypatch.setattr(YoloxXDetectionPipeline, "_build", staticmethod(never))
    with pytest.raises(ValueError, match="digest mismatch"):
        YoloxXDetectionPipeline.load_artifact(weights)


def test_missing_or_digestless_sidecar_is_refused(exported, tmp_path) -> None:
    out, _ = exported
    weights = _copy_artifact(out, tmp_path)
    sidecar = weights.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.pop("weights_sha256")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="weights_sha256"):
        YoloxXDetectionPipeline.load_artifact(weights)
    sidecar.unlink()
    with pytest.raises(ValueError, match="sidecar"):
        YoloxXDetectionPipeline.load_artifact(weights)


def test_pickle_paths_are_refused_on_save_and_on_load(adapted_pipe, tmp_path) -> None:
    with pytest.raises(ValueError, match="safetensors"):
        adapted_pipe.save_artifact(tmp_path / "adapter.pt")
    assert not any(tmp_path.iterdir())
    legacy = tmp_path / "legacy-adapter.pt"
    torch.save({"format": "dimer-yolox-detection-adapter/1", "state_dict": {}}, legacy)
    with pytest.raises(ValueError, match="pickle"):
        YoloxXDetectionPipeline.load_artifact(legacy)


def test_header_is_canonical_with_sorted_keys(exported) -> None:
    out, _ = exported
    raw = (out / ARTIFACT_FILE).read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    assert header_len % 8 == 0
    text = raw[8 : 8 + header_len].decode("utf-8").rstrip(" ")
    header = json.loads(text)
    assert text == json.dumps(header, sort_keys=True, separators=(",", ":"))
    assert list(header["__metadata__"]) == sorted(header["__metadata__"])
    assert header["__metadata__"] == {"format": ARTIFACT_FORMAT, "model_key": MODEL_KEY}


def test_saves_are_byte_identical_within_and_across_processes(adapted_pipe, exported, tmp_path) -> None:
    """safetensors 0.8 orders metadata keys per process, so determinism is also checked in a fresh interpreter."""
    import subprocess
    import sys
    from pathlib import Path

    out, descriptor = exported
    again = adapted_pipe.save_artifact(tmp_path / "in-process" / ARTIFACT_FILE)
    assert again["sha256"] == descriptor["sha256"]

    state_path = tmp_path / "state.safetensors"
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in adapted_pipe.model.state_dict().items()}, str(state_path))
    script = (
        "import sys, torch\n"
        "from safetensors.torch import load_file\n"
        "from yolox_x_detection_pipeline.pipeline import YoloxXDetectionPipeline as P\n"
        "model = P._build(3).eval()\n"
        "model.load_state_dict(load_file(sys.argv[1]), strict=True)\n"
        "pipe = P(model=model, device='cpu', class_names=('stop-sign', 'yield-sign', 'speed-limit-sign'),\n"
        "         source='test', base_state_digest='0' * 64, adapted=True)\n"
        "print(pipe.save_artifact(sys.argv[2], notes='unit test')['sha256'])\n"
    )
    src = str(Path(pipeline_module.__file__).resolve().parents[1])
    env = {**__import__("os").environ, "PYTHONPATH": src, "CUDA_VISIBLE_DEVICES": "-1"}
    digests = set()
    for index in range(1):
        target = tmp_path / f"proc{index}" / ARTIFACT_FILE
        result = subprocess.run(
            [sys.executable, "-c", script, str(state_path), str(target)],
            capture_output=True, text=True, env=env, check=True,
        )
        digests.add(result.stdout.strip())
        assert target.read_bytes() == (out / ARTIFACT_FILE).read_bytes()
    assert digests == {descriptor["sha256"]}


def test_weights_file_has_no_pickle_or_zip_framing(exported) -> None:
    """The file is exactly length prefix + JSON header + contiguous tensor bytes; nothing else fits in it."""
    out, _ = exported
    raw = (out / ARTIFACT_FILE).read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_len])
    assert raw[8:9] == b"{"
    spans = sorted(tuple(v["data_offsets"]) for k, v in header.items() if k != "__metadata__")
    assert spans[0][0] == 0
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert 8 + header_len + spans[-1][1] == len(raw)
    # Neither a pickle protocol marker nor a zip local-file header can open any part of the file.
    assert not raw.startswith((b"\x80", b"PK\x03\x04"))
    assert b"PK\x03\x04" not in raw[: 8 + header_len]
    assert not any(marker in raw[: 8 + header_len] for marker in (b"\x80\x02", b"\x80\x04", b"\x80\x05"))
