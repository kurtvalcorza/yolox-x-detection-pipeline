# Tutorials

## Registry

| Notebook | Profile | Mode | Spec | Standalone | Workflow | Runtime | Sample | BYOD | Status |
|---|---|---|---|---|---|---|---|---|---|
| [`yolox_x_detection_finetune_colab.ipynb`](yolox_x_detection_finetune_colab.ipynb) | `E2E` | `GUIDED` | DIMER Notebook Specification 2.5 (standalone, §4; isolated `uv` environment, no runtime restart) | yes | COCO detection, then validate → split → baseline → bounded SimOTA fine-tune → evaluate → new-data inference → export → fresh reload | CPU (GPU used automatically when present) | automatic, drawn in code | optional, two branches, both off by default; image by `BYOD_IMAGE_PATH` or upload, dataset by `BYOD_DATASET_DIR` (`annotations.json` + images) or upload, through adaptation, reload and a result JSON | Candidate (amendment-2 evaluation of record: fresh Colab T4, 2026-10-18, blob `6f13a083`, deterministic GPU; C1 selected, fresh test 3 +0.8775, *improved*; round-2 record, blob `a90023b2`: +0.872, history) |

Candidate status and the gates that remain open are recorded in
[`../docs/release-verification.md`](../docs/release-verification.md); clean-room execution on Kaggle Tesla T4 GPU is recorded in
[`../docs/release-verification.md`](../docs/release-verification.md).

## What the notebook does

It is **standalone**: it carries the repository's ten-module package verbatim, the pinned model
identity and the file manifest, and the exact runtime pins, so it keeps working after export even if
this repository changes. There is no clone, no `pip install -e .`, no DIMER worker or service, and no
credential. Its only external dependency is `github.com`, to download the pinned `yolox_x.pth` release
asset whose SHA-256 the notebook checks before anything opens it.

Because the profile is `E2E`, the default `Run all` path **actually adapts the model** — NOTEBOOK_SPEC
2.0 RUN7 and FT2 make a real bounded fine-tune mandatory rather than optional, so nothing is hidden
behind a default-off flag:

1. Install the pinned runtime and report versions.
2. Carry the package, module by module in dependency order.
3. Pin, stage and digest-verify the checkpoint.
4. Run COCO detection on a drawn scene and score it per object — including one object the pretrained
   model gets wrong.
5. Probe channel order (BGR vs RGB) and degenerate inputs.
6. Build and validate a 40-image labelled dataset over a three-class vocabulary that is not in COCO.
7. Split it, re-head the detector, and measure a baseline **before** adapting.
8. Run the bounded SimOTA fine-tune: the pre-registered three-config grid on a GPU (chosen on a 40-image validation
   pool), or the recorded selection C0 on a CPU (an exploratory run).
9. Re-check every pre-registered setting, then score a 200-image fresh test against the baseline with a paired
   image bootstrap; the 10-image exposed development test is shown for continuity and decides nothing.
10. Detect on images from an unseen seed.
11. Export the artifact (`yolox-x-detection-adapter.safetensors` state dict plus a `yolox-x-detection-adapter.json` sidecar with its SHA-256; no pickle), reload it from disk, and assert the reloaded model scores identically.
12. Write every result as JSON with provenance.
13. Optional BYOD: your own image, or your own labelled dataset through the same local stages.

Expect roughly a few minutes on a hosted CPU runtime; on the reference machine the committed notebook
executed in about 160 s with the checkpoint pre-staged and the pins already present, of which the fine-tune
is about 130 s. On a Kaggle T4 the previous version needed a restart after its install cell (154.5 s), then ran in 66.0 s; the regenerated notebook (2026-10-05 review fixes) installs into an isolated environment instead and ran one pass on a fresh Colab T4 on 2026-10-10 in 150.1 s, with 51 s of environment setup (blob `af87b360649d`, commit `d2b0bb2`; see `../docs/release-verification.md`). The pre-registered evaluation of record (blob `a90023b2`, commit `35dc06f`, fresh Colab T4, 2026-10-18) ran the three-config grid and the fresh test in 306.7 s, 104.5 s of it fine-tuning: C1 selected, fresh-test AP 0.9165 against the baseline's 0.0445, CI [0.8555, 0.8887], *improved*. The amendment-2 evaluation of record (blob `6f13a083`, commit `057d2d8`, fresh Colab T4, 2026-10-18, deterministic GPU training) ran in 290.3 s, 98.8 s of it fine-tuning (49 s environment setup): C1 selected, fresh test 3 AP 0.914 against the baseline's 0.0365, CI [0.8603, 0.8944], *improved*; reproduced exactly on 2026-10-18 (blob `ed433995`, 310.9 s).

## Reading the numbers

The notebook reports real measurements on **synthetic** data. Both the COCO demonstration scene and the
adaptation dataset are drawn in code with Pillow — flat colours, hard edges, no lighting, no occlusion.
An AP50 of 1.0 on the exposed development test says the plumbing works and the task is easy; it is not
evidence about photographs, and AP@[.50:.95] is the more honest summary. Even the fresh test is the same
renderer drawing the same shapes, so it measures in-distribution generalisation on drawn signs. Upstream's
published 51.5 COCO mAP for YOLOX-X is neither reproduced nor checked.

## Regenerating

The notebook is generated from repository sources and must never be edited by hand:

```bash
python tools/build_notebook.py            # rewrite tutorials/<notebook>
python tools/build_notebook.py --check    # fail if it has drifted from the package (PAR3)
```

`tests/test_notebook_parity.py` enforces the same thing in CI.
