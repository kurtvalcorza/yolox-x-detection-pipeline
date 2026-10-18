# Release verification

This file is the release gate for `tutorials/yolox_x_detection_finetune_colab.ipynb`. Static
validation (`tools/validate_release_assets.py`), the unit suite and CI are **not** clean-runtime
execution evidence; only an execution recorded under "Recorded executions" is.

## Gates

| # | Gate | State |
|---|---|---|
| 1 | Package unit suite passes offline, without the checkpoint | **met** — 128 tests |
| 2 | Vendored upstream code matches the pinned revision plus the declared edits | **met** — `tools/vendor_upstream.py --check` |
| 3 | Notebook matches the package byte-for-byte | **met** — `tools/build_notebook.py --check` (PAR3) |
| 4 | Static release-asset validation passes | **met** — `tools/validate_release_assets.py` |
| 5 | Notebook executes top-to-bottom in a fresh **local** kernel | **met** — pre-flight only, recorded below |
| 6 | Notebook executes top-to-bottom in a **supported hosted runtime** from a cold start | **open** — the recorded Kaggle T4 run (2026-09-15) needed one manual restart after the old install cell, so it is workflow history, not one-pass evidence (2026-10-05 review, YXX-M1; state corrected 2026-10-10). The 2026-10-05 note that this gate needs re-qualification with `restarted: false` is answered by the one-pass Colab T4 run of the regenerated notebook recorded below (2026-10-10, blob `af87b360649d`, commit `d2b0bb2`, no restart); the gate stays open until a reviewer/integrator accepts that record. |
| 7 | Weights uploaded to the DIMER Model Repository | **open** — owner action |

Gate 6 is what separates **Candidate** from **Release-grade**. The local run in gate 5 pre-staged the
checkpoint and used a venv that already held the pins; it therefore exercises the notebook's logic but
not its install cell, its 793 MB download, or a cold hosted runtime.

## Relationship to the YOLOX-S row

This repository is the large variant of a two-row pair. `kurtvalcorza/yolox-detection-pipeline` is
YOLOX-S, and the two share everything except `MODEL_DEPTH` / `MODEL_WIDTH`, the checkpoint asset and
the numbers below — same vendored upstream code at the same commit, same package, same notebook
structure, same sample data.

Two consequences worth stating plainly:

- **Every measurement below was taken on YOLOX-X.** None is carried over.
- **The fine-tuning grid was not re-run here.** `DEFAULT_EPOCHS = 6`, `DEFAULT_LEARNING_RATE = 1e-3`
  and `freeze_backbone=True` come from the grid measured on the S row, and this repository verified
  only that the chosen configuration works on X (it does: AP50 0.0384 → 1.0000). In particular the
  **unfrozen-collapse warning in the model card is the S row's measurement, not an X measurement** —
  a full fine-tune of X at this learning rate was never run, because it would be roughly an
  order of magnitude more CPU time and the S result is reason enough not to recommend it. Treat it as
  a warning inherited from a sibling, which is what it is.
- **Disclosure (2026-10-18, review R117-M1).** The S grid chose these defaults on **held-out** AP50 of
  `sign_dataset(24, seed=0)` split 18 / 6. `samples.py` is byte-identical in the two repositories and the
  generator is prefix-stable, so four of this repository's ten held-out images were that grid's selection
  images and three were its training images, and the "verified on X" check above was a further look at
  the same split. That 10-image split is now the **exposed development test**, reported for continuity
  only. Selection runs on a separate validation pool under
  [`yolox-x-preregistration.md`](yolox-x-preregistration.md), and only a fresh test is the measurement.

## Smoke evidence

Measured 2026-09-15 on the reference machine (Intel Core Ultra 9 275HX, Windows, Python 3.12.10,
`torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `numpy 2.5.3`, `pillow 11.3.0`) with
`CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`, float32.

### Load and verify

| Step | Result |
|---|---|
| `verify_snapshot` | 0.45 s, 1 file, 793,463,373 bytes |
| `torch.load(weights_only=True)` | succeeds; top-level keys `amp`, `model`, `optimizer`, `start_epoch` |
| `load_state_dict(strict=True)` | all 894 keys matched, none missing or unexpected |
| `from_pretrained` | 2.76 s (4.46 s on a cold file cache), 99,071,455 parameters (upstream README: 99.1 M) |

### COCO detection on the drawn tutorial scene

`detect` at the demo thresholds 0.3 / 0.3 → 4 detections in 0.42 s (0.42 s on the second call):

| Label | Score | Box |
|---|---|---|
| stop sign | 0.964 | 55, 135, 185, 267 |
| clock | 0.946 | 460, 131, 602, 271 |
| traffic light | 0.926 | 299, 120, 357, 253 |
| bench | 0.656 | 440, 499, 614, 584 |

`evaluation_report` against the drawn references, matching same-label only:

| Reference | `box_iou` | Matched score |
|---|---|---|
| stop sign | 0.974 | 0.964 |
| traffic light | 0.943 | 0.926 |
| clock | 0.975 | 0.946 |
| bench | 0.918 | 0.656 |
| **sports ball** | **0.000** | — (no `sports ball` detection at any threshold) |

4/5 references matched at IoU ≥ 0.5. At the evaluation thresholds 0.01 / 0.65 the same scene yields 5
detections.

**Where X differs from S on the same scene.** X localises the three rigid objects more tightly (IoU
0.974 / 0.943 / 0.975 against S's 0.946 / 0.914 / 0.946) and scores the bench higher (0.656 vs 0.545)
though with a looser box (0.918 vs 0.959). More interestingly, X is **cleaner about the object it
misses**: S proposes `kite` 0.369 and a spurious second `clock` 0.309 on the drawn football, while X
proposes nothing there at all above the threshold. Both miss the ball; only the smaller model invents
a replacement for it.

### Degenerate inputs

| Input | Demo pair 0.3/0.3 | Evaluation pair 0.01/0.65 |
|---|---|---|
| Blank white 640×640 | 0 detections | 0 detections |
| Uniform noise (NumPy default generator, seed 0) | 0 detections | 2 detections |

X returns two low-scoring boxes on noise at the lenient thresholds where S returns one. Neither
survives the demo thresholds.

### Channel order

The same letterboxed tensor, fed as RGB instead of upstream's BGR:

| | Detections | Stop sign | Clock | Traffic light | Bench |
|---|---|---|---|---|---|
| BGR (upstream) | 4 | 0.964 | 0.946 | 0.926 | 0.656 |
| RGB (the mistake) | 3 | 0.965 | 0.946 | 0.927 | **absent** |

**This is a weaker effect than on the S row and the difference is worth knowing.** On S, feeding RGB
changed four of six detections and moved the stop sign from 0.962 to 0.839. On X the three rigid
objects are essentially unmoved — the larger model is robust to the channel swap on them — and only
the bench is lost. Getting the channel order wrong on X is therefore *harder to notice*, not less
wrong: the failure is silent unless you happen to be looking at the object that drops.

### End-to-end adaptation

`sign_dataset(40, seed=0)` → 40 images, 82 boxes (`stop-sign` 21, `yield-sign` 34,
`speed-limit-sign` 27), no validation findings. `split_records(holdout=0.25, seed=0)` → 30 train /
10 held out. Re-heading onto `SIGN_CLASSES` took 1.47 s and reinitialised exactly six tensors:
`head.cls_preds.{0,1,2}.{weight,bias}`.

`finetune(epochs=6, batch_size=2, learning_rate=1e-3, seed=0, freeze_backbone=True)` — 130.0 s,
15 steps per epoch, 11,793,304 of 98,997,304 parameters trainable:

| Epoch | total | iou | conf | cls | l1 | num_fg |
|---|---|---|---|---|---|---|
| 1 | 5.69562 | 1.76349 | 2.05443 | 1.87770 | 0.0 | 7.73 |
| 2 | 3.54279 | 1.77989 | 1.02156 | 0.74134 | 0.0 | 7.65 |
| 3 | 2.47203 | 1.35271 | 0.57005 | 0.54926 | 0.0 | 7.98 |
| 4 | 2.21129 | 1.28636 | 0.43042 | 0.49451 | 0.0 | 8.02 |
| 5 | 2.13837 | 1.27499 | 0.38935 | 0.47403 | 0.0 | 8.14 |
| 6 | 1.61572 | 0.94787 | 0.32128 | 0.34658 | 0.0 | 8.65 |

`l1_loss` is 0.0 throughout by design: upstream enables the L1 box term only for the last 15 of 300
epochs, and this bounded run leaves `use_l1` off.

Held-out scores (10 images, evaluation thresholds, COCO 100-detection cap):

| Metric | Baseline | Adapted | Change |
|---|---|---|---|
| AP@[.50:.95] | 0.0384 | **0.7906** | +0.7522 |
| AP50 | 0.0384 | **1.0000** | +0.9616 |
| AP50 `stop-sign` | 0.0 | 1.0 | |
| AP50 `yield-sign` | 0.0 | 1.0 | |
| AP50 `speed-limit-sign` | 0.115 | 1.0 | |

**X's pre-adaptation baseline is much lower than S's** (AP50 0.0384 against 0.2222). Same explanation,
opposite sign: the baseline measures what a randomly initialised classification head does on top of
COCO-trained box and objectness heads, and X's wider head has proportionally more random weight to
overcome, so less accidental signal survives. It makes X's improvement look larger while its *final*
AP@[.50:.95] is slightly lower than S's (0.7906 against 0.8110) — which is the number to compare, and
on this synthetic task the eleven-times-larger model is not better.

New-data inference on three images from an unseen seed (99), demo confidence 0.3 / NMS 0.45:

| Image | Truth | Detections | Same-label `box_iou` |
|---|---|---|---|
| 0 | yield-sign ×2 | yield-sign 0.961, yield-sign 0.947 | 0.833, 0.819 |
| 1 | speed-limit-sign, yield-sign | yield-sign 0.969, speed-limit-sign 0.959 | 0.875, 0.852 |
| 2 | stop-sign | stop-sign 0.978 | 0.887 |

All five objects found with the correct label and no spurious boxes. X is more confident than S on
these (0.947–0.978 against 0.887–0.917) and slightly looser in its boxes (IoU 0.819–0.887 against
0.886–0.949) — a reminder that confidence and localisation are separate things.

### Artifact

`save_artifact` → 396,692,619 bytes over 894 tensors in 0.60 s, SHA-256 `11a91b35022d71b6…`.
`load_artifact` in a fresh object (0.50 s) reproduced AP 0.7906 / AP50 1.0000 **exactly**. An artifact
with a tampered `format` tag was refused with `ValueError`, and so was a YOLOX-S artifact (see below).

### Helper sanity

`average_precision` returns 1.0 for exact predictions and 0.0 for no predictions, penalises a false
positive ranked above a hit (AP50 0.5) and correctly does **not** penalise one ranked below all hits,
which is COCO's own semantics.

## A defect this row found

YOLOX-S and YOLOX-X are both assets of the upstream `0.1.1rc0` release, so they share `MODEL_ID` and
`MODEL_REVISION`. `load_artifact` checked only those, which meant **a YOLOX-S adapter passed the
identity check in this package** and then failed inside `load_state_dict` with a tensor-shape error —
a confusing way to learn you loaded the wrong file. `MODEL_KEY` distinguishes the variants and was
already written into every artifact, so it joined the check; the refusal now reads

```
ValueError: artifact was built on the 'yolox-s' variant, package pins 'yolox-x'
```

Fixed in both repositories, with a regression test in each. It was only visible because two variants
existed at once.

## Findings kept, not fixed

These are recorded behaviour, not defects, and should survive future edits:

1. **The drawn football is never detected as a `sports ball`** at any threshold. Unlike the S row, X
   proposes nothing else on that box either. The evaluation report scores it 0.000 rather than
   dropping the reference.
2. **The clock face needs numerals.** Probing on the S row found that a plain white disc with two
   hands is not read as a `clock` at all, and made the *ball* become the clock. The sample's numbered
   face is the result; X reads it at 0.946 with IoU 0.975.
3. **`bench` is the weakest of the four found objects** (0.656) and has the loosest box (IoU 0.918).
4. **Uniform noise yields two boxes** at the evaluation thresholds, none at the demo thresholds.
5. **The channel-order error is quieter on X than on S** (see above). Recorded because a quieter
   failure is a more dangerous one.

## Traps recorded for the fleet

- Upstream `postprocess` writes the xyxy conversion back into its argument **in place**, so a tensor
  produced under `torch.inference_mode()` raises `RuntimeError`. Upstream's own `tools/demo.py` uses
  `no_grad` for the same reason; this package does too, and says why at the call site.
- Omitting upstream's `head.initialize_biases(1e-2)` makes a re-headed fine-tune **collapse**: the
  objectness term saturates within a few steps and the model predicts one class at score 1.0 at every
  anchor. Found on the S row.
- Sibling variants that share an upstream release share a model id and revision; an artifact identity
  check needs the model key too (see above).
- The vendored `yolo_head` emits a `FutureWarning` for `torch.cuda.amp.autocast` under torch 2.14.
  It is upstream's code and is left verbatim rather than fixed, so the vendoring check keeps passing.
- A 793 MB release asset is slow over a single HTTP stream from GitHub; parallel range requests
  fetched it in about a minute where one stream was running at roughly 30 kB/s.

## Recorded executions

| Date | Runtime | Notebook | Blob | Result |
|---|---|---|---|---|
| 2026-09-15 | Local Windows CPU kernel (Python 3.12.10, `torch 2.14.0+cu130`, GPU hidden), pre-staged checkpoint, pins already present | `yolox_x_detection_finetune_colab.ipynb` | blob `a4642713bbb3126b1a1c90e55c64a4ef745a4e9a` at commit `e0631e0` | **PASS** - 22/22 code cells, 163.4 s; reproduced the smoke figures exactly (baseline AP 0.0384 / AP50 0.0384 -> adapted AP 0.7906 / AP50 1.0000, loss 5.69562 -> 1.61572, 11,793,304 of 98,997,304 parameters trained, COCO IoU 0.974/0.943/0.975/0.000/0.918, artifact 396,692,619 bytes over 894 tensors, fresh reload identical, all five new-data objects correctly labelled). Uniform noise at the evaluation thresholds gave `potted plant` 0.076 and `vase` 0.057. |
| 2026-09-15 | Kaggle GPU kernel (`kurtvalcorza/dimer-nb2-yolox-x-detection-finetune` v2), fresh container assigned **Tesla T4** (driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed in fresh IPython interpreter via `nbclient 0.10.2` | `yolox_x_detection_finetune_colab.ipynb` | blob `99c57d7bc1431e42bbb3a7328e273348027972c8` at commit `1943fbb` | **PASS after a manual restart** (not one-pass evidence; see gate 6) — 22/22 code cells, 220.6 s wall (pass 1: 154.5 s install + restart; pass 2: 66.0 s full run); reproduced smoke figures and geometry sanity: baseline AP 0.0384 / AP50 0.0384 -> adapted AP 0.8974 / AP50 1.0000, loss 5.6956 -> 1.6057, 11,793,304 of 98,997,304 parameters trained, COCO IoU 0.974/0.943/0.975/0.000/0.918, artifact 396,692,619 bytes over 894 tensors, fresh reload identical (reloaded AP 0.8974 / AP50 1.0000), tampered artifact refused, all five new-data sign objects correctly detected (same-label IoU 0.82–0.946). 2 files, 793 MB staged (`yolox_x.pth`). |
| 2026-10-10 (10:07:41 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-d2b0bb2-8e69`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (32 locked packages, setup 51 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`; default path, every form field at its default | `yolox_x_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `a16f3b9435ca`, `module_sha256` `a38e51295351…`, generator `build_notebook.py/2.1-swc-yolox`, `notebook_spec` 2.2) | blob `af87b360649db9c2816635dc4c06d3962a5c600d` at commit `d2b0bb2e7909622c28da22351b52dc4c805b2343` | **PASS** — one pass, no restart, 0 errors; 24/24 code cells in order (cells 4–13, the ten carried modules, print nothing); 150.1 s wall; `yolox_x.pth` 793,463,373 bytes digest-verified; COCO scene 4/5 matched (IoU 0.974/0.943/0.975/0.000/0.918), nothing proposed on the ball; RGB order 3 detections (0.965/0.946/0.927), bench lost; blank and noise 0 boxes at demo thresholds; 30/10 split; baseline AP 0.0384 / AP50 0.0384 → adapted AP 0.8974 / AP50 1.0 / AP75 1.0; loss 5.6956 → 1.6057 in 19.2 s, `num_fg` 7.64–8.68, 11,793,304 of 98,997,304 parameters trained; new-data seed 99: all five objects correctly labelled, no spurious boxes; artifact 396,692,619 bytes over 894 tensors, fresh reload identical, tampered artifact refused. Evidence in `docs/execution-evidence/2026-10-10-d2b0bb2/`: executed notebook SHA-256 `2cd0ebe0f94d7825793a6c31c5b6d9a62054090401aadb94cc7fe2345a304646`, `run_summary.json` `d6d156c1160cf6e05454ce3b40ce875990c709691c8d72f5d17f1663b5658709`, `exec.log` `96da52125a7ef93cb91ad138c115d10ed2c1e795183d7d047e719ac2a0349aff`. Not exercised: both BYOD branches, the Section 12 activity, a browser Run all |

## 2026-10-05 review fixes — verification pending

The 2026-10-05 review (`docs/reviews/2026-10-05-notebook-review/yolox_x_detection_finetune_colab_Review.md`) found that the Kaggle T4 row needed one manual restart after the in-kernel install cell, and that the notebook carried YOLOX-S statements as if they were YOLOX-X measurements. The notebook was then regenerated (`yolox_x_detection_finetune_colab_Fixes.md`):
- an isolated, hash-locked `uv` environment replaces the in-kernel install (spec 2.2);
- every statement about this model now matches this record or says it was measured on the S row: X proposes nothing on the football, the collapse is an S measurement and was not run on X, the IoU is 0.918–0.975, and the baseline is AP50 0.0384;
- the BYOD dataset branch reads `annotations.json` from `BYOD_DATASET_DIR` or the upload dialog;
- the guided layer was added;
- Section 8 starts every run from the verified base.

**The regenerated blob `af87b360649d` (commit `d2b0bb2`) ran once on a fresh Colab Tesla T4 on 2026-10-10 with no restart** (24/24 code cells, 0 errors, 150.1 s; see Recorded executions). The open items are reviewer/integrator acceptance of that one-pass record (gate 6) and the REL12 BYOD journey. `STATUS.md`, `README.md` and gate 6 were corrected on 2026-10-10 to match this record: status **Candidate**, gate 6 open (YXX-m3).

## 2026-10-18 review fixes (row 117): pre-registered grid and fresh test

The configuration is now chosen on a validation pool by the grid fixed in
[`yolox-x-preregistration.md`](yolox-x-preregistration.md), committed alone before any code. The fresh
test is scored only under those settings. The selection logic, tie rule, look ledger and atomic writes
are the same code as the sibling YOLOX-S repository (row 116); only the pools differ. Nothing below is
clean-runtime evidence.

**Pools** (`docs/yolox-x-detection-pools.json`; per-image SHA-256 over decoded pixels under Pillow 11.3.0):

| Pool | Definition | Images | Boxes (stop / yield / speed-limit) | Digest |
|---|---|---|---|---|
| Training | `sign_dataset(40, seed=0)` split, train part | 30 | 61 (16 / 28 / 17) | `60b0aaeea62c…` |
| Exposed development test | same split, held-out part | 10 | 21 (5 / 6 / 10) | `8729a7c00ce7…` |
| Validation | `sign_dataset(40, seed=1171)` | 40 | 81 (32 / 20 / 29) | `e98aec93282e…` |
| Fresh test | `sign_dataset(200, seed=1170)` | 200 | 413 (130 / 144 / 139) | `1f5ae3d9561b…` |

- Every pair of pools, and each pool against Section 10's seed-99 images, shares 0 images and 0 box
  layouts. The training and exposed digests equal the YOLOX-S row's (same generator, same seed).
- Nearest-neighbour mean absolute pixel difference (64×64 thumbnails), median per image: training vs
  fresh 0.0283, validation vs fresh 0.0280, training vs validation 0.0336; within training 0.035. The
  closest single pair (training vs fresh) is 0.0117. Disjoint images, the same renderer: the fresh test
  measures in-distribution generalisation on drawn signs.
- The September grid's 24 images are the first 24 of `sign_dataset(40, seed=0)` (checked by digest).

**Grid cost on a T4 (planning estimate, checked before pre-registering).** C0 took 19.2 s on the
2026-10-10 Colab T4 run (3.2 s per epoch), so the 24-epoch grid is about 80 s of fine-tuning, plus
about 2 minutes of evaluation (validation × 4 systems, fresh test × 2, one reload) and the 10 000-resample
bootstrap (20.1 s on the reference CPU). The whole run is expected at about 5–6 minutes, well under
the 55-minute line and the 3600 s cell timeout. Memory: batch 2 and the frozen backbone as in the
recorded T4 PASS; four YOLOX-X models (baseline + three configs, about 400 MB of float32 weights
each) stay resident. Peak memory has not been recorded.

**CPU development look (SCORE_ONCE_RULE item 8), fresh test MASKED.** 2026-10-18, local Windows CPU
(Intel Core Ultra 9 275HX, shared with other work, so times are slow), CI pins (`torch 2.14.0+cpu`,
Pillow 11.3.0), code at `5e64055` with the notebook regenerated from it (later commits change no
number). The committed notebook cells were run in-process with `DIMER_MASK_FRESH_TEST=1` and `RUN_GRID`
forced to `on`, so the grid ran on CPU. The report says `test_look` "not scored (fresh test masked for a
development check)", `preregistered_settings_hold` true, `settings_differing` empty, grid complete. The
fresh test was neither generated nor scored. These numbers were not used to change any setting.

| Config | Validation AP@[.50:.95] | Validation AP50 | Final `total_loss` | Fine-tune s (CPU) |
|---|---|---|---|---|
| Baseline (re-headed) | 0.0378 | 0.0378 | — | — |
| C0 6 ep, 1e-3 | 0.7465 | 1.0000 | 1.61572 | 181.4 |
| C1 10 ep, 1e-3 | **0.8838** | 1.0000 | 1.21424 | 279.8 |
| C2 8 ep, 2e-3 | 0.8272 | 1.0000 | 1.91754 | 214.8 |

- The rule selects **C1** on CPU: +0.1373 validation AP over C0 with no AP50 loss. The keep-baseline
  rule keeps the adapted model (+0.8460 over the baseline).
- Exposed development test (descriptive only): baseline 0.0384 / 0.0384, which matches every earlier
  record; C1 0.9037 / 1.0000. C0 reproduced the 2026-09-15 CPU final loss exactly (1.61572).
- New-data images (seed 99), C1: 5 of 5 with the right label, same-label IoU 0.911–0.940. Reload of the
  artifact gave identical validation scores (0.8838 / 1.0).
- Old → new: the default configuration changes from C0 (inherited from the S row's held-out grid) to
  the pre-registered rule's choice. CPU picks C1. The hosted T4 grid decides.

**Amendment 1 (2026-10-18, peer review of the YOLOX-S fix, mirrored here).** The decision is taken on a
T4 only (device class is a pre-registered setting; a CPU run with `RUN_GRID = "on"` or any other GPU is
exploratory), Section 9 recomputes the grid and fingerprint checks itself, an existing main report always
spends the slot, grid rows are written per run, and Sections 10–11 export the selected system. The CPU
look above ran before this amendment with the fresh test masked; the amendment changes no number in it.

**Evaluation of record:** none yet. Hosted fresh-test looks: 0. `RECORDED_EVALUATION` is `None` until
the first hosted T4 Run all with every setting at its default. That run is the evaluation of record,
and later identical hosted runs are reproductions.

**Redaction (review R117-m3, own commit).** In `execution-evidence/2026-10-10-d2b0bb2/exec.log`, lines
1 and 193 had the absolute path of the local run directory replaced with `<local-run-dir>/`.
- Captured SHA-256: `96da52125a7ef93cb91ad138c115d10ed2c1e795183d7d047e719ac2a0349aff`. This is the
  value the Recorded executions row above and `run_summary.json` record.
- Committed SHA-256: `9cabba86d6be8db1755a34b3cc1ea5151b49a7ed46319190a29e227b2382f22e`.

**Tie rule (review R117-m1).** `average_precision` now scores tied scores as one operating point, so
the result does not depend on image order. With distinct scores the numbers are unchanged.

**Status.** The notebook blob changes, so the tutorial stays **Candidate** (REL14); gate 6 needs a
one-pass hosted run of the new blob.
