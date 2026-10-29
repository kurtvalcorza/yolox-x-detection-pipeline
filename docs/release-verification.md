# Release verification

**Current status (2026-10-25): Candidate.** **Current blob (2026-10-25).** PR #10 (deterministic safetensors header, merged over the serving change) regenerated the notebook as blob `fbc426ed7eaf` (main `1e25d8b`). Its default path and its Section 14 activity (`FREEZE_BACKBONE = False`) are covered by two hosted Colab T4 runs of that exact blob on 2026-10-25, both one pass with no restart and 0 errors ("Hosted T4 runs of the current blob (2026-10-25, commit 1e25d8b)" at the end of this file). The remaining gates are REL12 (the BYOD journey) and REL2 (a browser Run all). Every hosted run of an earlier blob in this file, including the 2026-10-24 runs below, stays as history (REL14). **Safetensors serving (2026-10-24, history).** The model is now served from `yolox_x.safetensors`, a safetensors conversion of the upstream `.pth` whose SHA-256 (`9644228e0bc6…`) is pinned in code, instead of from the pickle itself (code commit `39160b4`). The regenerated notebook `yolox_x_detection_finetune_colab.ipynb` is blob `dac7f4057970` (commit `dda9bed`). Its default path and its Section 14 activity were covered by two hosted Colab T4 runs of that blob on 2026-10-24 ("Safetensors serving: hosted T4 (2026-10-24, commit dda9bed)" below).

The current blob `2bfb2176` (code commit `4ac5d70`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line, with no traceback, and stops Run all at that cell (previously the cell printed the refusal and Run all carried on); genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `fbc426ed`, are history for it (REL14).

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

**Note (2026-10-18, review R117-P5): the S-versus-X comparison above is withdrawn.** It rests on one
seeded CPU run of each model, on the ten-image split that is now the exposed development test, and the
gaps (0.0384 against 0.2222 at baseline, 0.7906 against 0.8110 after adaptation) are within the device
and run-to-run variation recorded since (about 0.1 AP between CPU and T4). The shared design and this
row's pre-registration make no claim comparing YOLOX-S and YOLOX-X. The per-model numbers above stand
only as historical records of the exposed split; the paragraph is kept unedited.

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
| 2026-10-18 (09:01:01 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-35dc06f-8e69`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (32 locked packages, setup 54 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`; default path, every form field at its default, grid on | `yolox_x_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `106113a6c651`, `module_sha256` `c288f824b737…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `a90023b2b9bb43db2055da1e60197b87aade7f40` at commit `35dc06f07ad914834eb56cca5495f1fa0ee38906` | **PASS — pre-registered evaluation of record** (REL13: Colab CLI 0.7.4, fresh Colab Tesla T4, 2026-10-18, `35dc06f` / `a90023b2`; first hosted run of the pre-registered settings and amendment 1, first scoring of the fresh test) — one pass, no restart, 0 errors; 25/25 code cells in order; 306.7 s wall; grid C0 / C1 / C2 validation AP 0.8510 / **0.9237** / 0.8533 (AP50 1.0 each), C1 selected (+0.0727 over C0), keep-baseline +0.8859; fresh test (200 images, 413 boxes) C1 AP 0.9165 vs baseline 0.0445, +0.872, paired bootstrap 95 % CI [0.8555, 0.8887], **improved**; `test_look` "pre-registered evaluation (this runtime)"; exposed development test C1 0.9535 / 1.0 (baseline 0.0384); artifact 396,695,319 bytes over 894 tensors, fresh reload identical, tampered artifact refused. Evidence in `docs/execution-evidence/2026-10-18-35dc06f/`: executed notebook SHA-256 `3ea0532da150ffccae8caac90ec48c6d2547b93e8fd73115479f3dcdd005eb34`, `run_summary.json` `e2fb1e5eb65464b45c183e4400d793c6227bac3ec6fb51d25453c64f0c52eb98`, `exec.log` captured `a13f7b7302ab6f6208b059897a6e9ef3b6020e3d7ac3de4793d026b8df5ac1ac` (committed redacted `3b6e47921f2c…`). Not exercised: both BYOD branches, the Section 12 activity, a browser Run all. See "Hosted T4 evaluation of record (2026-10-18)" |
| 2026-10-18 (13:43:42 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-057d2d8-8e69`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (32 locked packages, setup 49 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`; determinism flags on (amendment 2); default path, every form field at its default | `yolox_x_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `d3e8b320fa22`, `module_sha256` `bc9330519f98…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `6f13a083a88d4bc9fe67957b8d16f00e0fc27ada` at commit `057d2d8ff3bc30ede1e67f74d610274e02986513` | **PASS — amendment-2 evaluation of record** (first hosted run under amendment 2 that scores fresh test 3; first scoring of fresh test 3 by any run). One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 290.3 s wall. Grid hash `16768b30e593…`, equal to both masked determinism checks; grid C0 / C1 / C2 validation AP 0.8510 / 0.9237 / 0.8533 (AP50 1.0 each), C1 selected; fresh test 3 (seed 1172, 200 images, 377 boxes) C1 AP 0.914 vs baseline 0.0365, +0.8775, 95 % CI [0.8603, 0.8944], verdict *improved*; exposed development test C1 AP 0.9535 / AP50 1.0. Details and hashes: "Amendment-2 evaluation of record: hosted T4 (2026-10-18, commit 057d2d8)" below; evidence in `docs/execution-evidence/2026-10-18-057d2d8/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-18 (14:08:44 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-0c8bb09-8e69`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (32 locked packages, setup 51 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`; determinism flags on (amendment 2); default path, every form field at its default | `yolox_x_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `dc99dc71ee49`, `module_sha256` `80b966eefb12…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `ed433995dd1af1a8834ea0997e87b0c96a7a9d00` at commit `0c8bb09682367319f4e8895afa915b7567090dae` | **PASS — reproduction of the pre-registered evaluation** (reproduction 1 of the amendment-2 record `057d2d8`; second hosted look at fresh test 3). One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 310.9 s wall. Section 9 printed `test_look` "reproduction", wrote the `_reproduction_1` files and printed "matches the record". Grid hash `16768b30e593…`, equal to the record; every grid row (except fine-tune seconds) and all 24 epoch loss lines identical to the record; C1 selected; fresh test 3 C1 AP 0.914 vs baseline 0.0365, +0.8775, 95 % CI [0.8603, 0.8944], verdict *improved*, identical to the record. Details and hashes: "Amendment-2 reproduction 1: hosted T4 (2026-10-18, commit 0c8bb09)" below; evidence in `docs/execution-evidence/2026-10-18-0c8bb09/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-23 (13:59:33 UTC start) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-ca77e09-8e69`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (33 locked packages, setup 69 s), `cuda:0`; determinism flags on (amendment 2); default path, every form field at its default | `yolox_x_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `6069f14df1bc`, `module_sha256` `4bf6b6b5f596…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `d4c103eda7e171097451b0ab154f973e44671528` at commit `ca77e09394fd36634f697001dd9a0dee8a60dbc2` | **PASS — default path of the safetensors export** (first hosted run of this blob). One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 436.2 s wall. Grid hash `16768b30e593…`, equal to the record; C1 selected; Section 9 printed `test_look` "reproduction" and "matches the record" (fresh test 3 C1 AP 0.914 vs baseline 0.0365, +0.8775, 95 % CI [0.8603, 0.8944], *improved*). Section 11 wrote `outputs/yolox-x-detection-adapter.safetensors` (396,493,056 bytes, SHA-256 `a024ededec23…`) and `outputs/yolox-x-detection-adapter.json`, no `.pt`; fresh reload 0.9237 / 1.0, exactly the selected system's validation scores; flipped weight byte and edited format tag both refused. Details and hashes: "Safetensors artifact export: hosted T4 (2026-10-23, commit ca77e09)" below; evidence in `docs/execution-evidence/2026-10-23-ca77e09/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-23 (14:04:31 UTC start) | As the row above, on a second fresh Colab Tesla T4 VM (same session name, second account; setup 70 s); run copy with one edit, `FREEZE_BACKBONE = True` → `False` (Section 8 form, cell 44 of 56) | run copy of `yolox_x_detection_finetune_colab.ipynb` (git blob `ba036648`) | blob `d4c103eda7e171097451b0ab154f973e44671528` at commit `ca77e09394fd36634f697001dd9a0dee8a60dbc2` | **PASS — Section 14 activity (unfreeze the backbone)**. One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 427.7 s wall. 98,997,304 of 98,997,304 parameters trained; grid C0 / C1 / C2 validation AP 0.5213 / 0.7384 / 0.1137, C1 selected; Section 9 printed `test_look` "not scored (exploratory)" and "fresh test not scored: settings differ from the pre-registration: ['freeze_backbone']". Section 11 wrote the `.safetensors` (396,493,056 bytes, SHA-256 `049468fc01fa…`) and `.json` sidecar, no `.pt`; reload 0.7384 / 1.0; both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-23-ca77e09-activity/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-24 (10:52:02 UTC start) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM, committed blob fetched at the commit and checked before the VM was allocated, input unchanged; kernel Python 3.13.16, isolated CPython 3.12.12 (33 locked packages, setup 50 s), `cuda:0`; determinism flags on (amendment 2); default path, every form field at its default | `yolox_x_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `39160b4`) | blob `dac7f4057970bea0503281665bf25b82bb3511ca` at commit `dda9beda9fe8e4f748a5453c1881fa5a3a52bfdf` | **PASS — default path of the safetensors serving change.** One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 318.3 s wall. Grid hash `16768b30e593…`, equal to the record; C0 / C1 / C2 validation AP 0.8510 / 0.9237 / 0.8533, C1 selected; Section 9 printed `test_look` "reproduction" and "matches the record" (fresh test 3 C1 AP 0.914 vs baseline 0.0365, +0.8775, 95 % CI [0.8603, 0.8944], *improved*). Section 11 wrote `yolox-x-detection-adapter.safetensors` (396,493,056 bytes, 894 tensors) and its `.json` sidecar, no `.pt`; reload 0.9237 / 1.0; both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-24-dda9bed/`. |
| 2026-10-24 (09:26:10 UTC start) | As the row above, on a second fresh Colab Tesla T4 VM (second account; setup 46 s); run copy with one edit, `FREEZE_BACKBONE = True` → `False` | run copy of `yolox_x_detection_finetune_colab.ipynb` (git blob `d4cb10ca8a2d`) | blob `dac7f4057970bea0503281665bf25b82bb3511ca` at commit `dda9beda9fe8e4f748a5453c1881fa5a3a52bfdf` | **PASS — Section 14 activity (unfreeze the backbone)**. One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 399.9 s wall. 98,997,304 of 98,997,304 parameters trained; grid C0 / C1 / C2 validation AP 0.5213 / 0.7384 / 0.1137, C1 selected; Section 9 printed "fresh test not scored: settings differ from the pre-registration: ['freeze_backbone']". Section 11 wrote the `.safetensors` (396,493,056 bytes, SHA-256 `3e8fd66c49e0…`) and `.json` sidecar, no `.pt`; reload 0.7384 / 1.0; both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-24-dda9bed-activity/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-25 (00:52:43 UTC start) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM, committed blob fetched at the commit and checked before the VM was allocated, input unchanged; kernel Python 3.13.16, isolated CPython 3.12.12 (33 locked packages, setup 50 s), `cuda:0`; determinism flags on (amendment 2); default path, every form field at its default | `yolox_x_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `0fb271e`) | blob `fbc426ed7eafb2a3791d5521a57439e3fe19d648` at commit `1e25d8b05d110120f42d75b64b19937c718d1e4a` (main, the merge of PR #10) | **PASS — default path of the current blob.** One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 310.7 s wall. Grid hash `16768b30e593…`, equal to the record; C0 / C1 / C2 validation AP 0.8510 / 0.9237 / 0.8533, C1 selected; Section 9 printed `test_look` "reproduction" and "matches the record" (fresh test 3 C1 AP 0.914 vs baseline 0.0365, +0.8775, 95 % CI [0.8603, 0.8944], *improved*). Section 11 wrote `yolox-x-detection-adapter.safetensors` (396,493,056 bytes, 894 tensors) and its `.json` sidecar, no `.pt`; reload 0.9237 / 1.0; both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-25-1e25d8b/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-25 (00:32:54 UTC start) | As the row above, on a second fresh Colab Tesla T4 VM (setup 51 s); run copy with one edit, `FREEZE_BACKBONE = True` → `False` | run copy of `yolox_x_detection_finetune_colab.ipynb` (git blob `87384b6df63f`) | blob `fbc426ed7eafb2a3791d5521a57439e3fe19d648` at commit `1e25d8b05d110120f42d75b64b19937c718d1e4a` | **PASS — Section 14 activity (unfreeze the backbone)**. One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 425.3 s wall. 98,997,304 of 98,997,304 parameters trained; grid C0 / C1 / C2 validation AP 0.5213 / 0.7384 / 0.1137, C1 selected; Section 9 printed "fresh test not scored: settings differ from the pre-registration: ['freeze_backbone']". Section 11 wrote the `.safetensors` (396,493,056 bytes, SHA-256 `4090345ffc8e…`) and `.json` sidecar, no `.pt`; reload 0.7384 / 1.0; both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-25-1e25d8b-activity/`. Not exercised: both BYOD branches, a browser Run all |

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
recorded T4 PASS; five YOLOX-X models (the Section 4 demonstration model, the baseline and three
configs, about 400 MB of float32 weights each) stay resident during the grid, six while Section 11's
reloaded copy is alive (corrected 2026-10-18 from "four", review R117-P7). Peak memory has not been
recorded: the 2026-10-18 hosted run printed none, and it ran batch 2 without running out of memory.

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

**Evaluation of record:** the 2026-10-18 hosted Colab T4 run of `35dc06f` (blob `a90023b2`); see
"Hosted T4 evaluation of record (2026-10-18)" below. Hosted fresh-test looks: 1. Reproductions: none yet.
`RECORDED_EVALUATION` in `protocol.py` is still `None`; recording it there regenerates the notebook, and
until then a later identical hosted run prints "pre-registered evaluation (this runtime)" rather than
"reproduction". Later identical hosted runs are reproductions of this evaluation.
(2026-10-18, requote round: `RECORDED_EVALUATION` is now set; see "Requoted" at the end of the next
section.)

**Redaction (review R117-m3, own commit).** In `execution-evidence/2026-10-10-d2b0bb2/exec.log`, lines
1 and 193 had the absolute path of the local run directory replaced with `<local-run-dir>/`.
- Captured SHA-256: `96da52125a7ef93cb91ad138c115d10ed2c1e795183d7d047e719ac2a0349aff`. This is the
  value the Recorded executions row above and `run_summary.json` record.
- Committed SHA-256: `9cabba86d6be8db1755a34b3cc1ea5151b49a7ed46319190a29e227b2382f22e`.

**Tie rule (review R117-m1).** `average_precision` now scores tied scores as one operating point, so
the result does not depend on image order. With distinct scores the numbers are unchanged.

**Status.** The notebook blob changes, so the tutorial stays **Candidate** (REL14); gate 6 needs a
one-pass hosted run of the new blob.

## Hosted T4 evaluation of record (2026-10-18)

**REL13 label for every number in this section:** Colab CLI 0.7.4, fresh Colab Tesla T4, 2026-10-18
(start 09:01:01 UTC), commit `35dc06f07ad914834eb56cca5495f1fa0ee38906`, notebook blob
`a90023b2b9bb43db2055da1e60197b87aade7f40` (embedded `NOTEBOOK_SOURCE.repository_revision` `106113a`;
`106113a..35dc06f` only regenerates the notebook). Role: **pre-registered evaluation of record**, the
first hosted run of the pre-registered settings and amendment 1, and the first time any run scored the
200-image fresh test. Later hosted runs with identical settings are reproductions of this evaluation.
Executor: Colab CLI 0.7.4 sequential execution (`colab exec -f`), not a browser Run all. Forms and
uploads were not exercised. Status stays **Candidate**.

**Identity and runtime.**
- The VM ran a byte-for-byte copy of the committed notebook: input SHA-256 `59f6f5d4…`, git blob
  `a90023b2`, unchanged after the run (`input_unchanged` true).
- `inspect_run.py` passed: 56/56 cells with the same ids and order, and source identity OK. 25/25 code
  cells ran in order, 0 errors, 306.7 s wall (no execution counts: CLI).
- Hardware T4 (`colab status`, session `suite-yolox-35dc06f-8e69`). Section 8 printed
  `grid ON on Tesla T4`; `cuda:0`. Kernel Python 3.13.15; isolated CPython 3.12.12, 32 locked packages,
  setup 54 s; `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `numpy 2.5.3`; `module_sha256`
  `c288f824…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5.
- The device class string (`t4`) is a pre-registered setting but is not printed; the run reached the
  pre-registered label, which requires it.
- TF32: neither the notebook nor the package sets or prints a TF32 flag. The profile is float32, and
  the Tesla T4 (Turing) has no TF32 mode, so TF32 does not apply. That is inferred from the hardware,
  not printed by the run.
- Peak GPU memory was not printed (no `torch.cuda.max_memory_allocated()` in the run; review R117-P7).
  The grid finished without an out-of-memory error at batch 2.

**Score-once checks (as printed).**
- Section 9 printed `test_look` "pre-registered evaluation (this runtime)" and wrote
  `outputs/yolox_x_detection_evaluation_report.json`; the run-1 summary printed batch 2, backbone frozen,
  grid on, selected C1. Pools: training 30 images / 61 boxes, exposed development test 10 / 21,
  validation 40 / 81 (seed 1171), fresh test 200 / 413. Optimiser SGD (momentum 0.9, weight decay 5e-4),
  seed 0, 11,793,304 of 98,997,304 parameters trained.
- The report and results JSON stayed on the released VM. The `preregistered` boolean and
  `settings_differing` are therefore shown only through that label, not as committed files.

**Baseline (re-headed, before fine-tuning).** The six `head.cls_preds.{0,1,2}.{weight,bias}` tensors
could not transfer from the COCO checkpoint (printed in Section 7), so the baseline's classification
head is untrained, and an *improved* verdict was close to foregone (peer review R117-P1). The
informative parts are the size of the gain, the width of the interval, and that it held on images no
choice was made on.
- Validation: AP 0.0378, AP50 0.0378; per-class AP50 stop 0.0, yield 0.0, speed-limit 0.1134.
- Exposed development test: AP 0.0384, AP50 0.0384; per-class AP50 0.0 / 0.0 / 0.1151. This equals
  every earlier record (0.115 at three decimals).

**Grid (validation; this chose the configuration).**

| Config | Validation AP@[.50:.95] | AP50 | Gain over C0 | Final `total_loss` | Fine-tune s (T4) | CPU masked check AP (`5e64055`) | T4 − CPU |
|---|---|---|---|---|---|---|---|
| C0 6 ep, 1e-3 | 0.8510 | 1.0 | — | 1.60573 | 28.1 | 0.7465 | +0.1045 |
| C1 10 ep, 1e-3 | **0.9237** | 1.0 | +0.0727 | 1.18546 | 40.7 | 0.8838 | +0.0399 |
| C2 8 ep, 2e-3 | 0.8533 | 1.0 | +0.0023 | 1.78172 | 35.7 | 0.8272 | +0.0260 |

- **Selection, re-applied by hand.** C1 gains 0.0727 ≥ 0.03 with AP50 equal, so it qualifies. C2
  gains 0.0023 < 0.03, so it does not. C1 is selected, which matches the printed reason. Keep-baseline:
  C1 − baseline = 0.8859 ≥ 0.03 (printed 0.8859), so the adapted C1 is the selected system.
- AP50 is 1.0 for every config, so each class's AP50 is 1.0. The AP50 guard decided nothing
  (R117-P8). The C1 − C0 gap is one draw on 40 validation images; a later hosted reproduction could
  select differently.
- The grid's fine-tuning totalled 104.5 s against the pre-registration's planning estimate of about
  80 s (C0 alone took 28.1 s here against 19.2 s on 2026-10-10). The whole run took 306.7 s wall,
  inside the planned 5–6 minutes.
- On the T4, C0 and C1 share a seed and their first six epochs are identical to four decimals, as on
  CPU. C0's T4 loss curve (5.6956 → 1.6057) equals the 2026-10-10 Colab T4 run.

**Against the CPU masked check.** The CPU check also selected C1, so the decision is unchanged. Every
grid number differs.
- T4 validation AP is higher for all three configs: by 0.1045 (C0), 0.0399 (C1) and 0.0260 (C2).
- On CPU, C2 also qualified (+0.0807) but lost to C1's larger gain. On the T4 it does not qualify.
- The C1 margin over C0 is 0.0727 on the T4 against 0.1373 on CPU. Keep-baseline is 0.8859 against
  0.8460. The baseline itself is identical on both devices (validation 0.0378, exposed 0.0384).
- Only 2 of 24 epoch loss lines are identical (C0 and C1 epoch 1). C2 differs from epoch 1
  (8.0577 against 7.9960). Final loss: C0 1.60573 against 1.61572, C1 1.18546 against 1.21424, C2
  1.78172 against 1.91754.
- The exposed development test for C1 is 0.9535 on the T4 against 0.9037 on CPU (AP50 1.0 both).
- Fine-tuning took 104.5 s on the T4 against 676.0 s on CPU.
- The CPU check ran code `5e64055`; this run ran `106113a`. Between them `src/` changed only
  `protocol.py` (amendment 1: device class, ledger filter, slot rule), not training or scoring code, so
  the T4 − CPU gaps are read as device numerics (inferred from the diff, not re-run).
- The pre-registration's "CPU and T4 runs of C0 differ by about 0.1 AP" holds for C0 on validation
  (0.1045). It does not change the rule.

**Fresh test (primary endpoint; 200 images, 413 boxes).**

| System | AP@[.50:.95] | AP50 | Per-class AP50 (stop / yield / speed-limit) |
|---|---|---|---|
| Selected (C1) | **0.9165** | 1.0 | 1.0 / 1.0 / 1.0 |
| Pretrained baseline (re-headed) | 0.0445 | 0.0445 | 0.0 / 0.0 / 0.1336 |

- Difference +0.872; paired image bootstrap 95 % CI [0.8555, 0.8887] (n_boot 10 000 and seed 20261018
  are the pre-registered `protocol.py` constants; the run does not print them).
- Verdict **improved**: the lower bound 0.8555 is > 0 and the gain is ≥ 0.03.
- Re-derived from the printed values: 0.9165 − 0.0445 = 0.8720 (printed 0.872).
- Observed CI half-width: 0.0166. This replaces the pre-registration's planning estimate of ±0.07 AP.
  The minimum detectable change (planned about 0.10) is far below the observed gain.
- The bootstrap was not re-run, because the per-image detections stayed on the VM.
- The fresh test measures in-distribution generalisation (same renderer, same shapes). It is not a
  detection benchmark.

**Exposed development test (descriptive only; decides nothing).** Baseline AP 0.0384 / AP50 0.0384;
C1 AP 0.9535 / AP50 1.0000 (+0.9152 / +0.9616).

**Other printed checks.**
- COCO scene: 4/5 matched (IoU 0.974 / 0.943 / 0.975 / 0.000 / 0.918), equal to every earlier record.
  RGB order: 3 detections, bench lost. Blank: 0 boxes at both threshold pairs. Noise: 0 at the demo
  thresholds, 2 at the evaluation thresholds (`potted plant` 0.076, `vase` 0.057).
- New-data seed 99 with C1: 5/5 objects correctly labelled, scores 0.944–0.962, same-label IoU
  0.943–0.962, no spurious boxes.
- Artifact: 396,695,319 bytes, 894 tensors, SHA-256 `3bb88c6b…`. The fresh reload reproduced
  validation AP 0.9237 / AP50 1.0, and the tampered artifact was refused.
- BYOD branches off.

**Evidence** (`docs/execution-evidence/2026-10-18-35dc06f/`, `-text`). SHA-256 values were checked
end to end: the VM run directory, the local copy, and the committed blob.
- Executed notebook: `3ea0532da150ffccae8caac90ec48c6d2547b93e8fd73115479f3dcdd005eb34`.
- `run_summary.json`: `e2fb1e5eb65464b45c183e4400d793c6227bac3ec6fb51d25453c64f0c52eb98`.
- `exec.log`: in lines 1 and 255, the operator's home-directory run path was replaced with
  `<local-run-dir>/`.
  - Captured SHA-256: `a13f7b7302ab6f6208b059897a6e9ef3b6020e3d7ac3de4793d026b8df5ac1ac`.
  - Committed SHA-256: `3b6e47921f2c628df417437930b0286fa778436311d26ad377d1d91851500f20`.
  - `run_summary.json` does not record the log's hash.

**Not yet requoted.** The notebook (blob `a90023b2`) still quotes the 2026-10-10 C0 run and says no
hosted grid or fresh-test number exists; `RECORDED_EVALUATION` is still `None` and `RECORDED_SELECTION`
still `"C0"`. Until those are updated (a new blob), a later identical hosted run prints "pre-registered
evaluation (this runtime)"; this section records which run is the evaluation of record.

**Requoted (2026-10-18, requote round; no model run, nothing scored).** The paragraph above describes
blob `a90023b2`. The generator now carries the record:
- `protocol.RECORDED_EVALUATION` holds this run (commit `35dc06f`, blob `a90023b2`, selected C1;
  validation AP baseline 0.0378, C0 0.8510, C1 0.9237, C2 0.8533; fresh test n 200 / 413 boxes, C1
  0.9165 against baseline 0.0445, difference 0.872, 95 % CI [0.8555, 0.8887], "improved"), as Sections 8
  and 9 of the executed notebook printed it.
- Because a record exists, every later pre-registered scoring, including the first one in a new
  runtime, is labelled `reproduction`, writes `_reproduction_<n>` files (never the main report), and
  Section 9 prints whether it matches the record. Section 9 also saves the per-image results and the
  bootstrap's match tables (`*_fresh_test_per_image<suffix>.json`), display only.
- `RECORDED_SELECTION` is now `"C1"`: a grid-off CPU run trains C1 (10 epochs at 1e-3) and stays
  exploratory. In the masked CPU check (code `5e64055`) C1's fine-tune took 279.8 s.
- The notebook's worked answers quote this run under its REL13 label; figures shared with the
  2026-10-10 C0 run (COCO scene IoU, RGB swap, exposed baseline 0.0384, the C0 loss curve) are equal in
  this run and are re-credited to it. Section 9's reading rule now says the "improved" verdict was close
  to foregone because the baseline classification head is untrained (review R117-P1).
- Device class (review R117-P4): this row keeps `"t4"` (row 116 records `"cuda-t4"`). The value is a
  pre-registered setting (`PREREGISTERED_SETTINGS["device_class"]`, amendment 1) and the evaluation of
  record was gated on it, so aligning the vocabulary would change a pre-registered value; left as is.

## Amendment 2 (round 3, 2026-10-18): deterministic training and fresh test 3

See [`yolox-x-preregistration.md`](yolox-x-preregistration.md), amendment 2 (committed alone as `03a926b`).
- **Status of the round-2 record.** The hosted T4 evaluation of record of 2026-10-18 (`35dc06f`, fresh test
  seed 1170) stays valid as history and is carried unchanged as `ROUND2_RECORDED_EVALUATION`. It was the only
  hosted look at seed 1170, which is now the exposed test scored by that record. `RECORDED_EVALUATION` is
  `None` until the first hosted T4 run under amendment 2 scores fresh test 3.
- **Fresh test 3:** `sign_dataset(200, seed=1172)`, 200 images, 377 boxes, digest
  `098fa0dc920a166ff2ec3393bd3e175aa88e3a864525bb5862827a39e5ba7b36` (Pillow 11.3.0). It shares 0 images and
  0 box layouts with training, the exposed development test, validation, seed 1170 and seed 99; the closest
  single pair (vs seed 1170) is at mean absolute pixel difference 0.0109 against a within-training median of
  0.035. The seed-1170 digest re-computed to its committed value.
- **Determinism.** `enable_determinism` sets cuDNN deterministic, benchmark off,
  `torch.use_deterministic_algorithms(True)` (no `warn_only`) and seeds torch, CUDA, numpy and random;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set when the package is imported, before any CUDA use. The fine-tune
  batches in-process (no DataLoader workers). The flags are a pre-registered setting and recorded in every
  report with the device and a grid hash.
- **Ops without a deterministic CUDA kernel.** None found by inspection on the default path: with the
  backbone and neck frozen, gradients pass only through the head (convolutions, BatchNorm, BCE, IoU,
  indexing, `topk` in SimOTA), none of which is on PyTorch's list of ops that raise under deterministic
  mode. This is inferred, not run: there is no local GPU. Any such op raises a RuntimeError naming it in the
  masked hosted check, which then stops the round.
- **Before the new record:** two masked hosted T4 runs (`DIMER_MASK_FRESH_TEST=1`) must print the same grid
  hash, validation APs, selection and loss curves. If they differ, there is no run of record.
- **Evaluation of record under amendment 2:** the Colab CLI T4 run of 2026-10-18 at commit `057d2d8` (blob
  `6f13a083`); see "Amendment-2 evaluation of record" below. It ran after the masked determinism check passed
  (two masked hosted T4 runs, identical grid hash, validation AP, selection and loss curves; see "Masked
  determinism check" below). Hosted looks at fresh test 3: 2. Reproductions: 1 (commit `0c8bb09`, blob `ed433995`,
  identical to the record; see "Amendment-2 reproduction 1" below).
  *(State at the 057d2d8 revision, before the record existed:* `RECORDED_EVALUATION` is `None` and a grid-off
  CPU run keeps training the round-2 selection C1. Recording the amendment-2 record in `protocol.py`
  regenerates the notebook; until then a later identical hosted run labels itself "pre-registered evaluation
  (this runtime)", and this file is what says it is a reproduction.)
- **Requoted (2026-10-18, requote round 3):** `RECORDED_EVALUATION` now carries the `057d2d8` record (grid hash
  `16768b30e593…`, C1, fresh test 3 AP 0.914 vs 0.0365, +0.8775, CI [0.8603, 0.8944], *improved*) and the
  notebook is regenerated from it. A later pre-registered scoring of fresh test 3, including the first one in a
  new runtime, now labels itself "reproduction", writes `_reproduction_<n>` files and prints whether it matches
  the record. `RECORDED_SELECTION` stays C1 (both records selected it). The notebook's worked answers cite the
  `057d2d8` record; the figures they share with the round-2 record are equal (timings: 98.8 s grid, 290.3 s wall).

## Masked determinism check (amendment 2, 2026-10-18)

**REL13 label for this section:**
- Runtime: Colab CLI 0.7.4, two fresh Colab Tesla T4 VMs, one after the other, same session name
  `suite-yolox-1a65eb7-8e69` on different VMs (`…usw1b1-27nty9d5429i3`, then `…ass1c1-2ycje8zerh7x4`).
- Code: commit `1a65eb7` on the throwaway branch `fix/yolox-x-tutorial-review-2026-10-17-maskcheck`, notebook blob
  `c64d88660a4be0e475a4c0abed7a877800a9200b`. That blob is this PR's blob `6f13a083` plus one first code cell
  (`maskcheck-00`) that sets `DIMER_MASK_FRESH_TEST=1`; the diff is that one 13-line cell. The branch is **never
  merged** and may be deleted after this PR merges.
- Role: **not scored (fresh test masked for a development check)**. With the mask set, Section 9 never generates
  fresh test 3, so neither run generated or scored it. Neither suite log mentions seed 1172, the 377 boxes or a
  fresh-test per-image file; the outputs listing shows only the `_masked_1` report and results files.
- Evidence: the orchestrator's suite logs only. The two runs shared one local output directory, so the second
  run's files overwrote the first's; no evidence files are committed for them.

| Run | Started (UTC) | Suite result | Cells | Grid hash | Validation AP C0 / C1 / C2 | Selected | `test_look` | Fresh test 3 |
|---|---|---|---|---|---|---|---|---|
| 117m1 | 13:31:51 | PASS, 241 s start to release | 26/26 | `16768b30e5930007674d365a4b6a00d71b26927cc431bbebf96235988e38e544` | 0.8510 / 0.9237 / 0.8533 | C1 | not scored (fresh test masked …) | never generated |
| 117m2 | 13:36:38 | PASS, 239 s start to release | 26/26 | `16768b30e5930007674d365a4b6a00d71b26927cc431bbebf96235988e38e544` | 0.8510 / 0.9237 / 0.8533 | C1 | not scored (fresh test masked …) | never generated |

- The gate held: the two logs are identical apart from the VM name, process id, object addresses, setup and
  fine-tune timings. That covers the validation AP to every printed digit (C0 0.8510235357786575, C1
  0.9237344510180664, C2 0.8532741716506846), the final losses, all 24 epoch loss lines, the selection, the
  determinism flags and the grid hash. The grid hash covers the weight fingerprints, so the trained weights were
  identical too. Grid fine-tuning took 101.3 s and 101.9 s.
- C0 and C1 share their first six epoch lines exactly, as in the round-2 record.

## Amendment-2 evaluation of record: hosted T4 (2026-10-18, commit 057d2d8)

**REL13 label for every number in this section:**
- Runtime: Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4
  (session `suite-yolox-057d2d8-8e69`, VM `…usw4b0-3qzux89cf2yyi`), started 2026-10-18 13:43:42 UTC, 290.3 s wall
  (49 s environment setup). Section 8 printed `grid ON on Tesla T4` and the determinism flags:
  `cudnn_deterministic` True, `cudnn_benchmark` False, `use_deterministic_algorithms` True, `warn_only` False,
  `cublas_workspace_config` `:4096:8`. The device-class string (`t4`) is written to the report JSON, which stayed on
  the released VM; it is not printed. TF32 is not printed; the Tesla T4 (Turing) has no TF32 mode (inferred from the
  hardware). Peak GPU memory is not printed (no OOM at batch 2).
- Code: commit `057d2d8`, notebook blob `6f13a083`, embedded revision `d3e8b32` (`d3e8b32..057d2d8` only
  regenerates the notebook), `module_sha256` `bc9330519f98…`. Settings: the round-2 settings, amendment 1, amendment
  2 and every field at its default.
- Role: **amendment-2 evaluation of record.** It is the first hosted T4 run under amendment 2 that scores fresh
  test 3 and the first scoring of fresh test 3 by any run. Section 9 printed `test_look` "pre-registered evaluation
  (this runtime)". The record ran after the masked determinism check above passed.
- Evidence: `docs/execution-evidence/2026-10-18-057d2d8/`. The executed notebook and `run_summary.json` were
  copied byte for byte. In `exec.log`, lines 1 and 258 had the local run directory replaced with
  `<local-run-dir>/`; nothing else changed.
  - Captured SHA-256 of `exec.log`: `1d317691a47cf40b1825e57b5e125f141d1b5387bb19cae23c07be6b8171d5c8`. This is
    the value of the file in the run directory; `run_summary.json` records only the executed notebook's hash.
  - Committed SHA-256 of `exec.log`: `79aa8636b3cb7508b086db890b5e1d0ba3846741fb56c87e9161964f83431806`.
  - `run_summary.json`: `08aa52805cc57730b158a7b8a1fb7d538364f2478eae8b9c8342630b4ab83fa9`.
  - Executed notebook: `abbb4974da3a5ff97bc44b477c10e70d342f02f326d5a48b89b815d21583683e` (equal to
    `run_summary.json` `executed_sha256`).
  - The input copy in the run directory is byte-equal to the committed notebook (git blob `6f13a083`).
  - `inspect_run.py`: 56/56 cells, same ids and order, source identity OK.

**Grid (pre-registered; decided here).** Identical to both masked checks: grid hash `16768b30e593…`.

| Config | Validation AP@[.50:.95] | AP50 | Gain over C0 | Final `total_loss` | Fine-tune s (T4) | Round-2 record `35dc06f` AP |
|---|---|---|---|---|---|---|
| C0 (6 ep, 1e-3) | 0.8510 | 1.0 | — | 1.60573 | 27.3 | 0.8510 |
| C1 (10 ep, 1e-3) | **0.9237** | 1.0 | +0.0727 | 1.18546 | 38.4 | 0.9237 |
| C2 (8 ep, 2e-3) | 0.8533 | 1.0 | +0.0023 | 1.78172 | 33.1 | 0.8533 |

- C1 qualifies (gain ≥ 0.03, AP50 equal); C2 does not (+0.0023). C1 is selected, as in the round-2 record.
  Keep-baseline: 0.9237 − 0.0378 = 0.8859 ≥ 0.03, so the adapted C1 is the selected system.
- Grid fine-tuning totalled 98.8 s. Per-class validation AP50 is not printed per config; a mean AP50 of 1.0
  implies 1.0 for every class (arithmetic).
- Validation baseline: AP 0.0378 / AP50 0.0378; per-class AP50 stop 0.0, yield 0.0, speed-limit 0.1134. Equal to
  the round-2 record.

**Fresh test 3** (`sign_dataset(200, seed=1172)`, 200 images, 377 boxes, digest-checked before scoring):

| System | AP@[.50:.95] | AP50 | AP50 stop / yield / speed-limit |
|---|---|---|---|
| Selected C1 | 0.914 | 1.0 | 1.0 / 1.0 / 1.0 |
| Re-headed baseline | 0.0365 | 0.0365 | 0.0 / 0.0 / 0.1095 |

- Difference +0.8775, paired image-bootstrap 95 % CI [0.8603, 0.8944] (n_boot 10000, fixed seed). The lower bound
  is > 0 and the gain is ≥ 0.03, so the verdict is *improved*. The observed half-width is 0.0171.
- At IoU 0.5 every box was found on 200 of 200 images by C1 and on 12 of 200 by the baseline.
- The baseline's classification head is untrained (six `head.cls_preds` tensors could not transfer, printed in
  Section 7), so a large gain was expected. All pools come from the same drawing generator, so this measures
  in-distribution generalisation on drawn signs, not detection on real images.

**Exposed tests (decide nothing).**
- Exposed development test (the old 10 images, 21 boxes): baseline AP 0.0384 / AP50 0.0384 (per-class AP50 0.0 /
  0.0 / 0.1151) → C1 0.9535 / 1.0, change +0.9152 / +0.9616.
- Fresh test 2 (seed 1170, exposed, scored by the round-2 record): **not scored by this run**. Its only hosted
  numbers stay those of the round-2 record (+0.872, CI [0.8555, 0.8887]). Fresh test 2 and fresh test 3 are
  different pools, so +0.8775 is not a reproduction of +0.872.

**Other checks (this run).** COCO scene 4/5 matched (IoU 0.974 / 0.943 / 0.975 / 0.000 / 0.918); RGB order gives
3 detections, bench lost; blank 0 boxes at both thresholds, noise 0 at the demo thresholds and 2 at the evaluation
thresholds (potted plant 0.076, vase 0.057). New-data images: every printed truth label detected, same-label IoU
0.943–0.962. Artifact 396,695,319 bytes over 894 tensors, SHA-256 `42519d65c739…`; the fresh reload reproduces
validation AP 0.9237 / AP50 1.0 exactly; the tampered artifact is refused. BYOD off.

**Against the round-2 record.** Every grid number the round-2 record printed is equal here: the validation AP to
every printed digit, the final losses and all 24 epoch loss lines (`35dc06f` printed no grid hash, so the hash
cannot be compared). On this row the round-2 T4 grid was therefore not moved by GPU non-determinism, unlike row 116
(inferred from two runs). The exposed C1 AP (0.9535), the COCO, RGB, noise and new-data figures are also equal. The
artifact has the same size but a different SHA-256 (`3bb88c6b…` in round 2); the code differs between the runs, so
this is not attributed (not investigated). The round-2 record `35dc06f` stays as history.

## Amendment-2 reproduction 1: hosted T4 (2026-10-18, commit 0c8bb09)

**REL13 label for every number in this section:**
- Runtime: Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4
  (session `suite-yolox-0c8bb09-8e69`, VM `…euw4c1-1g562wfunxhc`), started 2026-10-18 14:08:44 UTC, 310.9 s wall,
  environment build 51 s. Section 8 printed `grid ON on Tesla T4`, `cuda:0` and the same determinism flags as the
  record.
- Code: commit `0c8bb09`, notebook blob `ed433995`, embedded revision `dc99dc7` (`dc99dc7..0c8bb09` only
  regenerates the notebook), `module_sha256` `80b966ee…`. `057d2d8..0c8bb09` carries the record into
  `RECORDED_EVALUATION` and requotes the worked answers; training, scoring and the pools are unchanged. Settings:
  the round-2 settings, amendment 1, amendment 2 and every field at its default.
- Role: **reproduction of the pre-registered evaluation** (reproduction 1 of the amendment-2 record `057d2d8`; the
  second hosted look at fresh test 3). Section 9 printed `test_look` "reproduction" and "reproduction of the
  pre-registered evaluation of 2026-10-18 (commit 057d2d8): matches the record". It wrote
  `yolox_x_detection_evaluation_report_reproduction_1.json`,
  `yolox_x_detection_fresh_test_per_image_reproduction_1.json` and (Section 11)
  `yolox_x_detection_results_reproduction_1.json`; no unsuffixed report, per-image or results file. The adapter
  keeps its usual name `yolox-x-detection-adapter.pt` (byte-identical to the record's, SHA-256 `42519d65c739…`).
  Nothing was changed in response to the record's test number.
- Evidence: `docs/execution-evidence/2026-10-18-0c8bb09/`. The executed notebook and `run_summary.json` were
  copied byte for byte. In `exec.log`, lines 1 and 259 had the local run directory replaced with
  `<local-run-dir>/`; nothing else changed.
  - Captured SHA-256 of `exec.log`: `80c3e669bab936bfd068ec5d0c44015fb5523a1c6e98951aa363054b9d2fbab2`.
  - Committed SHA-256 of `exec.log`: `ec391f2ef47d53fa433d2579aaeb3bec726b4ff3133790d6f01971c2898ea444`.
  - `run_summary.json`: `f5f73ed375a65cfc8f967ae0f9a6fce0d53298da7081f5b12b696b830c7b3a6f`.
  - Executed notebook: `d8a6bf1c24e00aa6af67f7b73fffaab39ffc1811a404a5ae1529107c1ad9a817` (equal to
    `run_summary.json` `executed_sha256`).
  - The input copy in the run directory is byte-equal to the committed notebook (git blob `ed433995`).
  - `inspect_run.py`: 56/56 cells, same ids and order, source identity OK.

**Match against the record.** A line diff of the two `exec.log` files (commit ids, VM name, object addresses
and temporary paths normalised) differs only in: environment build 49 → 51 s, worker pid, the embedded revision
and `module_sha256`, fine-tune seconds (C0 27.3 → 28.2, C1 38.4 → 39.7, C2 33.1 → 34.7), the `test_look` label,
the `_reproduction_1` file names, the new "matches the record" line and the results-file size (27,432 → 27,419
bytes, the shorter label). Everything else is identical:

| Check | Record `057d2d8` | Reproduction `0c8bb09` |
|---|---|---|
| Grid hash | `16768b30e5930007…` | `16768b30e5930007…` (equal) |
| Validation AP C0 / C1 / C2 | 0.8510 / 0.9237 / 0.8533 | equal (full precision) |
| Final `total_loss` C0 / C1 / C2 | 1.60573 / 1.18546 / 1.78172 | equal |
| Epoch loss lines (24) | — | all 24 identical |
| Selection | C1 (+0.0727; keep-baseline 0.8859) | C1, same reason |
| Fresh test 3 C1 / baseline AP | 0.914 / 0.0365 | 0.914 / 0.0365 |
| Difference, 95 % CI | +0.8775 [0.8603, 0.8944] | +0.8775 [0.8603, 0.8944] |
| Verdict | *improved* | *improved* |
| Exposed development test C1 AP / AP50 | 0.9535 / 1.0 | 0.9535 / 1.0 |
| Artifact | 396,695,319 B, 894 tensors, `42519d65…` | equal; reload 0.9237 / 1.0 |

`compare_with_record` checks the selection and the fresh-test fields only; the grid hash and loss curves were
compared from the logs. Determinism therefore held on a fourth T4 VM (after the two masked runs and the record).
Status stays **Candidate**.

## Safetensors artifact export (2026-10-23)

Code commit `6069f14` changes the exported adapter from one `torch.save` pickle (`yolox-x-detection-adapter.pt`, format
`dimer-yolox-detection-adapter/1`) to format `/2`: the state dict as `yolox-x-detection-adapter.safetensors` (written with
`safetensors.torch.save_file`, no pickle) and a plain-JSON sidecar `yolox-x-detection-adapter.json` holding the metadata and
the SHA-256 of the weights file. `load_artifact` checks the sidecar's format, identity, model key and that
SHA-256 before it opens the weights, then applies them with `load_state_dict(strict=True)`; `.pt` artifacts
are refused. Section 11's tamper check now flips one weight byte and edits the sidecar's format tag, and both
are refused. `safetensors==0.8.0` joins the pins and the hash lock.

The regenerated notebook `yolox_x_detection_finetune_colab.ipynb` is blob `d4c103eda7e171097451b0ab154f973e44671528`. **REL14:** every row under
"Recorded executions" ran an earlier blob; those rows, and the artifact sizes and digests quoted above, are
history for the `.pt` export and are not evidence for this blob. The hosted T4 runs of this blob are recorded in
the next section.

### Status at the time (history)

Superseded on 2026-10-24 by blob `dac7f4057970` (safetensors serving); see the last section of this file.

- Notebook then: blob `d4c103eda7e171097451b0ab154f973e44671528` at commit `ca77e09`.
- Default path: **covered** by a fresh Colab T4 run on 2026-10-23 (25/25 code cells, 0 errors, 436.2 s).
- Section 14 activity (unfreeze the backbone, `FREEZE_BACKBONE = False`): **covered** by a second fresh Colab T4
  run on 2026-10-23 (25/25 code cells, 0 errors, 427.7 s).
- Status: **Candidate**, pending only REL12 (the BYOD journey in a browser) and REL2.
- REL14: runs of earlier blobs, including every row above the two 2026-10-23 rows in "Recorded executions", are
  history and are not evidence for this blob.

## Safetensors artifact export: hosted T4 (2026-10-23, commit ca77e09)

**REL13 label for every number in this section:**
- Runtime: Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VMs in
  session `suite-yolox-ca77e09-8e69`. Default: VM `…usw4a1-2103p3avhkkch`, started 2026-10-23 13:59:33 UTC,
  436.2 s wall, environment build 69 s. Activity: VM `…usw1b0-20v5ohf8gp3b6`, started 14:04:31 UTC, 427.7 s wall,
  environment build 70 s. Both printed `grid ON on Tesla T4`, `cuda:0` and the amendment-2 determinism flags.
- Code: commit `ca77e09`, notebook blob `d4c103ed`, embedded revision `6069f14` (`6069f14..ca77e09` only
  regenerates the notebook), `module_sha256` `4bf6b6b5f596…`, 33 locked packages (`safetensors==0.8.0` added).
- Settings: default run, every field at its default. Activity run: a run copy (git blob `ba036648`, SHA-256
  `122483f7…`) that differs from blob `d4c103ed` only in cell 44, `FREEZE_BACKBONE = True` → `False`.

**Evidence.** The executed notebooks were renamed without `_output` and with the run label added; their bytes,
`run_summary.json` and `run_copy.ipynb` were copied byte for byte. In each `exec.log`, the two lines that named the
local run directory (default lines 1 and 262, activity lines 1 and 231) had it replaced with `<local-run-dir>/`;
nothing else changed. Both input copies in the run directories are byte-equal to the committed notebook (git blob
`d4c103ed`), and every executed notebook has 56/56 cells with sources equal to its input.

| File | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `2026-10-23-ca77e09/yolox_x_detection_finetune_colab_ca77e09_colab-cli-t4.ipynb` | `b8de576fa6643ddd29414f27dc38b1051f8c920655dc54173bd96c038a120c9b` (= `executed_sha256`) | same |
| `2026-10-23-ca77e09/run_summary.json` | `fc6a56b3024df4120b14a9dda2d0703c86c1f6d6cfbc7cf78d10a39ef04cf370` | same |
| `2026-10-23-ca77e09/exec.log` | `d30befb35346d3311f1ab92b7589cbc1837bcb9dadbbb162af1e84a6bc02a0aa` | `b56a01e625d4e418a9706af3fdf61152ae6e27f6e481a42802a9f46ca557c82a` (redacted) |
| `2026-10-23-ca77e09-activity/yolox_x_detection_finetune_colab_ca77e09_colab-cli-t4_activity.ipynb` | `d3680b0afb45b2a6098e9bddb35912c077d403462a20a8b7914fcdbc0bd5ca3a` (= `executed_sha256`) | same |
| `2026-10-23-ca77e09-activity/run_copy.ipynb` | `122483f7bcf355649fbba04d51ab8a5ec57584dad55287657ddcd45b6da1c05f` (= `run_copy_sha256`) | same |
| `2026-10-23-ca77e09-activity/run_summary.json` | `73286cfb68108ca89a2235ef60453739a37bfd43c7e1763b51dd61b57e35791a` | same |
| `2026-10-23-ca77e09-activity/exec.log` | `cd41607cf3a864ce8905ae0ac50ec721801d0818db3871c847a5fb99b7f3e7ba` | `7143cc2d9d00b41ea61b05fcc7e8557aba5b065d210d71dccd9146eae263a9a0` (redacted) |

**Default path.**
- Grid: grid hash `16768b30e5930007674d365a4b6a00d71b26927cc431bbebf96235988e38e544`, equal to the record. C0 / C1 /
  C2 validation AP 0.8510 / 0.9237 / 0.8533 (AP50 1.0 each), final `total_loss` 1.60573 / 1.18546 / 1.78172, equal
  to the record; fine-tune seconds 29.8 / 45.9 / 37.0. C1 selected (+0.0727 over C0; keep-baseline 0.8859).
  11,793,304 of 98,997,304 parameters trained.
- Section 9 printed `test_look` "reproduction", wrote the `_reproduction_1` files and printed "reproduction of the
  pre-registered evaluation of 2026-10-18 (commit 057d2d8): matches the record". Fresh test 3 (200 images, 377
  boxes): C1 AP 0.914 / AP50 1.0 against the baseline's 0.0365 / 0.0365, difference +0.8775, 95 % CI [0.8603,
  0.8944], verdict *improved*, identical to the record. Exposed development test C1 AP 0.9535 / AP50 1.0.
- Section 11, the artifact, as printed:

  ```
  "path": "outputs/yolox-x-detection-adapter.safetensors",
  "bytes": 396493056,
  "sha256": "a024ededec2337b8bb6485642c7d740fe42d99d0013b06844d3f81ab66134059",
  "sidecar": "outputs/yolox-x-detection-adapter.json",
  "sidecar_sha256": "6fcfb8565ab7251b92999573010cc5c0e32e0244d38328de9a657de91223b8ba",
  "format": "dimer-yolox-detection-adapter/2",
  "tensors": 894,
  ```

  The printed output listing is `['grid', 'work', 'yolox-x-detection-adapter.json',
  'yolox-x-detection-adapter.safetensors', 'yolox_x_detection_evaluation_report_reproduction_1.json',
  'yolox_x_detection_fresh_test_per_image_reproduction_1.json', 'yolox_x_detection_results_reproduction_1.json']`:
  no `.pt` file.
- Reload, exactly as printed: `{'reloaded_val_ap': 0.9237, 'reloaded_val_ap50': 1.0}` and "fresh reload reproduces
  the selected system validation scores exactly".
- Tamper refusals, exactly as printed:
  - "tampered artifact (one weight byte flipped) refused -> artifact weights digest mismatch for
    'tampered-artifact.safetensors': sidecar records a024ededec2337b8bb6485642c7d740fe42d99d0013b06844d3f81ab66134059,
    file has f041083ed67f502b73ecfbe906f82b7cf22e37c5003480d51c53b52750ed1021"
  - "tampered artifact (format tag edited) refused -> artifact format 'not-a-dimer-artifact/9' !=
    'dimer-yolox-detection-adapter/2'"

**Section 14 activity (`FREEZE_BACKBONE = False`).**
- The fine-tune grid ran: Section 8 printed `grid ON on Tesla T4: training ['C0', 'C1', 'C2']` and
  `freeze_backbone: false`, 98,997,304 of 98,997,304 parameters trained. C0 / C1 / C2 validation AP 0.5213 / 0.7384 /
  0.1137 (AP50 0.8864 / 1.0 / 0.3643), final `total_loss` 3.08909 / 2.27992 / 4.29648, fine-tune seconds 64.0 /
  101.8 / 86.7; grid hash `4fe23d36b7cc…`. C1 selected (+0.2171 over C0; keep-baseline 0.7006).
- Unfreezing the backbone gave a lower validation AP than the default head-only fine-tune (C1 0.7384 against
  0.9237) and a lower exposed development test AP (0.7727 against 0.9535; AP50 1.0 both).
- Section 9 printed `test_look` "not scored (exploratory)" and "fresh test not scored: settings differ from the
  pre-registration: ['freeze_backbone']"; it wrote `_exploratory_1` files and no fresh-test per-image file.
- Section 11 wrote `outputs/yolox-x-detection-adapter.safetensors` (396,493,056 bytes, SHA-256
  `049468fc01fa6862ff5d943c13266d310fdcd387afa915e4e645e973f4224276`) and `outputs/yolox-x-detection-adapter.json`
  (SHA-256 `95dec62f4807…`); no `.pt` in the output listing. Reload `{'reloaded_val_ap': 0.7384,
  'reloaded_val_ap50': 1.0}`, "fresh reload reproduces the selected system validation scores exactly". The flipped
  weight byte was refused with a digest mismatch (sidecar `049468fc…`, file `30a0bba76bb9…`) and the edited format
  tag with "artifact format 'not-a-dimer-artifact/9' != 'dimer-yolox-detection-adapter/2'".

Not exercised by either run: both BYOD branches (REL12) and a browser Run all. Status stays **Candidate**.

## Safetensors serving: hosted T4 (2026-10-24, commit dda9bed)

**REL13 label for every number in this section:**
- Runtime: Colab CLI sequential execution (`colab exec -f`, not a browser Run all), two fresh Colab Tesla T4 VMs.
  Default: started 2026-10-24 10:52:02 UTC, 318.3 s wall, environment build 50 s. Activity: started 09:26:10 UTC,
  399.9 s wall, environment build 46 s. Kernel Python 3.13.16, isolated CPython 3.12.12, 33 locked packages,
  `cuda:0`, amendment-2 determinism flags.
- Code: commit `dda9bed`, notebook blob `dac7f405`, embedded revision `39160b4` (the safetensors serving change).
- Settings: default run, every field at its default. Activity run: a run copy (git blob `d4cb10ca`, SHA-256
  `ffc8447f…`) that differs from blob `dac7f405` in exactly one line, `FREEZE_BACKBONE = True` → `False`.

**Model staging.** Both runs printed `{'fetched': ['yolox_x.pth']}`, `{'verified_files': 1, 'revision': '0.1.1rc0'}`
and `{'device': 'cuda:0', 'source': 'local-snapshot'}`. The output does not print the safetensors conversion or its
digest; it shows the staged `.pth`, the verification and the load.

**Evidence.** The executed notebooks were renamed without `_output` and with the run label added; their bytes,
`run_summary.json` and `run_copy.ipynb` were copied byte for byte. In each `exec.log`, the two lines that named the
local run directory had it replaced with `<local-run-dir>/` (byte-wise, no re-serialisation); nothing else changed.

| File | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `2026-10-24-dda9bed/yolox_x_detection_finetune_colab_dda9bed_colab-cli-t4_default.ipynb` | `395dd74a0d30b02383dce13db59a2d62e823fa0f9571894b70034e8ab9e578ef` (= `executed_sha256`) | same |
| `2026-10-24-dda9bed/exec.log` | `9af8766ec2ca5f99e8a5fcb9156fae9c69fb7460027d939c208b8f422f76fa90` | `438c38a4c4790ea39438b58a751d247546464809d3ae94c218fe691af49966ab` (redacted) |
| `2026-10-24-dda9bed/run_summary.json` | `0a150fa24ca6cccaa9976421d1601fea4283e55c64d02476659ae59aaa64e888` | same |
| `2026-10-24-dda9bed-activity/yolox_x_detection_finetune_colab_dda9bed_colab-cli-t4_activity.ipynb` | `5b8e9efbf56888f8e999244adc081780e5166159e051a8ffc3833cafdaad9d08` (= `executed_sha256`) | same |
| `2026-10-24-dda9bed-activity/exec.log` | `54c47f39664c839b1eea800a8e7332f82a36e0a4d88328a650d95e308532b482` | `491d7d5c9625e7b03c70ca94892606576a7f86d4832f57086bd54aa31ca0e77b` (redacted) |
| `2026-10-24-dda9bed-activity/run_summary.json` | `88af014b164dcce7cb0d1c46f9b3dd612e3e754828c6a15f3d788634f99a5af7` | same |
| `2026-10-24-dda9bed-activity/run_copy.ipynb` | `ffc8447f87b72cc3e2194a1784d1a310dc6ee711f5eada6a14cdbece55a38655` (= `run_copy_sha256`, git blob `d4cb10ca`) | same |

**Default path.**
- Grid: grid hash `16768b30e5930007674d365a4b6a00d71b26927cc431bbebf96235988e38e544`, equal to the record. C0 / C1 /
  C2 validation AP 0.8510 / 0.9237 / 0.8533 (AP50 1.0 each), final `total_loss` 1.60573 / 1.18546 / 1.78172, equal
  to the record; fine-tune seconds 27.2 / 39.1 / 34.7. C1 selected ("C1 beat C0 by 0.0727 validation AP"; keep-baseline
  0.8859). 11,793,304 parameters trained; C1 `total_loss` 5.6956 -> 1.1855 over 10 epochs.
- Section 9 printed `test_look` "reproduction" and "reproduction of the pre-registered evaluation of 2026-10-18
  (commit 057d2d8): matches the record". Fresh test 3 (200 images, 377 boxes): C1 AP 0.914 / AP50 1.0 against the
  baseline's 0.0365 / 0.0365, `ap_difference` 0.8775, `ci95` [0.8603, 0.8944], verdict *improved*. Exposed development
  test C1 AP 0.9535 / AP50 1.0 (baseline 0.0384 / 0.0384).
- Section 11 printed `"bytes": 396493056`, `"sha256": "4d44aae64dc0ba1cb33138b3d051c929f2b346c0a2c76a97326fc5ae35d179c6"`,
  `"sidecar_sha256": "897eab251cfbecc42e112737e63106e5583016a6bf219cd6cfc63d041b456002"`, `"tensors": 894`. The output
  listing is `['grid', 'work', 'yolox-x-detection-adapter.json', 'yolox-x-detection-adapter.safetensors', …]` plus the
  three `_reproduction_1` JSON files: no `.pt` file.
- Reload, as printed: `{'reloaded_val_ap': 0.9237, 'reloaded_val_ap50': 1.0}` and "fresh reload reproduces the selected
  system validation scores exactly". The flipped weight byte was refused ("artifact weights digest mismatch") and the
  edited format tag was refused ("artifact format 'not-a-dimer-artifact/9' != 'dimer-yolox-detection-adapter/2'").
- Other checks: COCO scene 4/5 drawn objects matched; new-data images: every truth label detected, same-label IoU
  0.943–0.962; both refusal probes fired.

**Section 14 activity (`FREEZE_BACKBONE = False`).**
- 98,997,304 of 98,997,304 parameters trained. C0 / C1 / C2 validation AP 0.5213 / 0.7384 / 0.1137 (AP50 0.8864 /
  1.0 / 0.3643), final `total_loss` 3.08909 / 2.27992 / 4.29648, fine-tune seconds 63.0 / 107.1 / 87.8; grid hash
  `4fe23d36b7cc…` (equal to the 2026-10-23 activity run). C1 selected (+0.2171 over C0; keep-baseline 0.7006).
- Unfreezing the backbone gave a lower validation AP than the default head-only fine-tune (C1 0.7384 against 0.9237)
  and a lower exposed development test AP (0.7727 against 0.9535; AP50 1.0 both).
- Section 9 printed `test_look` "not scored (exploratory)" and "fresh test not scored: settings differ from the
  pre-registration: ['freeze_backbone']".
- Section 11 wrote `yolox-x-detection-adapter.safetensors` (396,493,056 bytes, SHA-256 `3e8fd66c49e0…`) and its
  sidecar (SHA-256 `489ae5166773…`); no `.pt` in the output listing. Reload `{'reloaded_val_ap': 0.7384,
  'reloaded_val_ap50': 1.0}`, "fresh reload reproduces the selected system validation scores exactly"; both tamper
  checks refused.

### Status at the time (history)

- Notebook then: blob `dac7f4057970bea0503281665bf25b82bb3511ca` at commit `dda9bed` (model served from the pinned
  safetensors conversion).
- Default path: **covered** by a fresh Colab T4 run on 2026-10-24 (25/25 code cells, 0 errors, 318.3 s).
- Section 14 activity (`FREEZE_BACKBONE = False`): **covered** by a second fresh Colab T4 run on 2026-10-24 (25/25
  code cells, 0 errors, 399.9 s).
- Status: **Candidate**, pending only REL12 (the BYOD journey in a browser) and REL2.
- REL14: runs of earlier blobs, including the 2026-10-23 runs of blob `d4c103ed`, are history and are not evidence
  for this blob.

## Deterministic safetensors header (2026-10-24)

`safetensors==0.8.0` writes the `__metadata__` keys in hash-map order, which changes between processes
(eight processes saving the same four keys produced seven different orders), so two exports of the same
weights could differ in bytes and in `weights_sha256`. `save_artifact` now serializes with
`safetensors.torch.save` and re-emits the header as compact JSON with sorted keys, padded to an 8-byte
boundary; the tensor data block is unchanged. The format tag stays `dimer-yolox-detection-adapter/2`: the
reader is unchanged and earlier `/2` files still load. Covered by unit tests (byte-identical saves in two
fresh processes, canonical header, no pickle or zip framing). The hosted runs of this change are recorded in
the next section.

## Hosted T4 runs of the current blob (2026-10-25, commit 1e25d8b)

**REL13 label for every number in this section:**
- Runtime: Colab CLI sequential execution (`colab exec -f`, not a browser Run all), two fresh Colab Tesla T4 VMs.
  Default: started 2026-10-25 00:52:43 UTC, 310.7 s wall, environment build 50 s. Activity: started 00:32:54 UTC,
  425.3 s wall, environment build 51 s. Kernel Python 3.13.16, isolated CPython 3.12.12, 33 locked packages,
  `cuda:0`, amendment-2 determinism flags.
- Code: commit `1e25d8b` (main, the merge of PR #10: the deterministic safetensors header above, merged over the
  serving change), notebook blob `fbc426ed`, embedded revision `0fb271e`, `module_sha256` `66612b1d095c…`.
- Settings: default run, every field at its default. Activity run: a run copy (git blob `87384b6d`, SHA-256
  `43227f5f…`) that differs from blob `fbc426ed` in exactly one line, `FREEZE_BACKBONE = True` → `False`.
- Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true.

**Evidence.** The executed notebooks were renamed without `_output` and with the run label added; their bytes,
`run_summary.json` and `run_copy.ipynb` were copied byte for byte. In each `exec.log`, the two lines that named the
local run directory had it replaced with `<local-run-dir>/` (byte-wise, no re-serialisation); nothing else changed.

| File | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `2026-10-25-1e25d8b/yolox_x_detection_finetune_colab_1e25d8b_colab-cli-t4_default.ipynb` | `226b51942ae6281e85d81e79be73d495c7bea9f13a1296dc4df660dc6b6326ae` (= `executed_sha256`) | same |
| `2026-10-25-1e25d8b/exec.log` | `cbd2bfd6a3af1954bbefcb5dca01690be6038f2d3821bc26f1fdea52d93aaf70` | `61c719aef6bcc17fd70a93af8efbffe013678a33c8990ca6b7774d96329f087b` (redacted) |
| `2026-10-25-1e25d8b/run_summary.json` | `7d265aa0cc3516a0bc692959d40e5baec705ce94c3e992639446ff838868e8a1` | same |
| `2026-10-25-1e25d8b-activity/yolox_x_detection_finetune_colab_1e25d8b_colab-cli-t4_activity.ipynb` | `ec06dea9fea3d44a7d5c19c4a4ea2f1f75a8121825662aae86739d673adeb817` (= `executed_sha256`) | same |
| `2026-10-25-1e25d8b-activity/exec.log` | `e687fe1543c8d580dbc795d5dcaf90d0fd97c156ee326895e1f687056f0fca94` | `cf0327df7e3784e61e5fea2b7eda00779cab3e4fb02ff22f5eea33629d19a218` (redacted) |
| `2026-10-25-1e25d8b-activity/run_summary.json` | `12112881a23bc9f313c740602f0313bd2fe7f1d2110fa60d2a9f37177d46f5d1` | same |
| `2026-10-25-1e25d8b-activity/run_copy.ipynb` | `43227f5fb06655c9815ce699fbfe2c945e2f605262249ca1717f3e6a7146bce5` (= `run_copy_sha256`, git blob `87384b6d`) | same |

**Default path.**
- Grid: grid hash `16768b30e5930007674d365a4b6a00d71b26927cc431bbebf96235988e38e544`, equal to the record. C0 / C1 /
  C2 validation AP 0.8510 / 0.9237 / 0.8533 (AP50 1.0 each), final `total_loss` 1.60573 / 1.18546 / 1.78172; fine-tune
  seconds 27.9 / 39.2 / 33.9. C1 selected ("C1 beat C0 by 0.0727 validation AP"; "C1 beats the pretrained baseline by
  0.8859 validation AP"). 11,793,304 parameters trained; C1 `total_loss` 5.6956 -> 1.1855 over 10 epochs.
- Section 9 printed `test_look` "reproduction" and "reproduction of the pre-registered evaluation of 2026-10-18
  (commit 057d2d8): matches the record". Fresh test 3 (200 images, 377 boxes): C1 AP 0.914 / AP50 1.0 against the
  baseline's 0.0365 / 0.0365, `ap_difference` 0.8775, `ci95` [0.8603, 0.8944], verdict *improved*. Exposed development
  test C1 AP 0.9535 / AP50 1.0 (baseline 0.0384 / 0.0384).
- Section 11 printed `"bytes": 396493056`, `"sha256": "bdf2648f93b7aed379556e62826370afabc4eba8d4a710944ac251855e8424fb"`,
  `"sidecar_sha256": "18a2cf7c756085892ad4bbec7e1885f0b99009bd378ffc4f2feb2bf8b49f7a9d"`, `"tensors": 894`. The output
  listing is `['grid', 'work', 'yolox-x-detection-adapter.json', 'yolox-x-detection-adapter.safetensors', …]` plus the
  three `_reproduction_1` JSON files: no `.pt` file.
- Reload, as printed: `{'reloaded_val_ap': 0.9237, 'reloaded_val_ap50': 1.0}` and "fresh reload reproduces the selected
  system validation scores exactly". The flipped weight byte was refused ("artifact weights digest mismatch") and the
  edited format tag was refused ("artifact format 'not-a-dimer-artifact/9' != 'dimer-yolox-detection-adapter/2'").
- Other checks: COCO scene 4/5 drawn objects matched; new-data images: every truth label detected, same-label IoU
  0.943–0.962; both refusal probes fired.

**Section 14 activity (`FREEZE_BACKBONE = False`).**
- 98,997,304 of 98,997,304 parameters trained. C0 / C1 / C2 validation AP 0.5213 / 0.7384 / 0.1137 (AP50 0.8864 /
  1.0 / 0.3643), final `total_loss` 3.08909 / 2.27992 / 4.29648, fine-tune seconds 65.9 / 113.3 / 92.0; grid hash
  `4fe23d36b7cc…` (equal to the 2026-10-23 and 2026-10-24 activity runs). C1 selected (+0.2171 over C0; keep-baseline
  0.7006); C1 `total_loss` 7.6409 -> 2.2799.
- Exposed development test C1 AP 0.7727 / AP50 1.0. Section 9 printed `test_look` "not scored (exploratory)" and
  "fresh test not scored: settings differ from the pre-registration: ['freeze_backbone']".
- Section 11 wrote `yolox-x-detection-adapter.safetensors` (396,493,056 bytes, SHA-256 `4090345ffc8e…`) and its
  sidecar (SHA-256 `99133498995b…`); no `.pt` in the output listing. Reload `{'reloaded_val_ap': 0.7384,
  'reloaded_val_ap50': 1.0}`, "fresh reload reproduces the selected system validation scores exactly"; both tamper
  checks refused.

**Differences from the previous hosted runs (2026-10-24, blob `dac7f405`, commit `dda9bed`).** Every validation AP,
final loss, grid hash, selection line, fresh-test and exposed-test figure and reload score above equals the 2026-10-24
run of the same role. What differs: the adapter file digests at the same size and tensor count (default `bdf2648f…` /
sidecar `18a2cf7c…`, then `4d44aae6…` / `897eab25…`; activity `4090345f…` / `99133498…`, then `3e8fd66c…` /
`489ae516…`), which fits the header change of PR #10 (not checked tensor by tensor); the embedded revision; and
timing (default fine-tune 27.9 / 39.2 / 33.9 s, then 27.2 / 39.1 / 34.7 s; activity 65.9 / 113.3 / 92.0 s, then
63.0 / 107.1 / 87.8 s; wall 310.7 s and 425.3 s, then 318.3 s and 399.9 s). One default export exists for this blob,
so these runs do not test the header's byte-for-byte repeatability across sessions.

### Current status

- Current notebook: blob `fbc426ed7eafb2a3791d5521a57439e3fe19d648` at commit `1e25d8b` (main after PR #10 and #11).
- Default path: **covered** by a fresh Colab T4 run on 2026-10-25 (25/25 code cells, 0 errors, 310.7 s).
- Section 14 activity (`FREEZE_BACKBONE = False`): **covered** by a second fresh Colab T4 run on 2026-10-25 (25/25
  code cells, 0 errors, 425.3 s).
- Status: **Candidate**, pending only REL12 (the BYOD journey) and REL2 (a browser Run all).
- REL14: runs of earlier blobs, including the 2026-10-24 runs of blob `dac7f405`, are history and are not evidence
  for this blob.
