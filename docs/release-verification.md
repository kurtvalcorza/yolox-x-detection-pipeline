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
| 6 | Notebook executes top-to-bottom in a **supported hosted runtime** from a cold start | **open** |
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
| 2026-09-15 | Local Windows CPU kernel (Python 3.12.10, `torch 2.14.0+cu130`, GPU hidden), pre-staged checkpoint, pins already present | `yolox_x_detection_finetune_colab.ipynb` | `__LOCAL_ROW__` | `__LOCAL_RESULT__` |
| — | Supported hosted runtime, cold start | — | — | **not yet run** (gate 6) |
