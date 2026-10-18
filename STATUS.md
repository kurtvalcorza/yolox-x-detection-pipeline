# Release status

Current status: **Candidate**

The repository is complete and internally consistent: the package and its unit tests, the vendored
upstream provenance check, the standalone `E2E` notebook and its parity check, the model card, and the
static release-asset validation all pass, and the notebook has been executed once top-to-bottom in a
fresh local CPU kernel.

A hosted run is also recorded: on 2026-09-15 a Kaggle T4 kernel executed an earlier blob of the
notebook (22/22 code cells), but only after a manual restart following the old in-kernel install cell.
That run is workflow history, not one-pass clean-runtime evidence, so gate 6 is open. The 2026-10-05
review fixes regenerated the notebook with an isolated `uv` environment that needs no restart; the regenerated
blob `af87b360649d` (commit `d2b0bb2`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-10 (Colab CLI 0.7.4 sequential execution, 24/24 code cells, 150.1 s; baseline AP 0.0384 / AP50 0.0384 → adapted AP 0.8974 / AP50 1.0; fresh reload identical). The pre-registered evaluation of record ran on 2026-10-18 on a fresh Colab Tesla T4 (Colab CLI 0.7.4, commit `35dc06f`, blob `a90023b2`, 25/25 code cells, 0 errors, 306.7 s): the validation grid selected C1 (C0 / C1 / C2 validation AP 0.8510 / 0.9237 / 0.8533), and on the 200-image fresh test C1 scored AP 0.9165 against the re-headed baseline's 0.0445, a difference of +0.872 with a paired bootstrap 95 % CI [0.8555, 0.8887], verdict *improved*. The baseline's classification head is untrained, so a large gain was expected; the fresh test is drawn by the same generator, so this is in-distribution evidence, not a detection benchmark. Details and evidence hashes: "Hosted T4 evaluation of record (2026-10-18)" in [`docs/release-verification.md`](docs/release-verification.md). Under amendment 2 (deterministic GPU training, fresh test 3), the **amendment-2 evaluation of record** ran on 2026-10-18: blob `6f13a083a88d` (commit `057d2d8`) on a fresh Colab Tesla T4 (Colab CLI 0.7.4, 25/25 code cells, 0 errors, 290.3 s), after two masked hosted runs printed the same grid hash. The grid selected C1 again, with every validation number equal to the round-2 record (C0 0.8510 / C1 0.9237 / C2 0.8533). On fresh test 3 (200 images, 377 boxes, seed 1172), C1 scored AP@[.50:.95] 0.914 against the baseline's 0.0365: +0.8775, 95 % CI [0.8603, 0.8944], verdict *improved*. The round-2 record above stays as history. See "Masked determinism check" and "Amendment-2 evaluation of record" in the same file. Status stays Candidate until a reviewer/integrator accepts the hosted records; the REL12 BYOD journey is not recorded.

That local execution is **pre-flight evidence, not clean-runtime evidence**. A green CI run is static
and unit validation only; neither it nor a local kernel run establishes that the notebook completes in
a supported hosted runtime from a cold start. Promotion to **Release-grade** requires a clean-room
execution on a supported runtime recorded under "Recorded executions" in
[`docs/release-verification.md`](docs/release-verification.md), which also lists the gates that remain
open.
