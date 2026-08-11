# MalVision — Progress & Handoff

Status snapshot so any new working session (or a new machine) can pick up where we left
off. Everything below is already committed to this repo — the repo *is* the source of
truth, not any chat history.

## What this project is

MalVision classifies malware families by turning malware binaries into grayscale images
and running them through a fine-tuned ResNet18 (transfer learning). No malware is executed
— static, image-based analysis only. Full spec: `malvision.md`. Task plan and roles:
`briefing.md`.

## Roles

- **Y — Cybersecurity (me, the repo owner).** Malware research, dataset selection,
  confusion-matrix interpretation, Grad-CAM security analysis, report writing.
- **E — AI/ML.** Image pipeline, dataset class, model, training, evaluation, Grad-CAM code.

## Done so far

**Phase 1 — Y (complete, committed):**
- `report/malware_families.md` — profiles the 7 target families (Ramnit, Lollipop, Kelihos,
  Gatak, Obfuscator.ACY, Tracur, Vundo): infection, payload, persistence, propagation,
  platform, impact.
- `report/dataset_selection.md` — Malimg vs MaleVis vs BIG 2015. Key finding: our 7 families
  are **BIG 2015 classes**, not Malimg classes. Recommends a two-stage plan.

**Stage A — E (complete, committed):**
- Full pipeline in `src/` (image_gen, dataset, model, train, evaluate, gradcam) trained on
  **Malimg** (25 placeholder families). Val accuracy ~99%, near-diagonal confusion matrix,
  Grad-CAM overlays for all classes in `outputs/`.
- Reviewed and accepted. Follow-ups filed as **GitHub issue #1**.

## What's next — Stage B (the real target)

Move to **BIG 2015**, restricted to our 7 documented families, so Y's analysis becomes
meaningful.

**Data source decided: the pre-converted Fusion dataset (no 200GB download).**
The raw BIG 2015 competition data is ~200GB after extraction (50GB `.bytes` + 150GB `.asm`),
which does not fit on the available machines. Instead we use a Kaggle dataset that already
has BIG 2015 converted to 224×224 PNGs:

- **`marcesalas/fusion-dataset-59-malware-families-in-png-format`** — 2.24 GB total, 32,601
  PNGs (224×224), 59 families, already split into train/valid/test. Combines BIG 2015 +
  Malimg + MaleVis. MIT license.
- It contains all 7 of our families as pure BIG 2015 folders: `Ramnit`, `Lollipop`, `Vundo`,
  `Tracur`, `Gatak`, `Obfuscator.ACY`, plus `Kelihos_ver1` and `Kelihos_ver3` (merge these
  two into one "Kelihos"). Simda is not needed and is skipped.

Steps:

1. Download the Fusion dataset:
   `kaggle datasets download -d marcesalas/fusion-dataset-59-malware-families-in-png-format`
2. Run `python src/prepare_big2015_fusion.py <extracted_fusion_dir>` — it keeps only the 8
   BIG 2015 family folders, merges `Kelihos_ver1` + `Kelihos_ver3` → `Kelihos`, and writes
   `data/processed/<family>/`. (Tested against a synthetic fixture; selects the 7 classes and
   drops decoys correctly.)
3. **`src/image_gen.py` is not needed for Stage B** — the images are already 224×224 PNG.
   (`dataset.py` already does `.convert('L')`, so RGB-vs-grayscale doesn't matter.)
4. Retrain on the 7 families (`python src/train.py`), then Y fills in the two analysis
   templates already scaffolded: `report/model_errors_analysis.md` (confusion-matrix
   interpretation) and `report/gradcam_analysis.md` (Grad-CAM security reading). Note the
   PE-header caveat documented there: BIG 2015 `.bytes` ship with the PE header stripped, so
   Grad-CAM cannot be attributed to the header.
5. From issue #1, still relevant: persist metrics to `outputs/metrics.json`; track global-best
   checkpoint across stages. The `.bytes` hex-parsing fix (issue #1 item 1) is **no longer a
   Stage B blocker** since we skip raw binaries — keep it only if we ever want to reproduce the
   conversion from raw BIG 2015.

Reference: the Fusion dataset comes from the paper "Deep Learning Applied to Imbalanced Malware
Datasets Classification" (MobileNet fine-tuning; BIG 2015 98.71%). Useful as a comparison
baseline in the report.

## Open items

- **GitHub issue #1** — Stage A review follow-ups. Item 1 (`.bytes` parsing) deprioritized;
  items 2–3 (persist metrics, global-best checkpoint) still apply.
- Y's Phase 2/4/5 analysis tasks are blocked until Stage B results exist.

## How to resume in a new session

Open this repo folder and tell the assistant:
> "This is the MalVision project. I'm Y (cybersecurity). Read PROGRESS.md, briefing.md, and
> report/, then continue from Stage B."
