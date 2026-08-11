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
meaningful. Blockers/steps:

1. **Fix `.bytes` parsing (issue #1, item 1).** BIG 2015 `.bytes` files are hex-ASCII text,
   not raw binary. `src/image_gen.py` currently reads raw `uint8` → garbage on BIG 2015.
   Must parse hex before Stage B.
2. **Download BIG 2015** from Kaggle (`kaggle competitions download -c malware-classification`),
   extract only `.bytes`, organize into `data/raw/<family>/` using `trainLabels.csv`.
   Class map: 1 Ramnit, 2 Lollipop, 3 Kelihos_ver3, 4 Vundo, 5 Simda (**skip**), 6 Tracur,
   7 Kelihos_ver1, 8 Obfuscator.ACY, 9 Gatak. Merge classes 3+7 into one "Kelihos".
3. **Retrain** on the 7 families, then Y does: confusion-matrix interpretation
   ("Security Analysis of Model Errors"), Grad-CAM vs PE sections (`report/gradcam_analysis.md`).
4. Also from issue #1: persist metrics to `outputs/metrics.json`; track global-best checkpoint
   across stages.

## Open items

- **GitHub issue #1** — Stage A review follow-ups (assigned to E's work).
- Y's Phase 2/4/5 analysis tasks are blocked until Stage B results exist.

## How to resume in a new session

Open this repo folder and tell the assistant:
> "This is the MalVision project. I'm Y (cybersecurity). Read PROGRESS.md, briefing.md, and
> report/, then continue from Stage B."
