# Security Analysis of Model Errors (Phase 4 — Y)

**Status: template + pre-analysis. Fill the marked spots after Stage B training produces
`outputs/confusion_matrix.png` and the per-class metrics.**

This section interprets the confusion matrix from a cybersecurity angle: which families the
model confuses, and why that makes sense given how the malware is built. Family background is
in `report/malware_families.md`.

## Setup

- Dataset: BIG 2015 (via the Fusion PNG dataset), 7 families — Ramnit, Lollipop, Kelihos
  (ver1+ver3 merged), Vundo, Tracur, Obfuscator.ACY, Gatak.
- Metric focus: macro precision/recall/F1 (the classes are imbalanced), plus the confusion
  matrix for pairwise error structure.

## Headline results (fill after Stage B)

- Accuracy: `__`
- Macro F1: `__`
- Weakest class by F1: `__`
- Strongest class by F1: `__`

## Pre-analysis — what we expect, and why

These predictions come from the malware research, before seeing the numbers. State whether
each held after Stage B.

1. **Obfuscator.ACY is expected to be the weakest class — and that is the point.**
   Obfuscation and encryption deliberately flatten byte structure, so different Obfuscator.ACY
   samples wrap different payloads and do not share a consistent visual texture. Low recall
   here is an anticipated, documentable limitation of image-based classification, not a
   pipeline bug. If it *is* misclassified, note which classes it leaks into — those are the
   payload families hiding under the obfuscation.

2. **Families that share a builder, packer, or code lineage are the likely confusion pairs.**
   Image-based classification works because one family reuses code and therefore shares
   texture. The flip side: two families built on common tooling look alike. Call out any
   symmetric confusion (A→B and B→A) in the matrix and tie it to a shared-code explanation.

3. **Ramnit should separate cleanly.** As a file-infecting worm it carries a consistent
   injected code stub, which tends to give a dense, recognizable texture. Expect high recall.

4. **Kelihos internal split is no longer a risk.** We merged ver1 and ver3 into one class, so
   the ver1↔ver3 confusion that shows up in raw 9-class BIG 2015 studies does not apply here.
   If Kelihos still underperforms, the cause is external confusion with another family, not the
   version split.

5. **Class imbalance will skew per-class recall.** BIG 2015 is imbalanced (Ramnit and
   Kelihos_ver3 are large; Tracur and Vundo are smaller). Smaller classes with lower recall may
   reflect sample scarcity as much as visual similarity — separate the two when writing up.
   The training pipeline already applies inverse-frequency class weights, so note whether that
   helped the small classes.

## Per-pair error walkthrough (fill after Stage B)

For each off-diagonal cell above a threshold (e.g. ≥5 samples), write one line:

> `<TrueFamily>` → `<PredictedFamily>` (`<count>`): `<security explanation — shared packer /
> common code base / obfuscation / sample scarcity>`.

## Takeaways for the report

- Restate the Obfuscator.ACY result as the headline limitation of the visual approach.
- Summarize whether "shared code base predicts confusion" held.
- One sentence on how class imbalance and class weights shaped the per-class numbers.
