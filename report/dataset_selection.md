# Dataset Selection

This document evaluates the three candidate datasets for MalVision, records a key finding
about which families each dataset actually contains, and gives a recommendation with a
handoff plan for E. It closes the Phase 1 cybersecurity (Y) deliverable in `briefing.md`.

## Requirements (from briefing)

The chosen dataset must have:

- clearly labeled family directories,
- at least ~100 samples per class,
- a balanced or at least manageable class distribution,
- and, ideally, be quick to start with so Phase 1 is not blocked on handling raw binaries.

## Candidate comparison

| Property | Malimg | MaleVis | BIG 2015 (Microsoft) |
|---|---|---|---|
| Format | Grayscale images (ready) | RGB images (ready) | Raw `.bytes` + `.asm` (not images) |
| Samples | ~9,339–9,458 | 14,226 | 10,868 |
| Classes | 25 malware families | 26 (25 malware + 1 benign) | 9 malware families |
| Class balance | **Imbalanced** (some families dominate) | **Balanced** (~500/class; benign 1,832) | Imbalanced |
| Matches our grayscale approach? | **Yes** — exact match | Partly (RGB, not grayscale) | Yes, after conversion |
| Effort to start | **Lowest** (already images) | Low (already images) | **High** (~half TB, build images from raw) |
| Author / origin | Nataraj et al., 2011 | Hacettepe University | Microsoft / Kaggle, 2015 |

## Key finding — the family sets do not match

This is the important handoff point for E. The seven families we documented in
`malware_families.md` (Ramnit, Lollipop, Kelihos, Gatak, Obfuscator.ACY, Tracur, Vundo)
are **BIG 2015 classes**, not Malimg classes.

- **BIG 2015** has 9 classes: Ramnit, Lollipop, Kelihos_ver3, Vundo, Simda, Tracur,
  Kelihos_ver1, Obfuscator.ACY, Gatak. Our 7 families are exactly this set minus Simda,
  with the two Kelihos versions treated as one. So our family research maps directly onto
  BIG 2015.
- **Malimg** has a completely different 25-family set (Allaple.A, Yuner.A, the Lolyda and
  Swizzor variants, and so on). None of our seven documented families appear in it.
- **MaleVis** has yet another 25-family set. It overlaps our list only loosely (for
  example Hlux is an alias for Kelihos), so it does not line up cleanly either.

Consequence: if we train on Malimg, the confusion-matrix interpretation and the Grad-CAM
security analysis (Y's Phase 4 and Phase 5 work) would be about families we have **not**
researched. The family documentation only pays off on BIG 2015.

## Recommendation — two-stage plan

**Stage A — pipeline bring-up on Malimg.**
Use Malimg for the first iteration exactly as the briefing suggests. It is already
grayscale, matches our approach byte-for-byte, and needs no raw-binary handling, so E can
get the full chain working: `image_gen` (or direct load) → `dataset` → `train` →
`evaluate` → `gradcam`. Treat this stage as an engineering shakeout, not as the scientific
result. The families here are placeholders.

**Stage B — real target on BIG 2015.**
Switch to BIG 2015 for the results that go in the report, restricted to our seven
documented families. This is where E's `image_gen.py` earns its place — BIG 2015 ships raw
`.bytes` files, which is exactly the binary-to-grayscale conversion the briefing wants E to
build. And because Y has already profiled these families, the Phase 4 error analysis and
the Phase 5 Grad-CAM interpretation become substantive instead of generic.

**MaleVis — noted alternative, not primary.**
MaleVis is attractive because it is balanced (~500/class) and already in image form. Two
reasons it is a secondary option: it is RGB while our stated method is grayscale, and its
family set still does not match our documentation. Keep it as a fallback if BIG 2015's size
or raw-binary handling becomes a blocker.

## Class imbalance — mitigation

Both Malimg and BIG 2015 are imbalanced. Carry these into the modeling phases:

- Use **stratified** train/val/test splitting (already planned: 70/15/15, `random_state=42`)
  so every split preserves class ratios.
- Apply **class weights** in `CrossEntropyLoss` (inverse frequency) so rare families are
  not drowned out.
- Report **macro-averaged** precision/recall/F1 (already in the plan) so a large easy class
  cannot inflate the headline number.
- If a BIG 2015 family falls below the ~100-sample floor after filtering to our seven,
  flag it to E rather than silently keeping a tiny class.

## Handoff to E

- **Stage A:** point E at the Malimg image folders; no reorganization needed beyond
  confirming one directory per family.
- **Stage B:** deliver the BIG 2015 subset as `data/raw/<family_name>/` with the raw files
  inside, limited to the seven documented families. E converts these to
  `data/processed/<family_name>/` via `image_gen.py`.
- Both dataset roots stay untracked in git (see `.gitignore`); only manifests/label CSVs
  should be committed.

## Sources

- Malimg — Nataraj et al. (2011), "Malware Images: Visualization and Automatic
  Classification"; dataset descriptions via ResearchGate and arXiv:1903.11551.
- MaleVis — Hacettepe University MaleVis project page (14,226 RGB images, 26 classes);
  arXiv:2409.19461.
- BIG 2015 — Microsoft Malware Classification Challenge (Kaggle); data description via
  ResearchGate (10,868 samples, 9 families, `.bytes`/`.asm`).
