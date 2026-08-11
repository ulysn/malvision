# Grad-CAM Security Interpretation (Phase 5 — Y)

**Status: template + pre-analysis. Fill the marked spots after Stage B produces
`outputs/gradcam/<family>/` overlays.**

Grad-CAM shows which pixel regions — i.e. which byte regions of the binary — the model relies
on when it predicts a family. The security question: do those regions line up with meaningful
parts of the executable?

## Critical caveat — the PE header is stripped

BIG 2015 `.bytes` files were released **with the PE header removed** ("to ensure sterility",
per the competition data description). So we **cannot** claim the model attends to the PE
header — that region does not exist in this data. This is important and worth stating plainly
in the report, because a lot of image-based malware papers point at "the header region" and
that argument is not available to us here.

What we *can* interpret instead:

- **Relative position within the byte stream** — does the model focus on early code regions vs
  later data/appended regions?
- **High-entropy / packed regions** — encrypted or compressed blocks show up as fine, noisy
  texture. If Grad-CAM lights those up, the model is keying on the packing itself.
- **Repeated structural bands** — file-infectors and families with padding produce regular
  banded patterns; highlighted bands suggest the model found the reused code stub.

## Method

- Grad-CAM target layer: ResNet18 `layer4[-1]` (already wired in `src/gradcam.py`).
- 3–5 representative test samples per family in `outputs/gradcam/<family>/`.
- Cross-reference each family's highlighted regions against its documented behavior in
  `report/malware_families.md` and against published analyses.

## Pre-analysis — what we expect per family

State whether each held after inspecting the overlays.

- **Ramnit** (file-infecting worm): expect focused, consistent hot regions across samples — the
  injected code stub. Consistency across samples is the tell.
- **Gatak** (steganographic stealer): its data is hidden inside payload logic; expect the
  hotspot on a specific structured region rather than spread out.
- **Obfuscator.ACY**: expect **unstable, diffuse heatmaps** that jump around between samples.
  Because obfuscation destroys stable structure, there is no consistent region to latch onto.
  This visually explains the poor Phase 4 accuracy — pair the two sections in the report.
- **Kelihos / Vundo / Tracur / Lollipop**: expect moderately consistent hotspots tied to their
  reused components. Note any family whose heatmap is surprisingly consistent or surprisingly
  scattered.

## Per-family findings (fill after Stage B)

For each family, 2–3 sentences:

> `<Family>`: the model focuses on `<where in the image / relative region>`. This corresponds to
> `<packed region / reused stub / data section / nothing consistent>`. `<Does it match the
> family's known behavior? yes/no and why.>`

## Takeaways for the report

- Lead with the PE-header caveat so the interpretation is honest about what the data allows.
- The Obfuscator.ACY "diffuse heatmap ↔ low accuracy" link is the strongest, most novel point —
  it connects Phase 4 and Phase 5 into one story.
- Close with whether image-based Grad-CAM gives a security analyst anything actionable, or only
  a post-hoc rationalization. Say it straight either way.
