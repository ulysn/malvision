  MalVision — Project Brief & Task Assignment

  Project Summary

  MalVision classifies malware families by converting malware binary files into grayscale images and running them
  through a fine-tuned CNN (transfer learning). No malware is executed at any point — this is purely static, image-based
   analysis.

  Stack: Python, PyTorch, OpenCV, NumPy, Pandas, Matplotlib, Grad-CAM, Git/GitHub, Jupyter Notebook.

  ---
  Team

  ┌────────┬───────────────────────┐
  │ Person │        Domain         │
  ├────────┼───────────────────────┤
  │ E      │ AI / Machine Learning │
  ├────────┼───────────────────────┤
  │ Y      │ Cybersecurity         │
  └────────┴───────────────────────┘

  There are 5 sequential phases. Some tasks within a phase can run in parallel. The critical dependency is that Y must
  finalize the dataset before E can begin model training. Both must coordinate at integration points.

  ---
  Phase 1 — Setup & Foundation (Parallel, Week 1)

  E — Environment & Architecture Study

  - Set up the Python environment: PyTorch, torchvision, OpenCV, NumPy, Pandas, Matplotlib, pytorch-grad-cam.
  - Initialize the GitHub repository with this folder structure:
  malvision/
  ├── data/          # raw binary files, split folders (train/val/test)
  ├── src/
  │   ├── image_gen.py      # binary → grayscale image
  │   ├── dataset.py        # PyTorch Dataset class
  │   ├── model.py          # model definition
  │   ├── train.py          # training loop
  │   ├── evaluate.py       # metrics
  │   └── gradcam.py        # explainability
  ├── notebooks/     # exploratory work
  ├── outputs/       # saved models, plots, confusion matrices
  └── report/
  - Study ResNet18 architecture. Understand what ImageNet pre-trained weights give you and why they transfer to
  texture-based images.
  - Read the paper that started this field: "Visualization of Executable Files for Malware Analysis" (Nataraj et
  al., 2011) — it is the conceptual foundation of this entire project.

  Y — Malware Research & Dataset Selection

  - Research and write a documentation file (report/malware_families.md) covering each of these 7 families:
    - Ramnit, Lollipop, Kelihos, Gatak, Obfuscator.ACY, Tracur, Vundo
    - For each: infection method, payload, persistence mechanism, propagation vector, target platform, real-world
  impact.
  - Evaluate the three candidate datasets:
    - Malimg Dataset (Nataraj et al.) — 9,458 samples, 25 families, already converted to images. Fastest to get started
  with.
    - MaleVis Dataset — RGB visualization dataset, more recent.
    - Microsoft Malware Classification Challenge (BIG 2015) — raw binary files, 9 families, ~500GB. Requires image
  generation from scratch.
  - Recommendation to Y: Start with Malimg Dataset for the first iteration. It is well-studied, balanced enough, and
  removes the need to handle raw binaries in phase 1. If time allows, extend to BIG 2015 for E to practice the image
  generation pipeline.
  - Confirm the dataset has: clearly labeled family directories, sufficient samples per class (aim for >100 per class
  minimum), and a balanced or manageable class distribution.
  - Deliver to E: the final dataset folder organized as data/raw/<family_name>/ with files inside.

  ---
  Phase 2 — Image Generation Pipeline (E leads, Y reviews) — Week 2

  This phase is E's most critical deliverable before training. Y must review the output visually.

  E — Binary to Grayscale Image Converter (src/image_gen.py)

  Core logic:
  # Conceptual pseudocode — E implements the full version
  def binary_to_image(filepath, size=224):
      with open(filepath, 'rb') as f:
          bytes_data = np.frombuffer(f.read(), dtype=np.uint8)
      side = int(np.sqrt(len(bytes_data))) + 1
      padded = np.pad(bytes_data, (0, side*side - len(bytes_data)))
      img = padded.reshape(side, side)
      img_resized = cv2.resize(img, (size, size))
      return img_resized

  E must implement:
  1. The full converter with 224×224 output (for ResNet18 input compatibility).
  2. A batch script that processes an entire dataset folder.
  3. Save images as .png files into data/processed/<family_name>/.
  4. A notebook (notebooks/01_image_exploration.ipynb) showing sample images from each family — this is a visual sanity
  check.

  Important note for E: Do NOT use random crop or horizontal/vertical flip as data augmentation. Malware byte sequences
  are positional — flipping a malware image creates a meaningless artifact. Safe augmentations: brightness adjustment,
  Gaussian noise, slight rotation (±5 degrees max).

  Y — Visual Validation

  - Open the notebook E produces.
  - For each malware family, confirm the images show visually distinct texture patterns.
  - Ramnit and Kelihos should look noticeably different from each other. If all images look identical (uniform gray or
  all-white), there is a bug in E's pipeline — flag it immediately.
  - Document 1–2 sentences per family describing what the visual pattern looks like (dense texture, sparse, banded,
  etc.). This goes into the report.

  ---
  Phase 3 — Dataset Class & Preprocessing (E) — Week 2–3

  E — PyTorch Dataset (src/dataset.py)

  Implement a MalwareDataset(Dataset) class:
  - Reads from data/processed/<family>/ folders.
  - Converts grayscale images to 3-channel tensors (ResNet18 expects 3 channels — duplicate the grayscale channel across
   RGB).
  - Applies transforms:
    - Required: Resize(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) — use ImageNet
   stats since you're using ImageNet weights.
    - Optional: contrast enhancement, histogram equalization (apply before tensor conversion).
  - Implements __len__ and __getitem__.

  Implement the dataset split:
  - 70% train / 15% validation / 15% test using stratified splitting (preserve class balance across splits).
  - Use sklearn.model_selection.train_test_split with stratify=labels.
  - Save split indices or CSV manifests so the exact split is reproducible. Use random_state=42.

  Deliver: a notebook (notebooks/02_dataset_check.ipynb) showing class distribution bar charts for all three splits.

  ---
  Phase 4 — Model, Training & Evaluation (E builds, Y interprets) — Week 3–4

  E — Model Definition (src/model.py)

  Implement build_model(num_classes, freeze_backbone=True):
  - Load resnet18(pretrained=True) from torchvision.
  - Replace model.fc with nn.Linear(512, num_classes).
  - If freeze_backbone=True, freeze all layers except model.fc.

  Training strategy — implement all three stages:
  - Stage 1: Freeze backbone, train classifier only. 10–15 epochs.
  - Stage 2: Unfreeze layer3 and layer4 (last two residual blocks). Lower learning rate (1e-4 or lower). 10–15 epochs.
  - Stage 3 (optional): Unfreeze entire network. Very low learning rate (1e-5). 5–10 epochs.

  Use: Adam optimizer, CrossEntropyLoss, learning rate scheduler (StepLR or CosineAnnealingLR).

  E — Training Loop (src/train.py)

  - Log train loss, train accuracy, val loss, val accuracy per epoch.
  - Save the best model checkpoint (lowest val loss) to outputs/best_model.pth.
  - Plot training curves at the end.

  E — Evaluation (src/evaluate.py)

  Run on the test set only (never tune based on test set). Report:
  - Accuracy, Precision (macro), Recall (macro), F1-score (macro), ROC-AUC (one-vs-rest).
  - Confusion matrix — save as a heatmap image to outputs/confusion_matrix.png.

  Y — Result Interpretation

  Once E has confusion matrix and metrics:
  - Identify which malware families are most confused with each other. Explain why from a cybersecurity perspective
  (e.g., two families may share a common code base, packer, or obfuscation technique — this is a known phenomenon in
  malware genealogy).
  - If Obfuscator.ACY is consistently misclassified — this is expected. Obfuscation intentionally destroys visual
  texture patterns. Document this as a known limitation.
  - Provide a written section for the report: "Security Analysis of Model Errors."

  ---
  Phase 5 — Explainability (E builds, Y interprets) — Week 4–5

  E — Grad-CAM (src/gradcam.py)

  - Use the pytorch-grad-cam library targeting ResNet18's layer4.
  - For each malware family, pick 3–5 representative test samples.
  - Generate Grad-CAM heatmap overlaid on the grayscale image.
  - Save to outputs/gradcam/<family_name>/.

  The output should show which pixel regions (byte regions) the model focuses on when predicting that family.

  Y — Grad-CAM Security Analysis

  This is the most intellectually interesting part of the project. Y must answer:
  - Do the highlighted byte regions correspond to known malware sections (e.g., .text section — executable code, .data —
   initialized data, PE header)?
  - A PE executable has a structured layout. The first ~4KB is the PE header. If Grad-CAM consistently highlights the
  header region, that is meaningful.
  - Cross-reference with published malware analysis of the families. Do the active regions make sense?
  - Document findings in report/gradcam_analysis.md.

  ---
  Phase 6 — Model Comparison & Report (Both) — Week 5

  E — Multi-Model Comparison

  - After ResNet18 is working, run the same pipeline with EfficientNet-B0 (or ResNet34).
  - Compare: accuracy, training time, model size, F1-score.
  - Produce a comparison table for the report.

  Both — Final Deliverables Checklist

  ┌────────────────────────────────────────┬────────────────────────────┐
  │              Deliverable               │           Owner            │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Python implementation (all src/ files) │ E                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Image generation pipeline              │ E                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Training pipeline                      │ E                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Evaluation pipeline + metrics          │ E                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Trained model .pth file                │ E                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Performance report (metrics, curves)   │ E                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Confusion matrix heatmap               │ E                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Grad-CAM visualizations                │ E                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Malware family documentation           │ Y                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Security analysis of model errors      │ Y                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Grad-CAM security interpretation       │ Y                          │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Technical documentation / README       │ Both                       │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ GitHub repository                      │ E sets up, Both contribute │
  ├────────────────────────────────────────┼────────────────────────────┤
  │ Project presentation slides            │ Both                       │
  └────────────────────────────────────────┴────────────────────────────┘

  ---
  Critical Dependencies & Handoff Points

  Y finalizes dataset
         ↓
  E builds image generator → Y visually validates output
         ↓
  E trains model → Y interprets confusion matrix
         ↓
  E generates Grad-CAM → Y interprets security meaning
         ↓
  Both write report + presentation

  ---
  What is Out of Scope (do not spend time on these)

  Dynamic execution, reverse engineering, memory forensics, kernel analysis, live detection, behavior analysis, network
  traffic, sandbox, real-time antivirus. The project is static image-based classification only.

  ---
  Final Notes

  - E: The image generation and the 3-channel conversion step are the two most common failure points. Test them visually
   before training anything.
  - Y: The Grad-CAM interpretation section is what makes this project stand out academically. The security analysis of
  model errors (especially obfuscated families) is novel and publishable-quality insight — invest time here.
  - Both: The integration points above are non-negotiable sync moments. Do not proceed past a handoff until both sides
  have reviewed the output.