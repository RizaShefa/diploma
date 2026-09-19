# Implementation → Thesis Mapping

What each piece of the implementation supports in the written thesis: which
section it belongs to, which experiment produces it, what result it yields, and
which figure or table it becomes.

**None of the results below exist yet.** Every "Result" entry describes what the
listed command *will* produce once BUSI is in place and the scripts are run. No
numbers are recorded here that have not been measured.

---

## 1. Dataset integrity and leakage

| | |
|---|---|
| **Code** | `src/data/integrity.py`, `scripts/analyze_dataset.py --legacy-dataset` |
| **Thesis section** | Ch. 7 (Dataset collection and preparation) — needs substantial rewrite |
| **Experiment** | Audit the original 5,000-image dataset; measure leakage under the original splitting strategy; repeat under group-aware splitting |
| **Command** | `python scripts/analyze_dataset.py --legacy-dataset` |
| **Result** | Duplicate counts, near-duplicate rate at four thresholds, cluster reduction factor, train/test leakage percentage |
| **Figure** | Side-by-side near-duplicate image pair; leakage bar chart (grouped vs naive) |
| **Table** | Near-duplicate rate by correlation threshold |
| **Discussion** | *Why the previously reported 99.6% accuracy is not a generalisation estimate.* This is the strongest original contribution in the project — most student theses do not audit their own dataset. |

**Already measured** (on the legacy `dataset/benign/`, 2,500 images):

| Measure | Value |
|---|---|
| Exact byte-duplicate groups | 51 (103 files) |
| Near-twin at corr > 0.95 | 2,053 / 2,500 (82.1%) |
| Distinct clusters | ~891 |
| Test images with near-duplicate in train (original split) | **77.3%** |

⚠️ The malignant half of that dataset is missing from the project, so these
figures cover the benign class only. State that scope limit in the thesis.

---

## 2. Preprocessing parity (train/serve skew)

| | |
|---|---|
| **Code** | `src/preprocessing.py`, `tests/test_preprocessing.py` |
| **Thesis section** | Ch. 8 (Image preprocessing) and Ch. 9 (System design) |
| **Experiment** | Compare predictions under training-style vs serving-style preprocessing using the shipped model |
| **Result** | **Already measured:** 9/150 predictions flipped (6.0%); mean \|Δp\| = 0.068 |
| **Figure** | Before/after pipeline diagram |
| **Table** | Preprocessing steps: claimed vs originally implemented vs now implemented |
| **Discussion** | Offline metrics only describe live behaviour if the two pipelines are identical. A software-engineering defect that silently invalidates a scientific claim. |

Ch. 8 currently claims CLAHE, median filtering, anisotropic diffusion, histogram
equalisation and 224×224 resizing. Only resizing and normalisation were ever
implemented, at 64×64. Either implement the rest or rewrite the chapter.

---

## 3. Corrected experimental protocol

| | |
|---|---|
| **Code** | `src/data/splitting.py`, `src/training/trainer.py` |
| **Thesis section** | Ch. 10.4 (Training, validation and testing strategy) — currently describes a protocol that was not implemented |
| **Experiment** | Retrain the original architecture under the corrected protocol |
| **Command** | `python scripts/analyze_dataset.py && python scripts/train.py --config custom_cnn && python scripts/evaluate.py --model custom_cnn` |
| **Result** | Honest test metrics with confidence intervals |
| **Figure** | Split composition stacked bars (`dataset_split.png`) |
| **Table** | Protocol: as claimed in Ch. 10 vs as originally implemented vs as now implemented |
| **Discussion** | Separate validation set; stratified group-aware splitting; early stopping on validation loss; balanced class weights; seeded runs. |

Protocol corrections to document:

| Aspect | Original | Now |
|---|---|---|
| Validation data | the test set | separate split |
| Stratification | none | stratified by class |
| Grouping | none | near-duplicate clusters |
| Shuffling | `shuffle=False` on class-ordered data | shuffled each epoch |
| Epochs | fixed at 10 | early stopping on `val_loss` |
| Class weighting | none | balanced |
| Seeding | none | Python + NumPy + TF |

---

## 4. Architecture comparison

| | |
|---|---|
| **Code** | `src/models/registry.py`, `scripts/compare_models.py` |
| **Thesis section** | Ch. 10.1 (Model architecture selection) — the abstract's claim that the custom CNN "outperformed ResNet and VGG" currently has **no supporting experiment** |
| **Research question** | Does ImageNet transfer learning help on grayscale ultrasound, where the source domain differs sharply from the target? |
| **Command** | `train.py --config {custom_cnn,resnet50,efficientnetb0}` → `evaluate.py` → `compare_models.py` |
| **Result** | Sensitivity, specificity, precision, F1, ROC-AUC, PR-AUC with 95% CIs; parameter counts; inference latency |
| **Figure** | `comparison_sensitivity.png`, `comparison_specificity.png`, `comparison_roc_auc.png` |
| **Table** | Models × metrics with CIs, parameters and ms/image |
| **Discussion** | Accuracy/latency trade-off; whether capacity or data is the binding constraint at ~650 images; whether CI overlap makes any ranking unsupportable. |

Each model answers something specific:

| Model | Question |
|---|---|
| `custom_cnn` (33K params) | Baseline — the thesis's own architecture, corrected |
| `resnet50` (23.6M) | Does residual depth + ImageNet transfer help? |
| `efficientnetb0` (4.1M) | Does compound scaling beat depth at this data scale? |
| `custom_cnn_legacy64` (176K) | Resolution ablation — reproduces the original model exactly |

---

## 5. Evaluation metrics and confidence intervals

| | |
|---|---|
| **Code** | `src/evaluation/metrics.py` |
| **Thesis section** | Ch. 12 (Model evaluation and metrics) |
| **Experiment** | Full metric suite on the held-out test split with bootstrap CIs |
| **Result** | Sensitivity, specificity, precision, NPV, F1, balanced accuracy, ROC-AUC, PR-AUC, ECE |
| **Figure** | Confusion matrix, ROC, precision-recall, reliability diagram |
| **Table** | Metric ± 95% CI |
| **Discussion** | Why sensitivity leads for cancer detection; why PR-AUC is more honest than ROC-AUC under imbalance; why a point estimate from ~100 test images is not a result. |

Ch. 12's existing hand calculations are correct — `tests/test_metrics.py` pins
the implementation to them (TP=492, TN=504, FP=1, FN=3 → 99.6% / 99.8% / 99.4%).
The arithmetic was never the problem; the split that produced those counts was.

---

## 6. Threshold and operating-point analysis

| | |
|---|---|
| **Code** | `metrics.threshold_sweep`, `metrics.select_threshold_for_sensitivity` |
| **Thesis section** | New subsection in Ch. 12 |
| **Research question** | How does the decision threshold trade sensitivity against specificity, and what operating point is clinically appropriate? |
| **Result** | Threshold → sensitivity/specificity/PPV/NPV table; selected threshold for ≥95% sensitivity |
| **Figure** | `<model>_threshold.png`; operating point marked on the ROC curve |
| **Table** | Operating points at several thresholds |
| **Discussion** | A missed malignancy costs more than an unnecessary follow-up, so the threshold is chosen rather than left at 0.5. This is how medical AI is actually reported. |

---

## 7. Calibration

| | |
|---|---|
| **Code** | `metrics.expected_calibration_error`, `plots.calibration_figure` |
| **Thesis section** | New subsection in Ch. 12 |
| **Research question** | When the model says 90%, is it right 90% of the time? |
| **Result** | ECE and per-bin reliability data |
| **Figure** | Reliability diagram |
| **Discussion** | Softmax outputs are typically overconfident. Displaying a probability in the UI creates an obligation to show whether it means anything. The UI labels confidence `calibrated: false` and uses coarse bands rather than a false-precision percentage. |

---

## 8. Error analysis

| | |
|---|---|
| **Code** | `src/evaluation/errors.py` |
| **Thesis section** | New chapter |
| **Research question** | Are false negatives systematically smaller, darker or lower-contrast than correctly classified cases? |
| **Command** | `python scripts/evaluate.py --model <name>` (runs automatically) |
| **Result** | TP/TN/FP/FN counts; Cohen's *d* per image property between correct and incorrect predictions; confidence separation; abstention curve |
| **Figure** | Grad-CAM panel over false negatives; abstention coverage/accuracy curve |
| **Table** | Image property × correct mean × incorrect mean × Cohen's *d* × effect size |
| **Discussion** | This is what makes error analysis a research contribution rather than a gallery: it produces a falsifiable claim about *which* images fail. The abstention curve models a realistic workflow — decide when confident, refer otherwise. |

No p-values are reported: with ~100 test images and several properties examined,
effect sizes are the honest summary and avoid multiple-comparison problems.

---

## 9. Explainability (Grad-CAM)

| | |
|---|---|
| **Code** | `src/explainability/gradcam.py` |
| **Thesis section** | Ch. 5.5 (Model explainability) and Ch. 13.1 — which currently states Grad-CAM was *not* integrated |
| **Research question** | Does the model attend to the lesion, or to acquisition artifacts? |
| **Result** | Heatmaps per model; optionally pointing-game hit rate and IoU against BUSI masks |
| **Figure** | Grad-CAM before/after the leakage fix; Grad-CAM across architectures; Grad-CAM over false negatives |
| **Table** | Feature-map resolution by architecture (spatial precision limit) |
| **Discussion** | Why Grad-CAM over SHAP/LIME for ultrasound; why the 64×64 model's 6×6 map is too coarse to be clinically meaningful; why a heatmap is attention, not segmentation. |

⚠️ Expect the *leaked* model's Grad-CAM to look poor — possibly attending to
background or artifacts. **That is a result, not a failure**, and it is direct
evidence for the leakage argument. Capture before/after.

---

## 10. Dataset analysis and shortcut detection

| | |
|---|---|
| **Code** | `src/data/analysis.py` |
| **Thesis section** | Ch. 7.4 (Data quality and bias handling) |
| **Experiment** | Class distribution; dimension and intensity distributions by class; class-correlated artifact check |
| **Result** | Per-class counts and statistics; Cohen's *d* for area, mean intensity and aspect ratio between classes |
| **Figure** | `dataset_class_distribution.png`, `dataset_dimensions.png`, `dataset_intensity.png`, `dataset_samples.png`, `dataset_augmentation.png` |
| **Table** | Class × count × size range × mean intensity |
| **Discussion** | If a trivial image property separates the classes, a CNN can score well by reading an acquisition shortcut rather than pathology. The legacy dataset contained two distinct image sizes within one class folder — exactly the signal worth checking. |

---

## 11. Application and structured reporting

| | |
|---|---|
| **Code** | `app.py`, `src/inference/`, `src/reporting/` |
| **Thesis section** | Ch. 9 (System design), Ch. 11 (Implementation), Ch. 11.6 (Ethics) |
| **Result** | Working system: upload → validation → prediction → confidence → Grad-CAM → structured report |
| **Figure** | Architecture diagram; annotated screenshot of the result view |
| **Table** | API endpoints; security controls implemented vs not implemented |
| **Discussion** | Moving from "Yes Breast Cancer" to a probability statement with an explicit threshold, named model, validated metrics and stated limitations. |

Ch. 11.6 claims HIPAA/GDPR compliance, HTTPS, session management and
anonymisation. **None of that is implemented.** The README lists implemented vs
not-implemented controls; the thesis must match it.

---

## 12. Reproducibility

| | |
|---|---|
| **Code** | `configs/`, `src/config.py`, metadata artifacts, `tests/` |
| **Thesis section** | New appendix |
| **Result** | Every figure and table regenerable from a documented command sequence |
| **Table** | Script → output artifact |
| **Discussion** | Config inheritance as the mechanism for a controlled comparison; seeded runs; stale-split detection; 97 automated tests. |

---

## Claims requiring correction before submission

From the earlier audit, verified against the code:

| Ch. | Claim | Status |
|---|---|---|
| Abstract, 10.1 | Custom CNN "outperformed ResNet and VGG" | **No experiment existed.** Now runnable — run it or remove the claim |
| 8.1 | 224×224 resizing | Was 64×64. Now configurable; `standard_224` is the default |
| 8.2 | CLAHE, median filter, anisotropic diffusion | **Not implemented.** Rewrite, or implement and ablate |
| 8.3 | Rotation/flip/zoom/shift augmentation | **Was absent.** Now implemented, train-split only |
| 8.4, 10.3 | SMOTE, oversampling, class weighting | Only class weighting is implemented. Remove the other two |
| 10.2 | ImageNet "statistical priors" in early layers | **Not a real technique.** Replace with actual transfer learning (`resnet50`, `efficientnetb0`) |
| 10.2 | Batch normalisation | Was absent. Now present in `custom_cnn` |
| 10.4 | 70/15/15 stratified split, 5-fold CV | Was 80/20, unstratified, test used as validation. Split now correct; CV helper exists (`stratified_group_kfold`) but no CV script yet |
| 10.5 | Grid/random/Bayesian hyperparameter search | **Not implemented.** Remove, or run a small documented grid |
| 9.1, 9.3 | Confidence score returned to the frontend | **Was discarded.** Now returned and displayed |
| 9.3, 11.6 | HTTPS, logging, HIPAA/GDPR | Not implemented. Server-side logging now exists; the rest must be removed from the text |
| 7.2, 7.3 | 5,700 images, radiologist-annotated, masks, CSV metadata | Unverifiable for the old dataset. BUSI's provenance is citable |
| 13.1 | Grad-CAM "not yet integrated" | **Now integrated** — update this section |

---

## Suggested chapter additions

1. **Dataset Integrity and Leakage Analysis** (new) — items 1 and 10
2. **Comparative Architecture Evaluation** (extends Ch. 10) — item 4
3. **Explainability** (extends Ch. 5.5) — item 9
4. **Error Analysis** (new) — item 8
5. **Calibration and Operating-Point Selection** (extends Ch. 12) — items 6 and 7
6. **Reproducibility** (appendix) — item 12

Also: the thesis references "the sixth research question" in Ch. 13.1, but **no
research questions are stated anywhere in the document.** Add an explicit list
in the introduction, or remove the reference.
