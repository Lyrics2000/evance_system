# KSL Healthcare Translation Project — What the Notebooks Do

## The project in one line

A research/thesis pipeline for **Kenyan Sign Language (KSL) healthcare communication**. Because no KSL dataset exists yet, the notebooks use the public **WLASL** (Word-Level American Sign Language) dataset as a stand-in to benchmark five sign-recognition architectures, then build the language and validation layers that a real KSL system would need.

**Target end system (bidirectional):**

```
Deaf patient → video → landmarks → recognition model → gloss → healthcare sentence → doctor
Doctor → text/speech → healthcare sentence → KSL gloss → sign clips / avatar → patient
```

Notebooks 01–05 build the left-to-right direction's recognition stage. 06 builds gloss→text. 07 builds human validation. 08 designs the reverse direction.

## Environment

- `docker-compose.yml` runs two services: **MongoDB** (FiftyOne's backing store) and a **Jupyter** container exposing `:8888` (Lab) and `:5151` (FiftyOne app).
- `./dataset` is mounted at `/workspace/dataset`, so all model artifacts survive container restarts.
- Stack: FiftyOne + HuggingFace (data), MediaPipe Holistic (features), OpenCV (video I/O), scikit-learn / hmmlearn / PyTorch (models).
- Notebooks live in `dataset/` and `dataset/ipynb_checkpoints/`. **The most complete versions of 02–08 are in `dataset/ipynb_checkpoints/`** — the top-level `RandomForest.ipynb` is a truncated copy.

---

## Shared backbone (repeated in notebooks 01–05)

Every recognition notebook follows the same skeleton:

1. Create `ksl_project_data/{logs,features,models}`, configure timestamped file+stdout logging.
2. Load `Voxel51/WLASL` from the local FiftyOne DB if present, else pull from the HF hub; mark `persistent = True`.
3. `get_label_from_sample()` — defensive label lookup that probes `ground_truth`, `label`, `word`, `gloss`, etc., because field names vary by dataset version.
4. Select a small class subset and cap samples per class.
5. **MediaPipe Holistic** per frame → flatten pose (33) + left hand (21) + right hand (21) + face (468) landmarks × (x,y,z) = **1,629 floats per frame**. Sampled at `max_frames=60`, `frame_stride=2`. Missing body parts are zero-filled.
6. Cache extracted features to `features/*.joblib` / `*.npz` so the slow MediaPipe pass runs once.
7. Stratified 75/25 split, train, then report accuracy + weighted precision/recall/F1 + classification report + confusion matrix.
8. Save model, label encoder, scaler, and a `*_metadata.json` with the metrics.

---

## Notebook-by-notebook

### `Download_images.ipynb` — data acquisition (notebook 01a)

Loads WLASL through FiftyOne, sets it persistent, then **audits the local cache**: walks every sample, counts how many `filepath`s actually exist on disk vs. are missing, prints up to 20 missing examples, previews 5 samples, and launches the FiftyOne web app for visual inspection. This exists because the metadata downloads fine even when the video files don't.

### `RandomForest.ipynb` — traditional non-temporal baseline (01)

Collapses each video into **one fixed-length vector** by taking mean/std/min/max across frames of the 1,629-dim landmark stream → 6,516 features. Trains `RandomForestClassifier(n_estimators=300, class_weight="balanced")`.

- 6 classes, 79 videos → **accuracy 0.350, weighted F1 0.312**
- Deliberate weakness: averaging destroys movement order, which is exactly what a sign is.

### `02_hmm_wlasl_baseline.ipynb` — traditional temporal baseline

Keeps each video as a frame sequence. Applies `StandardScaler` + **PCA to 30 components** (HMMs can't handle 1,629 dims), then trains **one 5-state diagonal-covariance `GaussianHMM` per class**. Prediction = argmax of per-class log-likelihood.

- 5 classes, 67 sequences → **accuracy 0.235, weighted F1 0.090**
- The weakest model; precision of 0.055 means it collapsed onto one or two classes.

### `03_lstm_wlasl_landmark_baseline.ipynb` — deep temporal baseline

PyTorch **bidirectional 2-layer LSTM** (hidden 128, dropout 0.3) over the 60×1,629 sequence; final forward+backward hidden states concatenated into a 128-unit MLP head. Adam @ 1e-3, weight decay 1e-4, 30 epochs, batch 8. Plots train/test loss and accuracy curves to make overfitting visible.

- 5 classes, 67 videos → **accuracy 0.353, weighted F1 0.297**

### `04_stgcn_...baseline.ipynb` — graph spatial-temporal model

The most involved notebook. Instead of a flat vector, each frame is a **skeleton graph of 75 nodes** (33 pose + 21 + 21 hands; face dropped as too large) × 3 coordinates. It hand-builds the adjacency matrix from MediaPipe's pose and hand edge lists, adds self-loops, and symmetrically normalizes it.

Custom `STGCNBlock`: spatial graph convolution via `einsum("nctv,vw->nctw", x, A)` + 1×1 conv, then a temporal conv (kernel 9 along time), BatchNorm/ReLU/Dropout, with a residual path. Three blocks (3→64→128→128), global average pool, linear classifier. Data is transposed to `N×C×T×V` for PyTorch.

- 5 classes, 67 videos → **accuracy 0.412, weighted F1 0.293**
- Notably, no `torch-geometric` dependency — the graph conv is written by hand.

### `05_transformer_wlasl_landmark_baseline.ipynb` — attention baseline

Linear projection 1,629 → `d_model=128`, sinusoidal **positional encoding**, `nn.TransformerEncoder` (2 layers, 4 heads, FF 256, GELU, dropout 0.3), mean-pool over time, LayerNorm + linear head. Uses a hardcoded `FIXED_CLASSES = ["before","computer","cool","cousin","drink"]` to keep comparison fair.

- **accuracy 0.529, weighted F1 0.482 — the best recognition model in the study.**

### `06_gloss_to_healthcare_text_translation_baseline.ipynb` — language layer

Pivot point: recognition output (a gloss like `pain where`) → a natural sentence (`Where is the pain?`).

- Hand-authors a **20-phrase healthcare dataset** with `domain` (triage, emergency, medicine, pharmacy, follow-up, reception, assessment, diagnosis) and `priority` (critical/important/normal).
- **Model 1 — `ExactRuleBasedTranslator`**: normalized-gloss → sentence dictionary lookup.
- **Model 2 — `FuzzyTokenTranslator`**: Jaccard token-overlap against every stored gloss, threshold 0.5.
- Evaluates both against **simulated noisy recognition output** (reordered words, dropped words, wrong glosses).
- Result: **exact 0.0, fuzzy 1.0** — the headline argument that order-insensitive matching is required downstream of an imperfect recognizer.
- Also emits T5-format `input_text`/`target_text` CSVs (`t5_train.csv`, `t5_test.csv`) and a `future_language_model_plan.json` recommending **mT5-small** for eventual English/Kiswahili/KSL fine-tuning.

### `07_healthcare_phrase_validation_matrix.ipynb` — human validation instruments

No ML. Generates the paperwork for expert review, which is what makes this a healthcare project rather than a benchmark:

- **Validation matrix** — each phrase tagged with risk level (critical→high, mandatory review), clinical use case, and four review columns: KSL interpreter, deaf user, healthcare worker, ICT/AI expert.
- **Expert questionnaire** — Likert 1–5 items across Relevance, KSL Localization, and other sections, routed to five reviewer groups.
- **Scoring template**, plus **synthetic dummy responses** used only to demonstrate the analysis path (mean, std, % scoring ≥4, per-item and per-section rollups). Files are explicitly named `*_dummy.csv` and the metadata carries a warning not to report them as real evidence.
- **`validation_status_rules.json`** — thresholds for accepted / accepted-with-revision / rejected.

### `08_text_to_sign_generation_design.ipynb` — reverse direction

Mirrors 06 backwards: `healthcare text → KSL gloss → sign assets`.

- `ExactTextToGlossMapper` and `FuzzyTextToGlossMapper` (Jaccard, threshold 0.45).
- **Sign asset dictionary**: extracts every unique gloss token and maps it to placeholder paths (`assets/signs/<token>.mp4`, `assets/avatar_motions/<token>.json`) with `availability_status: placeholder_needed` — a pragmatic MVP that plays clip sequences instead of generating animation.
- `text_to_sign_pipeline()` chains exact → fuzzy → asset sequence and writes demo outputs.
- Emits a reverse T5 training set (`convert healthcare text to ksl gloss: ...`).
- **`future_diffusion_avatar_generation_design.json`** — design-only spec for a conditional diffusion motion generator producing `60 frames × 75 joints × 3 coords`, explicitly marked `design_only_not_trained`.
- **`bidirectional_webapp_api_design.json`** — `POST /api/sign-to-text/` and `POST /api/text-to-sign/` contracts for a future FastAPI/Django backend.

---

## Results summary

| # | Model | Type | Classes | Videos | Accuracy | Weighted F1 |
|---|---|---|---:|---:|---:|---:|
| 01 | Random Forest | Non-temporal ML | 6 | 79 | 0.350 | 0.312 |
| 02 | Gaussian HMM | Temporal probabilistic | 5 | 67 | 0.235 | 0.090 |
| 03 | BiLSTM | Deep temporal | 5 | 67 | 0.353 | 0.297 |
| 04 | ST-GCN | Graph spatial-temporal | 5 | 67 | 0.412 | 0.293 |
| 05 | Transformer Encoder | Attention | 5 | 67 | **0.529** | **0.482** |

Ranking (Transformer > ST-GCN > LSTM ≈ RF > HMM) matches the expected literature ordering, which is the point the comparison is making.

## Honest caveats worth stating in the write-up

- **67 videos across 5 classes** (~13 per class, ~17 test samples) — these numbers are directional, not statistically meaningful. A single test video shifts accuracy by ~6 points.
- Notebook 01 uses **6 classes and 79 videos**; 02–05 use 5 classes and 67. The Random Forest number is therefore not strictly comparable to the rest.
- The classes (`before`, `computer`, `cool`, `cousin`, `drink`) are **ASL benchmark words, not healthcare KSL** — acknowledged in the notebooks.
- 06's `exact = 0.0` is by construction: every simulated input was perturbed, so exact matching could not fire. It demonstrates brittleness rather than measuring it.
- 07's scores are synthetic. No real reviewers have been run.
- 08 trains nothing — it produces design artifacts and placeholder mappings.

## Housekeeping issues in the repo

- `dataset/ipynb_checkpoints/` is a real directory holding the canonical notebooks, while `dataset/ipynb_checkpoints/.ipynb_checkpoints/` holds Jupyter's autosaves. Duplicate `*-checkpoint-checkpoint.ipynb` files exist. Worth flattening before submission.
- Notebook 04's filename is doubled: `04_stgcn_wlasl_landmark_baseline04_stgcn_wlasl_landmark_baseline.ipynb`.
- Stray files `=4.9` and `=4.9,` came from a malformed `pip install` (unquoted version specifier).
- Two `requirements.txt` files disagree: root pins `numpy==1.26.4` / `mediapipe==0.10.21`; `dataset/requirements.txt` is a full `pip freeze` with `numpy==2.4.6` and no mediapipe. The root one is the intended environment.
- Numbering is inconsistent: `RandomForest.ipynb` is referred to as notebook 01 inside 06/07, but `Download_images.ipynb` is also a step-one notebook.
- `.env` is committed alongside `.env.example` — currently harmless (no secrets, only FiftyOne host/port), but should be gitignored.
