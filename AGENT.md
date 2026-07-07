# AGENT.md

## 1. Project Overview
This project is a **generic, modular, hybrid deep clustering system** for tabular customer data.

Goal:
- Build a Streamlit-based testing system (UI only for debugging/inference visualization)
- Perform end-to-end pipeline:
  ingestion → preprocessing → embedding → autoencoder → clustering → evaluation

The system must be:
- dataset-agnostic (NOT overfitted to any dataset)
- modular
- configurable
- robust for large-scale data (up to ~1M rows)

---

## 2. Core Architecture Principle

### 2.1 Modular Design
Each pipeline stage MUST be implemented as an independent module:

- ingestion
- schema_detection
- missing_handler
- outlier_handler
- preprocessing
- embedding
- autoencoder
- clustering
- evaluation

Each module must:
- have clear input/output contracts
- be independently testable
- be replaceable without affecting other modules

---

### 2.2 Hybrid Pipeline Strategy

The system uses a **HYBRID PIPELINE**:

#### Default pipeline (recommended flow):
ingestion →
schema_detection →
missing_handler →
outlier_handler →
preprocessing →
embedding →
autoencoder →
clustering →
evaluation

#### Flexibility rules:
- Modules can be SKIPPED
- Modules can be REORDERED (within logical constraints)
- Modules can be REPLACED

However:
- Autoencoder must always come BEFORE clustering
- Embedding must occur BEFORE autoencoder if text features exist

---

## 3. Execution Mode (Codex Behavior)

Codex must operate in HYBRID mode:

### Mode C - Hybrid Execution:
- Default: implement one module at a time (safe mode)
- Allowed: batch implementation of multiple modules when dependencies are clear
- Must always preserve module independence

---

## 4. Embedding System

- Embedding module located at:
  `Embeddings/embeddings.py`

Specifications:
- Model: intfloat/multilingual-e5-small
- Supports batch processing
- GPU-enabled
- Priority: accuracy > speed

Embedding must be:
- reusable
- dataset-agnostic
- plug-in component of pipeline

---

## 5. Autoencoder Module

Requirements:
- GPU training supported
- auto architecture selection (layer tuning)
- latent dimension optimization
- used for representation learning only (NOT reconstruction output usage)

Flow:
raw features → encoder → latent space → clustering

---

## 6. Clustering Module

- Primary algorithm: KMeans
- Auto-K selection:
  - Elbow method
  - Silhouette score

Must support:
- recomputation after parameter changes
- evaluation feedback loop

---

## 7. Error Handling Policy (CRITICAL)

All modules MUST implement:

- try/except wrapping
- stage-based logging
- safe fallback behavior

### Hybrid error strategy:
- Recoverable error → log + fallback + continue
- Critical error → stop current stage only (not full system)

No silent failures allowed.

---

## 8. Configuration Philosophy

- All behavior must be parameterized
- No hard-coded strategies
- Every module must expose:
  - default parameters
  - override options

Example:
missing_strategy="mean"
outlier_method="iqr"

---

## 9. Logging System

Each module must log:
- stage name
- input status
- output status
- errors (if any)

Format:
[STAGE] - message

---

## 10. Streamlit UI (Debug Mode Only)

UI is NOT production-facing.

Only used for:
- uploading dataset
- running pipeline
- inspecting intermediate outputs

No requirement for UI optimization at this stage.

---

## 11. Development Workflow (IMPORTANT)

Codex MUST follow:

1. Read AGENT.md first
2. Identify current pipeline stage
3. Implement module in isolation
4. Test module independently
5. Integrate into pipeline
6. Update AGENT.md progress section

---

## 12. Project Philosophy

- Build a reusable ML pipeline framework
- NOT a single-dataset solution
- Prioritize modularity, flexibility, and robustness
- Human-in-the-loop decisions for cleaning steps

## 13. Language
All user-facing interfaces MUST be in Vietnamese.

This includes:
- Streamlit UI text
- user messages
- warnings and results displayed to users

Internal system components must remain in English:
- logs
- code comments (optional)
- module names
- debugging output

---

## 14. Progress

- [x] ingestion
  - Implemented standalone module: `pipeline/ingestion.py`
  - Supports dataset-agnostic tabular ingestion from `CSV`, `XLSX`, and `JSON`
  - Accepts file paths, file-like objects, raw bytes, and inline text when format is provided
  - Includes stage-scoped logging, typed config/result contracts, fallback CSV decoding and delimiter handling, and metadata generation
  - Smoke-tested successfully with CSV, JSON, and file-like inputs
  - Excel dependency path validated; full `.xlsx` loading requires an installed engine such as `openpyxl` in the runtime environment

- [x] schema_detection
  - Implemented standalone module: `pipeline/schema_detection.py`
  - Accepts `DataFrame` output from ingestion without modifying ingestion logic
  - Infers `numeric`, `categorical`, `text`, and safe fallback `unknown` types with configurable thresholds
  - Computes per-column metadata including missing ratio, unique ratio, sample values, and inference confidence
  - Returns a structured schema object for downstream modules and logs stage summary plus inference issues
  - Smoke-tested on provided datasets and synthetic mixed-type data
- [x] missing_handler
  - Implemented standalone module: `pipeline/missing_handler.py`
  - Accepts `DataFrame` plus schema output and keeps ingestion/schema detection logic unchanged
  - Generates a structured missing-value preview report with per-column missing ratio, affected columns, planned strategy, and fallback events
  - Supports configurable per-type strategies for `numeric`, `categorical`, and `text`, with optional column overrides and simulated confirmation when no UI is active
  - Applies safe fallback behavior for recoverable failures, including `drop_rows` and `unknown` fill strategies
  - Smoke-tested on synthetic edge cases and integrated successfully with schema-derived real dataset handling
- [x] outlier_handler
  - Implemented standalone module: `pipeline/outlier_handler.py`
  - Accepts `DataFrame` plus schema output and applies IQR-based outlier detection only to schema-identified numeric columns
  - Generates a structured preview report with per-column bounds, outlier counts, percentages, and sample affected rows before any changes are applied
  - Supports configurable `cap`, `drop`, `group`, and `ignore` strategies with per-column overrides, confirmation simulation, and configurable IQR threshold
  - Handles recoverable per-column failures by logging warnings and falling back to safe skip or ignore behavior
  - Smoke-tested on synthetic strategy cases and integrated successfully with the ingestion → schema → missing → outlier flow
- [x] preprocessing
  - Implemented standalone module: `pipeline/preprocessing.py`
  - Accepts cleaned `DataFrame` plus schema output and builds a shape-safe numeric feature matrix without modifying prior modules
  - Scales numeric columns with configurable `standard`, `minmax`, or `robust` scalers and encodes categorical columns with configurable `onehot` or `label` encoding
  - Preserves text columns as a separate raw payload for the embedding stage and records all column transformations in a feature map
  - Handles recoverable per-column failures by skipping columns with warnings and falls back from high-cardinality one-hot expansion to label encoding when configured
  - Smoke-tested on synthetic mixed-type data and integrated successfully with the ingestion → schema → missing → outlier → preprocessing flow
- [x] embedding
  - Implemented standalone module: `pipeline/embedding.py`
  - Accepts `X_text` from preprocessing plus preprocessing feature metadata without modifying upstream modules or `Embeddings/embeddings.py`
  - Supports configurable `concat_text` and `concat_vector` merge strategies for multiple text columns and preserves row alignment with numeric-safe embedding matrices
  - Uses the existing GPU-enabled `EmbeddingEngine` as the integration boundary, passes configurable batch settings through, and logs stage start, text-column counts, batch activity, output shape, and skipped cases
  - Handles no-text payloads by returning an empty shape-safe matrix, supports configurable empty-text replacement, and falls back to row-level zero-vector recovery for recoverable embedding failures
  - Smoke-tested with no-text skip handling and synthetic multi-column merge scenarios using a stub embedding engine; full runtime execution in the current environment still requires installed embedding dependencies such as `torch`
- [x] autoencoder
  - Implemented standalone module: `pipeline/autoencoder.py`
  - Accepts `X_numeric` plus optional `X_embedding`, merges available feature sources along the feature axis, and validates row alignment without modifying earlier stages
  - Builds a dataset-agnostic symmetric autoencoder with automatically selected latent dimension and decreasing hidden layers derived from input dimensionality, while still allowing configurable overrides
  - Trains with configurable batched `Adam` + `MSE` on GPU when available through lazy-loaded PyTorch runtime, logs stage start, selected architecture, latent size, training setup, per-epoch loss, and final loss
  - Handles recoverable architecture and training issues with a simpler fallback architecture, sanitizes non-finite inputs and latent outputs, and returns stable `Z_latent`, the trained model, plus structured training metadata
  - Smoke-tested for package import safety without eager `torch` dependency, numeric/embedding feature merging, non-finite replacement, auto-architecture selection, and embedding-only input handling; full model training in the current environment still requires installed PyTorch
- [x] clustering
  - Implemented standalone module: `pipeline/clustering.py`
  - Accepts `Z_latent` from the autoencoder stage, validates shape and finite numeric values, and keeps all earlier modules unchanged
  - Uses `KMeans` as the primary algorithm with configurable `k_min`/`k_max`, `random_state`, `n_init`, and `max_iter`
  - Evaluates each valid K with both inertia and silhouette score, selects the highest-silhouette candidate as primary, and uses an elbow heuristic as secondary validation when the silhouette gap is within a configurable tolerance
  - Returns shape-safe `cluster_labels`, `best_k`, structured clustering metadata, and a stage report with tested K values, per-K scores, selected K, and warnings
  - Smoke-tested on synthetic multi-cluster latent data, invalid-input cases, and package import flow; full clustering requires installed `scikit-learn`
- [x] evaluation
  - Implemented standalone module: `pipeline/evaluation.py`
  - Accepts `Z_latent` plus `cluster_labels`, validates row alignment, finite numeric values, and requires at least 2 distinct clusters before computing quality metrics
  - Computes silhouette score as the primary metric, cluster distribution for per-cluster counts, and optional Davies-Bouldin plus Calinski-Harabasz scores for additional quality checks
  - Produces visualization-ready output with optional t-SNE reduction to `Z_2d`, preserves cluster labels for plotting, and adjusts perplexity with warning logs when the requested value exceeds the valid sample-size limit
  - Returns structured evaluation metrics, cluster distribution, visualization data, and a stage report with metric values, t-SNE settings, and warnings
  - Smoke-tested on synthetic clustered latent features, package import flow, and edge cases including single-cluster rejection; full evaluation requires installed `scikit-learn`



## 15. Streamlit UI (Post-Pipeline Stage)

The Streamlit UI is implemented ONLY after all core pipeline modules are completed.

Purpose:
- Upload dataset (CSV / XLSX / JSON)
- Execute pipeline end-to-end
- Display intermediate outputs (optional)
- Visualize clustering results

Scope:
- Simple interface (debug-focused)
- NOT production-grade UI
- Focus on functionality over design

UI Features:
- File uploader
- Run pipeline button
- Display:
  - cluster distribution
  - evaluation metrics
  - t-SNE visualization

Rules:
- UI must NOT interfere with pipeline logic
- UI must call pipeline as a black-box
- All user-facing text must be in Vietnamese
---

END OF AGENT SPEC
