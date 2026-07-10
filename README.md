# CLM-REM-protein

Predicting small-molecule binding affinity to REM sleep-regulatory protein targets using chemical language models (CLMs).

**Live demo:** [REM Sleep Selectivity Profiler](https://hannadmochowska-rem-sleep-profiler.hf.space) — paste a SMILES string and get a predicted binding affinity profile across six REM sleep-regulatory targets.

## Overview

REM sleep is regulated by cholinergic, GABAergic, orexinergic, and adenosinergic neurotransmitter systems. This project evaluates whether inference-only embeddings from MoLFormer-XL, paired with a per-target MLP regressor, can predict small-molecule binding affinity (pChEMBL) across six of the receptors that govern this circuitry: **CHRM2, CHRM4, GABRA1, HCRTR1, HCRTR2, and ADORA1**, using bioactivity data from ChEMBL v36 (N = 24,011 compound-target pairs).

Key findings:
- Inference-only MoLFormer-XL + MLP exceeded a Spearman R of 0.56 on four of six targets under a strict scaffold-aware (CCPart) split. End-to-end fine-tuning and a Morgan fingerprint baseline both collapsed under the same split.
- A leakage-free, panel-wide specificity matrix showed that cross-target transfer tracks known receptor pharmacology: CHRM2/CHRM4 (conserved orthosteric pocket) cross-predict strongly; HCRTR1/HCRTR2 (subtype-divergent pharmacology despite sharing most of their compound libraries) do not.
- Six reference compounds were used for pharmacological validation: four target-coverage ligands (suvorexant, scopolamine, diazepam, DPCPX) each correctly ranked their intended primary target highest (4/4), and two orexin subtype-selective antagonists (SB-334867, JNJ-10397049) correctly discriminated HCRTR1/HCRTR2 selectivity in both directions (2/2).

Full scientific methodology, results, and discussion are in the project report, [`REM_CLM_Paper.docx`](./REM_CLM_Paper.docx) (to be added to [`docs/`](./docs)). This README covers everything needed to run, reproduce, or extend the code.

## Final per-target performance

| Target | Spearman R | Pearson R | RMSE |
|---|---|---|---|
| HCRTR1 | 0.676 | 0.733 | 0.668 |
| HCRTR2 | 0.632 | 0.634 | 0.749 |
| CHRM2  | 0.615 | 0.604 | 1.475 |
| GABRA1 | 0.615 | 0.612 | 0.879 |
| ADORA1 | 0.385 | 0.396 | 0.955 |
| CHRM4  | 0.187 | 0.305 | 1.059 |

5-seed mean, cross-validation-selected configuration, CCPart scaffold split. CHRM4 predictions in particular should be treated as the least reliable of the six (see report, Limitations).

## Repository structure

```
.
├── app.py                          # Streamlit web app (REM Sleep Selectivity Profiler)
├── predict.py                      # Inference pipeline: SMILES validation, MoLFormer-XL embedding, prediction
├── requirements.txt                # Pinned dependencies
├── LICENSE                         # MIT license (code only; see License section below)
├── .streamlit/
│   └── config.toml                 # App theme/server config
│
├── HCRTR1/                         # Pilot: representation/fine-tuning comparison on HCRTR1 only
│   ├── ChemBERTa_2+MLP.ipynb
│   ├── ChemBERTa_2_fine_tuning.ipynb
│   ├── MoLFormer_XL+MLP.ipynb
│   ├── MoLFormer_XL_fine_tuning.ipynb
│   ├── MoLFormer_XL_fine_tuning_lr1e7_frozen6_5ep.ipynb    # LR=1e-7, 6 layers frozen, 5 epochs
│   ├── MoLFormer_XL_fine_tuning_lr2e5_frozen6_12ep.ipynb   # LR=2e-5, 6 layers frozen, 12 epochs
│   └── morgan_fp_baseline_HCRTR1.ipynb
│
├── MolFormer(FineTuned)/           # Final locked per-target hyperparameter tuning + specificity matrix
│   ├── MoLFormer_XL+MLP_ADORA1_mlp_tuning.ipynb
│   ├── MoLFormer_XL+MLP_CHRM2_mlp_tuning.ipynb
│   ├── MoLFormer_XL+MLP_CHRM4_mlp_tuning.ipynb
│   ├── MoLFormer_XL+MLP_GABRA1_mlp_tuning.ipynb
│   ├── MoLFormer_XL+MLP_HCRTR1_mlp_tuning.ipynb
│   ├── MoLFormer_XL+MLP_HCRTR2_mlp_tuning.ipynb
│   ├── config_log{ADORA1,CHRM2,CHRM4,GABRA1,HCRTR1,HCRTR2}.csv   # per-target tuning logs
│   ├── specificity_matrix.py       # 6x6 cross-target specificity matrix (global max_dissimilarity_2 split)
│   └── specificity_matrix.csv      # output matrix (matches report Table 4)
│
├── joblib files/                   # Deployed model artefacts (mirrors the tuning notebooks/logs above)
│   ├── MoLFormer_XL+MLP_{TARGET}_mlp_tuning.ipynb   # copies of the 6 notebooks above
│   ├── config_log{TARGET}.csv                        # copies of the 6 logs above
│   └── mlp_{ADORA1,CHRM2,CHRM4,GABRA1,HCRTR1,HCRTR2}.joblib   # 6 trained MLP regressors loaded by predict.py
│
├── train:test/                     # Final CCPart train/test CSVs per target (12 files)
│
├── docs/                           # Final project report (added once finalised; currently placeholder)
│
└── exploratory analysis/           # Early/superseded work, not part of the final pipeline
    ├── CLM_REM_protein-ChemBERTa-1.ipynb      # Early exploratory notebook
    ├── EGFR_MolFormer_classification_template.ipynb   # EGFR binary classification, not this project's REM targets or regression task
    └── MoLFormer_XL_with_full_fine_tuning_.ipynb
```

## Setup

```bash
git clone https://github.com/hannadmochowska/CLM-REM-protein.git
cd CLM-REM-protein
pip install -r requirements.txt
```

## Running the web app locally

```bash
streamlit run app.py
```

This loads the six trained models from `joblib files/` and MoLFormer-XL (`ibm/MoLFormer-XL-both-10pct`) via HuggingFace Transformers.

## Pipeline

The project report explains *why* this target panel and hypothesis were chosen and *what the results mean*. This section explains *how* the pipeline actually works, stage by stage, mapped to the files in this repo.

```
ChEMBL v36 SQLite  →  per-target extraction (SQL)  →  dedup + pChEMBL≥5 filter
        →  representation/model selection (HCRTR1 pilot)
        →  CCPart scaffold split  →  hyperparameter tuning (leakage-corrected via 5-fold CV)
        →  6 locked per-target MLP models
        →  global max_dissimilarity_2 split  →  6x6 specificity matrix
        →  predict.py (inference pipeline)  →  app.py (Streamlit UI)  →  HuggingFace Spaces
```

### 1. Data extraction

Records were pulled from a local copy of the ChEMBL v36 release (SQLite format), obtained once from the [EBI ChEMBL FTP mirror](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest). There is no single canonical download script in this repo — each notebook/script queries the local database directly:

```sql
SELECT cs.canonical_smiles, act.pchembl_value
FROM activities act
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
JOIN compound_structures cs ON act.molregno = cs.molregno
WHERE td.chembl_id = '{chembl_id}'
  AND act.pchembl_value IS NOT NULL
  AND act.standard_relation = '='
```

pChEMBL is calculated by ChEMBL principally for Ki, Kd, and IC50 assay types. Target ChEMBL IDs:

| Target | ChEMBL ID |
|---|---|
| ADORA1 | CHEMBL226 |
| CHRM2  | CHEMBL211 |
| CHRM4  | CHEMBL1821 |
| GABRA1 | CHEMBL1962 |
| HCRTR1 | CHEMBL5113 |
| HCRTR2 | CHEMBL4792 |

### 2. Preprocessing

Per target: duplicate compound-target pairs are collapsed by taking the **median pChEMBL per canonical SMILES**, then rows below **pChEMBL ≥ 5** are dropped. Final dataset sizes: GABRA1 = 357, CHRM4 = 1,816, CHRM2 = 2,316, ADORA1 = 4,951, HCRTR1 = 7,090, HCRTR2 = 7,481 (24,011 total).

### 3. Representation & model selection (`HCRTR1/`)

Before locking a final architecture, four strategies were compared on a single pilot target (HCRTR1), since this is itself a test of the memorisation-vs-generalisation hypothesis:

| Notebook | What it tests |
|---|---|
| `ChemBERTa_2+MLP.ipynb` | ChemBERTa-2 (DeepChem/ChemBERTa-77M-MLM), inference-only embeddings + MLP head |
| `ChemBERTa_2_fine_tuning.ipynb` | ChemBERTa-2, full end-to-end fine-tuning |
| `MoLFormer_XL+MLP.ipynb` | MoLFormer-XL, inference-only embeddings + MLP head (the eventual winner) |
| `MoLFormer_XL_fine_tuning.ipynb` | MoLFormer-XL, full end-to-end fine-tuning (default hyperparameters) |
| `MoLFormer_XL_fine_tuning_lr1e7_frozen6_5ep.ipynb` | MoLFormer-XL fine-tuning, conservative retry: LR=1e-7, bottom 6 layers frozen, 5 epochs |
| `MoLFormer_XL_fine_tuning_lr2e5_frozen6_12ep.ipynb` | MoLFormer-XL fine-tuning, retry: LR=2e-5, bottom 6 layers frozen, 12 epochs |
| `morgan_fp_baseline_HCRTR1.ipynb` | Classical baseline: Morgan/ECFP fingerprints (radius 2, 2048 bits) + Ridge / random forest / MLP |

**Result:** both fine-tuned models (ChemBERTa-2 and MoLFormer-XL, across all three hyperparameter attempts) achieved strong validation Pearson R (up to 0.88) but collapsed to near-zero or negative R on the scaffold-split test set — attributed to catastrophic forgetting from fine-tuning a large pretrained transformer on only a few thousand single-target compounds. The Morgan fingerprint baseline showed the same collapse (scaffold R at or below 0 for 2 of 3 regressors). Inference-only MoLFormer-XL + MLP was the only configuration that held up under scaffold splitting (R = 0.70, vs. 0.40 for the best fine-tuned model and 0.40 for the best fingerprint model), and outperformed inference-only ChemBERTa-2 (R = 0.62). This is why inference-only MoLFormer-XL + MLP was locked in as the architecture for all six targets.

### 4. Final model architecture

- **Embedding:** MoLFormer-XL (`ibm/MoLFormer-XL-both-10pct`, 45M params, IBM Research), frozen, inference-only. Output is a 768-dimensional pooler embedding (masked mean pooling).
- **Regressor:** `sklearn.neural_network.MLPRegressor`, one independently trained per target.
- **Critical detail:** embeddings must be generated with `batch_size=1` to exactly replicate training conditions. Batching with padding alters the embedding vectors due to attention masking — this silently degrades prediction quality with no error or warning. A **PR #7 attention fix** is applied across all 12 encoder layers to correct a masking inconsistency in the upstream MoLFormer-XL implementation. Both of these matter at inference time (`predict.py`), not just training.

### 5. Scaffold-aware split (CCPart)

Each target's compounds are partitioned with structural clustering: ECFP4, Tanimoto similarity, threshold 0.4, Butina clustering via the `hestia` CCPart implementation. Whole clusters of structurally similar compounds are assigned to either train or test, so no test compound shares a scaffold neighbourhood with any training compound.

### 6. Hyperparameter tuning (`MolFormer(FineTuned)/`)

Per-target MLP hyperparameters (hidden-layer architecture, activation, L2 alpha, learning rate) were tuned in four sequential steps, 5 seeds each. **The first pass scored candidate configurations directly on the held-out CCPart test set — this is data leakage.** It was caught and the entire tuning procedure was redone using 5-fold cross-validation on the training set only, with the test set touched exactly once for final evaluation. The `MoLFormer_XL+MLP_{TARGET}_mlp_tuning.ipynb` notebooks and `config_log{TARGET}.csv` files in this repo reflect the corrected, leakage-free procedure.

Final locked configuration per target:

| Target | Architecture | Activation | Alpha | Learning rate |
|---|---|---|---|---|
| ADORA1 | (512, 256, 128) | tanh | 0.01 | 0.001 |
| CHRM2  | (512, 256, 128) | relu | 0.01 | 0.0001 |
| CHRM4  | (512, 256, 128) | relu | 0.01 | 0.0001 |
| GABRA1 | (512, 256, 128) | relu | 0.001 | 0.001 |
| HCRTR1 | (512, 256, 128) | tanh | 0.001 | 0.001 |
| HCRTR2 | (512, 256, 128) | tanh | 0.0001 | 0.0005 |

All six converged on the same hidden-layer architecture; activation, alpha, and learning rate are target-specific. The trained models from this stage are the `mlp_{TARGET}.joblib` files in `joblib files/`.

### 7. Specificity matrix (`specificity_matrix.py`)

Tests whether each target-specific model learned genuine, receptor-selective structure-activity relationships, or just shared dataset artefacts. This needs a different split than Section 5: two target pairs here share a large fraction of their compound libraries (HCRTR1 and HCRTR2 share 85.5% and 81.0% of their compounds respectively), so a per-target CCPart split isn't sufficient to guarantee a leakage-free *cross*-target comparison — a compound in one target's training set could sit in another target's test set.

Instead, `specificity_matrix.py`:
1. Pools all 17,149 unique compounds across the six target datasets.
2. Computes a full pairwise ECFP4 Tanimoto similarity matrix.
3. Builds a single global ~20% test set by **MaxMin diversity picking** (seeded by the compound least similar to all others, then iteratively extended), applied identically across all six targets.
4. Verifies **zero** compound overlap between any target's training set and any other target's test set, across all 15 pairwise combinations.
5. Retrains each of the six locked models (Section 6 configs) on its own global-split training data, then evaluates all six on all six global-split test sets to build the 6x6 matrix, saved to `specificity_matrix.csv`.

Because this is a different split from Section 5/6, the diagonal of this matrix is **not** directly comparable to each target's headline performance number — it's a smaller, differently-selected test set. The point of this matrix is the off-diagonal (cross-target) values, not the diagonal.

### 8. Inference pipeline (`predict.py`)

Loads MoLFormer-XL and the six `.joblib` MLP regressors, and exposes three functions used by the Streamlit front-end:

- `validate_smiles()` — parses and canonicalises input SMILES with RDKit, flags invalid structures instead of failing silently
- `embed()` — generates the 768-dim MoLFormer-XL embedding (`batch_size=1`, PR #7 fix applied — see Section 4)
- `predict()` — runs the six per-target MLP regressors on the embedding and returns pChEMBL predictions

### 9. Web interface (`app.py`)

Streamlit front-end with two input modes toggled by a radio button: SMILES paste (one per line) or `.txt` file upload. `st.session_state` preserves the selected input mode across reruns, since Streamlit re-executes the full script on every widget interaction. Models are cached with `@st.cache_resource` so MoLFormer-XL is loaded once per server session, not once per prediction.

Results are shown two ways:
- A colour-coded heatmap table: a custom three-stop ramp, red `[201, 72, 91]` at pChEMBL 4.5, yellow `[233, 196, 106]` at the midpoint, green `[47, 158, 99]` at pChEMBL 9.0, with automatic dark/light text based on luminance.
- An interactive Plotly bar chart: single-compound profile or grouped comparison across multiple compounds.

A download button exports predictions as CSV. The four pharmacological reference compounds used in validation (suvorexant, scopolamine, diazepam, DPCPX) are built into the interface as copy-paste example structures.

### 10. Deployment

Deployed on HuggingFace Spaces, Docker runtime, CPU Basic instance (2 vCPU, 16 GB RAM). Dependencies pinned in `requirements.txt` to avoid known incompatibilities: `transformers<5.0` (the onnx module was removed in v5), `tokenizers>=0.20.0` (required for Python 3.13 compatibility), and `matplotlib` (required by Streamlit's `background_gradient` formatter used in the heatmap table). App theme/server config: `.streamlit/config.toml`.

## Citation

Dmochowska, H. (2026). *Predicting Small-Molecule Binding Affinity to REM Sleep-Regulatory Protein Targets Using Chemical Language Models*. MSc Artificial Intelligence in Medicine, University College Dublin.

## License

Code is released under the [MIT License](./LICENSE). ChEMBL data is distributed under [CC BY-SA 3.0](https://www.ebi.ac.uk/about/terms-of-use); see [Mendez et al., 2019](https://doi.org/10.1093/nar/gky1075).

## Acknowledgments

Supervised by Denis Shields (ShieldsLab, UCD). Developed as part of the MSc Artificial Intelligence in Medicine program at University College Dublin.
