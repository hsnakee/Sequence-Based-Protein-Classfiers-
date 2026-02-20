# Protein Sequence Classifier

A production-grade, modular machine learning framework for binary classification of protein sequences using classical ML, neural networks, and transformer embeddings.
Ai Assitant used : Claude Sonnet | IDE : Pycharm

---

##  Biological Motivation

This framework was designed with real biological classification tasks in mind. For example:

**SNARE vs non-SNARE classification**: SNARE proteins mediate membrane fusion in vesicle trafficking. Identifying novel SNAREs from unannotated proteomes requires distinguishing them from structurally similar coiled-coil proteins. This tool trains classifiers on known SNARE (class A) and non-SNARE (class B) sequences, then predicts labels for unannotated sequences (class C).

Other applicable tasks:
- Antimicrobial peptide (AMP) vs non-AMP
- Signal peptide vs non-signal peptide
- Secreted vs intracellular proteins
- Enzyme vs non-enzyme binary classification
- Any binary protein family classification task

---

##  Project Structure

```
protein-seq-classifier/
├── src/
│   ├── data_loading.py        # Dataset assembly, train/val/test splitting
│   ├── fasta_utils.py         # FASTA parsing, cleaning, validation
│   ├── feature_extractors.py  # k-mer, one-hot, physicochemical, composition
│   ├── embedding_models.py    # ProtBERT, ESM-2 with disk caching
│   ├── classical_models.py    # LR, SVM, RF, GBM, XGBoost, LightGBM
│   ├── neural_models.py       # CNN, BiLSTM, Transformer (PyTorch)
│   ├── training.py            # Unified training orchestrator
│   ├── evaluation.py          # Metrics, plots, comparison tables
│   ├── ensemble.py            # Voting & stacking ensembles
│   ├── predict.py             # Prediction pipeline for new sequences
│   └── utils.py               # Logging, seeding, config, device detection
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_models.ipynb
│   ├── 04_neural_models.ipynb
│   ├── 05_transformer_embeddings.ipynb
│   └── 06_ensemble.ipynb
├── configs/
│   └── default.yaml           # All configurable parameters
├── classify.py                # CLI: classify new sequences
├── train.py                   # CLI: train classifiers
├── requirements.txt
└── README.md
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/protein-seq-classifier.git
cd protein-seq-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install neural model support

```bash
# CPU-only PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 5. (Optional) Install transformer embeddings

```bash
# ProtBERT:
pip install transformers sentencepiece

# ESM-2:
pip install fair-esm
```

---

##  Quick Start

### Training

```bash
python train.py \
  --classA data/snare.fasta \
  --classB data/non_snare.fasta \
  --config configs/default.yaml \
  --output results/
```

**Skip neural models** (faster):
```bash
python train.py \
  --classA data/snare.fasta \
  --classB data/non_snare.fasta \
  --no-neural
```

**Custom class names:**
```bash
python train.py \
  --classA data/snare.fasta \
  --classB data/non_snare.fasta \
  --name-a "SNARE" \
  --name-b "Non-SNARE"
```

### Classification

```bash
python classify.py \
  --model results/classical_RandomForest.pkl \
  --pipeline results/feature_pipeline.pkl \
  --input data/unknown.fasta \
  --output predictions.csv
```

With custom threshold:
```bash
python classify.py \
  --model results/classical_RandomForest.pkl \
  --pipeline results/feature_pipeline.pkl \
  --input data/unknown.fasta \
  --output predictions.csv \
  --threshold 0.7
```

---

##  Output Files

After training, the `results/` directory contains:

```
results/
├── feature_pipeline.pkl         # Fitted classical ML feature extractor
├── onehot_pipeline.pkl          # Fitted one-hot encoder (for neural models)
├── classical_LogisticRegression.pkl
├── classical_RandomForest.pkl
├── classical_SVM.pkl
├── classical_GradientBoosting.pkl
├── classical_XGBoost.pkl
├── classical_LightGBM.pkl
├── neural_CNN.pt
├── neural_BiLSTM.pt
├── neural_Transformer.pt
├── config.json                  # Config snapshot
├── metadata.json                # Class labels and best threshold
├── model_comparison.csv         # Full metric table
├── train.log
└── plots/
    ├── RandomForest_confusion_matrix.png
    ├── RandomForest_roc_curve.png
    ├── RandomForest_pr_curve.png
    ├── RandomForest_calibration.png
    ├── RandomForest_feature_importance.png
    ├── model_comparison_f1_weighted.png
    └── ...
```

Predictions CSV columns:
```
seq_id | description | sequence_length | predicted_class | class_name | confidence | prob_ClassA | prob_ClassB
```

---

## 📈 Model Comparison Table (Template)

| Model               | Accuracy | F1 (weighted) | ROC-AUC | PR-AUC | MCC   |
|---------------------|----------|---------------|---------|--------|-------|
| RandomForest        | 0.923    | 0.921         | 0.971   | 0.968  | 0.843 |
| XGBoost             | 0.918    | 0.917         | 0.969   | 0.963  | 0.835 |
| LightGBM            | 0.915    | 0.914         | 0.966   | 0.960  | 0.830 |
| GradientBoosting    | 0.910    | 0.908         | 0.962   | 0.955  | 0.820 |
| SVM                 | 0.905    | 0.903         | 0.956   | 0.948  | 0.810 |
| LogisticRegression  | 0.891    | 0.889         | 0.943   | 0.935  | 0.782 |
| CNN                 | 0.912    | 0.910         | 0.963   | 0.957  | 0.824 |
| BiLSTM              | 0.908    | 0.906         | 0.960   | 0.953  | 0.816 |
| Transformer         | 0.915    | 0.913         | 0.965   | 0.959  | 0.830 |
| VotingEnsemble      | 0.931    | 0.930         | 0.975   | 0.972  | 0.862 |

---

##  Feature Extractors

All extractors are pluggable and implement `fit/transform` (sklearn-compatible):

| Extractor              | Features | Description                                         |
|------------------------|----------|-----------------------------------------------------|
| `KmerExtractor`        | Variable | k-mer frequency vectors (k=2,3,4 by default)       |
| `CompositionExtractor` | 20–420   | Amino acid (and optional dipeptide) composition     |
| `PhysicochemicalExtractor` | 16–21 | Hydrophobicity, charge, MW, aromaticity statistics |
| `GlobalStatsExtractor` | 8        | Length, log length, group fractions                 |
| `OneHotExtractor`      | L×20     | Padded/truncated one-hot matrix                     |
| `ProtBERTEmbedder`     | 1024     | Pre-trained ProtBERT mean-pooled embeddings         |
| `ESMEmbedder`          | 1280     | Pre-trained ESM-2 mean-pooled embeddings            |

### Adding a New Feature Extractor

1. Subclass `BaseFeatureExtractor` in `src/feature_extractors.py`:

```python
class MyExtractor(BaseFeatureExtractor):
    def fit(self, sequences, y=None):
        # compute vocabulary / stats
        return self

    def transform(self, sequences):
        # return np.array of shape (n, n_features)
        ...

    @property
    def feature_names(self):
        return ["feat_1", "feat_2", ...]
```

2. Add it to `build_feature_pipeline()` in `feature_extractors.py`:

```python
extractors.append(("my_extractor", MyExtractor(**params)))
```

---

##  Adding New Models

### Classical Model

Add a `build_mymodel()` function in `src/classical_models.py`:

```python
def build_mymodel(config, seed=42):
    from sklearn.neighbors import KNeighborsClassifier
    cfg = config.get("classical_models", {}).get("knn", {})
    return ModelWrapper(
        name="KNN",
        estimator=KNeighborsClassifier(),
        param_grid={"n_neighbors": [3, 5, 11]},
        cv_folds=cfg.get("cv_folds", 5),
        seed=seed,
    )
```

Call it from `build_all_classical_models()`.

### Neural Model

Add a new architecture in `src/neural_models.py`, then register it in `build_all_neural_models()`.

---

##  Performance Tips

- **Small datasets (<500 sequences)**: Use classical ML only (`--no-neural`). LR + RF are fast and effective.
- **Class imbalance**: Set `class_weight: "balanced"` in config (already default). For extreme imbalance, try SMOTE or threshold tuning.
- **Long sequences (>1000 AA)**: Set `max_length` in config for neural models; or use global/composition features for classical ML.
- **Speed**: Disable models you don't need in `configs/default.yaml` by setting `enabled: false`.
- **CV folds**: Reduce `cv_folds` from 5 to 3 for faster hyperparameter search.

##  GPU Tips

PyTorch models auto-detect CUDA if available. Check status:
```python
from utils import get_device
print(get_device("auto"))
```

Force CPU mode: set `neural_models.device: "cpu"` in config.

##  Embedding Cache

Transformer embeddings are expensive. They are cached per-sequence in `.embedding_cache/`:
- **ProtBERT** → `.embedding_cache/protbert/<md5>.npy`
- **ESM-2** → `.embedding_cache/esm/<md5>.npy`

The cache survives restarts. To clear:
```python
from embedding_models import clear_embedding_cache
clear_embedding_cache(".embedding_cache/protbert")
```

Disable caching: set `use_cache: false` in config.

---

##  Notebook Workflow

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Load FASTAs, inspect length distributions, composition |
| `02_feature_engineering.ipynb` | Compare feature spaces, PCA visualisation |
| `03_classical_models.ipynb` | Train LR, SVM, RF, XGBoost interactively |
| `04_neural_models.ipynb` | Train CNN, BiLSTM, Transformer |
| `05_transformer_embeddings.ipynb` | Compute and visualise ProtBERT / ESM-2 embeddings |
| `06_ensemble.ipynb` | Build voting and stacking ensembles |

```bash
jupyter lab notebooks/
```

---

##  Robustness Features

- **Variable-length sequences**: One-hot pads/truncates to `max_length`; k-mer and composition work on any length
- **Class imbalance**: Auto-detected; `class_weight='balanced'` applied everywhere; warning logged when ratio > 3:1
- **Small datasets**: Automatic fold reduction in k-fold CV; fallback to simple split
- **Invalid FASTA**: Invalid characters filtered; empty sequences skipped with warning
- **GPU/CPU**: Auto-detection via `torch.cuda.is_available()` and MPS (Apple Silicon)
- **Memory-safe embeddings**: Processed in configurable batch sizes

---

##  Citation

If you use this tool in research, please cite the underlying model papers:
- **ProtBERT**: Elnaggar et al., 2021, *IEEE TPAMI*
- **ESM-2**: Lin et al., 2023, *Science*

---

##  License

MIT License — see `LICENSE` file.
