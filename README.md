# Spotify Review Sentiment Classifier

Binary sentiment analysis (Negative / Positive) on 50,924 Spotify Google Play reviews — from raw text to a fine-tuned DistilBERT model, with a full evaluation report.

---

## Results at a glance

| Model | Macro-F1 | F1-Negative | F1-Positive | PR-AUC |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (`LR_TF_bi`) | 0.8914 | 0.8556 | 0.9272 | 0.9814 |
| **DistilBERT fine-tuned** | **0.9569** | **0.9418** | **0.9720** | **0.9968** |

DistilBERT wins by **+6.6 pp Macro-F1** — gains concentrated on negation-bearing examples (*"not working"*, *"no longer good"*) and low-frequency vocabulary that was invisible to the TF-IDF vocabulary.

---

## Project structure

```
├── Notebooks/
│   ├── 01_EDA.ipynb                        # Data loading, VADER labelling, class balance, QC
│   ├── 02_POS_Tagging.ipynb                # spaCy POS analysis, ADJ/ADV vocabulary separation
│   ├── 03_Baseline_Modeling.ipynb          # TF-IDF ablation (8 configs), best model = LR_TF_bi
│   ├── 04_Transformer-Based_NLP_Model.ipynb# DistilBERT fine-tuning, diagnostics, comparison
│   ├── 05_Model_Evaluation_Report.ipynb    # Stakeholder report: error analysis, SHAP, fairness
│   └── src/
│       └── pipeline.py                     # Shared preprocessing (contractions, URL removal, lemmatisation)
├── Data/
│   └── reviews_spotify_kaggle.csv          # Raw reviews (Content, Score, ~60k rows)
├── Models/
│   ├── LR_TF_bi_best_model.joblib          # Serialised sklearn pipeline (vectoriser + LR)
│   ├── m3_ablation_study.csv               # CV results for all 8 ablation configurations
│   ├── m5_model/                           # DistilBERT checkpoint (HuggingFace format)
│   └── m5_tokenizer/                       # WordPiece tokenizer
├── Figures/                                # All plots saved by notebooks (PNG, 150 dpi)
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon

# 2. Run notebooks in order
jupyter notebook Notebooks/01_EDA.ipynb
# ... through to 05_Model_Evaluation_Report.ipynb

# 3. Export the stakeholder report as HTML (no code)
jupyter nbconvert --to html Notebooks/05_Model_Evaluation_Report.ipynb --no-input
```

> **Training time:** NB04 (DistilBERT) takes ~90 min on a Colab T4 GPU. The checkpoint is already saved in `Models/m5_model/` — NB05 loads it automatically without re-training.

---

## Methodology

### Labels
VADER silver labels applied to cleaned text:
- **Positive** → compound ≥ 0.30
- **Negative** → compound ≤ −0.10
- **Neutral** → excluded (compound ∈ (−0.10, 0.30))

Final binary corpus: **50,924 reviews** (67% Positive / 33% Negative).

### Splits
Stratified 70 / 10 / 20 train / val / test split, `random_state=42`. Identical indices used in both NB03 and NB04 — results are directly comparable.

### Classical baseline (NB03)
8-configuration ablation over {BoW, TF-IDF} × {unigrams, bigrams} × {Naive Bayes, Logistic Regression}. Best configuration: **TF-IDF bigrams + LR** (`C=10`, `class_weight='balanced'`), selected by 5-fold CV Macro-F1.

Key finding: the classifier axis (NB → LR) gives the largest single gain (+0.099 F1); the n-gram axis (uni → bi) adds +0.018 by capturing negation bigrams.

### Transformer (NB04)
**DistilBERT-base-uncased** (66 M parameters, 6 layers) — chosen over BERT-base (110 M, ~6 h training) and RoBERTa-base (125 M, ~7 h) to fit a 2-hour Colab T4 session while retaining 97% of BERT-base's GLUE score.

Full fine-tuning, 3 epochs, AdamW (lr=2e-5, weight decay=0.01), linear warmup over 10% of steps, batch size 32, max sequence length 128.

Training converged smoothly (max inter-epoch F1 swing = 0.012). Best checkpoint at epoch 3 (val Macro-F1 = **0.9603**).

### Evaluation (NB05)
- **Error analysis** — 6-category failure mode breakdown (OOV, negation, label noise, domain jargon, lexical ambiguity, short fragments)
- **Explainability** — `shap.LinearExplainer` on LR; 6 plots including beeswarm, waterfall (FP example), and LR-coefficient vs SHAP agreement
- **Fairness** — 6 subgroup slices (negation-bearing, short, long, high-OOV, domain jargon, all); max gap ~5 pp on negation subgroup
- **Calibration** — reliability diagram + ECE score; borderline zone (prob 0.40–0.60) routed to human review (~8% of traffic)

---

## Failure modes & risks

| Failure mode | Share of errors | Business impact |
|---|---|---|
| OOV vocabulary (LR only) | 43.6% | Silent misclassification of non-standard vocabulary |
| Negation blindness (LR) | 42.9% | Bugs framed with negation escape triage |
| Domain jargon | 7.9% | Context-dependent terms (*"no ads"*, *"premium"*) miscategorised |
| Label noise ceiling | structural | Shared by both models; requires gold annotation to fix |

DistilBERT eliminates the OOV failure mode structurally (WordPiece subword tokenisation) and reduces negation errors via contextual self-attention.

---

## Deployment recommendation

Deploy `LR_TF_bi` as a **monitored triage assistant** now (Macro-F1 = 0.8914, ECE < 0.03, fully explainable). Upgrade to DistilBERT for production (Macro-F1 = 0.9569).

Conditions: human review for all predictions with probability ∈ (0.40, 0.60); weekly confidence distribution monitoring; retrain trigger if rolling Macro-F1 drops > 3 pp.

---

## Reproducibility

All experiments use `SEED = 42` (Python, NumPy, PyTorch, CUDA). DataLoader workers are seeded via `worker_init_fn`. Reloading the saved DistilBERT checkpoint produces bit-identical predictions (verified in NB04 Step 8).

---

## Dependencies

See `requirements.txt`. Core: `torch`, `transformers`, `scikit-learn`, `shap`, `spacy`, `nltk`, `pandas`, `matplotlib`, `seaborn`.
