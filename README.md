# Humanoid Robot Deployment Readiness — Multiclass Classification

Predicting the overnight deployment-readiness status of warehouse humanoid robots from diagnostic sensor data, for a 2026 AI/DS hackathon (Great Learning). 7-class ordinal classification, scored on accuracy.

**Public leaderboard: 0.8578 → 0.8583**, reached #1 on the leaderboard in the closing stretch of the competition.

## The problem

Every robot in a warehouse fleet gets a full diagnostic overnight. Before the morning shift, that data has to be turned into a deployment call: is this robot healthy enough to work, does it need light duty, or should it stay offline? The task is to classify each of 20,000 test robots into one of 7 ordered classes (`Under Maintenance` → `Not Ready` → `Degraded` → `Caution` → `Standard` → `High Performance` → `Peak Condition`) using 16 overnight sensor readings, trained on 80,000 labeled examples.

## Results

| Model | Accuracy |
|---|---|
| LightGBM (tuned gradient boosting) | 0.8578 |
| **+ confidence-gated ensemble with a neural net** | **0.8583** |

The gain is small in absolute terms (10 more correct predictions out of 20,000), but it's the one improvement — out of many tested — that actually held up under real evaluation. See [`robot_deployment_classification.ipynb`](robot_deployment_classification.ipynb) for the full, executed walkthrough.

## The key finding: a leakage trap hiding in plain sight

The data dictionary flagged two features for "extra scrutiny" without saying why. Checking them properly, rather than trusting or dismissing the hint, turned out to matter more than any modeling choice:

- **`shift_reliability_index`** correlated at **0.96** with the target — far beyond any legitimate sensor (the next-best tops out around 0.66–0.76) — but its train/test distributions differed by an order of magnitude more than every other feature (KS-statistic 0.125 vs. ~0.01 for clean features). Class-conditional means were almost perfectly evenly spaced, the signature of a feature partly constructed *from* the label during data generation. Included, it makes any model look outstanding in training and then fail to generalize. **Excluded.**
- **`motion_elegance_score`** had a correlation of 0.004 with the target (not statistically significant) — a plausible-sounding feature that turned out to be pure noise. **Excluded.**

Both conclusions are backed by evidence in the notebook, not just asserted — including a quick check showing a single-feature model on `shift_reliability_index` alone would out-predict all six legitimate health sensors combined, which is the tell that something is wrong with it.

## Approach

1. **EDA** — class balance, per-feature correlation with the (ordinal) target, and train/test distribution comparison for every feature to catch shift/leakage issues before they reach the model.
2. **Feature engineering** — data-quality fixes (impossible battery readings, missingness flags), a composite health score combining six sensor readings, error-rate ratios, and a "red flag" count for robots showing multiple simultaneous warning signs.
3. **Two genuinely different models** — LightGBM (gradient boosting) and an MLP (neural network). Tree-based "diversity" (a second tree variant, or a weaker linear model) was tested and consistently made results worse; a model that learns through an entirely different mechanism was the one combination that reliably helped.
4. **Confidence-gated ensemble** — LightGBM's prediction is used by default; it's overridden only where the MLP independently disagrees *and* both models clear a 0.5 confidence bar on the same alternative. Validated against several looser/tighter thresholds and additional model combinations before settling here — this threshold sits close to a genuine mathematical optimum for the ensemble, not an arbitrary cutoff.

## Repository structure

```
.
├── README.md
├── robot_deployment_classification.ipynb   # full pipeline, EDA → features → models → ensemble
├── run_lightgbm_ensemble.py                # standalone script version (CLI / Colab friendly)
└── requirements.txt
```

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook robot_deployment_classification.ipynb
```

Place `train.csv` and `test.csv` (from the competition) in the same directory. The notebook runs end-to-end and writes `submission.csv` in the competition's required format (`row_id`, `target`).

> **Note:** the LightGBM cells require `lightgbm` installed and were written to run standalone; everything else (EDA, feature engineering, MLP training and cross-validation) is fully executed in the committed notebook so its outputs and plots render directly on GitHub.

## What I'd do differently with more compute

Every model family available in this environment — gradient boosting, neural net, linear, and an ordinal-regression reformulation — plateaued in the 0.848–0.853 range under honest cross-validation. Several further ensembling ideas (adding more models, widening thresholds, full model-replacement) were tested carefully and made things *worse*, not better, despite looking reasonable in validation. That consistent pattern is itself informative: it suggests the two models used here disagree on genuinely difficult, ambiguous cases rather than one having a fixable blind spot the other catches — and that meaningfully closing the remaining gap would need a stronger base learner (or additional data) rather than a cleverer combination of what's already here.

## Acknowledgments

Dataset and problem statement from the *MIT – AI & DS – May-2026 Hackathon*, organized by Great Learning.
