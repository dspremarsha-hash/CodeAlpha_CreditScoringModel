# Credit Scoring Model

**CodeAlpha Machine Learning Internship — Task 1**

Predicts whether a loan applicant is a **good** or **bad** credit risk using their financial and personal history — the same kind of decision a bank makes before approving a loan.

## Dataset

**German Credit Data** (StatLog / UCI) — 1000 real loan applicants, 20 features (loan amount, duration, savings, employment history, housing status, age, etc.) plus a binary target: `credit_risk` (1 = good, 0 = bad). Class distribution: 700 good / 300 bad (imbalanced).

## Approach

1. **Feature engineering** — one-hot encoded 13 categorical columns (e.g. `purpose`, `housing`, `savings`) into 41 binary columns, combined with 7 numeric columns → 48 total features.
2. **Train/test split** — 80/20, stratified to preserve the 70/30 class ratio in both sets.
3. **Scaling** — `StandardScaler` fit on train only, applied to test (avoids data leakage).
4. **Modeling** — Logistic Regression, trained two ways:
   - **Baseline** (default)
   - **Class-balanced** (`class_weight='balanced'`) — weights the minority (bad-credit) class more heavily during training.
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix.

## Results

| Metric | Baseline | Class-Balanced |
|---|---|---|
| Accuracy | 0.710 | 0.665 |
| Precision | 0.781 | 0.817 |
| Recall | 0.814 | 0.671 |
| F1-Score | 0.797 | 0.737 |
| ROC-AUC | 0.751 | 0.755 |

**Bad-credit recall specifically** (the metric that matters most for risk management): **47% (baseline) → 65% (balanced)**.

### Key insight: a real trade-off, not just a metric

The baseline model has higher overall accuracy, but it misses over half of the genuinely risky applicants (approves them anyway) because it leans on the majority "good credit" class. The class-balanced model catches significantly more risky applicants (fewer costly bad loans) at the cost of rejecting more good applicants (lost business). **Which model is "better" depends on the bank's risk appetite** — this isn't purely a technical choice, it's a business one. ROC-AUC (threshold-independent) stays roughly the same across both, confirming the underlying model quality didn't change — only the decision threshold behavior did.

![Confusion Matrix](outputs/confusion_matrix.png)
![Feature Importance](outputs/feature_importance.png)

### What drives the prediction

Top features pushing toward *good* credit: no checking account on file, buying a used car, having a guarantor. Top features pushing toward *bad* credit: loan purpose = retraining, being a foreign worker, renting rather than owning.

## How to run

```bash
pip install -r requirements.txt
python credit_scoring.py
```

## Files

- `credit_scoring.py` — full pipeline (load → encode → split → scale → train → evaluate → visualize)
- `german_credit.csv` — dataset
- `outputs/` — confusion matrix and feature importance charts
- `requirements.txt` — dependencies
