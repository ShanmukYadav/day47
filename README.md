# Week 08 · Thursday — RNNs + Sequential Data

**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**

---

## What This Assignment Covers

This notebook addresses two real-world sequential prediction problems given to Vikram Anand, Head of AI at a fintech firm:

1. **Stock Price Forecasting** — LSTM for next-day close price prediction on Indian equities (RELIANCE, INFOSYS, TCS, HDFC, WIPRO)
2. **Churn Prediction** — Sequential vs tabular model comparison on customer chat interaction history *(requires `chat_logs.csv` — see note below)*

---

## Approach Summary

### Sub-step 1 — Sequence Construction (Easy ✅)
- Loaded `stock_prices.csv` (3,750 rows · 5 tickers · Jan 2022 – Nov 2024)
- Chose **RELIANCE** as the target stock
- **Window size: 30 trading days** (~6 weeks) — captures medium-term momentum without excessive noise
- **Split strategy: strictly chronological** — 80% train / 10% val / 10% test. A random split would introduce look-ahead bias (future data leaks into training), inflating reported accuracy and producing a model that fails in production
- MinMaxScaler fitted **only on training data** to prevent data leakage

### Sub-step 3 — LSTM Model (Medium ✅)
- 2-layer stacked LSTM with hidden size 64, dropout 0.2
- Adam optimiser (lr=1e-3) with ReduceLROnPlateau scheduler
- Early stopping (patience=10) with best-checkpoint saving
- Gradient clipping (max_norm=1.0) for training stability
- **Evaluation metric: MAPE** (Mean Absolute Percentage Error) — scale-invariant and directly readable as % deviation from true price. Deployment bar: MAPE < 1.5%

### Sub-step 4 — Churn Model (Medium — placeholder)
- Framework designed; requires `chat_logs.csv`
- Hypothesis: test LSTM on raw chat sequences vs GBM on aggregated features; prefer GBM if AUC gap < 1%

### Sub-step 5 — Risk Ranking & Cost Model (Medium — framework provided)
- Cost model: `C_FP = ₹200` (discount voucher) vs `C_FN = ₹2,000` (LTV lost)
- Optimal threshold derived from cost ratio; outreach list produced by ranking predicted churn probability

### Sub-step 6 — AR Baseline vs LSTM (Hard 🔴 optional, attempted ✅)
- Autoregressive baseline: linearly-weighted average of last 30 days
- Compared against LSTM on identical test period
- Includes auto-diagnosis: if AR wins, suggests adding technical features; if LSTM wins, explains what non-linear pattern it captured

---

## Repository Structure

```
week-08/
└── thursday/
    ├── W8_Thursday_Assignment.ipynb   ← main notebook (all sub-steps)
    ├── stock_prices.csv               ← dataset (5 equities, 2022–2024)
    ├── README.md                      ← this file
    ├── best_lstm.pt                   ← saved model checkpoint (generated on run)
    ├── split_visualisation.png        ← output: chronological split chart
    ├── training_curve.png             ← output: loss curves
    ├── lstm_predictions.png           ← output: predicted vs actual
    └── model_comparison.png           ← output: LSTM vs AR baseline
```

> `chat_logs.csv` is NOT committed — download from LMS and place in this directory before running Sub-steps 2, 4, and 5.

---

## How to Run

### 1. Prerequisites

**Python version:** 3.10 or higher

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

> For GPU training, replace `cpu` with `cu121` (CUDA 12.1) in the torch install URL.

### 3. Place datasets

```
week-08/thursday/
├── stock_prices.csv     ← already included
└── chat_logs.csv        ← download from LMS, place here
```

### 4. Launch notebook

```bash
cd week-08/thursday
jupyter notebook W8_Thursday_Assignment.ipynb
```

Run cells top-to-bottom. The notebook is self-contained — all outputs (plots, metrics, model checkpoint) are generated in the same directory.

### 5. Expected runtime

| Section | Time (CPU) |
|---|---|
| Data loading & sequence construction | < 5 seconds |
| LSTM training (50 epochs max, early stop) | 2–4 minutes |
| AR baseline comparison | < 5 seconds |
| Full notebook end-to-end | ~5 minutes |

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
torch>=2.0
jupyter>=1.0
```

Full pinned versions in `requirements.txt`.

---

## Key Design Decisions

| Decision | Choice | Justification |
|---|---|---|
| Window size | 30 days | Captures ~6-week momentum; shorter misses structure, longer adds noise |
| Split strategy | Chronological 80/10/10 | Time-series — random split leaks future, invalidates all metrics |
| Scaler fit | Train only | Fitting on val/test leaks their distribution statistics into training |
| LSTM layers | 2 | First layer: price patterns; second: higher-order temporal dependencies |
| Dropout | 0.2 | Regularises against overfitting on ~600-sample training set |
| Evaluation metric | MAPE | Scale-invariant; readable as % error; appropriate across different price levels |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients in deep LSTM |

---

## AI Usage Policy Compliance

All AI-assisted sub-steps include:
- The exact prompt used
- A critique documenting what was changed and why

See the **"AI Usage Log"** section at the bottom of the notebook.

---

## Submission Checklist

- [x] Notebook in `week-08/thursday/` (not root)
- [x] README with run instructions, Python version, and packages
- [ ] At least 3 commits with descriptive messages (commit after each sub-step)
- [x] No `.env`, API keys, `__pycache__`, or `.ipynb_checkpoints` committed
- [x] AI usage log included in notebook

---

*Deadline: Friday 09:15 AM — push to `week-08/thursday/` and paste link in LMS*
