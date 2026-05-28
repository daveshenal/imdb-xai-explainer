IMDb Top 1000 - Explainable Ratings (XAI)

A step-by-step Jupyter notebook that builds a full machine-learning and explainability workflow on the IMDb Top 1000 dataset. It trains a regression model to predict ratings (IMDb or your personal ratings) and explains predictions using SHAP (and optionally LIME).

### Contents

- `analyze_dataset.ipynb`: end‑to‑end workflow with preprocessing, modeling, evaluation, SHAP/LIME explanations, and visualizations
- `imdb_top_1000.csv`: dataset used by the notebook

### What it does

- Cleans and preprocesses fields like `Runtime`, `Released_Year`, and `Gross`
- Encodes categorical variables (`Genre`, `Certificate`, `Director`) with One‑Hot
- Trains a Random Forest (optionally XGBoost) to predict ratings
- Evaluates with RMSE/MAE/R²
- Explains predictions with SHAP global (summary, bar) and local (waterfall, force) plots; optional LIME example
- Compares top drivers for high vs. low predicted ratings

### Quickstart (Conda)

```bash
# Create environment (Python 3.10 recommended)
conda create -n imdb-xai python=3.10 -y
conda activate imdb-xai

# Install dependencies
pip install -U pip \
  pandas numpy scikit-learn matplotlib seaborn \
  shap xgboost lime jupyterlab ipykernel

# Launch JupyterLab
jupyter lab
```

Open `analyze_dataset.ipynb` and run cells top-to-bottom.

### Using your personal ratings (optional)

1. Create `my_personal_ratings.csv` with columns:
   - `Series_Title` (must match dataset title strings)
   - `Personal_Rating` (e.g., 1–10)
2. In the notebook, set `USE_PERSONAL_RATINGS = True` and run the target-selection cell.

### Notes and tips

- If SHAP errors about dtypes/sparse inputs, the notebook already converts transformed features to dense `float64` before calling SHAP.
- If you don’t need XGBoost or LIME, you can skip installing them.
- Categorical encoding groups rare categories when supported by your scikit‑learn version.

### Troubleshooting

- Package import errors: re-run pip install lines; ensure the correct environment is active.
- Display issues on force/waterfall plots: some SHAP backends differ by version; the notebook includes fallbacks.
- Data mismatches: confirm `Series_Title` matches exactly between your personal ratings and the dataset.

### License

For personal/educational use. Dataset sourced from IMDb Top 1000 (public CSV shared widely for learning purposes). Ensure you comply with IMDb data usage guidelines in your context.
