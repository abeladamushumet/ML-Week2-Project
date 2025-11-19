# ML-Week2-Project

Credit approval classification comparing K-Nearest Neighbors (KNN) and Decision Tree classifiers on the UCI Credit Approval dataset.

This repository contains a Jupyter notebook `credit_card_analysis.ipynb` that loads the dataset, performs preprocessing, tunes models using cross-validation and a train/validation/test split, evaluates model performance with common metrics, and summarizes the results.

## Contents
- `credit_card_analysis.ipynb` — main analysis notebook (imports, EDA, CV/grid search, train/val/test split, evaluation, conclusion).
- `dataset/credit_card_data-headers.txt` — dataset used in the notebook.

## Summary
We compared two classifiers (KNN and Decision Tree) using stratified 5-fold cross-validation and a 60/20/20 train/validation/test split.

Key results from the most recent run (values come from notebook output):

- Cross-validation (ROC AUC):
	- KNN best CV ROC AUC = 0.907696
	- Decision Tree best CV ROC AUC = 0.909872

- Validation (60/20/20 split):
	- Best KNN on validation: params = {n_neighbors: 5, weights: 'distance', p: 2}, val ROC AUC = 0.886182
	- Best Decision Tree on validation: params = {max_depth: 3, min_samples_split: 2, min_samples_leaf: 1}, val ROC AUC = 0.915019

- Test set (final evaluation):
	- KNN: Accuracy = 0.8626, Precision = 0.8475, Recall = 0.8475, F1 = 0.8475, ROC AUC = 0.9393
	- Decision Tree: Accuracy = 0.8473, Precision = 0.8679, Recall = 0.7797, F1 = 0.8214, ROC AUC = 0.9389

Confusion matrices (test set):

KNN:
```
[[63  9]
 [ 9 50]]
```

Decision Tree:
```
[[65  7]
 [13 46]]
```

Classification reports (test set):

KNN:
```
							precision    recall  f1-score   support

					0       0.88      0.88      0.88        72
					1       0.85      0.85      0.85        59

	 accuracy                           0.86       131
	macro avg       0.86      0.86      0.86       131
weighted avg       0.86      0.86      0.86       131
```

Decision Tree:
```
							precision    recall  f1-score   support

					0       0.83      0.90      0.87        72
					1       0.87      0.78      0.82        59

	 accuracy                           0.85       131
	macro avg       0.85      0.84      0.84       131
weighted avg       0.85      0.85      0.85       131
```

Interpretation:

- Both models show strong discrimination (ROC AUC ≈ 0.91 on CV). In this run, KNN achieved the highest test ROC AUC and slightly better accuracy, while Decision Tree produced competitive results and slightly higher precision but lower recall.
- KNN benefits from proper scaling (StandardScaler used) and a tuned k (here k=5, distance weighting). Decision Tree offers interpretability and stable CV performance with a small max depth.
- Model selection depends on the metric of interest: choose KNN if overall discrimination and balanced accuracy matter; choose Decision Tree if interpretability or slightly higher precision is prioritized.

## How to run
1. Create a Python environment (Python 3.8+ recommended) and install dependencies:

```powershell
# from the repository root (Windows PowerShell)
python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn joblib
```

2. Open the notebook in VS Code or Jupyter and run cells in order. The notebook expects the dataset at `dataset/credit_card_data-headers.txt` (path is configured in the notebook).

3. To reproduce the reported run exactly, run all cells from top to bottom. The notebook performs GridSearchCV and manual validation tuning; running can take a few minutes depending on CPU.

## Notes & next steps
- Results are sensitive to random splits and hyperparameter choices. To produce more stable estimates, consider repeated cross-validation or nested CV.
- For production use, add model calibration, persistent model artifacts (joblib), and tests. Consider adding a `requirements.txt` for reproducibility.
- Optionally, the notebook can be extended with a small code cell that programmatically injects the computed metrics into the final conclusion so the README or exported reports can be generated automatically.

