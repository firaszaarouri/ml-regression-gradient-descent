# Multi-Linear Regression via Gradient Descent

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/ML-From%20Scratch-green)
![sklearn](https://img.shields.io/badge/Validated-scikit--learn-orange)

> Full implementation of multi-linear regression using gradient descent from scratch, applied to two real-world datasets. Custom algorithm matches scikit-learn's analytical solution exactly.

## Results

| Dataset | R² (test) | MSE |
|---------|-----------|-----|
| Energy (CCPP) | **0.9301** | 20.89 |
| Loans (P2P) | **0.6341** | 5.22 |

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tabulate jupyter
jupyter notebook ML_Regression_Complete.ipynb
```

## Contents

- `ML_Regression_Complete.ipynb` — Full notebook (36 cells)
- `dataEnergy.csv` — Combined Cycle Power Plant dataset (9,568 samples)
- `dataLoans.csv` — P2P lending dataset (2,500 samples)

## Key Concepts

- Gradient descent from scratch (model, cost, gradient, update rule)
- Proper train/test split + StandardScaler (no data leakage)
- Convergence analysis across learning rates and iterations
- Residual analysis and regression assumption validation
- 5-fold cross-validation
- Sklearn comparison (results are identical ✓)

## Author
Firas — MSc Data Analytics, Cranfield University / PhD Candidate, LIP6 Sorbonne
