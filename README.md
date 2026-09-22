\# Retail Demand Forecasting



An end-to-end machine learning project for 28-day retail demand forecasting using the M5 Forecasting dataset.



The project compares traditional machine learning and deep learning approaches for multi-series demand forecasting and is being developed into a reproducible forecasting pipeline with experiment tracking, testing, model serving, and deployment.



\## Project Status

The core forecasting and model-serving pipeline is implemented.

Completed components:

- M5 data loading and preprocessing
- Store-category demand aggregation
- Exploratory data analysis
- Leakage-safe feature engineering
- Time-based train/validation/test splitting
- Seasonal-naive forecasting baseline
- Random Forest and XGBoost model development
- Expanding-window cross-validation and hyperparameter tuning
- Final model selection and untouched test evaluation
- MLflow experiment tracking and model persistence
- FastAPI model-serving API
- Automated data-loading and API tests

The selected Random Forest model achieved:

| Metric | Random Forest | Seasonal Naive |
|---|---:|---:|
| MAE | 126.05 | 157.90 |
| RMSE | 212.98 | 262.85 |
| WAPE | 8.60% | 10.77% |
| Bias | -4.74% | -3.91% |

The Random Forest reduced WAPE by approximately 20% relative to the
horizon-safe seasonal-naive benchmark.

### Remaining Work

- Containerize the application with Docker
- Add CI/CD with GitHub Actions
- Finalize documentation and repository cleanup



\## Dataset



This project uses the M5 Forecasting dataset, which contains hierarchical retail sales data from Walmart stores in the United States.



Large raw data files are not stored in this repository. Instructions for obtaining and preparing the dataset will be added as the project is developed.

