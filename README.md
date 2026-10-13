# Credit Card Fraud Detection Pipeline

An end-to-end machine learning project that detects fraudulent credit card
transactions. It covers EDA, imbalance handling, model comparison, threshold
tuning and a reusable training/prediction pipeline with a Streamlit demo app.

## Dataset
- Source: [Kaggle Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, only 492 fraud (0.17%) before removing duplicates
- Features: Time, Amount, and V1 to V28 (anonymized PCA components)

## Approach
1. **EDA**: studied the class imbalance, Amount/Time behaviour and the features most correlated with fraud (V17, V14, V12, V10).
2. **Preprocessing**: removed duplicates, stratified 80/20 split, RobustScaler on Time and Amount (fit on training data only to avoid leakage).
3. **Imbalance handling**: compared class weights, SMOTE and undersampling.
4. **Models**: Logistic Regression, Random Forest and XGBoost.
5. **Evaluation**: used PR-AUC, precision, recall and F1 instead of accuracy, because ROC-AUC and accuracy look excellent even for weak models on such imbalanced data.
6. **Threshold tuning**: chose the decision threshold on a validation set, then evaluated once on the untouched test set.
7. **Pipeline**: scaler + XGBoost in one scikit-learn Pipeline, so raw transactions can be scored directly.

## Results (test set)
| Metric | Score |
|---|---|
| PR-AUC | 0.83 |
| Precision | 0.9861 |
| Recall | 0.7474 |
| F1 | 0.8503 |

Out of 95 frauds in the test set, the model caught 71 and
missed 24, with 1 false alarm.

## Project structure
```
├── src/
│   ├── config.py       # paths and settings
│   ├── preprocess.py   # load, clean, split
│   ├── model.py        # pipeline definition
│   ├── train.py        # training + threshold tuning + evaluation
│   └── predict.py      # prediction on new data
├── notebooks/          # EDA and experiments
├── app.py              # Streamlit demo
└── requirements.txt
```

## How to run
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Download creditcard.csv from Kaggle into the data/ folder, then:
python -m src.train            # trains and saves the pipeline
python -m src.predict          # quick demo
streamlit run app.py           # web app
```

## Key learnings
- Accuracy is useless on imbalanced data; PR-AUC and recall/precision matter.
- Split first, then scale and resample, to avoid data leakage.
- The decision threshold is a business choice, not a fixed 0.5.
- Pick the threshold on validation data, never on the test set.

## Limitations and future work
- Only about 95 fraud cases in the test set, so scores are noisy; cross-validation would give a more stable estimate.
- Features are anonymized, so results cannot be explained in business terms.
- Next steps: hyperparameter tuning, SHAP explanations, and serving the model through an API.