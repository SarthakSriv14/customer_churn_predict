# Customer Churn Prediction

A small, presentation-aligned machine-learning project that predicts whether a telecom customer may leave. It uses the IBM Telco Customer Churn sample dataset, XGBoost classification, a short preprocessing/feature-engineering pipeline, evaluation metrics, and a Streamlit dashboard.

## What it includes

- **Data preprocessing:** converts `TotalCharges` to a number, handles missing values with imputation, removes duplicate rows and the customer ID.
- **EDA:** churn distribution and churn rates across contract, payment, tenure, and spending groups.
- **Feature engineering:** tenure groups, spending tiers, service count, and average charge per tenure month.
- **Preprocessing:** one-hot encoding for categorical features and scaling for numerical features inside an sklearn pipeline.
- **Training:** an XGBoost binary classifier with an 80/20 stratified train/test split.
- **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, a confusion matrix, and a ROC curve.
- **Interpretation:** global XGBoost feature importance.
- **Retention support:** simple suggestions based on the customer's profile, such as a contract offer, onboarding check-in, plan review, or support add-on.
- **Dashboard:** explore the data, see model metrics, and enter a customer profile for a prediction.

## Project structure

```text
customer_churn_prediction/
├── app.py                 # Streamlit dashboard and prediction form
├── train_model.py         # Data loading, features, model training, evaluation
├── requirements.txt       # Small set of dependencies
├── data/
│   └── README.md          # Dataset download information
├── model/                 # Model artifact generated after training
└── outputs/               # Charts and classification report generated after training
```

## Run on Windows (VS Code)

Open a terminal in this folder and run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The first training run needs internet access to download the dataset. The CSV is saved to `data/Telco-Customer-Churn.csv`; subsequent runs use the local copy. If downloading fails, download the IBM Telco CSV manually and save it at that exact path.

When Streamlit starts, open the local URL shown in the terminal, usually `http://localhost:8501`.

## Target and metrics

The target column is `Churn`: `Yes` is encoded as 1 (customer churned) and `No` as 0 (customer stayed). The test set is held out before the model pipeline is fitted. Recall is included because failing to flag a customer who actually churns can be costly, while precision helps quantify unnecessary retention outreach.

The code uses a 0.5 probability threshold for the churn/no-churn classification. The dashboard's risk bands are presentation aids: Low below 40%, Medium from 40% to below 70%, and High at 70% or above.

## Dataset note

The standard IBM Telco sample contains customer demographics and account/service/billing fields, but it does **not** contain exact age, location, or late-payment history. The implementation uses fields that genuinely exist in this CSV rather than fabricating missing attributes. If your course has a different dataset that includes those columns, the project should be adjusted to use that exact file.

## Outputs

After training, the `outputs/` folder contains:

- `churn_distribution.png`
- `churn_by_contract.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `roc_curve.png`
- `classification_report.txt`

The model is saved as `model/churn_model.joblib`. Metrics shown in the app are calculated from the held-out test set; they are not hard-coded sample scores.

## Scope and future work

This version keeps the code small while implementing the core pipeline in the presentation. Hyperparameter optimization, SHAP-based individual explanations, real-time API deployment, automated outreach integration, and ongoing model monitoring remain future enhancements rather than unfinished hidden dependencies.

For education and demonstration only; predictions are estimates, not guarantees or automated business decisions.
