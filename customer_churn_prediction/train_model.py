"""Train and evaluate the Customer Churn Prediction model."""

from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "model" / "churn_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"

# Use the IBM sample first, with a public mirror as a fallback.
DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_data() -> pd.DataFrame:
    """Load the Telco CSV, downloading it once if it is not present locally."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        last_error = None
        for url in DATA_URLS:
            try:
                print(f"Downloading dataset from: {url}")
                urlretrieve(url, DATA_PATH)
                break
            except Exception as exc:
                last_error = exc
                if DATA_PATH.exists():
                    DATA_PATH.unlink()
        else:
            raise RuntimeError(
                "Could not download the Telco dataset. Check your internet connection, "
                "or download the CSV manually and save it as:\n"
                f"{DATA_PATH}"
            ) from last_error

    df = pd.read_csv(DATA_PATH)
    required = {"Churn", "tenure", "MonthlyCharges", "TotalCharges"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"The dataset is missing required columns: {sorted(missing)}")

    # Empty TotalCharges entries are expected for a small number of new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop_duplicates().copy()
    df = df[df["Churn"].isin(["Yes", "No"])].copy()

    # customerID is an identifier, not a useful behavior pattern for the model.
    if "customerID" in df.columns:
        df = df.drop(columns="customerID")

    if df.empty:
        raise ValueError("No valid customer records were found in the dataset.")
    return df.reset_index(drop=True)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few transparent features that match the presentation."""
    df = df.copy()

    if "tenure" in df.columns:
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 120],
            labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
        )

    if "MonthlyCharges" in df.columns:
        df["SpendingTier"] = pd.cut(
            df["MonthlyCharges"],
            bins=[-1, 35, 70, 100, float("inf")],
            labels=["Low", "Medium", "High", "Very high"],
        )

    existing_services = [col for col in SERVICE_COLUMNS if col in df.columns]
    if existing_services:
        df["ServiceCount"] = sum((df[col] == "Yes").astype(int) for col in existing_services)

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["AverageChargePerTenureMonth"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    return df


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Create preprocessing and XGBoost in one pipeline to prevent data leakage."""
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # Included to reflect the preprocessing approach in the project slides.
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    churn_count = int((y_train == 1).sum())
    retained_count = int((y_train == 0).sum())
    class_weight = retained_count / max(churn_count, 1)

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def save_plots(df: pd.DataFrame, bundle: dict) -> None:
    """Save a few useful charts for the report and presentation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Churn class distribution
    counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Retained", "Churned"], counts.values)
    ax.set_title("Customer Churn Distribution")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Churn rate by contract
    contract_rate = (df.groupby("Contract", observed=False)["Churn"]
                     .apply(lambda values: (values == "Yes").mean() * 100))
    fig, ax = plt.subplots(figsize=(7, 4))
    contract_rate.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("Contract type")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=150)
    plt.close(fig)

    # 3. Confusion matrix
    matrix = confusion_matrix(bundle["test_labels"], bundle["test_predictions"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Retained", "Churned"])
    ax.set_yticks([0, 1], ["Retained", "Churned"])
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 4. Global feature importance
    feature_df = pd.DataFrame(bundle["feature_importance"][:12])
    if not feature_df.empty:
        feature_df = feature_df.sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feature_df["feature"], feature_df["importance"])
        ax.set_title("Most Important Features (XGBoost)")
        ax.set_xlabel("Relative importance")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)

    # 5. ROC curve
    fpr, tpr, _ = roc_curve(bundle["test_labels"], bundle["test_probabilities"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {bundle['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)


def train_model() -> dict:
    """Run cleaning, feature engineering, training, evaluation, and export."""
    df = create_features(load_data())
    X = df.drop(columns="Churn")
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training XGBoost on {len(X_train):,} records; testing on {len(X_test):,} records...")
    model = build_model(X_train, y_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).astype(int)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }

    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    feature_importance = sorted(
        [
            {
                "feature": str(name).replace("numeric__", "").replace("categorical__", ""),
                "importance": round(float(score), 6),
            }
            for name, score in zip(transformed_names, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    bundle = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "churn_rate": round(float((y == 1).mean()), 4),
        "test_labels": y_test.astype(int).tolist(),
        "test_predictions": predictions.tolist(),
        "test_probabilities": probabilities.astype(float).tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    save_plots(df, bundle)

    report = classification_report(y_test, predictions, target_names=["Retained (0)", "Churned (1)"])
    (OUTPUT_DIR / "classification_report.txt").write_text(
        "CUSTOMER CHURN PREDICTION — XGBOOST\n\n"
        f"Dataset rows: {len(df)}\nTraining rows: {len(X_train)}\nTesting rows: {len(X_test)}\n\n"
        "TEST METRICS\n" + "\n".join(f"{key}: {value:.4f}" for key, value in metrics.items())
        + "\n\nCLASSIFICATION REPORT\n" + report,
        encoding="utf-8",
    )

    print("\nModel training complete.")
    print(f"Dataset rows: {len(df):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    for name, score in metrics.items():
        print(f"{name.replace('_', ' ').title():<12}: {score:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Charts and report saved to: {OUTPUT_DIR}")
    return bundle


if __name__ == "__main__":
    train_model()
