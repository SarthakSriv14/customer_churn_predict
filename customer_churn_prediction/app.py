"""Simple Streamlit dashboard for the Customer Churn Prediction project."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from train_model import DATA_PATH, MODEL_PATH, create_features, load_data


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("Customer Churn Prediction")
st.caption("XGBoost classification • Customer behavior analysis • Retention planning")

if not MODEL_PATH.exists():
    st.warning("The trained model is not available yet. Run these commands in the project folder:")
    st.code("python train_model.py\nstreamlit run app.py", language="bash")
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return load_data()


bundle = load_bundle()
dataset = load_dataset()
featured_dataset = create_features(dataset)
metrics = bundle["metrics"]


tab_overview, tab_explore, tab_predict = st.tabs(["Model Overview", "Explore Data", "Predict a Customer"])

with tab_overview:
    st.subheader("Model performance")
    metric_cols = st.columns(5)
    display_metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
    ]
    for col, (label, key) in zip(metric_cols, display_metrics):
        col.metric(label, f"{metrics[key]:.1%}")
    st.caption("Metrics are calculated on a stratified 20% test set that was not used for training.")

    top_left, top_right = st.columns(2)
    with top_left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            bundle["test_labels"],
            bundle["test_predictions"],
            labels=[0, 1],
            display_labels=["Retained", "Churned"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title("Actual vs Predicted")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Recall matters because false negatives are customers who churn without being flagged.")

    with top_right:
        st.subheader("Most important features")
        importance_df = pd.DataFrame(bundle["feature_importance"][:12])
        if not importance_df.empty:
            importance_df = importance_df.sort_values("importance", ascending=True)
            importance_df["feature"] = importance_df["feature"].str.replace("_", " ", regex=False)
            st.bar_chart(importance_df.set_index("feature")["importance"])
        st.caption("Importance is global: it shows which features the model used most overall, not a causal explanation.")

    with st.expander("View ROC curve"):
        fig, ax = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_predictions(
            bundle["test_labels"], bundle["test_probabilities"], name="XGBoost", ax=ax
        )
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_explore:
    st.subheader("Dataset overview")
    data_cols = st.columns(4)
    data_cols[0].metric("Customers", f"{len(dataset):,}")
    data_cols[1].metric("Churn rate", f"{(dataset['Churn'] == 'Yes').mean():.1%}")
    data_cols[2].metric("Features used", str(len(bundle["feature_columns"])))
    data_cols[3].metric("Test records", f"{bundle['test_rows']:,}")

    st.subheader("Churn distribution")
    churn_counts = dataset["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    churn_counts.index = ["Retained", "Churned"]
    st.bar_chart(churn_counts)

    left, right = st.columns(2)
    with left:
        st.subheader("Churn rate by contract")
        contract_rates = (dataset.groupby("Contract", observed=False)["Churn"]
                          .apply(lambda values: (values == "Yes").mean() * 100))
        st.bar_chart(contract_rates.rename("Churn rate (%)"))

        st.subheader("Churn rate by payment method")
        payment_rates = (dataset.groupby("PaymentMethod", observed=False)["Churn"]
                         .apply(lambda values: (values == "Yes").mean() * 100))
        st.bar_chart(payment_rates.rename("Churn rate (%)"))

    with right:
        st.subheader("Churn rate by tenure group")
        tenure_rates = (featured_dataset.groupby("TenureGroup", observed=False)["Churn"]
                        .apply(lambda values: (values == "Yes").mean() * 100))
        st.bar_chart(tenure_rates.rename("Churn rate (%)"))

        st.subheader("Churn rate by spending tier")
        spending_rates = (featured_dataset.groupby("SpendingTier", observed=False)["Churn"]
                          .apply(lambda values: (values == "Yes").mean() * 100))
        st.bar_chart(spending_rates.rename("Churn rate (%)"))

    st.caption("EDA helps identify patterns and guide feature engineering; these relationships alone do not prove cause and effect.")

with tab_predict:
    st.subheader("Check an individual customer")
    st.write("Enter a customer's subscription and billing details to estimate churn probability.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 6)
            contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        with col2:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with col3:
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=150.0, value=70.0, step=1.0)

        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": round(tenure * monthly_charges, 2),
        }
        customer_df = create_features(pd.DataFrame([customer]))
        customer_df = customer_df.reindex(columns=bundle["feature_columns"])
        probability = float(bundle["model"].predict_proba(customer_df)[0, 1])
        prediction = probability >= 0.5
        risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"

        st.subheader("Prediction result")
        result_cols = st.columns(3)
        result_cols[0].metric("Churn probability", f"{probability:.1%}")
        result_cols[1].metric("Risk level", risk)
        result_cols[2].metric("Predicted outcome", "Likely to churn" if prediction else "Likely to stay")
        st.progress(min(max(probability, 0.0), 1.0))

        if risk == "High":
            st.error("High risk: consider contacting this customer soon.")
        elif risk == "Medium":
            st.warning("Medium risk: monitor engagement and consider a targeted offer.")
        else:
            st.success("Low risk: continue regular customer engagement.")

        # Simple, transparent business rules for retention suggestions.
        actions = []
        if contract == "Month-to-month":
            actions.append("Offer a suitable discount or benefit for moving to a one-year contract.")
        if tenure <= 12:
            actions.append("Arrange an onboarding or customer-success check-in for this newer customer.")
        if monthly_charges >= 70:
            actions.append("Review the plan and explain lower-cost or better-value bundle options.")
        if internet != "No" and (security != "Yes" or support != "Yes"):
            actions.append("Consider a trial or bundle for online security and technical support.")
        if payment == "Electronic check":
            actions.append("Explain the convenience of automatic bank transfer or card payment.")
        if not actions:
            actions.append("Continue regular engagement and monitor future changes in service usage.")

        st.markdown("**Suggested retention actions**")
        for action in actions:
            st.markdown(f"- {action}")
        st.caption("This is an educational model estimate. Retention suggestions are simple rules, not automated business decisions.")

st.divider()
st.caption("Project scope: XGBoost classification, EDA, feature engineering, model evaluation, global feature importance, and rule-based retention suggestions.")
