import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Paths (dynamic based on your folder structure)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'fraud_detection_pipeline.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

# -----------------------------
# Load model and scaler
# -----------------------------
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Banking Fraud Detection")

st.markdown("""
Enter transaction details below to predict whether it is a Fraudulent Transaction or Legitimate Transaction.
""")

# ---- User Inputs ----
tx_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=100.0)
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, value=5000.0, step=100.0)
newbalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, value=4000.0, step=100.0)
oldbalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, value=1000.0, step=100.0)
transaction_hour = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)
step = st.number_input("Step (Transaction Step)", min_value=1, value=1)
flag_dest_new_account = st.selectbox("Destination is a new account?", ["No", "Yes"])
threshold = st.slider("Fraud threshold (probability)", 0.0, 1.0, 0.50, 0.01)

# ---- Prepare dataframe in correct column order ----
input_df = pd.DataFrame({
    'amount': [amount],
    'oldbalanceOrg': [oldbalanceOrg],
    'newbalanceOrig': [newbalanceOrig],
    'oldbalanceDest': [oldbalanceDest],
    'transaction_hour': [transaction_hour],
    'step': [step]
})

# ---- Scale input ----
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Scaling failed: {e}")
    st.stop()

# ---- Predict on button click ----
if st.button("Predict Fraud"):
    try:
        # Model probability
        proba_model = float(model.predict_proba(input_scaled)[0][1])
    except Exception:
        pred = int(model.predict(input_scaled)[0])
        proba_model = 1.0 if pred == 1 else 0.0

    # Rule-based adjustments
    rule_risk = 0.0
    if tx_type in {"TRANSFER", "CASH_OUT"} and abs(oldbalanceOrg - amount) < 1e-6 and abs(newbalanceOrig) < 1e-6:
        rule_risk = max(rule_risk, 0.90)
    if tx_type in {"TRANSFER", "CASH_OUT"} and abs(oldbalanceDest) < 1e-6:
        rule_risk = max(rule_risk, 0.70)
    if flag_dest_new_account == "Yes":
        rule_risk = max(rule_risk, 0.80)

    # Final risk
    final_risk = max(proba_model, rule_risk)

    # Determine label
    pred_label = "Fraudulent Transaction" if final_risk >= threshold else "Legitimate Transaction"

    # Display result
    st.subheader("Prediction Result")
    if final_risk >= threshold:
        st.error(pred_label)
    else:
        st.success(pred_label)

    # Optional: details
    with st.expander("Details"):
        st.write({
            "model_probability": round(proba_model, 4),
            "rule_risk": round(rule_risk, 4),
            "final_risk_used": round(final_risk, 4),
            "threshold": threshold
        })
