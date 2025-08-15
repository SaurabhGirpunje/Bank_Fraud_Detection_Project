import os, sys
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import ADASYN

# -------------------------------
# Paths and Config
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert at front so imports take precedence

from src.config import DATA_PROCESSED, MODELS_DIR

print('File loading done and preprocessing started.')

# -------------------------------
# Preprocessing
# -------------------------------
def scale_features(train_df, test_df, features_to_scale):
    scaler = StandardScaler()
    train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
    test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
    return train_df, test_df, scaler

# -------------------------------
# Train final model
# -------------------------------
def train_final_model(X_train, y_train, scaler, best_params):
    best_params["scale_pos_weight"] = sum(y_train == 0) / sum(y_train == 1)
    best_params["random_state"] = 42
    best_params["eval_metric"] = "logloss"
    best_params["n_jobs"] = -1

    pipeline = Pipeline([
        ("scaler", scaler),
        ("xgb", XGBClassifier(**best_params))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# -------------------------------
# Evaluate model
# -------------------------------
def evaluate_model(pipeline, X, y, dataset_name="Dataset"):
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:,1]
    print(f"====== {dataset_name} METRICS ======")
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print("\nClassification Report:\n", classification_report(y, preds, digits=4))
    print("ROC-AUC Score:", roc_auc_score(y, probs))
    return preds, probs

# -------------------------------
# Save objects
# -------------------------------
def save_pickle(obj, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")
    return path

# -------------------------------
# Main execution (optional)
# -------------------------------
if __name__ == "__main__":
    # Load train/test
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED, "test.csv"))
    print("Data loaded successfully.")

    features_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                         'oldbalanceDest', 'transaction_hour', 'step']

    train_df, test_df, scaler = scale_features(train_df, test_df, features_to_scale)
    print("Feature scaling completed.")

    X_train = train_df[features_to_scale]
    y_train = train_df['isFraud']
    X_test = test_df[features_to_scale]
    y_test = test_df['isFraud']
    print("Train and test data prepared and scaled.")

    # # Apply ADASYN oversampling
    # adasyn = ADASYN(sampling_strategy=0.2,  # fraud = 20% of non-fraud
    #             n_neighbors=5,
    #             random_state=42)
    # print("ADASYN oversampling applied.")

    # X_train, y_train = adasyn.fit_resample(X_train, y_train)

    Best_Parameters = {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.09063874396765441, 
                       'subsample': 0.9584461997882531, 'colsample_bytree': 0.8675132864407697}
    print(f"Using Best Parameters: {Best_Parameters}")

    final_pipe = train_final_model(X_train, y_train, scaler, Best_Parameters)
    print("Final model trained.")

    # Evaluate
    print("Evaluating on train set...")
    evaluate_model(final_pipe, X_train, y_train, dataset_name="TRAIN")

    print("Evaluating on test set...")
    evaluate_model(final_pipe, X_test, y_test, dataset_name="TEST")

    # Save scaler and pipeline
    save_pickle(scaler, "scaler.pkl")
    save_pickle(final_pipe, "fraud_detection_pipeline.pkl")
    print("Scaler and pipeline saved.")
