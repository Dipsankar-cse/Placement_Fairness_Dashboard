import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

def train():
    df = pd.read_csv("dataset/placement_data.csv")
    
    # One-hot encode categorical protected feature for training
    df_encoded = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    
    X = df_encoded.drop(columns=["Placed"])
    y = df_encoded["Placed"]
    sensitive_features = df["Gender"]
    
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=42
    )
    
    # 1. Base Unmitigated Model
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    
    base_dp = demographic_parity_difference(y_test, y_pred_base, sensitive_features=A_test)
    print(f"Base Model Demographic Parity Gap: {base_dp:.4f}")
    
    # 2. Fairlearn Bias Mitigation
    mitigator = ExponentiatedGradient(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        constraints=DemographicParity()
    )
    mitigator.fit(X_train, y_train, sensitive_features=A_train)
    
    y_pred_fair = mitigator.predict(X_test)
    fair_dp = demographic_parity_difference(y_test, y_pred_fair, sensitive_features=A_test)
    print(f"Mitigated Model Demographic Parity Gap: {fair_dp:.4f}")
    
    # Save trained models
    os.makedirs("models", exist_ok=True)
    joblib.dump(base_model, "models/base_model.pkl")
    joblib.dump(mitigator, "models/fair_model.pkl")
    print("Models saved successfully in models/ directory!")

if __name__ == "__main__":
    train()