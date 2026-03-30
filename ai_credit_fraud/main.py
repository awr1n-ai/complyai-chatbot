import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score

def generate_synthetic_data(n_samples=5000):
    """
    Generate synthetic data for credit scoring and fraud detection.
    """
    np.random.seed(42)
    
    # Features simulating user profiles and transaction details
    age = np.random.randint(18, 70, n_samples)
    income = np.random.uniform(20000, 150000, n_samples)
    credit_history_length = np.random.uniform(0, 30, n_samples) # in years
    debt_to_income = np.random.uniform(0.1, 0.8, n_samples)
    transaction_amount = np.random.exponential(scale=500, size=n_samples)
    location_mismatch = np.random.choice([0, 1], p=[0.9, 0.1], size=n_samples)
    
    # Base risk score (hidden internal mechanics)
    # Higher age/income/history -> lower risk. Higher debt -> higher risk.
    risk_score = (
        (70 - age) / 52 * 0.2 + 
        (150000 - income) / 130000 * 0.3 + 
        (30 - credit_history_length) / 30 * 0.2 + 
        debt_to_income * 0.3
    )
    
    # Target 1: Credit Score (Range ~300 to 850)
    # Lower risk -> higher credit score
    credit_scores = 850 - (risk_score * 550)
    credit_scores = np.clip(credit_scores + np.random.normal(0, 30, n_samples), 300, 850)
    
    # Target 2: Fraud (0 or 1)
    # Fraud is more likely if high transaction amount + location mismatch
    fraud_prob = location_mismatch * 0.4 + (transaction_amount > 2000) * 0.3 + np.random.uniform(0, 0.1, n_samples)
    is_fraud = (fraud_prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_history_length': credit_history_length,
        'debt_to_income': debt_to_income,
        'transaction_amount': transaction_amount,
        'location_mismatch': location_mismatch,
        'credit_score': credit_scores,
        'is_fraud': is_fraud
    })
    
    return df

def train_credit_scoring_model(X_train, y_train):
    """
    Train a regression model to predict the credit score.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_fraud_detection_model(X_train, y_train):
    """
    Train a classification model to detect fraudulent transactions.
    """
    # Using 'balanced' class_weight to address potential class imbalances in fraud data
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("1. Generating Synthetic Data...")
    df = generate_synthetic_data(10000)
    print(f"Dataset Shape: {df.shape}")
    print(f"Fraud Rate: {df['is_fraud'].mean() * 100:.2f}%")
    
    # Independent Features & Dependent Targets
    features = ['age', 'income', 'credit_history_length', 'debt_to_income', 'transaction_amount', 'location_mismatch']
    X = df[features]
    y_credit = df['credit_score']
    y_fraud = df['is_fraud']
    
    # Standardize/Normalize features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_credit_train, y_credit_test, y_fraud_train, y_fraud_test = train_test_split(
        X_scaled, y_credit, y_fraud, test_size=0.2, random_state=42
    )
    
    # ----------------------------------------
    # Credit Scoring Model (Regression)
    # ----------------------------------------
    print("\n2. Training Credit Scoring Model (Regression)...")
    credit_model = train_credit_scoring_model(X_train, y_credit_train)
    
    # Evaluation
    credit_preds = credit_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_credit_test, credit_preds))
    print(f"Credit Scoring RMSE (Root Mean Squared Error): {rmse:.2f} points")
    
    # ----------------------------------------
    # Fraud Detection Model (Classification)
    # ----------------------------------------
    print("\n3. Training Fraud Detection Model (Classification)...")
    fraud_model = train_fraud_detection_model(X_train, y_fraud_train)
    
    # Evaluation
    fraud_preds = fraud_model.predict(X_test)
    fraud_probs = fraud_model.predict_proba(X_test)[:, 1]
    
    print("\nFraud Detection Report:")
    print(classification_report(y_fraud_test, fraud_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_fraud_test, fraud_probs):.4f}")
    
    # ----------------------------------------
    # Feature Importance Insights
    # ----------------------------------------
    print("\n4. Feature Importances for Fraud Detection:")
    importances = fraud_model.feature_importances_
    for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f" - {name}: {imp:.4f}")
