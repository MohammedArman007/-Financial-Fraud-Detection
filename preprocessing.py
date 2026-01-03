import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df = df.copy()

    # Drop transaction_id (not useful for training)
    if "transaction_id" in df.columns:
        df = df.drop(columns=["transaction_id"])

    # Separate target variable
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=["merchant", "location"], drop_first=True)

    # Scale numeric columns
    scaler = StandardScaler()
    numeric_cols = ["amount"]
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Recombine features + target
    df_processed = pd.concat([X, y], axis=1)

    return df_processed, X.columns  # also return feature names
