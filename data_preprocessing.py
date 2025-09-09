import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib, os

def load_data(path="data/raw/creditcard.csv"):
    return pd.read_csv(path)

def preprocess_data(df, save_scaler=True, scaler_path="models/scaler.pkl"):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    if save_scaler:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled, y_train_res, y_test
