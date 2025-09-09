import os, json, joblib
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_and_save_models, evaluate_models

def main():
    os.makedirs("models", exist_ok=True)
    df = load_data("data/raw/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df, save_scaler=True, scaler_path="models/scaler.pkl")
    models = train_and_save_models(X_train, y_train, save_dir="models")
    results = evaluate_models(models, X_test, y_test)
    best_name = max(results.items(), key=lambda kv: kv[1]["roc_auc"])[0]
    joblib.dump(models[best_name], "models/best_model.pkl")
    with open("models/feature_list.json", "w") as f:
        json.dump(df.drop(columns=["Class"]).columns.tolist(), f, indent=2)
    print("Best model saved:", best_name)
    for n, stats in results.items():
        print(f"{n}: ROC-AUC={stats['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
