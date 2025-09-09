import joblib, os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def train_and_save_models(X_train, y_train, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    models = {
        "logreg": LogisticRegression(max_iter=500),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(probability=True, kernel="rbf"),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        joblib.dump(model, os.path.join(save_dir, f"{name}_fraud_model.pkl"))
    return trained

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {"accuracy": acc, "roc_auc": auc, "report": classification_report(y_test, y_pred, output_dict=True)}
    return results
