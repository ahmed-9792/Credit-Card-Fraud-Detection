import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(models, X_test, y_test):
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fraud Detection")
    plt.legend(loc="lower right")
    plt.show()
