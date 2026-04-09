import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

def print_evaluation(model, X_test, y_test, threshold=None):
    """
    Evaluates the model and prints metrics.
    If threshold is provided, uses it for binary classification instead of 0.5.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    
    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))
    
def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plots the Precision-Recall curve to help find the best threshold.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall vs Threshold")
    plt.legend()
    plt.show()

def get_feature_importances(model, num_cols, cat_cols):
    """
    Extracts and prints feature importances from a Random Forest pipeline.
    """
    cat_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_features = num_cols + list(cat_features)
    
    importances = model.named_steps['model'].feature_importances_
    
    feature_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    feature_df = feature_df.sort_values(by="Importance", ascending=False)
    
    return feature_df
