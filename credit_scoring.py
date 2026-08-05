"""
Credit Scoring Model - CodeAlpha ML Internship (Task 1)

Predicts whether a loan applicant is a "good" or "bad" credit risk
using the German Credit dataset (1000 real applicants, UCI/StatLog).

Pipeline: load -> encode -> split -> scale -> train -> evaluate -> visualize
"""

import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

RANDOM_STATE = 42


def load_data(path="german_credit.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_features(df):
    """One-hot encode all text columns; leave numeric columns untouched."""
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"Encoded {len(categorical_cols)} categorical columns "
          f"-> {df_encoded.shape[1]} total columns")
    return df_encoded


def split_data(df):
    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows "
          f"(class balance preserved via stratify)")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, scaler, class_weight=None):
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight=class_weight)
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(model, scaler, X_test, y_test, label="Model"):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n=== {label} — Test Set Performance ===")
    for name, value in metrics.items():
        print(f"{name.capitalize():<10}: {value:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["bad credit (0)", "good credit (1)"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics, y_pred, confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(cm, out_path="outputs/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Bad credit (0)", "Good credit (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Credit Scoring Model")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, out_path="outputs/feature_importance.png", top_n=10):
    coefs = pd.Series(model.coef_[0], index=feature_names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2E86AB" if v > 0 else "#C73E1D" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Weight (positive = pushes toward good credit)")
    ax.set_title(f"Top {top_n} Most Influential Features")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df = load_data()
    df_encoded = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # Baseline model
    scaler = StandardScaler()
    model = train_model(X_train, y_train, scaler)
    metrics, y_pred, cm = evaluate_model(model, scaler, X_test, y_test, label="Baseline Logistic Regression")

    # Class-balanced model (better at catching actual bad-credit applicants)
    scaler_bal = StandardScaler()
    model_bal = train_model(X_train, y_train, scaler_bal, class_weight="balanced")
    metrics_bal, y_pred_bal, cm_bal = evaluate_model(
        model_bal, scaler_bal, X_test, y_test, label="Class-Balanced Logistic Regression"
    )

    # Save visuals from the balanced model (the one we recommend, see README)
    plot_confusion_matrix(cm_bal)
    plot_feature_importance(model_bal, X_train.columns)

    # Save models for reuse
    joblib.dump(model, "logistic_model_baseline.pkl")
    joblib.dump(model_bal, "logistic_model_balanced.pkl")
    joblib.dump(scaler_bal, "scaler.pkl")

    print("\nDone. Models and charts saved.")


if __name__ == "__main__":
    main()
