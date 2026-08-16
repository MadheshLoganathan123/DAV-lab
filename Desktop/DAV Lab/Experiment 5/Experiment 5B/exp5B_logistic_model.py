

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              classification_report)

# ── Load Datasets ─────────────────────────────────────────────────────────────
uci_diabetes  = pd.read_csv("../../Experiment 3/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../../Experiment 3/pima_diabetes.csv")

print("UCI Diabetes Dataset Sample:")
print(uci_diabetes.head())
print(f"\nUCI  Shape: {uci_diabetes.shape}")
print("\nPima Indians Diabetes Dataset Sample:")
print(pima_diabetes.head())
print(f"\nPima Shape: {pima_diabetes.shape}")

# ── Select Features and Target Variable ───────────────────────────────────────
features = ["Glucose", "BloodPressure", "BMI"]
target   = "Outcome"   # Binary: 0 = No Diabetes, 1 = Diabetes

print(f"\nFeatures : {features}")
print(f"Target   : {target}")

# ── Function: Build, Validate, and Visualise Logistic Regression Model ─────────
def build_logistic_model(df, features, target, dataset_name, ax):
    """
    Trains a Logistic Regression model, evaluates it, and plots
    the confusion matrix on the provided matplotlib Axes.
    """
    X = df[features]
    y = df[target]

    # Split 80 % training / 20 % testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model (max_iter=1000 ensures convergence)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n{'-' * 55}")
    print(f"{dataset_name} - Logistic Regression Results:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Confusion Matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"],
                ax=ax, linewidths=0.5)
    ax.set_title(
        f"{dataset_name}\nAccuracy={accuracy:.4f}  F1={f1:.4f}",
        fontsize=10
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    return {
        "Dataset"  : dataset_name,
        "Accuracy" : accuracy,
        "Precision": precision,
        "Recall"   : recall,
        "F1 Score" : f1
    }

# ── Run Models on Both Datasets ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("MODEL BUILDING AND VALIDATION - LOGISTIC REGRESSION")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Logistic Regression – Confusion Matrices\nUCI and Pima Diabetes Datasets",
             fontsize=13, fontweight='bold')

results = []
results.append(build_logistic_model(uci_diabetes,  features, target,
                                    "UCI Diabetes Dataset",          axes[0]))
results.append(build_logistic_model(pima_diabetes, features, target,
                                    "Pima Indians Diabetes Dataset", axes[1]))

plt.tight_layout()
plt.savefig("output.png", bbox_inches='tight')
plt.close()
print("\nPlot saved: output.png")

# ── Comparison Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("COMPARISON SUMMARY")
print("=" * 55)
summary_df = pd.DataFrame(results).set_index("Dataset")
print(summary_df.to_string())

print("\nRESULT:")
print("The Logistic Regression Model predicts diabetes presence (Outcome: 0 or 1).")
print("Accuracy, Precision, Recall, and F1 Score indicate classification performance.")
print("Confusion matrices show the distribution of true/false positives and negatives.")
print("Differences in metrics between the two datasets highlight variations in the")
print("classification ability of the model on each dataset.")
