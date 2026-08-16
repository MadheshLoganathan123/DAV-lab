"""
Experiment 5 A) Model Building and Validation – Building and Validating Linear Models

AIM:
To build and validate Linear Regression Models using the UCI Diabetes and Pima Indians
Diabetes datasets.

REQUIREMENTS:
    Python: Version 3.13.2
    Jupyter Notebook: Version 7.3.2

THEORY:
    - Python: An interpreted, general-purpose, high-level programming language.
    - Jupyter Notebook: An interactive environment for executing Python code.
    - NumPy: Fundamental library for numerical computing.
    - Pandas: Data analysis library providing DataFrame and Series structures.
    - Matplotlib: Visualization library for creating static, animated, and interactive plots.
    - Seaborn: Statistical data visualization library built on Matplotlib.
    - Scikit-Learn (sklearn): A machine learning library providing tools for regression,
      classification, clustering, and model evaluation.
    - Linear Regression: Models the relationship between a dependent variable (target)
      and one or more independent variables (features). The model learns coefficients
      that minimize the sum of squared residuals.
    - UCI Diabetes Dataset: Contains medical predictor variables and a target variable
      indicating diabetes presence.
    - Pima Indians Diabetes Dataset: Contains health-related attributes such as glucose
      level, BMI, and blood pressure, with a target variable for diabetes diagnosis.
    - Model Validation Metrics:
      1. R² Score (Coefficient of Determination): Measures how well the model explains
         variability in the target variable. Range: 0 to 1 (higher is better).
      2. Mean Squared Error (MSE): Measures the average squared difference between
         actual and predicted values (lower is better).
      3. Mean Absolute Error (MAE): Measures the average absolute difference between
         actual and predicted values (lower is better).

PROCEDURE:
    1. Open Jupyter Notebook and import required libraries.
    2. Load the UCI Diabetes and Pima Indians Diabetes datasets.
    3. Select relevant numerical features: Glucose, BloodPressure, BMI.
    4. Select the target variable: Age.
    5. Split each dataset into training (80%) and testing (20%) sets.
    6. Train a Linear Regression Model using sklearn.
    7. Make predictions on the test set.
    8. Evaluate model performance using R² Score, MSE, and MAE.
    9. Visualize Actual vs. Predicted values for both datasets.

CODE IMPLEMENTATION:
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
target   = "Age"    # Target variable (continuous → suitable for linear regression)

print(f"\nFeatures : {features}")
print(f"Target   : {target}")

# ── Function: Build, Validate, and Visualise Linear Regression Model ──────────
def build_linear_model(df, features, target, dataset_name, ax):
    """
    Trains a Linear Regression model, evaluates it, and plots
    Actual vs. Predicted values on the provided matplotlib Axes.
    """
    X = df[features]
    y = df[target]

    # Split 80 % training / 20 % testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{'-' * 50}")
    print(f"{dataset_name} - Linear Regression Results:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MSE      : {mse:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  Coefficients:")
    for feat, coef in zip(features, model.coef_):
        print(f"    {feat:30s}: {coef:.4f}")

    # Actual vs Predicted scatter plot
    ax.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='k',
               linewidths=0.4, label="Predicted vs Actual")
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color='red', linewidth=2, linestyle='--', label="Perfect Prediction")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{dataset_name}\nR²={r2:.4f}  MSE={mse:.2f}  MAE={mae:.2f}")
    ax.legend(fontsize=8)

    return {"Dataset": dataset_name, "R² Score": r2, "MSE": mse, "MAE": mae}

# ── Run Models on Both Datasets ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("MODEL BUILDING AND VALIDATION - LINEAR REGRESSION")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Linear Regression – Actual vs. Predicted (Age)\nUCI and Pima Diabetes Datasets",
             fontsize=13, fontweight='bold')

results = []
results.append(build_linear_model(uci_diabetes,  features, target, "UCI Diabetes Dataset",          axes[0]))
results.append(build_linear_model(pima_diabetes, features, target, "Pima Indians Diabetes Dataset", axes[1]))

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
print("The Linear Regression Model establishes relationships between the independent")
print("variables (Glucose, BloodPressure, BMI) and the target variable (Age).")
print("R² Score, MSE, and MAE indicate model performance. Differences between the")
print("two datasets highlight variations in data distribution and predictive capability.")
