"""
Experiment 4 C) Hypothesis Testing – T-Test on Diabetes Datasets

AIM:
To perform a T-test on the UCI Diabetes and Pima Indians Diabetes datasets to compare
the means of numerical variables and determine statistical significance.

REQUIREMENTS:
    Python: Version 3.13.2
    Jupyter Notebook: Version 7.3.2

THEORY:
    - Python: An interpreted, general-purpose, high-level programming language.
    - Jupyter Notebook: An interactive environment for executing Python code.
    - NumPy: Fundamental library for numerical computing.
    - Pandas: Data analysis library providing DataFrame and Series structures.
    - SciPy.stats: Module providing statistical functions including hypothesis tests.
    - UCI Diabetes Dataset: Contains medical predictor variables (Glucose, BloodPressure,
      BMI, Insulin, Age) and a binary target variable (Outcome).
    - Pima Indians Diabetes Dataset: Contains health-related attributes of Pima Indian
      women (glucose level, blood pressure, BMI) with diabetes diagnosis.
    - T-Test: A statistical hypothesis test used to compare the means of two groups
      and determine if they are significantly different.
    - Types of T-tests:
      1. Independent (Unpaired) T-test: Compares means of two independent datasets.
      2. Paired T-test: Compares means within the same dataset before/after an event.
    - Here we use the Independent (Unpaired) T-test (Welch's t-test: equal_var=False).
    - Decision Rule:
      * p < 0.05  → Significant difference exists between the two groups.
      * p >= 0.05 → No significant difference.

PROCEDURE:
    1. Open Jupyter Notebook and import required libraries.
    2. Load the UCI Diabetes and Pima Indians Diabetes datasets.
    3. Select relevant numerical columns: Glucose, BloodPressure, BMI.
    4. Perform an Independent T-test on each selected feature.
    5. Display T-statistic and P-value for each variable.
    6. Interpret statistical significance at α = 0.05.

CODE IMPLEMENTATION:
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ── Load Datasets ─────────────────────────────────────────────────────────────
uci_diabetes  = pd.read_csv("../../Experiment 3/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../../Experiment 3/pima_diabetes.csv")

print("UCI Diabetes Dataset Sample:")
print(uci_diabetes.head())
print(f"\nUCI  Shape : {uci_diabetes.shape}")
print("\nPima Indians Diabetes Dataset Sample:")
print(pima_diabetes.head())
print(f"\nPima Shape : {pima_diabetes.shape}")

# ── Select Numerical Columns ──────────────────────────────────────────────────
numerical_columns = ["Glucose", "BloodPressure", "BMI"]
alpha = 0.05

print("\n" + "=" * 60)
print("HYPOTHESIS TESTING - INDEPENDENT T-TEST")
print("=" * 60)
print(f"Significance Level (alpha) : {alpha}")
print("H0: The means of the two datasets are NOT significantly different.")
print("H1: The means of the two datasets ARE significantly different.")

# ── Perform Independent T-test ────────────────────────────────────────────────
t_test_results = {}
for col in numerical_columns:
    t_stat, p_value = ttest_ind(
        uci_diabetes[col],
        pima_diabetes[col],
        equal_var=False   # Welch's t-test – does not assume equal variances
    )
    t_test_results[col] = {
        "T-statistic": round(t_stat, 6),
        "P-value"    : round(p_value, 6)
    }

# ── Convert Results to DataFrame ──────────────────────────────────────────────
t_test_df = pd.DataFrame(t_test_results).T

# ── Display Results ───────────────────────────────────────────────────────────
print("\n--- T-Test Results ---")
print(t_test_df.to_string())

# ── Detailed Interpretation ───────────────────────────────────────────────────
print("\n--- Detailed Interpretation ---")
for col in numerical_columns:
    t_stat  = t_test_results[col]["T-statistic"]
    p_value = t_test_results[col]["P-value"]
    uci_mean  = uci_diabetes[col].mean()
    pima_mean = pima_diabetes[col].mean()
    significance = "SIGNIFICANT" if p_value < alpha else "NOT SIGNIFICANT"

    print(f"\n{col}:")
    print(f"  UCI  Mean  : {uci_mean:.4f}")
    print(f"  Pima Mean  : {pima_mean:.4f}")
    print(f"  T-statistic: {t_stat:.6f}")
    print(f"  P-value    : {p_value:.6f}")
    print(f"  Result     : {significance} (p {'<' if p_value < alpha else '>='} {alpha})")

print("\nRESULT:")
print("The Independent T-test compares Glucose, BloodPressure, and BMI between the")
print("UCI Diabetes and Pima Indians Diabetes datasets.")
print("Variables with p-value < 0.05 show a statistically significant difference")
print("in means between the two datasets; others do not.")
