"""
Experiment 4 D) Hypothesis Testing – ANOVA on Diabetes Datasets

AIM:
To perform ANOVA (Analysis of Variance) on the UCI Diabetes and Pima Indians Diabetes
datasets to analyze differences between multiple group means.

REQUIREMENTS:
    Python: Version 3.13.2
    Jupyter Notebook: Version 7.3.2

THEORY:
    - Python: An interpreted, general-purpose, high-level programming language.
    - Jupyter Notebook: An interactive environment for executing Python code.
    - NumPy: Fundamental library for numerical computing.
    - Pandas: Data analysis library providing DataFrame and Series structures.
    - SciPy.stats: Module providing scipy.stats.f_oneway() for One-Way ANOVA.
    - UCI Diabetes Dataset: Contains various medical predictor variables (Glucose,
      BloodPressure, BMI, Insulin, Age) with a binary outcome variable.
    - Pima Indians Diabetes Dataset: Contains health-related attributes of Pima Indian
      women (glucose level, blood pressure, BMI) with diabetes diagnosis.
    - Analysis of Variance (ANOVA): A statistical method used to compare the means of
      multiple groups and determine if there are significant differences.
    - Types of ANOVA:
      1. One-Way ANOVA: Compares means of three or more independent groups.
      2. Two-Way ANOVA: Examines the effect of two categorical independent variables
         on a dependent variable.
    - Decision Rule:
      1. p < 0.05  → Significant difference exists between groups.
      2. p >= 0.05 → No significant difference.

PROCEDURE:
    1. Open Jupyter Notebook and import required libraries.
    2. Load the UCI Diabetes and Pima Indians Diabetes datasets.
    3. Select relevant numerical columns: Glucose, BloodPressure, BMI.
    4. Perform One-Way ANOVA on each selected feature.
    5. Display F-statistic and P-value for each variable.
    6. Interpret statistical significance at α = 0.05.

CODE IMPLEMENTATION:
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway

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
print("HYPOTHESIS TESTING - ONE-WAY ANOVA")
print("=" * 60)
print(f"Significance Level (alpha) : {alpha}")
print("H0: The means across groups are NOT significantly different.")
print("H1: At least one group mean IS significantly different.")

# ── Perform One-Way ANOVA ─────────────────────────────────────────────────────
anova_results = {}
for col in numerical_columns:
    f_stat, p_value = f_oneway(
        uci_diabetes[col],
        pima_diabetes[col]
    )
    anova_results[col] = {
        "F-statistic": round(f_stat, 6),
        "P-value"    : round(p_value, 6)
    }

# ── Convert Results to DataFrame ──────────────────────────────────────────────
anova_df = pd.DataFrame(anova_results).T

# ── Display Results ───────────────────────────────────────────────────────────
print("\n--- ANOVA Results ---")
print(anova_df.to_string())

# ── Detailed Interpretation ───────────────────────────────────────────────────
print("\n--- Detailed Interpretation ---")
for col in numerical_columns:
    f_stat  = anova_results[col]["F-statistic"]
    p_value = anova_results[col]["P-value"]
    uci_mean  = uci_diabetes[col].mean()
    pima_mean = pima_diabetes[col].mean()
    significance = "SIGNIFICANT" if p_value < alpha else "NOT SIGNIFICANT"

    print(f"\n{col}:")
    print(f"  UCI  Mean  : {uci_mean:.4f}")
    print(f"  Pima Mean  : {pima_mean:.4f}")
    print(f"  F-statistic: {f_stat:.6f}")
    print(f"  P-value    : {p_value:.6f}")
    print(f"  Result     : {significance} (p {'<' if p_value < alpha else '>='} {alpha})")

print("\nRESULT:")
print("One-Way ANOVA compares the means of Glucose, BloodPressure, and BMI between")
print("the UCI Diabetes and Pima Indians Diabetes datasets.")
print("Variables with F-statistic having p-value < 0.05 show statistically significant")
print("differences across the groups; others exhibit no significant variation.")
