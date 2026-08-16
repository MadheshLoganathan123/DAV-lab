
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ztest

# ── Load Dataset ──────────────────────────────────────────────────────────────
uci_diabetes = pd.read_csv("../../Experiment 3/uci_diabetes.csv")

print("UCI Diabetes Dataset Sample:")
print(uci_diabetes.head())
print(f"\nDataset Shape : {uci_diabetes.shape}")
print(f"Glucose - Mean: {uci_diabetes['Glucose'].mean():.4f}")
print(f"Glucose - Std : {uci_diabetes['Glucose'].std():.4f}")
print(f"Glucose - N   : {len(uci_diabetes['Glucose'])}")

# ── Hypothesis Setup ──────────────────────────────────────────────────────────
population_mean = 100
alpha = 0.05          # 5 % significance level

print("\n" + "=" * 55)
print("HYPOTHESIS TESTING - Z-TEST")
print("=" * 55)
print(f"H0 (Null Hypothesis)       : Mean Glucose = {population_mean}")
print(f"H1 (Alternative Hypothesis): Mean Glucose != {population_mean}")
print(f"Significance Level (alpha) : {alpha}")

# ── Perform Z-Test ────────────────────────────────────────────────────────────
z_stat, p_value = ztest(uci_diabetes["Glucose"], value=population_mean)

# ── Display Results ───────────────────────────────────────────────────────────
print("\n--- Z-Test Results ---")
print(f"Z-Statistic : {z_stat:.4f}")
print(f"P-Value     : {p_value:.6f}")

# ── Interpretation ────────────────────────────────────────────────────────────
print("\n--- Interpretation ---")
if p_value < alpha:
    print(f"Decision: REJECT the null hypothesis (p = {p_value:.6f} < alpha = {alpha})")
    print("Conclusion: The mean Glucose level in the UCI Diabetes dataset is")
    print(f"            SIGNIFICANTLY DIFFERENT from {population_mean}.")
else:
    print(f"Decision: FAIL TO REJECT the null hypothesis (p = {p_value:.6f} >= alpha = {alpha})")
    print("Conclusion: There is NO significant difference between the sample mean")
    print(f"            Glucose level and {population_mean}.")

print("\nRESULT:")
print("The Z-test determines whether the mean Glucose level in the UCI Diabetes")
print("dataset is significantly different from the reference value of 100.")
print("If p-value < 0.05 the null hypothesis is rejected, indicating a significant")
print("difference; otherwise there is no significant difference.")
