

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# ── Load Dataset ──────────────────────────────────────────────────────────────
uci_diabetes = pd.read_csv("../../Experiment 3/uci_diabetes.csv")

print("UCI Diabetes Dataset Sample:")
print(uci_diabetes.head())
print(f"\nDataset Shape: {uci_diabetes.shape}")

# ── Helper: print descriptive stats for an attribute ──────────────────────────
def print_stats(series, label):
    print(f"\n{label} Statistics:")
    print(f"  Mean              : {series.mean():.4f}")
    print(f"  Standard Deviation: {series.std():.4f}")
    print(f"  Min               : {series.min():.4f}")
    print(f"  Max               : {series.max():.4f}")

print_stats(uci_diabetes["Glucose"], "Glucose")
print_stats(uci_diabetes["BMI"], "BMI")

# ── Plot Normal Curves for Glucose and BMI ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Normal Distribution Curves – UCI Diabetes Dataset", fontsize=14, fontweight='bold')

attributes = [("Glucose", "steelblue", axes[0]),
              ("BMI",     "darkorange", axes[1])]

for col, color, ax in attributes:
    data = uci_diabetes[col]
    mu   = data.mean()
    sigma = data.std()

    # Histogram with KDE
    sns.histplot(data, kde=True, stat="density", color=color,
                 alpha=0.4, linewidth=0, ax=ax, label="Histogram + KDE")

    # Theoretical normal distribution curve
    x = np.linspace(data.min(), data.max(), 300)
    ax.plot(x, norm.pdf(x, mu, sigma), color='red', linewidth=2,
            label=f"Normal Curve\nµ={mu:.2f}, σ={sigma:.2f}")

    ax.set_title(f"Normal Curve – {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("output.png", bbox_inches='tight')
plt.close()

print("\nPlot saved: output.png")
print("\nRESULT:")
print("The normal curves show the distribution of Glucose and BMI in the UCI Diabetes")
print("dataset, indicating data spread, central tendency, and degree of skewness.")
print("Both attributes approximate a normal distribution, with Glucose showing a slight")
print("right-skew and BMI appearing approximately symmetric.")
