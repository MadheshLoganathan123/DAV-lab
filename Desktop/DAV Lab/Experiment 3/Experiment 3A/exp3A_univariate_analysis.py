import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Import Datasets
uci_diabetes = pd.read_csv("../uci_diabetes.csv")
pima_diabetes = pd.read_csv("../pima_diabetes.csv")

# Display Dataset Samples
print("UCI Diabetes Dataset Sample:")
print(uci_diabetes.head())
print("\nPima Indians Diabetes Dataset Sample:")
print(pima_diabetes.head())

# Define Relevant Numerical Columns
numerical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Univariate Analysis Function
def univariate_analysis(df, columns):
    stats = {}
    for col in columns:
        stats[col] = {
            "Mean": np.mean(df[col]),
            "Median": np.median(df[col]),
            "Mode": df[col].mode()[0],
            "Variance": np.var(df[col], ddof=1),
            "Standard Deviation": np.std(df[col], ddof=1),
            "Skewness": skew(df[col]),
            "Kurtosis": kurtosis(df[col])
        }
    return pd.DataFrame(stats).T

# Perform Univariate Analysis
uci_stats = univariate_analysis(uci_diabetes, numerical_columns)
pima_stats = univariate_analysis(pima_diabetes, numerical_columns)

# Display Results
print("\nUCI Diabetes Dataset Statistics:")
print(uci_stats)
print("\nPima Indians Diabetes Dataset Statistics:")
print(pima_stats)
