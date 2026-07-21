import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

# Load the Datasets
uci_diabetes = pd.read_csv("../uci_diabetes.csv")
pima_diabetes = pd.read_csv("../pima_diabetes.csv")

# Display first few rows
print("UCI Diabetes Dataset Sample:\n", uci_diabetes.head())
print("\nPima Indians Diabetes Dataset Sample:\n", pima_diabetes.head())

# Perform Linear Regression (Glucose vs. BMI)
def linear_regression_analysis(df, x_column, y_column, dataset_name):
    X = df[[x_column]] # Independent variable
    Y = df[y_column] # Dependent variable
    
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    r2 = r2_score(Y, Y_pred)
    
    print(f"\n{dataset_name} - Linear Regression (Predicting {y_column} using {x_column}):")
    print(f"R² Score: {r2:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', alpha=0.5, label='Actual Data')
    plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{dataset_name}: {x_column} vs. {y_column}")
    plt.legend()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_linear_regression.png")
    plt.close()

# Apply Linear Regression on both datasets
linear_regression_analysis(uci_diabetes, "Glucose", "BMI", "UCI Diabetes Dataset")
linear_regression_analysis(pima_diabetes, "Glucose", "BMI", "Pima Indians Diabetes Dataset")

# Perform Logistic Regression (Predicting Diabetes)
def logistic_regression_analysis(df, features, target, dataset_name):
    X = df[features]
    Y = df[target]
    
    # Splitting dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    print(f"\n{dataset_name} - Logistic Regression (Predicting {target} using {features}):")
    print(f"Accuracy Score: {accuracy:.4f}")

# Select features and target
features = ["Glucose", "BloodPressure", "BMI", "Age"]
target = "Outcome"

# Apply Logistic Regression on both datasets
logistic_regression_analysis(uci_diabetes, features, target, "UCI Diabetes Dataset")
logistic_regression_analysis(pima_diabetes, features, target, "Pima Indians Diabetes Dataset")
