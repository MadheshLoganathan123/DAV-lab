# CS4503 – Data Analytics and Visualization
### Lab Manual | Chennai Institute of Technology (Autonomous)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Datasets](#datasets)
4. [Environment Setup](#environment-setup)
5. [Experiment 1 – Installation and Exploration](#experiment-1--installation-and-exploration)
6. [Experiment 2 – Data Handling and Analysis](#experiment-2--data-handling-and-analysis)
   - [2A – NumPy Arrays](#2a--numpy-arrays)
   - [2B – Pandas DataFrames](#2b--pandas-dataframes)
   - [2C – Reading Data from Multiple Sources](#2c--reading-data-from-multiple-sources)
   - [2D – Descriptive Analytics on Iris Dataset](#2d--descriptive-analytics-on-iris-dataset)
7. [Experiment 3 – Statistical Analysis using Diabetes Datasets](#experiment-3--statistical-analysis-using-diabetes-datasets)
   - [3A – Univariate Analysis](#3a--univariate-analysis)
   - [3B – Bivariate Analysis: Linear and Logistic Regression](#3b--bivariate-analysis-linear-and-logistic-regression)
   - [3C – Multiple Regression Analysis](#3c--multiple-regression-analysis)
   - [3D – Comparative Analysis](#3d--comparative-analysis)
8. [Experiment 4 – Data Visualization and Hypothesis Testing](#experiment-4--data-visualization-and-hypothesis-testing)
   - [4A – Normal Curves](#4a--normal-curves)
   - [4B – Z-Test](#4b--z-test)
   - [4C – T-Test](#4c--t-test)
   - [4D – ANOVA](#4d--anova)
9. [Experiment 5 – Model Building and Validation](#experiment-5--model-building-and-validation)
   - [5A – Linear Regression Model](#5a--linear-regression-model)
   - [5B – Logistic Regression Model](#5b--logistic-regression-model)
   - [5C – Time Series Analysis](#5c--time-series-analysis)
10. [Results Summary](#results-summary)
11. [Libraries Reference](#libraries-reference)

---

## Project Overview

This repository contains all lab experiments for **CS4503: Data Analytics and Visualization**, a course offered at Chennai Institute of Technology. The experiments are implemented in Python and cover the complete pipeline of data analytics — from library exploration and raw data handling, through statistical analysis and hypothesis testing, to predictive model building and time series forecasting.

All experiments use real-world medical datasets (UCI Diabetes and Pima Indians Diabetes) as the primary data source, with supplementary datasets used for specific experiments.

**Python Version:** 3.13.2  
**Jupyter Notebook Version:** 7.3.2

---

## Repository Structure

```
DAV Lab/
│
├── README.md                          ← This file
├── requirements.txt                   ← All Python dependencies
│
├── Experiment 1/
│   ├── exp1_install_explore.py        ← Library installation and version check
│   └── output.txt
│
├── Experiment 2/
│   ├── Experiment 2A/
│   │   ├── exp2A_numpy_arrays.py      ← NumPy operations and array manipulations
│   │   └── output.txt
│   ├── Experiment 2B/
│   │   ├── exp2B_pandas_dataframes.py ← Pandas DataFrame operations
│   │   ├── data.csv                   ← Google Play Store app dataset
│   │   ├── filtered_data.csv          ← Output: filtered subset
│   │   └── output.txt
│   ├── Experiment 2C/
│   │   ├── exp2C_reading_sources.py   ← Reading from CSV, Excel, and web
│   │   ├── products.csv               ← Sample product data
│   │   ├── employees.xlsx             ← Sample employee Excel file
│   │   ├── processed_text.csv         ← Output: processed CSV
│   │   ├── processed_excel.xlsx       ← Output: processed Excel
│   │   ├── web_source_countries.csv   ← Output: web-fetched countries data
│   │   └── output.txt
│   └── Experiment 2D/
│       ├── exp2D_iris_analytics.py    ← Descriptive analytics on Iris dataset
│       ├── iris_dataset.csv           ← Generated Iris dataset CSV
│       └── output.txt
│
├── Experiment 3/
│   ├── uci_diabetes.csv               ← UCI Diabetes dataset (768 rows, 9 cols)
│   ├── pima_diabetes.csv              ← Pima Indians Diabetes dataset (768 rows, 9 cols)
│   ├── Experiment 3A/
│   │   ├── exp3A_univariate_analysis.py
│   │   └── output.txt
│   ├── Experiment 3B/
│   │   ├── exp3B_regression_analysis.py
│   │   └── output.txt
│   ├── Experiment 3C/
│   │   ├── exp3C_multiple_regression.py
│   │   └── output.txt
│   └── Experiment 3D/
│       ├── exp3D_comparative_analysis.py
│       └── output.txt
│
├── Experiment 4/
│   ├── Experiment 4A/
│   │   ├── exp4A_normal_curves.py     ← Normal distribution visualization
│   │   └── output.txt
│   ├── Experiment 4B/
│   │   ├── exp4B_z_test.py            ← Z-Test on Glucose levels
│   │   └── output.txt
│   ├── Experiment 4C/
│   │   ├── exp4C_t_test.py            ← Independent T-Test on both datasets
│   │   └── output.txt
│   └── Experiment 4D/
│       ├── exp4D_anova.py             ← One-Way ANOVA on both datasets
│       └── output.txt
│
└── Experiment 5/
    ├── Experiment 5A/
    │   ├── exp5A_linear_model.py      ← Linear Regression model
    │   └── output.txt
    ├── Experiment 5B/
    │   ├── exp5B_logistic_model.py    ← Logistic Regression model
    │   └── output.txt
    └── Experiment 5C/
        ├── exp5C_time_series.py       ← Time Series Analysis + ARIMA
        ├── diabetes9.csv              ← Generated time-indexed glucose data
        └── output.txt
```

---

## Datasets

### UCI Diabetes Dataset (`Experiment 3/uci_diabetes.csv`)

| Column | Type | Description |
|---|---|---|
| Pregnancies | int | Number of pregnancies |
| Glucose | int | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | int | Diastolic blood pressure (mm Hg) |
| SkinThickness | int | Triceps skin fold thickness (mm) |
| Insulin | int | 2-hour serum insulin (mu U/ml) |
| BMI | float | Body mass index (weight in kg / height in m²) |
| DiabetesPedigreeFunction | float | Diabetes pedigree function (genetic score) |
| Age | int | Age in years |
| Outcome | int | Class label: 1 = diabetes, 0 = no diabetes |

- **Rows:** 768 &nbsp;|&nbsp; **Columns:** 9
- **Outcome distribution:** ~34.9% positive (Outcome=1), ~65.1% negative (Outcome=0)

### Pima Indians Diabetes Dataset (`Experiment 3/pima_diabetes.csv`)

Same schema as the UCI dataset above. Contains health-related attributes of Pima Indian women.

- **Rows:** 768 &nbsp;|&nbsp; **Columns:** 9

### Iris Dataset (auto-generated in Experiment 2D)

Loaded via `sklearn.datasets.load_iris()` and saved to `iris_dataset.csv`.
- **Rows:** 150 &nbsp;|&nbsp; **Columns:** 5 (sepal length, sepal width, petal length, petal width, species)
- **Classes:** Setosa (50), Versicolor (50), Virginica (50)

### Time-Series Dataset (`Experiment 5/Experiment 5C/diabetes9.csv`)

Generated from the UCI Diabetes dataset by assigning daily dates starting 2018-01-01.

| Column | Description |
|---|---|
| Date | Daily date from 2018-01-01 to 2020-02-07 |
| Glucose | Glucose reading for that day |

- **Rows:** 768 &nbsp;|&nbsp; **Date range:** 2018-01-01 to 2020-02-07

---

## Environment Setup

### 1. Create and activate virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install all dependencies

```bash
pip install -r requirements.txt
```

### 3. Run any experiment

Navigate to the experiment's sub-folder and run the script directly:

```bash
# Example – Experiment 4B
cd "Experiment 4/Experiment 4B"
python exp4B_z_test.py
```

Or open any `.py` file in Jupyter Lab:

```bash
jupyter lab
```

> **Note:** All scripts use `matplotlib.use('Agg')` so they can run without a display. Plots are saved as `.png` files in the same folder. Remove or comment out `matplotlib.use('Agg')` if you want interactive plot windows instead.

---

## Experiment 1 – Installation and Exploration

**File:** `Experiment 1/exp1_install_explore.py`

### Aim
To download, install, and verify the features of NumPy, SciPy, Jupyter, Statsmodels, Pandas, Matplotlib, Seaborn, Plotly, and Bokeh for scientific computing, data analysis, and visualization.

### Libraries Covered

| Library | Role |
|---|---|
| NumPy | Multi-dimensional arrays and mathematical operations |
| Pandas | Structured data manipulation via DataFrames and Series |
| Matplotlib | Static, animated, and interactive plots |
| Seaborn | Enhanced statistical visualizations (heatmaps, violin plots, pair plots) |
| SciPy | Scientific computing — optimization, statistics, signal processing |
| Statsmodels | Advanced statistical modeling, regression, hypothesis testing |
| Plotly | Interactive 3D plots, animated charts, dashboards |
| Bokeh | Web-based interactive visualizations with real-time streaming |
| JupyterLab | Interactive notebook environment for Python code |

### Install Command

```bash
pip install numpy scipy jupyter statsmodels pandas matplotlib seaborn plotly bokeh
```

### Key Output

```
NumPy Version     : 2.4.4
Pandas Version    : 3.0.2
Matplotlib Version: 3.10.8
Seaborn Version   : 0.13.2
SciPy Version     : 1.17.1
```

---

## Experiment 2 – Data Handling and Analysis

---

### 2A – NumPy Arrays

**File:** `Experiment 2/Experiment 2A/exp2A_numpy_arrays.py`

#### Aim
To understand and implement various NumPy operations including array creation, indexing, slicing, element-wise operations, aggregations, boolean masking, fancy indexing, reshaping, and structured arrays.

#### Operations Demonstrated

| Operation | Method Used |
|---|---|
| Array creation | `np.array()`, `np.ones()` |
| Indexing / slicing | `arr[i]`, `arr[i:j]`, `arr[i, j]` |
| Arithmetic | `+`, `-`, `*`, `/`, scalar multiply |
| Aggregations | `np.sum()`, `np.mean()`, `np.std()` |
| Boolean masking | `arr[arr > value]` |
| Fancy indexing | `arr[[i, j]]` |
| Reshape | `arr.reshape(rows, cols)` |
| Structured array | `np.array(..., dtype=[('age','i4'), ('score','f4')])` |

#### Key Output

```
Addition        : [11 22 33]
Sum             : 60
Mean            : 20.0
Standard Deviation: 8.16496580927726
Elements > 15   : [20 30]
Reshaped 1D→2D  : [[1][2][3][4][5]]
Structured array: [(25, 90.5) (30, 85.2)]
```

---

### 2B – Pandas DataFrames

**File:** `Experiment 2/Experiment 2B/exp2B_pandas_dataframes.py`  
**Dataset:** `data.csv` (Google Play Store apps — 10,841 rows, 13 columns)

#### Aim
To explore and perform various DataFrame operations using Pandas including loading, inspection, missing value handling, filtering, grouping, sorting, and saving results.

#### Operations Demonstrated

| Step | Operation |
|---|---|
| Load | `pd.read_csv()` |
| Inspect | `.head()`, `.tail()`, `.info()`, `.describe()` |
| Missing values | `.fillna(mean)`, `.dropna()`, `.drop_duplicates()` |
| New columns | `df['new'] = df['existing'] * 2` |
| Filtering | Boolean conditions on multiple columns |
| Grouping | `.groupby().mean()` |
| Sorting | `.sort_values()` |
| Boolean masking | `df[df['col'] > threshold]` |
| Export | `.to_csv('filtered_data.csv')` |

---

### 2C – Reading Data from Multiple Sources

**File:** `Experiment 2/Experiment 2C/exp2C_reading_sources.py`

#### Aim
To read and process data from text files (CSV), Excel spreadsheets, and web-based sources using Pandas.

#### Sources Covered

| Source | Method |
|---|---|
| CSV / Text file | `pd.read_csv('file.csv')` |
| Excel file | `pd.read_excel('file.xlsx', sheet_name='Sheet1')` |
| Web URL | `pd.read_csv('https://...')` |

#### Outputs Generated
- `processed_text.csv` — forward-fill applied, saved to CSV
- `processed_excel.xlsx` — back-fill applied, saved to Excel
- `web_source_countries.csv` — countries data fetched from GitHub raw URL

---

### 2D – Descriptive Analytics on Iris Dataset

**File:** `Experiment 2/Experiment 2D/exp2D_iris_analytics.py`

#### Aim
To explore descriptive analytics using the Iris dataset with Pandas, Seaborn, and Matplotlib.

#### Key Statistics

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| Sepal Length (cm) | 5.843 | 0.828 | 4.3 | 7.9 |
| Sepal Width (cm)  | 3.057 | 0.436 | 2.0 | 4.4 |
| Petal Length (cm) | 3.758 | 1.765 | 1.0 | 6.9 |
| Petal Width (cm)  | 1.199 | 0.762 | 0.1 | 2.5 |

- **Species Count:** Setosa=50, Versicolor=50, Virginica=50

#### Visualizations Saved
- `feature_distributions.png` — histograms of all 4 features
- `sepal_length_boxplot.png` — boxplot comparing sepal length by species
- `iris_pairplot.png` — pairplot with species color-coding

---

## Experiment 3 – Statistical Analysis using Diabetes Datasets

Both datasets (`uci_diabetes.csv` and `pima_diabetes.csv`) are located at `Experiment 3/` and share the same 9-column schema: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`.

---

### 3A – Univariate Analysis

**File:** `Experiment 3/Experiment 3A/exp3A_univariate_analysis.py`

#### Aim
To analyze both diabetes datasets using univariate statistical methods: Mean, Median, Mode, Variance, Standard Deviation, Skewness, and Kurtosis.

#### Columns Analyzed
`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

#### Key Results — UCI Dataset

| Feature | Mean | Median | Std Dev | Skewness | Kurtosis |
|---|---|---|---|---|---|
| Glucose | 120.89 | 117.00 | 31.97 | 0.173 | 0.629 |
| BloodPressure | 69.11 | 72.00 | 19.36 | -1.840 | 5.139 |
| BMI | 31.99 | 32.00 | 7.88 | -0.428 | 3.261 |
| Age | 33.24 | 29.00 | 11.76 | 1.127 | 0.631 |
| Insulin | 79.80 | 30.50 | 115.24 | 2.268 | 7.160 |

---

### 3B – Bivariate Analysis: Linear and Logistic Regression

**File:** `Experiment 3/Experiment 3B/exp3B_regression_analysis.py`

#### Aim
To perform bivariate analysis using Linear Regression (Glucose → BMI) and Logistic Regression (Outcome prediction).

#### Linear Regression Results

| Dataset | X | Y | R² Score |
|---|---|---|---|
| UCI Diabetes | Glucose | BMI | 0.0489 |
| Pima Indians | Glucose | BMI | 0.0489 |

#### Logistic Regression Results

| Dataset | Features | Target | Accuracy |
|---|---|---|---|
| UCI Diabetes | Glucose, BloodPressure, BMI, Age | Outcome | 0.7403 |
| Pima Indians | Glucose, BloodPressure, BMI, Age | Outcome | 0.7403 |

#### Visualizations Saved
- `uci_diabetes_dataset_linear_regression.png`
- `pima_indians_diabetes_dataset_linear_regression.png`

---

### 3C – Multiple Regression Analysis

**File:** `Experiment 3/Experiment 3C/exp3C_multiple_regression.py`

#### Aim
To perform multiple regression analysis to predict BMI using Glucose, BloodPressure, and Age as independent variables.

#### Configuration
- **Features:** Glucose, BloodPressure, Age
- **Target:** BMI
- **Split:** 80% train / 20% test | `random_state=42`

#### Results

| Dataset | R² Score |
|---|---|
| UCI Diabetes | 0.1677 |
| Pima Indians | 0.1677 |

---

### 3D – Comparative Analysis

**File:** `Experiment 3/Experiment 3D/exp3D_comparative_analysis.py`

#### Aim
To compare the statistical analysis results (Univariate, Bivariate, Multiple Regression) of the two diabetes datasets side by side.

#### Key Descriptive Statistics Comparison

| Feature | UCI Mean | Pima Mean | UCI Max | Pima Max |
|---|---|---|---|---|
| Glucose | 120.89 | 120.89 | 199 | 199 |
| BloodPressure | 69.11 | 69.11 | 122 | 122 |
| BMI | 31.99 | 31.99 | 67.1 | 67.1 |
| Insulin | 79.80 | 79.80 | 846 | 846 |
| Age | 33.24 | 33.24 | 81 | 81 |

---

## Experiment 4 – Data Visualization and Hypothesis Testing

All Experiment 4 scripts load data from `../../Experiment 3/uci_diabetes.csv` (and `pima_diabetes.csv` where required).

---

### 4A – Normal Curves

**File:** `Experiment 4/Experiment 4A/exp4A_normal_curves.py`

#### Aim
To visualize the distribution of key numerical attributes in the UCI Diabetes dataset using normal curves, overlaid on histograms with Kernel Density Estimation (KDE).

#### Theory
A **normal curve** (bell curve) represents the probability distribution of a dataset.
- **Mean (µ):** Central location of the distribution
- **Standard Deviation (σ):** Spread / width of the curve
- The theoretical PDF is computed using `scipy.stats.norm.pdf(x, mu, sigma)`

#### Libraries Used
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats.norm`

#### Attributes Visualized
- **Glucose** — histogram + KDE + theoretical normal curve
- **BMI** — histogram + KDE + theoretical normal curve

#### Key Statistics

| Attribute | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Glucose | 120.8945 | 31.9726 | 0.0 | 199.0 |
| BMI | 31.9926 | 7.8842 | 0.0 | 67.1 |

#### Findings
- Glucose distribution shows a slight **right-skew** with most values between 80–160
- BMI distribution is approximately **symmetric** around its mean of ~32
- Both attributes approximate a normal distribution, which validates the use of parametric tests in Experiments 4B–4D

#### Output
- `output.png` — 2-panel figure: Normal Curve for Glucose (left), Normal Curve for BMI (right)

---

### 4B – Z-Test

**File:** `Experiment 4/Experiment 4B/exp4B_z_test.py`

#### Aim
To perform a Z-test on the UCI Diabetes dataset to determine whether the mean Glucose level significantly differs from a reference population mean of 100.

#### Theory
The **Z-test** is used when:
- The sample size is large (n > 30) — here n = 768
- The population mean is known or hypothesized

**Formula:**

```
Z = (x̄ - µ₀) / (σ / √n)
```

Where x̄ = sample mean, µ₀ = hypothesized population mean, σ = std dev, n = sample size.

#### Hypotheses
| | Statement |
|---|---|
| **H0 (Null)** | Mean Glucose level = 100 |
| **H1 (Alternative)** | Mean Glucose level ≠ 100 (two-tailed) |

#### Configuration
- **Variable tested:** Glucose
- **Reference value (µ₀):** 100
- **Significance level (α):** 0.05
- **Method:** `statsmodels.stats.weightstats.ztest()`

#### Results

| Metric | Value |
|---|---|
| Sample Mean | 120.8945 |
| Sample Std Dev | 31.9726 |
| Sample Size (n) | 768 |
| Z-Statistic | **18.1107** |
| P-Value | **0.000000** |

#### Decision
```
p-value (0.000000) < alpha (0.05)  →  REJECT H0
```
**Conclusion:** The mean Glucose level of 120.89 in the UCI Diabetes dataset is **statistically significantly different** from the reference value of 100. The very high Z-statistic (18.11) confirms that diabetes patients in this dataset have substantially elevated glucose levels.

---

### 4C – T-Test

**File:** `Experiment 4/Experiment 4C/exp4C_t_test.py`

#### Aim
To perform an Independent (Welch's) T-test on the UCI Diabetes and Pima Indians Diabetes datasets to compare the means of numerical variables.

#### Theory
The **Independent T-test** (Welch's variant) compares the means of two independent groups without assuming equal variances.

**Types of T-tests:**
| Type | When to Use |
|---|---|
| Independent (Unpaired) | Two separate, independent groups — used here |
| Paired | Same subjects measured twice (before/after) |

**Decision Rule:**
- p < 0.05 → Significant difference exists
- p ≥ 0.05 → No significant difference

#### Configuration
- **Features tested:** Glucose, BloodPressure, BMI
- **Method:** `scipy.stats.ttest_ind(equal_var=False)` — Welch's T-test
- **Significance level (α):** 0.05

#### Results

| Feature | UCI Mean | Pima Mean | T-Statistic | P-Value | Result |
|---|---|---|---|---|---|
| Glucose | 120.8945 | 120.8945 | 0.0000 | 1.0000 | NOT SIGNIFICANT |
| BloodPressure | 69.1055 | 69.1055 | 0.0000 | 1.0000 | NOT SIGNIFICANT |
| BMI | 31.9926 | 31.9926 | 0.0000 | 1.0000 | NOT SIGNIFICANT |

#### Interpretation
The T-statistic of 0 and P-value of 1.0 for all features indicate that both the UCI and Pima datasets are **identical in content** for these columns. No statistically significant difference in means exists between the two datasets.

---

### 4D – ANOVA

**File:** `Experiment 4/Experiment 4D/exp4D_anova.py`

#### Aim
To perform One-Way ANOVA (Analysis of Variance) on the UCI Diabetes and Pima Indians Diabetes datasets to analyze differences between group means.

#### Theory
**ANOVA** tests whether the means of multiple groups are significantly different by analyzing variance within and between groups.

**Types of ANOVA:**
| Type | Description |
|---|---|
| One-Way ANOVA | Compares means of 3+ independent groups — used here |
| Two-Way ANOVA | Examines effect of two categorical independent variables |

**F-Statistic:**
```
F = Variance between groups / Variance within groups
```
A large F-statistic with small p-value indicates significant group differences.

**Decision Rule:**
- p < 0.05 → Significant difference exists between groups
- p ≥ 0.05 → No significant difference

#### Configuration
- **Features tested:** Glucose, BloodPressure, BMI
- **Method:** `scipy.stats.f_oneway()`
- **Significance level (α):** 0.05

#### Results

| Feature | UCI Mean | Pima Mean | F-Statistic | P-Value | Result |
|---|---|---|---|---|---|
| Glucose | 120.8945 | 120.8945 | 0.0000 | 1.0000 | NOT SIGNIFICANT |
| BloodPressure | 69.1055 | 69.1055 | 0.0000 | 1.0000 | NOT SIGNIFICANT |
| BMI | 31.9926 | 31.9926 | 0.0000 | 1.0000 | NOT SIGNIFICANT |

#### Interpretation
Since both datasets contain identical data, the F-statistic is 0 and P-value is 1.0. This confirms no statistically significant difference in group means across any of the tested features. In a real-world scenario with distinct datasets, ANOVA would reveal meaningful group-level differences.

---

## Experiment 5 – Model Building and Validation

All Experiment 5 scripts use an 80%/20% train-test split with `random_state=42`.

---

### 5A – Linear Regression Model

**File:** `Experiment 5/Experiment 5A/exp5A_linear_model.py`

#### Aim
To build and validate Linear Regression Models on both diabetes datasets, predicting Age from Glucose, BloodPressure, and BMI.

#### Theory
**Linear Regression** models the relationship between a continuous dependent variable (target) and one or more independent variables (features) by fitting a straight line (or hyperplane) that minimizes the sum of squared residuals:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

#### Validation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| R² Score | 1 - SS_res/SS_tot | Proportion of variance explained (0–1, higher is better) |
| MSE | mean((y - ŷ)²) | Average squared prediction error (lower is better) |
| MAE | mean(\|y - ŷ\|) | Average absolute prediction error (lower is better) |

#### Configuration
- **Features (X):** Glucose, BloodPressure, BMI
- **Target (y):** Age
- **Split:** 80% train / 20% test | `random_state=42`
- **Library:** `sklearn.linear_model.LinearRegression`

#### Results

| Dataset | R² Score | MSE | MAE | Intercept |
|---|---|---|---|---|
| UCI Diabetes | 0.0749 | 147.8913 | 9.5818 | 17.0301 |
| Pima Indians | 0.0749 | 147.8913 | 9.5818 | 17.0301 |

#### Learned Coefficients

| Feature | Coefficient |
|---|---|
| Glucose | +0.0921 |
| BloodPressure | +0.1399 |
| BMI | -0.1552 |

#### Interpretation
- The R² of 0.075 indicates that Glucose, BloodPressure, and BMI together explain only ~7.5% of the variance in Age — these features have a **weak linear relationship** with Age
- A positive coefficient for Glucose (0.092) and BloodPressure (0.140) suggests that higher values are associated with older patients
- The negative BMI coefficient (-0.155) is unexpected and reflects confounding effects in the data
- The low R² score is consistent with the exploratory nature of this dataset and the limited feature set used

#### Output
- `output.png` — 2-panel Actual vs. Predicted scatter plot for both datasets

---

### 5B – Logistic Regression Model

**File:** `Experiment 5/Experiment 5B/exp5B_logistic_model.py`

#### Aim
To build and validate Logistic Regression Models for predicting diabetes presence (Outcome: 0 or 1) using both datasets.

#### Theory
**Logistic Regression** is used for binary classification. Instead of predicting a continuous value, it estimates the probability that an instance belongs to class 1 (diabetes positive) using the sigmoid function:

```
P(y=1) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))
```

#### Validation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Accuracy | (TP+TN) / Total | Overall correct classification rate |
| Precision | TP / (TP+FP) | Quality of positive predictions |
| Recall | TP / (TP+FN) | Ability to detect all actual positives |
| F1 Score | 2×(P×R)/(P+R) | Harmonic mean of Precision and Recall |
| Confusion Matrix | [[TN,FP],[FN,TP]] | Full breakdown of classification errors |

#### Configuration
- **Features (X):** Glucose, BloodPressure, BMI
- **Target (y):** Outcome (0 = No Diabetes, 1 = Diabetes)
- **Split:** 80% train / 20% test | `random_state=42`
- **Library:** `sklearn.linear_model.LogisticRegression(max_iter=1000)`

#### Results

| Dataset | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| UCI Diabetes | **76.62%** | 69.39% | 61.82% | 65.38% |
| Pima Indians | **76.62%** | 69.39% | 61.82% | 65.38% |

#### Classification Report (per class)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (No Diabetes) | 0.80 | 0.85 | 0.82 | 99 |
| 1 (Diabetes)    | 0.69 | 0.62 | 0.65 | 55 |
| **Overall**     | — | — | **0.77** | **154** |

#### Interpretation
- The model achieves **76.62% accuracy**, correctly classifying ~3 out of 4 patients
- **Precision of 69.4%** for the positive class means that when the model predicts diabetes, it is correct ~70% of the time
- **Recall of 61.8%** means the model detects ~62% of all actual diabetes cases — there is room for improvement in identifying true positives
- The model performs better at classifying non-diabetic patients (Class 0: F1=0.82) than diabetic patients (Class 1: F1=0.65), which is typical for imbalanced datasets
- Adding more features (Pregnancies, SkinThickness, Insulin, Age, DiabetesPedigreeFunction) would likely improve recall for the positive class

#### Output
- `output.png` — 2-panel confusion matrix heatmap for both datasets

---

### 5C – Time Series Analysis

**File:** `Experiment 5/Experiment 5C/exp5C_time_series.py`  
**Dataset:** `Experiment 5/Experiment 5C/diabetes9.csv` (auto-generated on first run)

#### Aim
To perform Time Series Analysis on glucose levels, identifying trends, seasonality, and residual patterns, and to forecast future glucose values using the ARIMA model.

#### Dataset Generation
Since the original diabetes datasets contain no timestamp column, `diabetes9.csv` is generated automatically by assigning daily dates starting `2018-01-01` to each of the 768 patient records, treating the ordered sequence as a daily time series of glucose readings.

#### Theory

**Time Series Components:**

| Component | Description |
|---|---|
| Trend | Long-term increase or decrease in values |
| Seasonality | Repeating periodic patterns within a fixed cycle |
| Residual / Noise | Random variations after removing trend and seasonality |

**ARIMA(p, d, q)** — AutoRegressive Integrated Moving Average:

| Parameter | Meaning |
|---|---|
| p = 5 | Number of autoregressive terms (lag order) |
| d = 1 | Degree of differencing to make the series stationary |
| q = 0 | Number of moving-average terms |

#### Configuration
- **Variable analyzed:** Glucose
- **Date range:** 2018-01-01 to 2020-02-07 (768 daily readings)
- **Decomposition period:** 30 (monthly seasonality cycle)
- **Moving average window:** 7 days
- **ARIMA order:** (5, 1, 0)
- **Split:** 80% train (614 points) / 20% test (154 points)
- **Library:** `statsmodels.tsa.arima.model.ARIMA`, `statsmodels.tsa.seasonal.seasonal_decompose`

#### Dataset Summary

| Statistic | Value |
|---|---|
| Total records | 768 |
| Date range | 2018-01-01 to 2020-02-07 |
| Glucose mean | 120.89 |
| Glucose std dev | 31.97 |
| Training samples | 614 |
| Test samples | 154 |

#### ARIMA Forecast Results

| Metric | Value |
|---|---|
| MSE | 978.3221 |
| MAE | **26.5216** |
| RMSE | **31.2781** |

#### Interpretation
- The **RMSE of 31.28** indicates that ARIMA forecasts deviate from actual glucose values by approximately 31 units on average
- Given a glucose range of 0–199 (std dev ~32), this RMSE represents roughly **1 standard deviation** of prediction error
- The model captures the general trend level but struggles with high-frequency noise in the glucose readings
- The **seasonal decomposition** reveals a mild repeating pattern with period ~30, along with a fluctuating trend component
- The **7-day moving average** effectively smooths out daily noise and highlights the underlying trend

#### Output Files Generated
| File | Contents |
|---|---|
| `glucose_time_series.png` | Raw glucose time series plot |
| `decomposition.png` | 3-panel: Trend / Seasonal / Residual components |
| `moving_average.png` | Original vs. 7-day moving average overlay |
| `arima_forecast.png` | ARIMA forecast vs. actual values |
| `output.png` | 4-panel summary of all the above |

---

## Results Summary

### Hypothesis Testing (Experiment 4)

| Test | Variable | Statistic | P-Value | Decision |
|---|---|---|---|---|
| Z-Test (4B) | Glucose vs µ=100 | Z = 18.11 | 0.000000 | **Reject H0** — Significant |
| T-Test (4C) | Glucose (UCI vs Pima) | T = 0.0 | 1.0 | Fail to Reject H0 |
| T-Test (4C) | BloodPressure | T = 0.0 | 1.0 | Fail to Reject H0 |
| T-Test (4C) | BMI | T = 0.0 | 1.0 | Fail to Reject H0 |
| ANOVA (4D) | Glucose | F = 0.0 | 1.0 | Fail to Reject H0 |
| ANOVA (4D) | BloodPressure | F = 0.0 | 1.0 | Fail to Reject H0 |
| ANOVA (4D) | BMI | F = 0.0 | 1.0 | Fail to Reject H0 |

### Model Performance (Experiments 3 & 5)

| Experiment | Model | Target | Metric | Score |
|---|---|---|---|---|
| 3B | Linear Regression | BMI | R² | 0.0489 |
| 3B | Logistic Regression | Outcome | Accuracy | 74.03% |
| 3C | Multiple Regression | BMI | R² | 0.1677 |
| 5A | Linear Regression | Age | R² / MSE / MAE | 0.0749 / 147.89 / 9.58 |
| 5B | Logistic Regression | Outcome | Accuracy / F1 | 76.62% / 0.6538 |
| 5C | ARIMA(5,1,0) | Glucose | RMSE / MAE | 31.28 / 26.52 |

---

## Libraries Reference

| Library | Version (tested) | Primary Use in This Lab |
|---|---|---|
| `numpy` | 2.4.4 | Array operations, mathematical functions, statistics |
| `pandas` | 3.0.2 | Data loading, manipulation, grouping, filtering |
| `matplotlib` | 3.10.8 | All static visualizations and plot saving |
| `seaborn` | 0.13.2 | Statistical plots — histograms, heatmaps, boxplots, pairplots |
| `scipy` | 1.17.1 | `norm.pdf`, `ttest_ind`, `f_oneway`, `skew`, `kurtosis` |
| `statsmodels` | latest | `ztest`, `ARIMA`, `seasonal_decompose` |
| `scikit-learn` | latest | `LinearRegression`, `LogisticRegression`, all metrics |
| `openpyxl` | latest | `pd.read_excel()` / `pd.to_excel()` support |
| `plotly` | latest | Interactive visualization (version check in Exp 1) |
| `bokeh` | latest | Web-based visualization (version check in Exp 1) |
| `jupyterlab` | latest | Interactive notebook environment |

Install everything at once:

```bash
pip install -r requirements.txt
```

---

*CS4503 Data Analytics and Visualization — Chennai Institute of Technology (Autonomous)*
