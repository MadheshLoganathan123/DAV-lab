"""
Experiment 2 D) Exploring Descriptive Analytics Using the Iris Dataset

AIM:
To explore descriptive analytics using the Iris dataset with Python's Pandas and Seaborn
libraries.

NOTE: The Iris dataset is loaded via scikit-learn's bundled sklearn.datasets.load_iris(),
which works fully offline and is equivalent in content to the standard iris CSV referenced
in the manual (150 samples, 4 features, 3 species).
"""
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))
df = df.drop(columns=['target'])
df.to_csv('iris_dataset.csv', index=False)

# Display basic information and summary statistics
print("Basic Information:")
df.info()
print("\nSummary Statistics:")
print(df.describe())

# Perform univariate analysis - species count
print("\nSpecies Count:")
print(df['species'].value_counts())

# Visualize data distributions using histograms
df.hist(figsize=(8, 6), edgecolor='black')
plt.suptitle('Feature Distributions')
plt.savefig('feature_distributions.png', bbox_inches='tight')
plt.close()

# Boxplot for Sepal Length
plt.figure()
sns.boxplot(data=df, x='species', y='sepal length (cm)')
plt.title('Sepal Length Comparison')
plt.savefig('sepal_length_boxplot.png', bbox_inches='tight')
plt.close()

# Pairplot to analyze feature relationships
pairplot = sns.pairplot(df, hue='species')
pairplot.savefig('iris_pairplot.png')
plt.close()

print("\nSaved: feature_distributions.png, sepal_length_boxplot.png, iris_pairplot.png")
print("\nRESULT: Descriptive analytics on the Iris dataset completed successfully,")
print("providing insights into feature distributions and species differentiation.")
