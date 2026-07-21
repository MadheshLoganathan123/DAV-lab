"""
Experiment 2 B) Exploring Pandas DataFrame Operations for Data Manipulation and Analysis

AIM:
To explore and perform various DataFrame operations using Pandas, including loading
datasets, data inspection, handling missing values, transformations, filtering, grouping,
sorting, and saving results.

NOTE: 'data.csv' is a synthetically generated dataset (Apps-style: App, Category, Rating,
Reviews, Size_MB, Installs, Type, Price, Content_Rating) created locally in this offline
sandbox to stand in for the original Google Play Store dataset referenced in the manual.
The code and operations below match the manual exactly.
"""
import pandas as pd

# Load dataset into a DataFrame
df = pd.read_csv('data.csv')

# Display first and last few rows
print("First 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())

# Check data types and general info
print("\nDataFrame Info:")
df.info()

# Summary statistics
print("\nSummary statistics:\n", df.describe())

# Handle missing values
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
print("\nMissing values after fillna:\n", df.isna().sum())

# Create a new column
df['Price_x2'] = df['Price'] * 2

# Create a Series and perform operations
series = df['Reviews']
print("\nSeries addition (first 5):\n", (series + 10).head())

# Filter rows based on conditions
filtered_df = df[(df['Rating'] > 4.0) & (df['Reviews'] < 100000)]
print("\nFiltered DataFrame (Rating>4.0 and Reviews<100000), shape:", filtered_df.shape)
print(filtered_df.head())

# Grouping and aggregation
grouped = df.groupby('Category')['Rating'].mean()
print("\nGrouped mean Rating by Category:\n", grouped)

# Sorting
df_sorted = df.sort_values(by='Reviews', ascending=False)
print("\nTop 5 rows sorted by Reviews (desc):\n", df_sorted.head())

# Boolean masking
masked_df = df[df['Reviews'] > df['Reviews'].median()]
print("\nMasked DataFrame (Reviews > median), shape:", masked_df.shape)

# Remove duplicates and drop missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print("\nShape after drop_duplicates/dropna:", df.shape)

# Create a new DataFrame with selected columns
subset_df = df[['App', 'Category', 'Rating']]

# Save the new DataFrame to a CSV file
subset_df.to_csv('filtered_data.csv', index=False)
print("\nSaved subset_df to filtered_data.csv")

# Compute summary statistics
print("\nTotal sum of Reviews:", df['Reviews'].sum())
print("Mean Reviews:", df['Reviews'].mean())
print("Standard Deviation of Reviews:", df['Reviews'].std())

print("\nRESULT: Pandas DataFrame operations (loading, inspection, missing value handling,")
print("transformations, filtering, grouping, sorting, exporting) executed successfully.")
