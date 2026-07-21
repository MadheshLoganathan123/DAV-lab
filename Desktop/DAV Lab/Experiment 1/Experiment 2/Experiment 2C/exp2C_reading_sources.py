"""
Experiment 2 C) Reading Data from Text Files, Excel, and the Web

AIM:
To read and process data from various sources, including text files, Excel spreadsheets,
and web-based data, using Python's Pandas library.

NOTE ON THE "WEB" SOURCE:
This sandbox has no internet access, so a real pd.read_csv(<URL>) call cannot be executed
here. 'web_source_countries.csv' is used locally as a stand-in with the same structure a
web CSV would have. In a normal (internet-connected) environment, simply replace the
pd.read_csv() call below with a live URL, e.g.:

    web_df = pd.read_csv('https://raw.githubusercontent.com/<user>/<repo>/main/countries.csv')
"""
import pandas as pd

# Read data from a text/CSV file
text_df = pd.read_csv('products.csv')

# Read data from an Excel file
excel_df = pd.read_excel('employees.xlsx', sheet_name='Sheet1')

# Read data from a web-based source
# Live version (requires internet):
# web_df = pd.read_csv('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv')
web_df = pd.read_csv('web_source_countries.csv')  # local stand-in, offline sandbox

# Display data
print("Text/CSV data:\n", text_df.head())
print("\nExcel data:\n", excel_df.head())
print("\nWeb-sourced data:\n", web_df.head())

# Handle missing values
text_df = text_df.ffill()
excel_df = excel_df.bfill()
web_df = web_df.dropna()

# Save processed data
text_df.to_csv('processed_text.csv', index=False)
excel_df.to_excel('processed_excel.xlsx', index=False)

print("\nSaved processed_text.csv and processed_excel.xlsx")
print("\nRESULT: Successfully read and processed data from text, Excel, and web-style sources.")
