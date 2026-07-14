import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
input_file = os.path.join(BASE_DIR, "data", "CROP PREDICTION DATA.xlsx")

df = pd.read_excel(input_file, header=2)

print("="*60)
print("DATASET INFORMATION")
print("="*60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

print("\nFirst 5 Rows:")
print(df.head())