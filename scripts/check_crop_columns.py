import pandas as pd

df = pd.read_excel("output/Crop_Normalized.xlsx")

print(df.columns.tolist())
print(df.shape)
print(df.head())