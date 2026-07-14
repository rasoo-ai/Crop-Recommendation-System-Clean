import pandas as pd
import geopandas as gpd

print("=" * 60)
print("CROP DATASET")
print("=" * 60)

crop = pd.read_excel("output/cleaned_data.xlsx")

print("\nSample State Names:")
print(sorted(crop["State_Name"].dropna().unique())[:20])

print("\nSample District Names:")
print(sorted(crop["District_Name"].dropna().unique())[:20])

print("\nSample Tehsil Names:")
print(sorted(crop["Tehsil_Name"].dropna().unique())[:20])

print("\n" + "=" * 60)
print("BOUNDARY DATASET")
print("=" * 60)

boundary = gpd.read_file("data/SubDistricts_2011.geojsonl")

print("\nSample State Names:")
print(sorted(boundary["stname"].dropna().unique())[:20])

print("\nSample District Names:")
print(sorted(boundary["dtname"].dropna().unique())[:20])

print("\nSample Tehsil Names:")
print(sorted(boundary["sdtname"].dropna().unique())[:20])