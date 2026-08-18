import pandas as pd

FILE = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\output\Tehsil_GPS.xlsx"

df = pd.read_excel(FILE)

unmatched = df[df["Latitude"].isna()]

print("Total unmatched:", len(unmatched))

unique = unmatched[
    ["State_Name", "District_Name", "Tehsil_Name"]
].drop_duplicates()

print("Unique unmatched tehsils:", len(unique))

OUT = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\output\Unique_Unmatched_Tehsils.xlsx"

unique.to_excel(OUT, index=False)

print("\nSaved:")
print(OUT)
