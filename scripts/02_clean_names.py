import pandas as pd
import os
import re

# --------------------------------------------------
# Project Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "data", "CROP PREDICTION DATA.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "cleaned_data.xlsx")

print("=" * 60)
print("STEP 2 : CLEANING STATE, DISTRICT AND TEHSIL NAMES")
print("=" * 60)

# Read Excel (header is on row 3)
df = pd.read_excel(INPUT_FILE, header=2)

print(f"\nRows Loaded : {len(df)}")

# ---------------------------------------------
# Function to clean names
# ---------------------------------------------
def clean_name(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Remove text inside brackets
    text = re.sub(r"\(.*?\)", "", text)

    # Replace &, / and -
    text = text.replace("&", "and")
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9 ]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Cleaning State names...")
df["State_clean"] = df["State_Name"].apply(clean_name)

print("Cleaning District names...")
df["District_clean"] = df["District_Name"].apply(clean_name)

print("Cleaning Tehsil names...")
df["Tehsil_clean"] = df["Tehsil_Name"].apply(clean_name)

print("\nSaving cleaned dataset...")

df.to_excel(OUTPUT_FILE, index=False)

print("\nDone!")
print("Saved to:")
print(OUTPUT_FILE)
