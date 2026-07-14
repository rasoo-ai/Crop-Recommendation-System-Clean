import pandas as pd
from rapidfuzz import process, fuzz
import os

print("=" * 60)
print("STEP 11 : FUZZY MATCHING")
print("=" * 60)

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
print("Loading datasets...")

df = pd.read_excel("output/Still_Unmatched.xlsx")
boundary = pd.read_excel("output/Tehsil_GPS_Exact.xlsx")

# ---------------------------------------------------
# Clean boundary reference list
# ---------------------------------------------------
print("Preparing reference dictionary...")

boundary["KEY"] = (
    boundary["STATE_NORM"].astype(str) + "|" +
    boundary["DISTRICT_NORM"].astype(str) + "|" +
    boundary["TEHSIL_NORM"].astype(str)
)

boundary_keys = boundary["KEY"].dropna().unique().tolist()

# ---------------------------------------------------
# Matching function
# ---------------------------------------------------
def fuzzy_match(row):
    key = (
        str(row["STATE_NORM"]) + "|" +
        str(row["DISTRICT_NORM"]) + "|" +
        str(row["TEHSIL_NORM"])
    )

    match = process.extractOne(
        key,
        boundary_keys,
        scorer=fuzz.WRatio
    )

    if match:
        best_match, score, _ = match

        if score >= 90:
            return best_match, score

    return None, None


# ---------------------------------------------------
# Apply fuzzy matching
# ---------------------------------------------------
print("Running fuzzy matching... (this may take time)")

df[["FUZZY_KEY", "MATCH_SCORE"]] = df.apply(
    lambda row: pd.Series(fuzzy_match(row)),
    axis=1
)

# ---------------------------------------------------
# Split matched and unmatched
# ---------------------------------------------------
matched_df = df[df["MATCH_SCORE"].notna()].copy()
unmatched_df = df[df["MATCH_SCORE"].isna()].copy()

# ---------------------------------------------------
# Merge matched back with boundary data
# ---------------------------------------------------
print("Merging results...")

matched_df = matched_df.merge(
    boundary,
    left_on="FUZZY_KEY",
    right_on="KEY",
    how="left"
)

# ---------------------------------------------------
# Save outputs
# ---------------------------------------------------
os.makedirs("output", exist_ok=True)

matched_df.to_excel("output/Fuzzy_Matched.xlsx", index=False)
unmatched_df.to_excel("output/Final_Unmatched.xlsx", index=False)

# ---------------------------------------------------
# Summary
# ---------------------------------------------------
print()
print("=" * 60)
print("FUZZY MATCHING COMPLETED")
print("=" * 60)

print("Fuzzy Matched :", len(matched_df))
print("Still Unmatched :", len(unmatched_df))

print()
print("Saved files:")
print("output/Fuzzy_Matched.xlsx")
print("output/Final_Unmatched.xlsx")