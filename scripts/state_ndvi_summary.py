import pandas as pd

print("=" * 60)
print("STATE NDVI SUMMARY")
print("=" * 60)

# Load NDVI data
df = pd.read_csv("output/ALL_INDIA_TEHSIL_NDVI.csv")

# Create state-wise summary
summary = (
    df.groupby("STATE_NORM")["MEAN_NDVI"]
    .agg(
        Average_NDVI="mean",
        Maximum_NDVI="max",
        Minimum_NDVI="min",
        Tehsil_Count="count"
    )
    .reset_index()
)

# Save summary
summary.to_csv("output/STATE_NDVI_SUMMARY.csv", index=False)

print(summary)
print("\nSummary saved to output/STATE_NDVI_SUMMARY.csv")
print("=" * 60)
print("COMPLETED")
print("=" * 60)
