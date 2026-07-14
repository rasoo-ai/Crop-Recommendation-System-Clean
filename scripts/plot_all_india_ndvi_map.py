import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("PLOT ALL INDIA NDVI MAP")
print("=" * 60)

# Load boundary file
gdf = gpd.read_file("output/boundary_normalized.geojson")

# Load NDVI CSV
df = pd.read_csv("output/ALL_INDIA_TEHSIL_NDVI.csv")

# Merge NDVI with boundaries

gdf = gdf.merge(df, on=["STATE_NORM", "TEHSIL_NORM"], how="left")

gdf = gdf.set_geometry("geometry_x")
gdf = gdf.rename_geometry("geometry")

print(gdf.columns.tolist())

# Restore active geometry
if "geometry_x" in gdf.columns:
    gdf = gdf.set_geometry("geometry_x")
    gdf = gdf.rename_geometry("geometry")

# Plot
fig, ax = plt.subplots(figsize=(12, 14))

gdf.plot(
    column="MEAN_NDVI",
    cmap="YlGn",
    linewidth=0.1,
    edgecolor="black",
    legend=True,
    missing_kwds={"color": "lightgrey"},
    ax=ax,
)

ax.set_title("India Tehsil-wise NDVI", fontsize=18)
ax.set_axis_off()

plt.savefig("output/ALL_INDIA_NDVI_MAP.png", dpi=300, bbox_inches="tight")
plt.close()

print("Map saved successfully!")
print("=" * 60)
print("COMPLETED")
print("=" * 60)