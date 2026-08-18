import matplotlib.pyplot as plt
import rioxarray

print("=" * 60)
print("STEP 14F : PLOT NDVI MAP")
print("=" * 60)

# Load NDVI raster
ndvi = rioxarray.open_rasterio("output/TELANGANA_NDVI.tif").squeeze()

# Reduce resolution for plotting only
ndvi_small = ndvi.coarsen(x=10, y=10, boundary="trim").mean()

plt.figure(figsize=(10, 8))

ndvi_small.plot(
    cmap="RdYlGn",
    vmin=-1,
    vmax=1,
    add_colorbar=True
)

plt.title("Telangana NDVI")
plt.axis("off")

plt.savefig(
    "output/TELANGANA_NDVI_Map.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Map saved successfully!")
print("=" * 60)
print("STEP 14F COMPLETED")
print("=" * 60)
