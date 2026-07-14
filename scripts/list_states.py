import geopandas as gpd

print("=" * 60)
print("STEP 14A : LIST STATES")
print("=" * 60)

# Load boundary dataset
gdf = gpd.read_file("output/boundary_normalized.geojson")

print("\nAvailable columns:")
print(gdf.columns.tolist())

# Find the state column automatically
possible_columns = [
    "STATE_NORM",
    "State_clean",
    "State_Name",
    "STATE_NAME",
    "STATE",
    "State",
    "state",
    "stname"
]

state_col = None

for col in possible_columns:
    if col in gdf.columns:
        state_col = col
        break

if state_col is None:
    raise Exception("State column not found!")

print(f"\nUsing state column: {state_col}")

states = sorted(gdf[state_col].dropna().unique())

print(f"\nTotal States: {len(states)}\n")

for i, state in enumerate(states, 1):
    print(f"{i}. {state}")