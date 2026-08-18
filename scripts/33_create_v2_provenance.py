import pandas as pd

output = "output/Dataset_V2_Provenance.xlsx"

columns = [
    "Record_ID",
    "Crop",
    "State_Name",
    "District_Name",
    "Tehsil_Name",
    "Source_Name",
    "Source_URL",
    "Source_Date",
    "Data_Type",
    "Fields_Verified",
    "Notes",
]

pd.DataFrame(columns=columns).to_excel(
    output,
    index=False
)

print("Created:")
print(output)
