import re

with open('pages/3_Tehsil_Analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    ('st.title("📍 Tehsil-wise Crop Analysis")', 'st.title("Tehsil-wise Crop Analysis")'),
    ('"📍 Generate Tehsil Recommendation"', '"Generate Tehsil Recommendation"'),
    ('"🗺️ Generate State Map"', '"Generate State Map"'),
    ('["🔍 Tehsil Lookup", "🗺️ State Map"]', '["Tehsil Lookup", "State Map"]'),
    ('st.subheader("🗺️ Crop Recommendation Map")', 'st.subheader("Crop Recommendation Map")'),
    ('"📍 Smart Kisan', '"Smart Kisan'),
]

for old, new in replacements:
    content = content.replace(old, new)

with open('pages/3_Tehsil_Analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
