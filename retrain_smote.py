import pandas as pd, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

print('Loading...')
df = pd.read_excel('output/Crop_Normalized.xlsx')
df = df.dropna(subset=['Crop'])
crop_counts = df['Crop'].value_counts()
df = df[df['Crop'].isin(crop_counts[crop_counts >= 5].index)]
print('Rows:', len(df), '| Crops:', df['Crop'].nunique())

feature_cols = [
    'Soil_Type','pH_Value','Nitrogen_Value (N)','Phosphorus_Value (P)',
    'Potassium_Value (K)','Electrical_Conductivity (EC)',
    'Organic_Carbon (%)','Soil_Moisture (%)','Zinc (%)','Iron (%)',
    'Manganese (%)','Copper (%)','Boron (%)','Sulphur (%)',
    'Rainfall_cm','temperature_celsius','humidity_percentage',
    'State_Name','Agro_Climatic Zone'
]
X = df[feature_cols].copy()
y = df['Crop']
for c in X.select_dtypes(include='number').columns:
    X[c] = pd.to_numeric(X[c],errors='coerce').fillna(X[c].median())
for c in X.select_dtypes(include='object').columns:
    X[c] = X[c].fillna(X[c].mode()[0])
encoders = {}
for c in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    encoders[c] = le

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y)

print('SMOTE...')
X_res,y_res = SMOTE(random_state=42,k_neighbors=3).fit_resample(X_train,y_train)

print('Training 100 trees...')
model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_res,y_res)

preds = model.predict(X_test)
acc = accuracy_score(y_test,preds)
f1  = f1_score(y_test,preds,average='macro',zero_division=0)
print('Accuracy:',round(acc*100,2),'%')
print('Macro F1:',round(f1,3))
print(classification_report(y_test,preds,zero_division=0))

if f1 > 0.692:
    joblib.dump(model,'output/crop_prediction_model_smote.pkl')
    joblib.dump(encoders,'output/smote_encoders.pkl')
    print('SAVED: crop_prediction_model_smote.pkl')
else:
    print('No improvement - keep balanced model')
