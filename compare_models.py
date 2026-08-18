import joblib, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_excel('output/Crop_Normalized.xlsx')
target_col = [c for c in df.columns if 'crop' in c.lower() or 'label' in c.lower() or c == df.columns[-1]][0]
df = df.dropna(subset=[target_col])
y = df[target_col]
X = df.drop(columns=[target_col])
for c in X.select_dtypes(include='number').columns:
    X[c] = X[c].fillna(X[c].median())
for c in X.select_dtypes(include='object').columns:
    X[c] = X[c].fillna(X[c].mode()[0])
sx = X.sample(min(500,len(X)), random_state=42)
sy = y.loc[sx.index]

models = {
    'current':      'output/crop_prediction_model_balanced.pkl',
    'balanced':     'output/crop_prediction_model_balanced.pkl',
    'class_weight': 'output/crop_prediction_model_class_weight.pkl',
    'extra_trees':  'output/crop_prediction_model_extra_trees.pkl',
}

best_f1, best_name, results = 0, '', {}
print('Model            | Accuracy   | Macro F1')
print('-' * 42)
for name, path in models.items():
    try:
        m = joblib.load(path)
        p = m.predict(sx)
        acc = accuracy_score(sy, p)
        f1  = f1_score(sy, p, average='macro', zero_division=0)
        results[name] = (acc, f1, path)
        if f1 > best_f1:
            best_f1, best_name = f1, name
        print(name.ljust(16) + ' | ' + str(round(acc*100,2)).rjust(7) + '%  | ' + str(round(f1,3)))
    except Exception as e:
        print(name.ljust(16) + ' | SKIP: ' + str(e)[:40])

print('-' * 42)
print('BEST: ' + best_name + ' (F1=' + str(round(best_f1,3)) + ')')
print('PATH: ' + results[best_name][2])
