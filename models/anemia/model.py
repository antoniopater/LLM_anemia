import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocess.anemia.preprocess import changeLabels, scaleData, transformData

# gdzie zapisujemy model
out_dir    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(out_dir, 'modelXGBoost.pkl')

# 1) Wczytanie i enkodowanie etykiet
print("Start training pipeline…")
df = changeLabels()    # czyta synthetic_data_with_probs.csv obok

# 2) Przygotowanie cech i targetu
features = ['RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT','WBC']
X = df[features].values
y = df['Label_num'].values
print("Klasy w y:", sorted(set(y)))

# 3) Podział na train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Skalowanie
X_train_s = scaleData(X_train)    # fit + save
X_test_s  = transformData(X_test) # tylko transform

# 5) Trening
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train_s, y_train)
print("Model wytrenowany.")

# 6) Zapis
joblib.dump(model, MODEL_PATH)
print(f"Model zapisano jako {MODEL_PATH}")

# 7) Ewaluacja
y_pred = model.predict(X_test_s)
acc    = accuracy_score(y_test, y_pred)
print(f"Accuracy na teście: {acc:.4f}")
