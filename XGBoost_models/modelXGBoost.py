import joblib  # Upewnij się, że masz joblib: pip install joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Wczytaj dane z pliku CSV
df = pd.read_csv("synthetic_data_vae.csv")

# Lista kolumn one-hot z etykietami
label_cols = ['Label_Anemia Makrocytarna', 'Label_Anemia Mikrocytarna', 'Label_Anemia Normocytarna', 'Label_Healthy']

# Utwórz pojedynczą kolumnę 'Label' na podstawie kolumn one-hot
df['Label'] = df[label_cols].idxmax(axis=1)

# Usuń prefiks 'Label_' z nazwy etykiety
df['Label'] = df['Label'].str.replace('Label_', '')

# Mapowanie etykiet na wartości numeryczne
label_mapping = {
    "Anemia Mikrocytarna": 0,
    "Anemia Makrocytarna": 1,
    "Anemia Normocytarna": 2,
    "Healthy": 3
}
df['Label_num'] = df['Label'].map(label_mapping)

# Wybierz cechy (features) oraz target
features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
X = df[features]
y = df['Label_num']

# Podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja i trenowanie modelu XGBoost_models
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Zapisz model do pliku
joblib.dump(model, "xgb_model.pkl")
print("Model zapisany jako 'xgb_model.pkl'.")

# Możesz wczytać model z pliku w dowolnym momencie
loaded_model = joblib.load("xgb_model.pkl")
print("Model został wczytany z pliku.")

# Przykładowa predykcja na zbiorze testowym
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy wczytanego modelu: {accuracy:.4f}")
