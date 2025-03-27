import joblib  # Upewnij się, że masz joblib: pip install joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import preprocessData
from KNN_with_better_preprocess import preprocess

df = preprocessData.changeLabels()

features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
X = df[features]
y = df['Label_num']

# Podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
X_train_pca = preprocess.fit_scale_pca(X_train)
X_test_pca = preprocess.fit_scale_pca(X_test)

# Inicjalizacja i trenowanie modelu XGBoost_models
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train_pca, y_train)

# Zapisz model do pliku
joblib.dump(model, "xgb_model.pkl")
print("Model zapisany jako 'xgb_model.pkl'.")

# Możesz wczytać model z pliku w dowolnym momencie
loaded_model = joblib.load("xgb_model.pkl")
print("Model został wczytany z pliku.")

# Przykładowa predykcja na zbiorze testowym
y_pred = loaded_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy wczytanego modelu: {accuracy:.4f}")
