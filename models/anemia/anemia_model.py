# train_model.py
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocess.anemia.preprocess_anemia import changeLabels, scaleData, transformData

# ustawienia katalogu i ścieżki do modelu
out_dir    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(out_dir, 'modelXGBoost.pkl')
CSV_PATH   = os.path.join(out_dir, 'dane-anemia.csv')

if __name__ == '__main__':
    print("Start training pipeline…")
    # 1) Wczytanie i enkodowanie etykiet z pliku dane-anemia.csv
    df = changeLabels(csv_path=CSV_PATH)

    # 2) Przygotowanie cech i targetu
    features = ['RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT','WBC']
    X = df[features].values
    y = df['Label_num'].values
    print("Klasy w y:", sorted(set(y)))

    # 3) Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Skalowanie danych
    X_train_s = scaleData(X_train)
    X_test_s  = transformData(X_test)

    # 5) Trening XGBoost
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train_s, y_train)
    print("Model wytrenowany.")

    # 6) Zapis modelu
    joblib.dump(model, MODEL_PATH)
    print(f"Model zapisano jako {MODEL_PATH}")

    # 7) Ewaluacja
    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)
    print(f"Accuracy na teście: {acc:.4f}")
