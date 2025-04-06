import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from preprocess.anemia import preprocess
from sklearn.model_selection import train_test_split
import preprocess.anemia.preprocess as preprocessData

name = os.path.basename(__file__)[:-3]
print(name)

df = preprocessData.changeLabels()

features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
X = df[features]
y = df['Label_num']
print("Unikalne klasy w Label_num:", sorted(df['Label_num'].unique()))
print("Liczba przykładów:")
print(df['Label'].value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
X_train_pca = preprocess.scaleData(X_train)
X_test_pca = preprocess.scaleData(X_test)

model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train_pca, y_train)

joblib.dump(model, f"{name}.pkl")
print(f"Model zapisany jako '{name}.pkl'.")

loaded_model = joblib.load(f"{name}.pkl")
print("Model został wczytany z pliku.")

y_pred = loaded_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy wczytanego modelu: {accuracy:.4f}")
