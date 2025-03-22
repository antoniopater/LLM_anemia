from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pickle
import preprocessData

df = preprocessData.changeLabels()

features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
X = df[features]
y = df['Label_num']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = preprocessData.scaleData(X_train)
X_test = preprocessData.scaleData(X_test)
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Zapis modelu do pliku
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model zapisany jako knn_model.pkl")

# Wczytywanie modelu z pliku
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Ewaluacja modelu
accuracy = loaded_model.score(X_test, y_test)
print(f"Accuracy KNN: {accuracy}")