import pickle
from os.path import split

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import preprocess

df = preprocess.chngeLabels()
features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
X = df[features]
y = df['Label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
X_train_pca = preprocess.fit_scale_pca(X_train)
X_test_pca = preprocess.fit_scale_pca(X_test)

model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train_pca, y_train)

accuracy = model.score(X_test_pca, y_test)
print(f"Accurazy: {accuracy}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(preprocess.scaler, f)
with open("pca.pkl", "wb") as f:
    pickle.dump(preprocess.pca, f)
