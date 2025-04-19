import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from preprocess.anemia import preprocess
from sklearn.model_selection import train_test_split
import preprocess.anemia.preprocess as preprocessData
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

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

known_classes = model.classes_
print("Klasy znane przez model:", known_classes)

# One-hot encoding tylko tych klas, które zna model
y_test_bin = label_binarize(y_test, classes=known_classes)
y_score = model.predict_proba(X_test_pca)

# Przygotuj nazwy klas (upewnij się, że odpowiadają indeksom)
class_names = ['Aplastyczna', 'Hemolityczna', 'Makrocytarna', 'Mikrocytarna', 'Normocytarna', 'Healthy',
               'Trombocytopenia']
# Dopasuj nazwy do znanych klas
class_names_subset = [class_names[i] for i in known_classes]
print("Nazwy klas w y_test:", class_names_subset)
# Oblicz ROC i AUC dla każdej klasy
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(known_classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Rysuj wykres
plt.figure(figsize=(10, 8))
colors = plt.get_cmap('tab10')

for i in range(len(known_classes)):
    plt.plot(fpr[i], tpr[i], lw=2, color=colors(i),
             label=f'{class_names_subset[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywe ROC dla każdej klasy (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve_multiclass.png')
print("Krzywe ROC zapisane jako: roc_curve_multiclass.png")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=known_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_subset)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Macierz pomyłek')
plt.grid(False)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("Unikalne klasy w y_test:", sorted(np.unique(y_test)))
print("Klasy znane przez model:", model.classes_)
