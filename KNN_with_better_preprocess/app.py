# app.py
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pickle
import preprocess

# Wczytujemy model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Wczytujemy dopasowane scaler i pca do preprocessData
with open("scaler.pkl", "rb") as f:
    preprocess.scaler = pickle.load(f)
with open("pca.pkl", "rb") as f:
    preprocess.pca = pickle.load(f)

label_mapping_inv = {
    0: "Anemia Mikrocytarna",
    1: "Anemia Makrocytarna",
    2: "Anemia Normocytarna",
    3: "Healthy"
}

features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']


def predict():
    try:
        # Pobieramy wartości z pól
        input_values = [float(entries[f].get()) for f in features]
        # Tworzymy DF z jedną próbką
        df_input = pd.DataFrame([input_values], columns=features)

        # Transformujemy nową próbkę tym samym scalerem i PCA
        X_input_pca = preprocess.transform_scale_pca(df_input)

        # Predykcja
        prediction = model.predict(X_input_pca)
        predicted_label = label_mapping_inv.get(prediction[0], "Unknown")
        result_label.config(text=f"Predykcja: {predicted_label}")

    except Exception as e:
        messagebox.showerror("Błąd", str(e))


root = tk.Tk()
root.title("BloodAI - Predykcja anemii")

entries = {}
for i, feature in enumerate(features):
    label = tk.Label(root, text=feature)
    label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries[feature] = entry

predict_button = tk.Button(root, text="Przewiduj", command=predict)
predict_button.grid(row=len(features), column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="Wynik predykcji zostanie wyświetlony tutaj")
result_label.grid(row=len(features) + 1, column=0, columnspan=2, pady=10)

root.mainloop()
