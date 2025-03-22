import tkinter as tk
from tkinter import messagebox
import pickle
import pandas as pd
import preprocessData

# Ładowanie modelu z pliku
with open("/Users/antonio/Desktop/LLM_anemia_2/KNN_model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Odwrotna mapa etykiet - mapuje numer na nazwę etykiety
label_mapping_inv = {
    0: "Anemia Mikrocytarna",
    1: "Anemia Makrocytarna",
    2: "Anemia Normocytarna",
    3: "Healthy"
}

# Lista cech używanych w modelu
features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']

def predict():
    try:
        # Pobieramy wartości z pól tekstowych i konwertujemy na float
        input_values = []
        for feature in features:
            value = float(entries[feature].get())
            input_values.append(value)
        # Konwersja danych wejściowych do DataFrame
        df_input = pd.DataFrame([input_values], columns=features)
        # Skalowanie danych przy użyciu funkcji scaleData z preprocessData
        scaled_input = preprocessData.scaleData(df_input)
        # Predykcja modelu
        prediction = model.predict(scaled_input)
        predicted_label = label_mapping_inv.get(prediction[0], "Unknown")
        result_label.config(text=f"Predykcja: {predicted_label}")
    except Exception as e:
        messagebox.showerror("Błąd", str(e))

# Konfiguracja głównego okna GUI
root = tk.Tk()
root.title("BloodAI - Predykcja anemii")

entries = {}
# Tworzymy pola dla każdej cechy
for i, feature in enumerate(features):
    label = tk.Label(root, text=feature)
    label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries[feature] = entry

# Przycisk wywołujący predykcję
predict_button = tk.Button(root, text="Przewiduj", command=predict)
predict_button.grid(row=len(features), column=0, columnspan=2, pady=10)

# Etykieta do wyświetlania wyniku
result_label = tk.Label(root, text="Wynik predykcji zostanie wyświetlony tutaj")
result_label.grid(row=len(features)+1, column=0, columnspan=2, pady=10)

root.mainloop()
