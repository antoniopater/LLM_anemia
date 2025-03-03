import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# 1. Wczytanie modelu i tokenizatora
MODEL_PATH = "./medicalbert-finetuned"  # Ścieżka do zapisanego modelu
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Etykiety klas (dostosuj do swoich danych)
label_encoder = LabelEncoder()
label_encoder.fit(["microcytic", "macrocytic", "normocytic", "no_anemia"])  # Dodaj "no_anemia" dla przypadku bez anemii

# 2. Funkcja do klasyfikacji
def classify_anemia(input_text):
    # Tokenizacja tekstu
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Predykcja
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Mapowanie wyniku na etykietę
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# 3. Interfejs użytkownika
def main():
    print("Witaj w aplikacji do klasyfikacji anemii!")
    print("Podaj parametry krwi, a aplikacja określi, czy to anemia i jaki jej typ.")

    # Przykładowe parametry krwi
    print("\nPrzykładowy format:")
    print("Pacjent, 45 lat, z objawami zmęczenia i osłabienia. Badania laboratoryjne: RBC: 0.75, HGB: 0.68, HCT: 0.72, MCV: 0.65, MCH: 0.58, MCHC: 0.62, RDW: 1.25, Mikrocyty: ↑, PLT: 1.35.")

    # Pobranie danych od użytkownika
    input_text = input("\nWprowadź parametry krwi: ")

    # Klasyfikacja
    result = classify_anemia(input_text)

    # Wyświetlenie wyniku
    if result == "no_anemia":
        print("\nWynik: Nie stwierdzono anemii.")
    else:
        print(f"\nWynik: Stwierdzono anemię typu {result}.")

# Uruchomienie aplikacji
if __name__ == "__main__":
    main()