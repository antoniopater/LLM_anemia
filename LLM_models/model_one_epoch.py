import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Mapowanie etykiet na wartości liczbowe
label2idx = {
    "normal": 0,
    "anemia_microcytic": 1,
    "anemia_macrocytic": 2,
    "anemia_normocytic": 3
}


class AnemiaDataset(Dataset):
    def __init__(self, csv_path, text_column=None, label_column="anemia_type"):
        print("Ładowanie danych z CSV:", csv_path)
        self.df = pd.read_csv(csv_path)

        if text_column is not None and text_column in self.df.columns:
            self.texts = self.df[text_column].tolist()
            print(f"Użyto kolumny tekstowej: {text_column}")
        else:
            # Łączymy wszystkie kolumny poza etykietą w jeden tekst
            print("Brak dedykowanej kolumny tekstowej, łączenie pozostałych kolumn.")
            self.texts = self.df.drop(columns=[label_column]).apply(
                lambda row: " ".join([f"{col}: {row[col]}" for col in row.index]),
                axis=1
            ).tolist()

        # Mapujemy etykiety tekstowe na liczby
        self.labels = self.df[label_column].map(label2idx).values.astype(np.int64)
        print("Dane zostały załadowane. Liczba próbek:", len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class ClassifierAnemia:
    def __init__(self, num_classes=4, lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu",
                 csv_path="medical_data_anemia_patterns.csv", text_column=None, batch_size=16):
        self.device = device
        print(f"Używany device: {self.device}")

        # Inicjalizacja tokenizera i modelu MedicalBERT.
        # Zastąp "medicalai/ClinicalBERT" właściwym identyfikatorem modelu z Hugging Face.
        print("Ładowanie tokenizera i modelu MedicalBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.model = AutoModelForSequenceClassification.from_pretrained("medicalai/ClinicalBERT",
                                                                        num_labels=num_classes)
        self.model = self.model.to(self.device)
        print("Model został załadowany.")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Przygotowanie datasetu i dataloadera
        print("Przygotowywanie datasetu i DataLoadera...")
        self.dataset = AnemiaDataset(csv_path, text_column=text_column)
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        print("Inicjalizacja zakończona.")

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print("Rozpoczynam trening jednej epoki...")
        for batch_idx, (texts, labels) in enumerate(train_loader):
            # Tokenizacja danych wejściowych
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            labels = torch.tensor(labels).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = self.criterion(outputs.logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print("Epoka zakończona.")
        return epoch_loss, epoch_acc

    def validation(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        print("Rozpoczynam ewaluację...")
        with torch.no_grad():
            for batch_idx, (texts, labels) in enumerate(val_loader):
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                labels = torch.tensor(labels).to(self.device)

                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                if batch_idx % 10 == 0:
                    print(f"  Walidacja - Batch {batch_idx}: Loss = {loss.item():.4f}")

        val_loss = running_loss / total
        val_acc = correct / total
        print("Ewaluacja zakończona.")
        return val_loss, val_acc

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print(f"Model zapisany do: {model_path}")

    def train_model(self, train_loader, val_loader, epochs, model_path="best_model.pth", patience=2, min_delta=0.001):
        best_val_acc = 0.0
        epochs_without_improvement = 0

        print("Rozpoczynam trening modelu...")
        for epoch in range(epochs):
            print(f"\n=== Epoka {epoch + 1}/{epochs} ===")
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validation(val_loader)

            print(f"Epoch [{epoch + 1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            # Sprawdzenie, czy nastąpiła istotna poprawa
            if val_acc - best_val_acc > min_delta:
                best_val_acc = val_acc
                self.save_model(model_path)
                print("  Najlepszy model zapisany.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"  Brak istotnej poprawy przez {epochs_without_improvement} epok.")

            # Jeśli przez 'patience' epok nie było poprawy, zakończ trening
            if epochs_without_improvement >= patience:
                print("Early stopping: brak istotnej poprawy przez kolejne epoki.")
                print("Kończenie treningu.")
                break

        print("Trening zakończony.")


if __name__ == "__main__":
    # Ustawienia
    csv_path = "../trainingData/trainingData/medical_data_anemia_patterns.csv"
    text_column = None  # Jeżeli nie ma kolumny tekstowej, dane są konwertowane z cech tabelarycznych
    batch_size = 32
    num_epochs = 10
    lr = 1e-4

    # Przygotowanie DataLoaderów: dla treningu i walidacji
    print("Przygotowywanie DataLoaderów...")
    dataset = AnemiaDataset(csv_path, text_column=text_column)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("DataLoadery gotowe.")

    # Inicjalizacja modelu
    classifier = ClassifierAnemia(csv_path=csv_path, text_column=text_column, batch_size=batch_size, lr=lr)

    # Trening modelu z walidacją i early stopping
    classifier.train_model(train_loader, val_loader, epochs=num_epochs, model_path="../best_medicalbert_model.pth",
                           patience=2, min_delta=0.001)
