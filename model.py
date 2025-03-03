import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Wczytanie danych
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = data["fine_tuning_data"]["samples"]["sample_examples"]
    texts = [example["input_text"] for example in examples]
    labels = [example["label"] for example in examples]
    return texts, labels

# 2. Przygotowanie datasetu
class MedicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}  # Poprawione tworzenie tensorów
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 3. Funkcja do treningu
def train(model, train_loader, val_loader, optimizer, device, epochs=3):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # Walidacja
        model.eval()
        val_preds, val_labels = [], []
        for batch in val_loader:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_accuracy}")

# 4. Główna funkcja
def main():
    # Wczytanie danych
    texts, labels = load_data("medicalbert_finetuning_data.json")

    # Kodowanie etykiet
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Tokenizacja
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Podział na zbiór treningowy i walidacyjny
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels_encoded, test_size=0.2, random_state=42
    )

    # Przygotowanie datasetów
    train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
    val_encodings = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")

    train_dataset = MedicalDataset(train_encodings, train_labels)
    val_dataset = MedicalDataset(val_encodings, val_labels)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Wczytanie modelu
    model = AutoModelForSequenceClassification.from_pretrained(
        "dmis-lab/biobert-v1.1", num_labels=len(label_encoder.classes_)
    )

    # Optymalizator
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Trening
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader, val_loader, optimizer, device, epochs=3)

    # Zapisanie modelu i tokenizatora
    model.save_pretrained("./medicalbert-finetuned")
    tokenizer.save_pretrained("./medicalbert-finetuned")
    print("Model został zapisany w folderze './medicalbert-finetuned'.")

# Uruchomienie
if __name__ == "__main__":
    main()