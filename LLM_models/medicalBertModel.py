import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load trainingData
df = pd.read_csv("../trainingData/trainingData/medical_data_anemia_patterns.csv")  # Replace with your actual trainingData path

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(df["anemia_type"].unique())}
df["label"] = df["anemia_type"].map(label_mapping)

# Split trainingData into features and target
X = df.drop(columns=["anemia_type", "label"]).values
y = df["label"].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert numeric trainingData to text format
def convert_row_to_text(row):
    # Create a string in the format "FeatureName: value"
    text_representation = " ".join([f"{col}:{value}" for col, value in zip(df.columns[:-1], row)])
    return text_representation

# Apply the conversion to each row
text_data = [convert_row_to_text(row) for row in X_scaled]

# Split trainingData into train and test sets
X_train, X_test, y_train, y_test = train_test_split(text_data, y, test_size=0.2, random_state=42)

# Convert to PyTorch Datasets
class TabularDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}

# Load tokenizer and model
model_name = "distilbert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping)).to(device)

# Prepare the datasets
train_dataset = TabularDataset(X_train, y_train, tokenizer)
test_dataset = TabularDataset(X_test, y_test, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./resultsNew",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="../logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
