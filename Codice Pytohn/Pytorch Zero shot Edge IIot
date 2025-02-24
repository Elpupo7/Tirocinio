import os
import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

df_full = pd.read_csv("preprocessed_DNN.csv", delimiter=",")

attack_dfs = {attack_type: df_full[df_full["Attack_type"] == attack_type]
              for attack_type in df_full["Attack_type"].unique()}

# Tutti quelli che hanno 1000 campioni
attack_samples = {
    attack_type: (
        attack_dfs["Fingerprinting"] if attack_type == "Fingerprinting"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else df.sample(n=1000, random_state=42)
    )
    for attack_type, df in attack_dfs.items()
}

df1 = pd.concat(list(attack_samples.values())).sample(frac=1, random_state=42)
print("Dimensione del nuovo DataFrame:", df1.shape)
print("Distribuzione di Attack_label:")
print(df1["Attack_label"].value_counts())
print('\n')
print(df1["Attack_type"].value_counts())

attack_types = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'DDoS_HTTP', 'Port_Scanning',
    'SQL_injection', 'Password', 'Backdoor', 'Uploading', 'Vulnerability_scanner',
    'XSS', 'Ransomware', 'DDoS_TCP', 'Fingerprinting', 'MITM'
]

# Mapping
label2id = {label: idx for idx, label in enumerate(attack_types)}
id2label = {idx: label for label, idx in label2id.items()}
df1["label"] = df1["Attack_type"].map(label2id)




dataset1 = Dataset.from_pandas(df1)
dataset = DatasetDict({"train": dataset1}) 

    # "Testualizzo" le colonne
def combine_features(example):
    exclude_keys = ['Unnamed: 0', 'Attack_label', 'Attack_type']
    text_parts = []
    for key, value in example.items():
        if key not in exclude_keys:
            text_parts.append(f"{key}: {value}")
    example["text"] = " ".join(text_parts)
    return example

dataset = dataset["train"].map(combine_features)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

from collections import Counter

print("Distribuzione nel train set:")
print(Counter(dataset["train"]["label"]))
print("Distribuzione nel test set:")
print(Counter(dataset["test"]["label"]))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizzazione
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

import torch
from torch.utils.data import DataLoader, Dataset

class AttackDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key in ["input_ids", "attention_mask"]}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = AttackDataset(tokenized_datasets["train"].to_dict(), tokenized_datasets["train"]["label"])
test_dataset = AttackDataset(tokenized_datasets["test"].to_dict(), tokenized_datasets["test"]["label"])
len(train_dataset)

# DataLoader PyTorch
batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
len(train_loader)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(attack_types),
    id2label=id2label,
    label2id=label2id
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device) 




def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Disattiva il calcolo dei gradienti
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            # Predizione
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# Valutazione sul dataset di train
train_loss, train_accuracy = evaluate(model, train_loader, device)
print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

# Valutazione sul dataset di test
test_loss, test_accuracy = evaluate(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
