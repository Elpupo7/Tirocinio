import pandas as pd
import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

df = pd.read_csv("windows10_dataset.csv",  delimiter=",")
df = df.rename(columns={'label': 'Attack_label'}) # Uso label già per il mapping delgi attacchi

print(df['type'].value_counts()) # type: sono i tipi di attacco
print('\n')
print(df['Attack_label'].value_counts()) # indica se una riga è un attacco o meno (1: ATTACCO, 0: NORMALE)

# Mapping attacchi
attack_types = [
    'normal', 'ddos','password','xss','injection', 'dos', 'scanning',
     'mitm'
]
# Dizionari di mapping
label2id = {label: idx for idx, label in enumerate(attack_types)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["type"].map(label2id)

dataset1 = Dataset.from_pandas(df)

# Testualizzo le colonne
def combine_features(example):
    exclude_keys = ['Unnamed: 0', 'Attack_label', 'type']
    text_parts = []
    for key, value in example.items():
        if key not in exclude_keys:
            text_parts.append(f"{key}: {value}")
    example["text"] = " ".join(text_parts)
    return example

dataset1 = dataset1.map(combine_features)
dataset = dataset1.train_test_split(test_size=0.2, seed=42) 

from collections import Counter

#print("Distribuzione nel train set:")
print(Counter(dataset["train"]["label"]))

print("Distribuzione nel test set:")
print(Counter(dataset["test"]["label"])) 

# Tokenizzazione del testo
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class AttackDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings 
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Converti ogni valore in un tensore di PyTorch
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key in ["input_ids", "attention_mask"]}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = AttackDataset(tokenized_datasets["train"].to_dict(), tokenized_datasets["train"]["label"])
test_dataset = AttackDataset(tokenized_datasets["test"].to_dict(), tokenized_datasets["test"]["label"])
len(train_dataset)

# DataLoader PyTorch
batch_size = 36
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

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 2
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_train_correct = 0
    total_train_samples = 0
    all_preds = []
    all_labels = []

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        # Calcolo delle predizioni
        preds = torch.argmax(logits, dim=1)
        total_train_correct += (preds == labels).sum().item()
        total_train_samples += labels.size(0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_accuracy = total_train_correct / total_train_samples
    avg_loss = total_loss / len(train_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1-Score: {f1:.4f}")


from sklearn.metrics import precision_recall_fscore_support
# Test loop
model.eval()
correct = 0
test_loss = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss 
        logits = outputs.logits
        test_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_loss /= len(test_loader)
test_accuracy = correct / total
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
print(f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}")



# Prova del modello
def predict_attack(model, sample_row, id2label, tokenizer, device):
  
    # Combina le colonne per ottenere il testo in input
    sample_text = combine_features(sample_row)["text"]

    # Tokenizzazione
    inputs = tokenizer(sample_text, padding="max_length", truncation=True, max_length=500, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    # Indice della classe con il punteggio più alto
    predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
    predicted_label = id2label[predicted_class]

    return predicted_label


# Prova del modello per un attacco
attack_df = df[df["Attack_label"] == 1]
sample_row = attack_df.sample(n=1, random_state=None).iloc[0]

# Prova del modello su un dato casuale (normale o attacco)
#sample_row = df.sample(n=1, random_state=None).iloc[0]

# Predizione
predicted_label = predict_attack(model, sample_row, id2label, tokenizer, device)
print(sample_row['Attack_label'])
print(sample_row['type'])
print("Predicted label:", predicted_label)
