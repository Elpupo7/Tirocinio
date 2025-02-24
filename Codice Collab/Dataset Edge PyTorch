from google.colab import files

!pip install -q kaggle

!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

!unzip DNN-EdgeIIoT-dataset.csv.zip

!rm DNN-EdgeIIoT-dataset.csv.zip

import pandas as pd

import numpy as np

df = pd.read_csv('DNN-EdgeIIoT-dataset.csv', low_memory=False)

from sklearn.utils import shuffle

drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4",

         "http.file_data","http.request.full_uri","icmp.transmit_timestamp",

         "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",

         "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)

df.dropna(axis=0, how='any', inplace=True)

df.drop_duplicates(subset=None, keep="first", inplace=True)

df = shuffle(df)

df.isna().sum()

print(df['Attack_type'].value_counts())

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)

encode_text_dummy(df,'http.request.method')

encode_text_dummy(df,'http.referer')

encode_text_dummy(df,"http.request.version")

encode_text_dummy(df,"dns.qry.name.len")

encode_text_dummy(df,"mqtt.conack.flags")

encode_text_dummy(df,"mqtt.protoname")

encode_text_dummy(df,"mqtt.topic")

df.to_csv('preprocessed_DNN.csv', encoding='utf-8')

!pip install datasets

# Importa i moduli necessari
import os
import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

df_full = pd.read_csv("preprocessed_DNN.csv", delimiter=",")

attack_dfs = {attack_type: df_full[df_full["Attack_type"] == attack_type]
              for attack_type in df_full["Attack_type"].unique()}

# Campioni per il train
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

 # "testualizzaizone"
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


# Tokenizzazione
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

import torch
from torch.utils.data import DataLoader, Dataset

class AttackDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings # è un tensore, perchè il metodo items() lo posso usare solo con i tensori
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

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)



# Training loop scheduler
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    for batch in train_loader: # Scorre batch per batch
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

    train_accuracy = total_train_correct / total_train_samples
    avg_loss = total_loss / len(train_loader) 
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")


# Test set
model.eval()  
correct = 0
# total = 0
test_loss = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        test_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)  

test_loss /= len(test_loader)
test_accuracy = correct / total 
print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")



# Prova del modello
def predict_attack(model, sample_row, id2label, tokenizer, device):

    sample_text = combine_features(sample_row)["text"]

    # Tokenizzazione dle testo
    inputs = tokenizer(sample_text, padding="max_length", truncation=True, max_length=500, return_tensors="pt")

    # Sposta i tensori sul device (CPU/GPU)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Evita il calcolo dei gradienti
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    # Predizione
    predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
    predicted_label = id2label[predicted_class]

    return predicted_label


# Carica il DataFrame e filtra i campioni desiderati
# df_full è stato definito all'inizio del codice, mediante: df_full = pd.read_csv("preprocessed_DNN.csv", delimiter=",")
attack_df = df_full[df_full["Attack_label"] == 1]

# Estrazione riga casuale tra gli attacchi
sample_row = attack_df.sample(n=1, random_state=None).iloc[0]

# Per una riga totalemnte casuale
#sample_row = df_full.sample(n=1, random_state=None).iloc[0]

# Predizione
predicted_label = predict_attack(model, sample_row, id2label, tokenizer, device)
print("Attack_label:", sample_row['Attack_label'])
print("Attack_type:", sample_row['Attack_type'])
print("Predicted label:", predicted_label)





