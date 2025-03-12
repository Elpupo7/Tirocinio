import os
import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset

df_full = pd.read_csv('preprocessed_DNN.cs', delimiter=",")
# Colonne rilevanti per ciascun protocollo
colonne_rilevanti = [
    # TCP
    'tcp.ack', 'tcp.ack_raw', 'tcp.checksum', 'tcp.connection.syn',
    'tcp.connection.synack', 'tcp.flags', 'tcp.flags.ack', 'tcp.len', 'tcp.seq',
    # UDP
    'udp.stream', 'udp.time_delta',
    # DNS
    'dns.qry.name', 'dns.qry.qu', 'dns.qry.type', 'dns.retransmission',
    'dns.retransmit_request', 'dns.retransmit_request_in',
    # HTTP
    'http.content_length', 'http.response', 'http.tls_port',
    'http.request.method-GET', 'http.request.method-POST',
    'http.request.method-OPTIONS', 'http.request.method-PROPFIND',
    'http.request.method-PUT', 'http.request.method-SEARCH',
    'http.request.method-TRACE',
    # MQTT
    'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len',
    'mqtt.msg_decoded_as', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.topic_len', 'mqtt.ver',
    # Modbus/TCP
    'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id',
    # Target / Etichetta
    'Attack_label', 'Attack_type'
]

# DataFrame con solo le colonne rilevanti
df = df_full[colonne_rilevanti]

attack_dfs = {attack_type: df[df["Attack_type"] == attack_type]
              for attack_type in df["Attack_type"].unique()}

attack_samples = {
    attack_type: (
        attack_dfs["Fingerprinting"] if attack_type == "Fingerprinting"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else df.sample(n=1000, random_state=42)
    )
    for attack_type, df in attack_dfs.items()
}

# Mapping
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

label2id = {label: idx for idx, label in enumerate(attack_types)}
id2label = {idx: label for label, idx in label2id.items()}
df1["label"] = df1["Attack_type"].map(label2id)

# Dataset Hugging Face
dataset1 = Dataset.from_pandas(df1)
dataset = DatasetDict({"train": dataset1})

# "Testualizzazione"
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
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


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


# DataLoader PyTorch
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
len(train_loader)


from transformers import DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
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


from sklearn.metrics import precision_recall_fscore_support

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

        # Predizione
        preds = torch.argmax(logits, dim=1)
        total_train_correct += (preds == labels).sum().item()
        total_train_samples += labels.size(0)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Calcolo matriche
    train_accuracy = total_train_correct / total_train_samples
    avg_loss = total_loss / len(train_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1-Score: {f1:.4f}")


model.eval()
correct = 0
test_loss = 0
total = 0
all_preds = []
all_labels = []
conta = 0
print("Inizio test")
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
      
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss  # Il modello ci dà già la loss
        logits = outputs.logits  # Logits per le predizioni
        test_loss += loss.item()
        # Predizione
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calcolo metriche
test_loss /= len(test_loader)
test_accuracy = correct / total
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
print(f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}")


# Prova del modello

def predict_attack(model, sample_row, id2label, tokenizer, device):
 
    sample_text = combine_features(sample_row)["text"]
    inputs = tokenizer(sample_text, padding="max_length", truncation=True, max_length=500, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    # Predizione 
    predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
    predicted_label = id2label[predicted_class]

    return predicted_label


# Prova del modello su più esempi assieme
attack_type = 'MITM'
attack_df = df[df["Attack_type"] == attack_type]
attack_df_limited = attack_df.sample(n=15)
print(len(attack_df))
print(type(attack_df))
print(attack_df["Attack_type"].value_counts())
for index, row in attack_df_limited.iterrows():
    if row['Attack_type'] == attack_type:
      predicted_label = predict_attack(model, row, id2label, tokenizer, device)
      print("Attack_label:", row['Attack_label'])
      print("Attack_type:", row['Attack_type'])
      print("Predicted label:", predicted_label)
      print('index: ', index,'\n')






