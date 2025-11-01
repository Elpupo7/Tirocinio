import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from peft import LoraConfig, get_peft_model

# Per addetsrare il modello SULLE SOLE COLONNE RILEVANTI
df_full = pd.read_csv('preprocessed_DNN.csv', delimiter=",")
print(len(df_full.columns))

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

# DF SOLE COLONNE RILEVANTI
df = df_full[colonne_rilevanti]
print(len(df.columns))

# Per usare il dataset predefinito
df = pd.read_csv('preprocessed_DNN.csv', delimiter=",")

print("Dimensione del nuovo DataFrame:", df.shape)
print("Distribuzione di Attack_label:")
print(df["Attack_label"].value_counts())
print('\n')
print(df["Attack_type"].value_counts())

LABEL_COLUMNS = ["Attack_type","Attack_label"]
MAX_LENGTH = 128
batch_size = 32
EPOCHS = 3
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

  def calculate_accuracy(labels, preds, threshold=0.5):
      preds_binary = (preds > threshold).astype(int)
      correct = np.sum(labels == preds_binary)
      total = labels.size
      accuracy = correct / total
      return accuracy * 100

attack_dfs = {attack_type: df[df["Attack_type"] == attack_type]
              for attack_type in df["Attack_type"].unique()}



attack_samples = {
    attack_type: (
        attack_dfs["Fingerprinting"] if attack_type == "Fingerprinting"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else df.sample(n=2000)
    )
    for attack_type, df in attack_dfs.items()
}

df1 = pd.concat(list(attack_samples.values())).sample(frac=1, random_state=42)
print("Dimensione del nuovo DataFrame:", df1.shape)
print("Distribuzione di Attack_label:")
print(df1["Attack_label"].value_counts())
print('\n')
print(df1["Attack_type"].value_counts())


label_encoders = {}
for col in LABEL_COLUMNS:
    label_encoders[col] = LabelEncoder()
    df1[col] = label_encoders[col].fit_transform(df1[col])

# Combine encoded label columns into a list of indices for each sample
df1["multilabel_target"] = df1.apply(lambda row: [row[col] for col in LABEL_COLUMNS], axis=1)

# Concatenate relevant feature columns into a single 'text' column for input to the model
df1["text"] = df1.apply(lambda row: " ".join([str(row[col]) for col in df1.columns if col not in LABEL_COLUMNS + ["multilabel_target"]]), axis=1)

print(df1["multilabel_target"].value_counts())


num_attack_type = len(label_encoders["Attack_type"].classes_)
num_attack_label = len(label_encoders["Attack_label"].classes_)
num_classes_split = [num_attack_type, num_attack_label] # Mi serve per poi caloclare le metriche macro in base al tipo di classe ( ATtack Type, Attack Label)

classi = [] # Mi serve per poi caloclare le metriche per le diverse classi di attacco 
for col, encoder in label_encoders.items():
    print(f"\n{col} - Classi encodate:")
    for i, label in enumerate(encoder.classes_):
        print(f"{label} -> {i}")
        classi.append(label)


num_classes = sum(len(label_encoders[col].classes_) for col in LABEL_COLUMNS)
print(f"[INFO] Number of total classes: {num_classes}")


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df1["text"], df1["multilabel_target"], test_size=0.4, random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)


class AttackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, num_classes, label_encoders, label_columns):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.label_encoders = label_encoders
        self.label_columns = label_columns  # Passato correttamente come attributo

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Inizializza il vettore one-hot con dimensione num_classes
        label_vector = np.zeros(self.num_classes)

        # Calcola gli offset in base alle classi di ciascuna colonna
        offsets = []
        current_offset = 0
        for col in self.label_columns:  
            offsets.append(current_offset)
            current_offset += len(self.label_encoders[col].classes_)

      
        for i, lbl in enumerate(self.labels[idx]):
            # Imposta a 1 nella posizione corretta tenendo conto dell'offset
            label_vector[offsets[i] + int(lbl)] = 1

        encodings = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_vector, dtype=torch.float),
        }


train_dataset = AttackDataset(
    train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LENGTH, num_classes, label_encoders, LABEL_COLUMNS
)
val_dataset = AttackDataset(
    val_texts.tolist(), val_labels.tolist(), tokenizer, MAX_LENGTH, num_classes,label_encoders, LABEL_COLUMNS
)
test_dataset = AttackDataset(
    test_texts.tolist(), test_labels.tolist(), tokenizer, MAX_LENGTH, num_classes, label_encoders, LABEL_COLUMNS
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Optimizer and Loss Function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.BCEWithLogitsLoss()

def train_model():
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Salva predizioni e label
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # **Binarizza le predizioni con soglia 0.5**
    all_preds_binary = (all_preds > 0.5).astype(int)

    # Calcola accuracy
    accuracy = calculate_accuracy(all_labels, all_preds) 

    # Calcola precision, recall e f1-score con 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds_binary, average="macro", zero_division=0
    )

    print(f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1-Score: {f1:.4f}")

    return total_loss / len(train_loader), accuracy

def evaluate_model(loader): #CORRETTA!!
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # **Binarizza le predizioni (0.5 come soglia)**
    all_preds_binary = (all_preds > 0.5).astype(int)

    # Calcola precision, recall e f1-score con 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds_binary, average="macro", zero_division=0
    )

    # Calcola accuracy
    accuracy = calculate_accuracy(all_labels, all_preds)  

    print(f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1-Score: {f1:.4f}")
    return total_loss / len(loader), all_labels, all_preds_binary, accuracy



train_loss_history = []
val_loss_history = []
print(f"[+] Training the model for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    print("\n")
    train_loss, train_accuracy  = train_model()
    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.2f}%")
    val_loss, val_labels, val_preds, val_accuracy = evaluate_model(val_loader)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"\nEpoch {epoch + 1}/{EPOCHS} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

# Plot Metrics
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid(alpha=0.3)



def metrics1(labels, pred, class_splits):
    preds_binary = (pred > 0.5).astype(int)

    start = 0
    for i, class_name in enumerate(LABEL_COLUMNS):
        n_classes = class_splits[i]  # Numero di classi nel gruppo
        end = start + n_classes

        # Estrai solo le classi corrispondenti
        labels_group = labels[:, start:end]
        preds_group = preds_binary[:, start:end]

        # Calcola le metriche con average='macro'
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_group, preds_group, average="macro", zero_division=0
        )

        print(f"Group {class_name} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        start = end


# Funzione per calcolare metriche
def metrics2(labels, pred):
    preds_binary = (pred > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_binary, average=None)
    for i, class_name in enumerate(classi):
        print(f"Class {class_name} | Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}\n")


print("[+] Evaluating on the test set...")
test_loss, test_labels, test_preds, test_accuracy = evaluate_model(test_loader)
print(f"Train test Loss: {test_loss:.4f} | Train test Accuracy: {test_accuracy:.2f}%")
metrics2(test_labels, test_preds)
metrics1(test_labels, test_preds, num_classes_split)
