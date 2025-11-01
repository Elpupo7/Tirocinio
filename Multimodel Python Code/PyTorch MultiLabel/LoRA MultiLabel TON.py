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
from peft import PeftModel
# Moduli sotto stanti sono usati per utilizzare gli stessi encoder usati per EDGE ( riga 132 a 135)
from google.colab import drive
import pickle

# Codice per usare il dataset con le SOLE COLONNE RILEVANTI
df_full = pd.read_csv('windows10_dataset.csv', delimiter=",")
df_full = df_full.rename(columns={'type': 'Attack_type'})
df_full = df_full.rename(columns={'label': 'Attack_label'})
print(len(df_full.columns))

# Colonne rilevanti
colonne_da_tenere = [
    'Network_I(Intel R _82574L_GNC)TCP_APS',
    'Network_I(Intel R _82574L_GNC) Packets Received Unknown',
    'Network_I(Intel R _82574L_GNC) Bytes Received sec',
    'Network_I(Intel R _82574L_GNC) Bytes Sent sec',
    'Network_I(Intel R _82574L_GNC) Packets Outbound Errors',
    'Network_I(Intel R _82574L_GNC) Packets Received Discarded',
    'Network_I(Intel R _82574L_GNC) Bytes Total sec',
    'Network_I(Intel R _82574L_GNC) Packets Outbound Discarded',
    'Network_I(Intel R _82574L_GNC) TCP RSC Exceptions sec',
    'Network_I(Intel R _82574L_GNC) Packets Sent Unicast sec',
    'Network_I(Intel R _82574L_GNC) Output Queue Length',
    'Network_I(Intel R _82574L_GNC) Packets Received sec',
    'Network_I(Intel R _82574L_GNC) Current Bandwidth',
    'Network_I(Intel R _82574L_GNC) Packets sec',
    'Network_I(Intel R _82574L_GNC) TCP Active RSC Connections',
    'Network_I(Intel R _82574L_GNC) Packets Sent sec',
    'Network_I(Intel R _82574L_GNC) Packets Received Unicast sec',
    'Network_I(Intel R _82574L_GNC) Packets Sent Non-Unicast sec',
    'Network_I(Intel R _82574L_GNC) Packets Received Non-Unicast sec',
    'Network_I(Intel R _82574L_GNC) TCP RSC Coalesced Packets sec',
    'Network_I(Intel R _82574L_GNC) Offloaded Connections',
    'Network_I(Intel R _82574L_GNC) Packets Received Errors',
    'Attack_label',
    'Attack_type'
]

df = df_full[colonne_da_tenere]
print(len(df.columns))


# Codice per usare il dataset PREDEFINITO
df = pd.read_csv('windows10_dataset.csv', delimiter=",")
df = df.rename(columns={'type': 'Attack_type'})
df = df.rename(columns={'label': 'Attack_label'})
print(len(df.columns))


print("Dimensione del nuovo DataFrame:", df.shape)
print("Distribuzione di Attack_label:")
print(df["Attack_label"].value_counts())
print('\n')
print(df["Attack_type"].value_counts())


# Formattazione attacchi di TON in modo da mappare correttamente le occorenze con i tipi di attacco in EDGE
df["Attack_type"] = df["Attack_type"].replace({
    "normal": "Normal",
    "ddos": "DDoS_UDP",
    "password": "Password",
    "xss": "XSS",
    "injection": "SQL_injection",
    "dos": "DDoS_TCP",
    "scanning": "Port_Scanning",
    "mitm": "MITM"
})

print(df["Attack_type"].value_counts())
print(df['Attack_label'].value_counts())


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

# Per usare un sottoinsieme del dataset di partenza
attack_samples = {
    attack_type: (
        attack_dfs["SQL_injection"] if attack_type == "SQL_injection"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else attack_dfs["DDoS_TCP"] if attack_type == "DDoS_TCP"
        else attack_dfs["Port_Scanning"] if attack_type == "Port_Scanning"
        else attack_dfs["XSS"] if attack_type == "XSS"
        else attack_dfs["Password"] if attack_type == "Password"
        else attack_dfs["DDoS_UDP"] if attack_type == "DDoS_UDP"
        else df.sample(n=5000, random_state=42)  # Spostato random_state all'interno di sample()
    )
    for attack_type, df in attack_dfs.items()
}


# attack_samples = {attack_type: df for attack_type, df in attack_dfs.items()} # Per usare l'intero dataset 


df1 = pd.concat(list(attack_samples.values())).sample(frac=1, random_state=42)
print("Dimensione del nuovo DataFrame:", df1.shape)
print("Distribuzione di Attack_label:")
print(df1["Attack_label"].value_counts())
print('\n')
print(df1["Attack_type"].value_counts())


drive.mount('/content/drive')
load_path = "/content/drive/My Drive/label_encoders/label_encoders.pkl"
with open(load_path, "rb") as f:
    label_encoders = pickle.load(f)


df1["Attack_type_encoded"] = label_encoders["Attack_type"].transform(df1["Attack_type"])
df1["Attack_label_encoded"] = label_encoders["Attack_label"].transform(df1["Attack_label"])

df1["multilabel_target"] = df1.apply(lambda row: [row["Attack_type_encoded"], row["Attack_label_encoded"]], axis=1)

df1["text"] = df1.apply(lambda row: " ".join([str(row[col]) for col in df1.columns if col not in LABEL_COLUMNS + ["multilabel_target"]]), axis=1)

print(df1["multilabel_target"].value_counts()) 

num_attack_type = len(label_encoders["Attack_type"].classes_)
num_attack_label = len(label_encoders["Attack_label"].classes_)
num_classes_split = [num_attack_type, num_attack_label] # Usato per il calcolo delle metriche macro di precision, recall e f1


classi = [] # Usato per stampare le metriche precision, recall e f1 per i diversi tipi di attacco 
for col, encoder in label_encoders.items():
    print(f"\n{col} - Classi encodate:")
    for i, label in enumerate(encoder.classes_):
        print(f"{label} -> {i}")
        classi.append(label)


num_classes = sum(len(label_encoders[col].classes_) for col in LABEL_COLUMNS)
print(f"[INFO] Number of total classes: {num_classes}")


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df1["text"], df1["multilabel_target"], test_size=0.3, random_state=42
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

        # Calcola gli offset per definire in modo corretto la rappresentazione one hot (perchè attack label è: 0 oppure 1)
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
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


print("Train loader", len(train_loader))
print("Val loader", len(val_loader))
print("Test loader",len(test_loader))




model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)



lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_lin", "v_lin", "k_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

# LoRa sul modello
model = get_peft_model(model, lora_config) 
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

    # Convert lists to numpy arrays
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


def evaluate_model(loader):
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

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # **Binarizza le predizioni (0.5 come soglia)**
    all_preds_binary = (all_preds > 0.5).astype(int)

    # Calcola le metriche con average="macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds_binary, average="macro", zero_division=0
    )
    accuracy = calculate_accuracy(all_labels, all_preds)  # Calculate accuracy

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
    print(f"\nEpoch {epoch + 1}/{EPOCHS} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)


# CONSENTE DI PRENDERE E CALCOLARE LE MACRO METRICHE SOLO SUGLI EFFETTIVI ATTACK TYPE PRESENTI DENTRO AL DATASET TON
def metrics1(labels, pred, class_splits):  #CORRETTA
    preds_binary = (pred > 0.5).astype(int)

    start = 0
    for i, class_name in enumerate(LABEL_COLUMNS):
        n_classes = class_splits[i]  # Numero di classi nel gruppo corrente
        end = start + n_classes

        # Estrai le colonne per il gruppo corrente
        labels_group = labels[:, start:end]
        preds_group = preds_binary[:, start:end]

        # Se il gruppo corrente è "Attack_type", seleziona solo le classi presenti in Dataset 2
        if class_name == "Attack_type":
            # Indici relativi all'interno del gruppo "Attack_type" di Dataset 1 per le classi di Dataset 2:
            selected_indices = [3, 4, 6, 7, 8, 9, 11, 14]
            # Nota: Poiché in questo gruppo gli indici partono da 0, questi numeri sono validi se il gruppo è completo
            labels_group = labels_group[:, selected_indices]
            preds_group = preds_group[:, selected_indices]


        # Calcola le metriche per il gruppo corrente
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_group, preds_group, average="macro", zero_division=0
        )

        print(f"Group {class_name} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        start = end  # Aggiorna "start" per il prossimo gruppo

# Calcola precision, recall, f1 per ogni tipo in Attack Type e Attack Label
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



