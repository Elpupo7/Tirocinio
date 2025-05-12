import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from tqdm import tqdm
import argparse
from peft import PeftModel
from peft import LoraConfig, get_peft_model


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = 'EdgeIoT_250_rows.csv'
DATASET = 'TON_350_rows.csv'
LABEL_COLUMNS = ["Attack_type", "Attack_label"]
EPOCHS = 2
BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = 'google/t5-efficient-tiny'  # Use a valid T5 model
OUTPUT_MODEL = f'models/t5_trained.pth'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)



  def calculate_accuracy(labels, preds, threshold=0.5):
      preds_binary = (preds > threshold).astype(int)
      correct = np.sum(labels == preds_binary)
      total = labels.size
      accuracy = correct / total
      return accuracy * 100


# Load dataset
print(f"[+] Loading dataset {DATASET}...")
df1 = pd.read_csv(DATASET, delimiter=",")


label_encoders = {}
for col in LABEL_COLUMNS:
    label_encoders[col] = LabelEncoder()
    df1[col] = label_encoders[col].fit_transform(df1[col])

# Combine encoded label columns into a list of indices for each sample
df1["multilabel_target"] = df1.apply(lambda row: [row[col] for col in LABEL_COLUMNS], axis=1)

# Concatenate relevant feature columns into a single 'text' column for input to the model
df1["text"] = df1.apply(lambda row: " ".join([str(row[col]) for col in df1.columns if col not in LABEL_COLUMNS + ["multilabel_target"]]), axis=1)

print(df1["multilabel_target"].value_counts())

#MULTILABEL TON
#from google.colab import drive
#import pickle

#load_path = "/content/drive/My Drive/label_encoders/label_encoders.pkl"

#with open(load_path, "rb") as f:
    #label_encoders = pickle.load(f)

#df1["Attack_type_encoded"] = label_encoders["Attack_type"].transform(df1["Attack_type"])
#df1["Attack_label_encoded"] = label_encoders["Attack_label"].transform(df1["Attack_label"])

#df1["multilabel_target"] = df1.apply(lambda row: [row["Attack_type_encoded"], row["Attack_label_encoded"]], axis=1)

#df1["text"] = df1.apply(lambda row: " ".join([str(row[col]) for col in df1.columns if col not in LABEL_COLUMNS + ["multilabel_target"]]), axis=1)

#print(df1["multilabel_target"].value_counts())


class T5ForMultiLabelClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.config    = T5Config.from_pretrained(model_name)
        self.t5        = T5ForConditionalGeneration.from_pretrained(model_name, config=self.config)

        self.classifier = nn.Linear(self.config.d_model, num_labels)
        self.loss_fn    = nn.BCEWithLogitsLoss()

    def forward(self, *args, **kwargs):
        """
        Accettiamo *args, **kwargs perché PEFT potrebbe
        passarci input_ids, attention_mask, inputs_embeds, ecc.
        """
        # Estrai le label multilabel (se presenti)
        labels = kwargs.pop("labels", None)

        # Esegui l'encoder anche se ci arrivano inputs_embeds:
        encoder_outputs = self.t5.encoder(
            *args,
            **{k: v for k, v in kwargs.items() if k in ["input_ids", "attention_mask", "inputs_embeds"]},
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state  # [B, seq_len, d_model]
        cls_repr = hidden_states[:, 0, :]                  # [B, d_model]

        logits = self.classifier(cls_repr)  # [B, num_labels]

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits

        return logits

model = T5ForMultiLabelClassification(
    model_name=MODEL_NAME,
    num_labels=num_classes
).to(DEVICE)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["SelfAttention.q", "SelfAttention.v"],  # moduli query e value
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)


model = get_peft_model(model, lora_config) 
model.to(DEVICE)

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
        self.label_columns = label_columns

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer 
optimizer = AdamW(model.parameters(), lr=2e-5)

#Quando uso i LoRA
#optimizer = AdamW(model.parameters(), lr=1e-4)


def train_model():
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Salva predizioni e label
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Binarizza le predizioni con soglia 0.5
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
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Binarizza le predizioni (0.5 come soglia)
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


plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid(alpha=0.3)

# Metrica per EDGE
def metrics_edge(labels, pred, class_splits):
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


# Metrica per TON

def metrics_ton(labels, pred, class_splits):  #CORRETTA
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

# Metriche per classe
def metrics(labels, pred):
    preds_binary = (pred > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_binary, average=None)
    for i, class_name in enumerate(classi):
        print(f"Class {class_name} | Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}\n")

print("[+] Evaluating on the test set...")
test_loss, test_labels, test_preds, test_accuracy = evaluate_model(test_loader)
print(f"Train test Loss: {test_loss:.4f} | Train test Accuracy: {test_accuracy:.2f}%")
metrics(test_labels, test_preds)
metrics_edge(test_labels, test_preds, num_classes_split) # Per EDGE
#metrics_ton(test_labels, test_preds, num_classes_split) # Per TON

















