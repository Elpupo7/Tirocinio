import pandas as pd
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
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


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = 'EdgeIoT_250_rows.csv'
#DATASET = 'TON_350_rows.csv' #Per TON
LABEL_COLUMNS = ["Attack_type","Attack_label"]
EPOCHS = 3
BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = 'distilgpt2'

  def calculate_accuracy(labels, preds, threshold=0.5):
      preds_binary = (preds > threshold).astype(int)
      correct = np.sum(labels == preds_binary)
      total = labels.size
      accuracy = correct / total
      return accuracy * 100

print(f"[+] Loading dataset {DATASET}...")
df1 = pd.read_csv(DATASET)


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

num_attack_type = len(label_encoders["Attack_type"].classes_)
num_attack_label = len(label_encoders["Attack_label"].classes_)
num_classes_split = [num_attack_type, num_attack_label]

classi = []
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

print(f"[+] Loading tokenizer and model: {MODEL_NAME}")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
# GPT2 pad token 
tokenizer.pad_token = tokenizer.eos_token

model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels= num_classes).to(DEVICE)
model.config.pad_token_id = model.config.eos_token_id # per far conoscere al modello il token di padding:


# LoRa per TON
#lora_config = LoraConfig(
#        r=16,
#        lora_alpha=32,
#        target_modules=["c_attn", "c_proj"],
#        lora_dropout=0.1,
#        bias="none",
#        task_type="SEQ_CLS"
#    )

#model = get_peft_model(model, lora_config)
#model.to(DEVICE)



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
        # Initialise the one-hot vector with size num_classes
        label_vector = np.zeros(self.num_classes)

        # Calculate offsets based on the classes of each column
        offsets = []
        current_offset = 0
        for col in self.label_columns:
            offsets.append(current_offset)
            current_offset += len(self.label_encoders[col].classes_)


        for i, lbl in enumerate(self.labels[idx]):
            # Set to 1 in the correct position, taking into account the offset
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

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.BCEWithLogitsLoss()


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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Save predictions and labels
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # **Binarise predictions with a threshold of 0.5**
    all_preds_binary = (all_preds > 0.5).astype(int)


    accuracy = calculate_accuracy(all_labels, all_preds)

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
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # **Binarise predictions with a threshold of 0.5**
    all_preds_binary = (all_preds > 0.5).astype(int)

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

# Metrica for EDGE
def metrics_edge(labels, pred, class_splits):
    preds_binary = (pred > 0.5).astype(int)

    start = 0
    for i, class_name in enumerate(LABEL_COLUMNS):
        n_classes = class_splits[i]  # Number of classes in the group
        end = start + n_classes

        # Extract only the corresponding classes
        labels_group = labels[:, start:end]
        preds_group = preds_binary[:, start:end]

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_group, preds_group, average="macro", zero_division=0
        )

        print(f"Group {class_name} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        start = end


# Metric for TON
def metrics_ton(labels, pred, class_splits):
    preds_binary = (pred > 0.5).astype(int)

    start = 0
    for i, class_name in enumerate(LABEL_COLUMNS):
        n_classes = class_splits[i]  # Number of classes in the current group
        end = start + n_classes

        # Extract the columns for the current group
        labels_group = labels[:, start:end]
        preds_group = preds_binary[:, start:end]

        # If the current group is ‘Attack_type’, select only the classes present in TON.
        if class_name == "Attack_type":
            # Relative indices within the ‘Attack_type’ group of Dataset 1 for the classes of TON:
            selected_indices = [3, 4, 6, 7, 8, 9, 11, 14]
    
            labels_group = labels_group[:, selected_indices]
            preds_group = preds_group[:, selected_indices]


        # Calculate metrics for the current group
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_group, preds_group, average="macro", zero_division=0
        )

        print(f"Group {class_name} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        start = end  # Update ‘start’ for the next group

# Metrics by class
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

















