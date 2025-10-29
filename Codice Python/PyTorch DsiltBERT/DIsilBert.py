import pandas as pd
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
    #r=16,
    #lora_alpha=32,
    #target_modules=["q_lin", "v_lin", "k_lin"],
    #lora_dropout=0.1,
    #bias="none",
    #task_type="SEQ_CLS"
#)

#model = get_peft_model(model, lora_config)
#model.to(DEVICE)

#NUOVA CLASSE ATTACKDATASET
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




