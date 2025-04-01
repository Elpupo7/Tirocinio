import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


max_length = 128
batch_size = 32
EPOCHS = 3
LABEL_COLUMNS = ["Attack_type","Attack_label"]

df = pd.read_csv('preprocessed_DNN.csv', delimiter=",")



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



df1["text"] = df1.apply(
    lambda row: "Identify the attack details based on the input features. "
                "Output format: "
                "Attack_type: <one of: Password, DDoS_UDP, Normal, Ransomware, Port_Scanning, XSS, DDoS_TCP, DDoS_HTTP, "
                "Backdoor, SQL_injection, DDoS_ICMP, Fingerprinting, Vulnerability_scanner, Uploading, MITM>; "
                "Attack_present: <1 if attack is present, 0 if normal traffic>. "
                "Input features: "
                + " ".join([f"{col}: {row[col]}" for col in df1.columns if col not in LABEL_COLUMNS]),
    axis=1
)
print(df1["text"].iloc[0])

df1["target_text"] = df1.apply(lambda row: f"Attack_type: {row['Attack_type']} ; Attack_present: {row['Attack_label']}", axis=1)
print(df1["target_text"].value_counts())


data = df1[["text", "target_text"]].rename(columns={"text": "input_text"}).to_dict(orient="records")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)




class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        input_encodings = self.tokenizer(
            sample["input_text"],
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        target_encodings = self.tokenizer(
            sample["target_text"],
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return {
            "input_ids": input_encodings.input_ids.squeeze(0),
            "attention_mask": input_encodings.attention_mask.squeeze(0),
            "labels": target_encodings.input_ids.squeeze(0)
        }


train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

train_data, val_data= train_test_split(train_data, test_size=0.2, random_state=42)

# Creazione dataset e dataloader
train_dataset = MyDataset(train_data, tokenizer, max_length=max_length)
val_dataset = MyDataset(val_data, tokenizer, max_length=max_length)
test_dataset = MyDataset(test_data, tokenizer, max_length=max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



def parse_prediction(text):
    # formato output deve essere "Attack_type: <valore> ; Attack_present: <valore>"
    try:
        parts = text.split(";")
        attack_type = parts[0].split("Attack_type:")[1].strip()
        attack_present = parts[1].split("Attack_present:")[1].strip()
        return attack_type, attack_present
    except Exception as e:
        return None, None



def train_model(train_dataloader):
    model.train()

    total_loss = 0
    all_attack_types_pred = []
    all_attack_types_true = []
    all_attack_present_pred = []
    all_attack_present_true = []
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Creiamo una copia dei target per le metriche
        labels_for_metrics = labels.clone()

        # Maschera i token di padding nei target per il calcolo della loss
        labels[labels == tokenizer.pad_token_id] = -100

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Generazione delle predizioni
        pred_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        target_texts = tokenizer.batch_decode(labels_for_metrics, skip_special_tokens=True)

        # Parsing delle predizioni e dei target
        for pred, target in zip(pred_texts, target_texts):
            pred_attack_type, pred_attack_present = parse_prediction(pred)
            true_attack_type, true_attack_present = parse_prediction(target)
            if pred_attack_type is not None and true_attack_type is not None:
                all_attack_types_pred.append(pred_attack_type)
                all_attack_types_true.append(true_attack_type)
                all_attack_present_pred.append(pred_attack_present)
                all_attack_present_true.append(true_attack_present)

        # Calcola accuratezza "grezza" (numero di previsioni esatte su tutte)
        correct_preds += sum([1 if pred == target else 0 for pred, target in zip(pred_texts, target_texts)])
        total_preds += len(pred_texts)

    # Calcolo metriche
    accuracy_attack_type = accuracy_score(all_attack_types_true, all_attack_types_pred)
    accuracy_attack_present = accuracy_score(all_attack_present_true, all_attack_present_pred)
    precision_attack_type = precision_score(all_attack_types_true, all_attack_types_pred, average='macro', zero_division=0)
    precision_attack_present = precision_score(all_attack_present_true, all_attack_present_pred, average='macro', zero_division=0)
    recall_attack_type = recall_score(all_attack_types_true, all_attack_types_pred, average='macro', zero_division=0)
    recall_attack_present = recall_score(all_attack_present_true, all_attack_present_pred, average='macro', zero_division=0)
    f1_attack_type = f1_score(all_attack_types_true, all_attack_types_pred, average='macro', zero_division=0)
    f1_attack_present = f1_score(all_attack_present_true, all_attack_present_pred, average='macro', zero_division=0)

    print(f"\nMetrics for Attack Type:")
    print(f"Accuracy: {accuracy_attack_type:.4f} | Precision: {precision_attack_type:.4f} | Recall: {recall_attack_type:.4f} | F1 Score: {f1_attack_type:.4f}")

    print(f"\nMetrics for Attack Present:")
    print(f"Accuracy: {accuracy_attack_present:.4f} | Precision: {precision_attack_present:.4f} | Recall: {recall_attack_present:.4f} | F1 Score: {f1_attack_present:.4f}")

    return total_loss / len(train_dataloader), accuracy_attack_type, accuracy_attack_present




def test_model(test_dataloader):
    conta = 0
    model.eval()
    total_loss = 0
    all_attack_types_pred = []
    all_attack_types_true = []
    all_attack_present_pred = []
    all_attack_present_true = []
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Creiamo una copia dei target per le metriche (senza mascheratura)
            labels_for_metrics = labels.clone()

            # Maschera i token di padding nei target per il calcolo della loss
            labels[labels == tokenizer.pad_token_id] = -100

            # Passaggio in avanti (forward pass)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Predizioni
            pred_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)
            pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            target_texts = tokenizer.batch_decode(labels_for_metrics, skip_special_tokens=True)


            for pred, target in zip(pred_texts, target_texts):
              pred_attack_type, pred_attack_present = parse_prediction(pred)
              true_attack_type, true_attack_present = parse_prediction(target)
              if pred_attack_type is not None and true_attack_type is not None:
                all_attack_types_pred.append(pred_attack_type)
                all_attack_types_true.append(true_attack_type)
                all_attack_present_pred.append(pred_attack_present)
                all_attack_present_true.append(true_attack_present)

            # Calcola accuratezza "grezza" (numero di previsioni esatte su tutte)
            correct_preds += sum([1 if pred == target else 0 for pred, target in zip(pred_texts, target_texts)])
            total_preds += len(pred_texts)

            # Calcolo metriche
            accuracy_attack_type = accuracy_score(all_attack_types_true, all_attack_types_pred)
            accuracy_attack_present = accuracy_score(all_attack_present_true, all_attack_present_pred)
            precision_attack_type = precision_score(all_attack_types_true, all_attack_types_pred, average='macro', zero_division=0)
            precision_attack_present = precision_score(all_attack_present_true, all_attack_present_pred, average='macro', zero_division=0)
            recall_attack_type = recall_score(all_attack_types_true, all_attack_types_pred, average='macro', zero_division=0)
            recall_attack_present = recall_score(all_attack_present_true, all_attack_present_pred, average='macro', zero_division=0)
            f1_attack_type = f1_score(all_attack_types_true, all_attack_types_pred, average='macro', zero_division=0)
            f1_attack_present = f1_score(all_attack_present_true, all_attack_present_pred, average='macro', zero_division=0)

    print(f"\nMetrics for Attack Type:")
    print(f"Accuracy: {accuracy_attack_type:.4f} | Precision: {precision_attack_type:.4f} | Recall: {recall_attack_type:.4f} | F1 Score: {f1_attack_type:.4f}")

    print(f"\nMetrics for Attack Present:")
    print(f"Accuracy: {accuracy_attack_present:.4f} | Precision: {precision_attack_present:.4f} | Recall: {recall_attack_present:.4f} | F1 Score: {f1_attack_present:.4f}")

    # Restituisci all_preds e all_labels per l'uso successivo
    return total_loss / len(test_dataloader), accuracy_attack_type, accuracy_attack_present, all_attack_types_pred, all_attack_types_true, all_attack_present_pred, all_attack_present_true


train_loss_history = []
val_loss_history = []

print(f"[+] Training the model for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    print("\n")
    train_loss, train_acc_attack_type, train_acc_attack_present = train_model(train_dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy - Attack Type: {train_acc_attack_type:.4f} | Attack Present: {train_acc_attack_present:.4f}")

    val_loss, val_acc_Att_type, val_acc_Att_pres, all_val_pred_att_type, all_val_ture_att_type, all_val_pred_att_present, all_val_true_att_present= test_model(val_dataloader)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"\nEpoch {epoch + 1}/{EPOCHS} | Validation Loss: {val_loss:.4f} | Validation Accuracy - Attack Type: {val_acc_Att_type:.4f} | Attack Present: {val_acc_Att_pres:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid(alpha=0.3)

test_loss, test_acc_Att_type, test_acc_Att_present, all_test_pred_att_type, all_test_true_att_type, all_test_pred_att_present, all_test_true_att_present = test_model(test_dataloader)

print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy - Attack Type: {test_acc_Att_type:.4f} | Attack Present: {test_acc_Att_present:.4f}")




def calculate_per_class_metrics(true_label, pred_label):

    unique_labels = sorted(set(true_label) | set(pred_label))
    precision, recall, f1, _ = precision_recall_fscore_support(true_label, pred_label, average=None)
    for i, class_name in enumerate(unique_labels):
        print(f"Class {class_name} | Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}\n")
    



print("Per-class metrics for Attack_present: (val set)")
calculate_per_class_metrics(all_test_true_att_present, all_test_pred_att_present)
print("\nPer-class metrics for Attack_type:")
calculate_per_class_metrics(all_test_true_att_type, all_test_pred_att_type)


def metrics2(labels, pred, class_name):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred, average="macro", zero_division=0
        )

        print(f"Group {class_name} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")


print("Per-class metrics for Attack_present: (test set)")
label_name="Attack_present"
metrics2(all_test_true_att_present, all_test_pred_att_present,label_name)
label_name2="Attack_type"
print("\nPer-class metrics for Attack_type: (test set")
metrics2(all_test_true_att_type, all_test_pred_att_type, label_name2,)





