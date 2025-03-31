import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration
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


max_length = 128
batch_size = 32
EPOCHS = 3
LABEL_COLUMNS = ["Attack_type","Attack_label"]


df = pd.read_csv('/preprocessed_DNN.csv', delimiter=",")
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


df1["text"] = df1.apply(lambda row: "Predict the malware network based on the following input (the possible case of predict are: Password 1, DDoS_UDP 1, Normal 0, Ransomware 1, Port_Scanning 1, XSS 1, DDoS_TCP 1, DDoS_HTTP 1, Backdoor 1, SQL_injection 1, DDoS_ICMP  1, Fingerprinting 1, Vulnerability_scanner 1, Uploading 1, MITM 1): "  
                        + " ".join([f"{col}: {row[col]}" for col in df1.columns if col not in LABEL_COLUMNS]), axis=1)

#print(df1["text"].iloc[0])
df1["target_text"] = df1.apply(lambda row: " ".join([str(row[col]) for col in LABEL_COLUMNS]), axis=1)
print(df1["target_text"]..value_counts())

data = df1[["text", "target_text"]].rename(columns={"text": "input_text"}).to_dict(orient="records")
#print(data[0])


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


train_dataset = MyDataset(train_data, tokenizer, max_length=max_length)
val_dataset = MyDataset(val_data, tokenizer, max_length=max_length)
test_dataset = MyDataset(test_data, tokenizer, max_length=max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))
print('\n')
print(len(train_dataloader))
print(len(val_dataloader))
print(len(test_dataloader))



def train_model(train_dataloader):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(train_dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Creiamo una copia dei target per le metriche (senza mascheratura)
        labels_for_metrics = labels.clone()

        # Maschera i token di padding nei target per il calcolo della loss
        labels[labels == tokenizer.pad_token_id] = -100

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Usando model.generate() per generare le predizioni
        pred_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        target_texts = tokenizer.batch_decode(labels_for_metrics, skip_special_tokens=True)

        all_preds.extend(pred_texts)
        all_labels.extend(target_texts)

        correct_preds += sum([1 if pred == target else 0 for pred, target in zip(pred_texts, target_texts)])
        total_preds += len(pred_texts)

    # Calcolo le metriche
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n")

    return total_loss / len(train_dataloader), accuracy


def test_model(test_dataloader):
    model.eval()  # Modalità valutazione (disabilita dropout e batchnorm)
    total_loss = 0
    all_preds = []
    all_labels = []
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


            all_preds.extend(pred_texts)
            all_labels.extend(target_texts)
          
            correct_preds += sum([1 if pred == target else 0 for pred, target in zip(pred_texts, target_texts)])
            total_preds += len(pred_texts)

        # Calcolo le metriche
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n")

    return total_loss/len(test_dataloader), all_labels, all_preds, accuracy

train_loss_history = []
val_loss_history = []
print(f"[+] Training the model for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    print("\n")
    train_loss, train_accuracy  = train_model(train_dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.2f}%")
    val_loss, val_labels, val_preds, val_accuracy = test_model(val_dataloader)

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

















