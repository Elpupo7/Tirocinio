import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from peft import PeftModel
from peft import LoraConfig, get_peft_model, TaskType


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = 'EdgeIoT_250_rows.csv'
#DATASET = 'TON_350_rows.csv' # Per il file TOn
TARGET_COLUMNS = ["Attack_type","Attack_label"]
EPOCHS = 120
BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = 'google/t5-efficient-tiny'

# Load dataset
print(f"[+] Loading dataset {DATASET}...")
df1 = pd.read_csv(DATASET, delimiter=",")
# Combine all non-target columns into a single text column
text_columns = [col for col in df1.columns if col not in TARGET_COLUMNS]
df1["text"] = df1[text_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

# Encode labels as strings (T5 expects text outputs)
unique_labels = sorted(df1[TARGET_COLUMNS[0]].unique())  # ordinato in modo deterministico
label_to_id = {label: str(i) for i, label in enumerate(unique_labels)}
id_to_label = {str(i): label for label, i in label_to_id.items()}
labels = [label_to_id[label] for label in df1[TARGET_COLUMNS[0]]]  # Assuming single target column

train_texts, test_texts, train_labels, test_labels = train_test_split(df1["text"], labels, test_size=0.4, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

  print(f"[+] Initializing T5 tokenizer and model: {MODEL_NAME}...")
  tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
  model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

# To train LORAs on TON
#lora_config = LoraConfig(
#        r=32,
#        lora_alpha=64,
#        target_modules=["SelfAttention.q", "SelfAttention.v"] ,
#        lora_dropout=0.1,
#        bias="none",
#        task_type="SEQ_2_SEQ_LM"
#    )
# Applica i LoRA Adapters al modello
#model = get_peft_model(model, lora_config)
#model.to(DEVICE) 

# Dataset Class
class T5TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Format input as "classify: <text>"
        input_text = f"classify: {text}"
        encodings = self.tokenizer(
            input_text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )

        # Format labels as text
        label_encodings = self.tokenizer(
            label, padding="max_length", truncation=True,
            max_length=10, return_tensors="pt"
        )

        label_ids = label_encodings["input_ids"].squeeze(0)
        # Replace pad tokens with -100 so that they are ignored in the loss computation
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": label_ids
        }



# Create datasets
train_dataset = T5TextClassificationDataset(train_texts.tolist(), train_labels, tokenizer, MAX_LENGTH)
val_dataset = T5TextClassificationDataset(val_texts.tolist(), val_labels, tokenizer, MAX_LENGTH)
test_dataset = T5TextClassificationDataset(test_texts.tolist(), test_labels, tokenizer, MAX_LENGTH)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer  for model
optimizer = AdamW(model.parameters(), lr=5e-5)

# Optimizer for LoRA
#optimizer = AdamW(model.parameters(), lr=1e-4)


# Training Loop
def train_model():
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Generate predictions
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
        preds = [tokenizer.decode(output, skip_special_tokens=True).strip().lower() for output in outputs]

        # Convert labels to text
        labels_text = []
        for label in labels:
          # Create a copy to avoid modifying the original tensor.
          label_ids = label.clone().tolist()
          # Replace all -100 values with the tokenizer's pad_token_id.
          label_ids = [token_id if token_id != -100 else tokenizer.pad_token_id for token_id in label_ids]
          decoded_label = tokenizer.decode(label_ids, skip_special_tokens=True).strip().lower()
          labels_text.append(decoded_label)

        all_labels.extend(labels_text)
        all_preds.extend(preds)
        #print(f"Predicted Text: {preds[:4]}")
        #print(f"Target Text: {labels_text[:4]}")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # Calculate accuracy
    correct = sum([1 for pred, label in zip(all_preds, all_labels) if pred == label])
    accuracy = correct / len(all_labels) * 100

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return total_loss / len(train_loader), accuracy

# Evaluation Function
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

            # Generate predictions
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
            preds = [tokenizer.decode(output, skip_special_tokens=True).strip().lower() for output in outputs]

            # Convert labels to text
            labels_text = []
            for label in labels:
                # Create a copy to avoid modifying the original tensor.
                label_ids = label.clone().tolist()
                # Replace all -100 values with the tokenizer's pad_token_id.
                label_ids = [token_id if token_id != -100 else tokenizer.pad_token_id for token_id in label_ids]
                decoded_label = tokenizer.decode(label_ids, skip_special_tokens=True).strip().lower()
                labels_text.append(decoded_label)

            all_labels.extend(labels_text)
            all_preds.extend(preds)
            #print(f"Predicted Text: {preds[:4]}")
            #print(f"Target Text: {labels_text[:4]}")
            # Calculate validation loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    # Calculate accuracy
    correct = sum([1 for pred, label in zip(all_preds, all_labels) if pred == label])
    accuracy = correct / len(all_labels) * 100

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return total_loss / len(loader), all_labels, all_preds, accuracy


train_loss_history = []
val_loss_history = []
val_accuracy_history = []

print(f"[+] Training the model for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
  train_loss, acc_train = train_model()
  val_loss, val_labels, val_preds, val_accuracy = evaluate_model(val_loader)

  train_loss_history.append(train_loss)
  val_loss_history.append(val_loss)
  val_accuracy_history.append(val_accuracy)

  print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Accuracy: {acc_train:.2f}% | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Test set 
print("\n[+] Evaluating on the test set...")
test_loss, test_labels, test_preds, test_accuracy = evaluate_model(test_loader)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='macro'
    )
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


print("\n")


ordered_ids = sorted(id_to_label.keys(), key=int)

# Calcola metriche per Attack type
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels,
    test_preds,
    labels=ordered_ids,
    average=None
)

# Metriche per Attack type
for idx, id_str in enumerate(ordered_ids):
    attack_name = id_to_label[id_str]
    print(f"Tipo di attacco: {attack_name}")
    print(f"  Precision: {precision[idx]:.4f}")
    print(f"  Recall:    {recall[idx]:.4f}")
    print(f"  F1-score:  {f1[idx]:.4f}")
    print("-----------------------------")











