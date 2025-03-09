
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer
# CODICE PER ADDESTRARE I LORA Adapter SUL DATASET TON CON LE COLE COLONNE RILEVANTI
df_full = pd.read_csv(windows10_dataset.csv)
df_full = df_full.rename(columns={'label': 'Attack_label'})

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
    'type'
]

# Creazione del nuovo DataFrame con solo le colonne selezionate
df = df_full[colonne_da_tenere]

# Formattazione attacchi di TON in modo da mappare correttamente le occorenze con i dizionari usati per addestrare il modello DistilBert di EDGE IIoT
df["type"] = df["type"].replace({
    "normal": "Normal",
    "ddos": "DDoS_UDP",
    "password": "Password",
    "xss": "XSS",
    "injection": "SQL_injection",
    "dos": "DDoS_TCP",
    "scanning": "Port_Scanning",
    "mitm": "MITM"
})

attack_dfs = {attack_type: df[df["type"] == attack_type]
              for attack_type in df["type"].unique()}


df1 = pd.concat(list(attack_dfs.values())).sample(frac=1, random_state=42)
print("Dimensione del nuovo DataFrame:", df1.shape)
print("Distribuzione di Attack_label:")
print(df1["Attack_label"].value_counts())
print('\n')
print(df1["type"].value_counts())

# Mapping
attack_types = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'DDoS_HTTP', 'Port_Scanning',
    'SQL_injection', 'Password', 'Backdoor', 'Uploading', 'Vulnerability_scanner',
    'XSS', 'Ransomware', 'DDoS_TCP', 'Fingerprinting', 'MITM'
]

label2id = {label: idx for idx, label in enumerate(attack_types)}
id2label = {idx: label for label, idx in label2id.items()}

df1["label"] = df1["type"].map(label2id) 

# Dataset di Hugging face
dataset1 = Dataset.from_pandas(df1)
dataset = DatasetDict({"train": dataset1})

# "Testualizzo"
def combine_features(example):
    exclude_keys = ['Unnamed', 'Attack_label', 'type']
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


import torch
from torch.utils.data import DataLoader, Dataset
class AttackDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings 
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key in ["input_ids", "attention_mask"]}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = AttackDataset(tokenized_datasets["train"].to_dict(), tokenized_datasets["train"]["label"])
test_dataset = AttackDataset(tokenized_datasets["test"].to_dict(), tokenized_datasets["test"]["label"])
len(train_dataset)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
len(train_loader)


from transformers import DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
# Modello DISTILBERT per PyTorch
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(attack_types),
    id2label=id2label,
    label2id=label2id
)
# Config LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
    lora_dropout=0.05, 
    bias="lora_only",  
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config) #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)
# Per addestrare solo i parametri LoRA
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5) 
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
        # Calcolo  predizioni
        preds = torch.argmax(logits, dim=1)
        total_train_correct += (preds == labels).sum().item()
        total_train_samples += labels.size(0) 

        # Memorizzo le predizioni e etichette per le metriche
        all_preds.extend(preds.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy()) 

        loss.backward()
        optimizer.step()
        scheduler.step()
      
    # Calcola l'accuracy
    train_accuracy = total_train_correct / total_train_samples
    avg_loss = total_loss / len(train_loader)

    # Calcolo altre metriche
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted') 
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Training Accuracy: {train_accuracy:.4f}") 
    print(f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1-Score: {f1:.4f}")


model.eval()  
correct = 0
test_loss_lora = 0
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

        loss = outputs.loss  
        logits = outputs.logits 

        test_loss_lora += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        # Memorizzo le predizioni e etichette per le metriche
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        conta += 1
        if conta % 20 == 0:
          print(f"test {conta}")
test_loss_lora /= len(test_loader)
test_accuracy_lora = correct / total
    # Calcolo altre metriche
precision_lora, recall_lora, f1_lora, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
print(f"Test Loss = {test_loss_lora:.4f}, Test Accuracy = {test_accuracy_lora:.4f}")
print(f"Precision = {precision_lora:.4f}, Recall = {recall_lora:.4f}, F1-Score = {f1_lora:.4f}")




# PROVA DEL MODELLO DistilBert con l'addestramento sui LORA
def predict_attack(model, sample_row, id2label, tokenizer, device):
    sample_text = combine_features(sample_row)["text"]
    inputs = tokenizer(sample_text, padding="max_length", truncation=True, max_length=500, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
    predicted_label = id2label[predicted_class]

    return predicted_label



attack_df = df[df["Attack_label"] == 1]
# Estrai una riga casuale che ha un attacco (ottenendo una Pandas Series)
sample_row = attack_df.sample(n=1, random_state=None).iloc[0]
# Estrazione di una riga casuale (tipo Pandas Series)
sample_row = df_full.sample(n=1, random_state=None).iloc[0]
# Esegui la predizione
predicted_label = predict_attack(model, sample_row, id2label, tokenizer, device)

# Stampa le informazioni della riga e la predizione
print("Attack_label:", sample_row['Attack_label'])
print("Attack_type:", sample_row['type'])
print("Predicted label:", predicted_label)
























