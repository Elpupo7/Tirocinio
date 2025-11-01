import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    DistilBertPreTrainedModel,
    DistilBertConfig
)
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS     = 5
LABEL_COLUMNS = ["Attack_type", "Attack_label"]
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = 'EdgeIoT_250_rows.csv"
#DATASET = 'TON_350_rows.csv"
df = pd.read_csv(DATASET)

# Mapping per EDGE
label_encoders = {}
for col in LABEL_COLUMNS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
  
df["label_bin"]   = df["Attack_label"].astype(int) 
df["label_multi"] = df["Attack_type"].astype(int)   


# Mapping TON
#import pickle
#with open("/content/drive/My Drive/label_encoders/label_encoders.pkl", "rb") as f:
    #label_encoders = pickle.load(f)

#df["label_bin"]   = label_encoders["Attack_label"].transform(
    #df["Attack_label"]
#)
#df["label_multi"] = label_encoders["Attack_type"].transform(
    #df["Attack_type"]
#)

# Classi di attacco e relativo mapping
#for cls, idx in zip(label_encoders["Attack_type"].classes_, range(len(label_encoders["Attack_type"].classes_))):
    #print(f"{cls} -> {idx}")


df['text'] = df.drop(columns=LABEL_COLUMNS + ['label_bin'] + ["label_multi"]).astype(str).agg(' '.join, axis=1)


train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.2, random_state=42)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


class AttackDataset(Dataset):
    def __init__(self, texts, labels_bin, labels_multi, tokenizer, max_length):
        self.texts        = texts
        self.labels_bin   = labels_bin
        self.labels_multi = labels_multi
        self.tokenizer    = tokenizer
        self.max_length   = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        enc  = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_bin":      torch.tensor(self.labels_bin.iloc[idx], dtype=torch.float),
            "label_multi":    torch.tensor(self.labels_multi.iloc[idx], dtype=torch.long),
        }



# Istanziamento dataset
train_dataset = AttackDataset(train_df["text"], train_df["label_bin"], train_df["label_multi"], tokenizer, MAX_LENGTH)
val_dataset   = AttackDataset(val_df["text"],   val_df["label_bin"],   val_df["label_multi"],   tokenizer, MAX_LENGTH)
test_dataset  = AttackDataset(test_df["text"],  test_df["label_bin"],  test_df["label_multi"],  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)


class DistilBertDualHead(DistilBertPreTrainedModel):
    def __init__(self, config, num_attack_types):
        super().__init__(config)
        self.distilbert        = DistilBertModel(config)
        self.dropout           = nn.Dropout(config.seq_classif_dropout)
        self.binary_classifier = nn.Linear(config.hidden_size, 1) # restituisce 1 logits per la classifficazione binaria
        self.multi_classifier  = nn.Linear(config.hidden_size, num_attack_types) # restituisce un logit per ogni tipo di attacco
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        label_bin=None,
        label_multi=None,
        **kwargs
    ):
        # Rimuoviamo labels (o altre chiavi indesiderate) da kwargs
        kwargs.pop("labels", None)
        # Se dovessero arrivare anche `token_type_ids`, `position_ids`, etc, li lasciamo passare
        bert_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        # Pool sul token [CLS] che riassume l’informazione di tutta la frase.
        pooled = bert_outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        logits_bin   = self.binary_classifier(pooled).squeeze(-1)
        logits_multi = self.multi_classifier(pooled)

        # Calcoliamo la loss

        loss = None
        if label_bin is not None and label_multi is not None:
            loss_fn_bin   = nn.BCEWithLogitsLoss()
            loss_fn_multi = nn.CrossEntropyLoss()
            loss_bin      = loss_fn_bin(logits_bin, label_bin.float())
            loss_multi    = loss_fn_multi(logits_multi, label_multi.long())
            loss = loss_bin + loss_multi

        return {"loss": loss, "logits_bin": logits_bin, "logits_multi": logits_multi}


num_attack_types = len(label_encoders["Attack_type"].classes_)


# Config LoRa 
#lora_config = LoraConfig(
    #r=16,
    #lora_alpha=32,
    #target_modules=["q_lin", "v_lin", "k_lin"],
    #lora_dropout=0.1,
    #bias="none",
    #task_type="SEQ_CLS"
#)

config = DistilBertConfig.from_pretrained(MODEL_NAME)
model  = DistilBertDualHead.from_pretrained(
    MODEL_NAME,
    config=config,
    num_attack_types=num_attack_types
)
# Solo quando addestri i LoRa esegui la prossima istruzione 
#model = get_peft_model(model, lora_config) 
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)
def train_model(train_loader, model, optimizer, device):
    """
    Esegue un'epoca di training.
    Ritorna:
      - avg_loss: perdita media su tutti i batch
      - metrics: dizionario con le metriche per task ('binary' e 'multi')
    """
    model.train()
    total_loss = 0


    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            label_bin=batch["label_bin"].to(device),
            label_multi=batch["label_multi"].to(device),
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # media loss
    avg_loss = total_loss / len(train_loader)


    return avg_loss


def evaluate_model(eval_loader, model, device):
    """
    Esegue la validazione sul validation o test set.
    Ritorna:
      - avg_loss: perdita media
      - metrics: dizionario con le metriche per task ('binary' e 'multi')
    """
    model.eval()
    total_loss = 0

    all_bin_labels, all_bin_preds   = [], []
    all_multi_labels, all_multi_preds = [], []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                label_bin=batch["label_bin"].to(device),
                label_multi=batch["label_multi"].to(device),
            )
            total_loss += out["loss"].item()

            probs_b = torch.sigmoid(out["logits_bin"].cpu())
            preds_b = (probs_b > 0.5).long()
            all_bin_preds.append(preds_b)
            all_bin_labels.append(batch["label_bin"].long())

            logits_m = out["logits_multi"].cpu()
            preds_m  = logits_m.argmax(dim=1)
            all_multi_preds.append(preds_m)
            all_multi_labels.append(batch["label_multi"])

    avg_loss = total_loss / len(eval_loader)

    # concateniamo tutti i batch
    all_bin_preds    = torch.cat(all_bin_preds)
    all_bin_labels   = torch.cat(all_bin_labels)
    all_multi_preds  = torch.cat(all_multi_preds)
    all_multi_labels = torch.cat(all_multi_labels)

    # calcolo metriche
    metrics = {
        "binary": {
            "accuracy":  accuracy_score(all_bin_labels, all_bin_preds),
            "precision": precision_score(all_bin_labels, all_bin_preds),
            "recall":    recall_score(all_bin_labels, all_bin_preds),
            "f1":        f1_score(all_bin_labels, all_bin_preds)
        },
        "multi": {
            "accuracy":  accuracy_score(all_multi_labels, all_multi_preds),
            "precision": precision_score(all_multi_labels, all_multi_preds, average="weighted"),
            "recall":    recall_score(all_multi_labels, all_multi_preds, average="weighted"),
            "f1":        f1_score(all_multi_labels, all_multi_preds, average="weighted")
        }
    }

    return avg_loss, metrics, all_bin_labels, all_bin_preds, all_multi_labels, all_multi_preds


train_loss_history, val_loss_history = [], []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_model(train_loader, model, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")
    train_loss_history.append(train_loss)

    val_loss, val_metrics, bin_labels, bin_preds, multi_labels, multi_preds  = evaluate_model(val_loader, model, device)

    print(f"\nRis per tipo il riconoscimento binario | \n"
    f"Val Loss: {val_loss:.4f} | \n"
    f"Binary Accuracy: {val_metrics['binary']['accuracy']:.4f} | \n"
    f"Binary Precision: {val_metrics['binary']['precision']:.4f} | \n"
    f"Binary Recall: {val_metrics['binary']['recall']:.4f} | \n"
    f"Binary F1: {val_metrics['binary']['f1']:.4f} | \n"
    f"\nRis per tipo di attacco | \n"
    f"Multi Accuracy: {val_metrics['multi']['accuracy']:.4f} | \n"
    f"Multi Precision: {val_metrics['multi']['precision']:.4f} | \n"
    f"Multi Recall: {val_metrics['multi']['recall']:.4f} | \n"
    f"Multi F1: {val_metrics['multi']['f1']:.4f}"
)
    val_loss_history.append(val_loss)

def metrics_per_task_EDGE(bin_labels, bin_preds, multi_labels, multi_preds, label_encoder_multi):
    """
    Stampa precision/recall/f1 per:
      - task binario (presence vs absence), sia per-class che aggregato
      - task multiclasse (tipo di attacco), sia per-class che aggregato
    """
    # Metriche Task Binario
    print("Binary Task")

    
    prec_b_all, rec_b_all, f1_b_all, support_b = precision_recall_fscore_support(
        bin_labels, bin_preds, average=None, zero_division=0
    )

    for i, label in enumerate(["Normale (0)", "Attacco (1)"]):
        print(f"{label:12s} | P: {prec_b_all[i]:.4f} – R: {rec_b_all[i]:.4f} – F1: {f1_b_all[i]:.4f} – sup: {support_b[i]}")

    # Media metriche per attacco
    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
        bin_labels, bin_preds, average="macro", zero_division=0
    )
    print(f"\nMacro – Precision: {prec_b:.4f} – Recall: {rec_b:.4f} – F1: {f1_b:.4f}\n")

    # Metriche Task per i tipi di attacchi
    print("Multiclass Task")

    prec_m, rec_m, f1_m, support_m = precision_recall_fscore_support(
        multi_labels, multi_preds, average=None, zero_division=0
    )
    for i, class_name in enumerate(label_encoder_multi.classes_):
        print(f"{class_name:15s} | P: {prec_m[i]:.4f} – R: {rec_m[i]:.4f} – F1: {f1_m[i]:.4f} – sup: {support_m[i]}\n")

    # Macro-average
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        multi_labels, multi_preds, average="macro", zero_division=0
    )
    print(f"\nMacro – Precision: {prec_macro:.4f} – Recall: {rec_macro:.4f} – F1: {f1_macro:.4f}")



def metrics_per_task_TON(bin_labels, bin_preds,
                     multi_labels, multi_preds,
                     label_encoder_multi):
    """
    Stampa precision/recall/F1 per tutte le classi
    e calcola la macro‐media solo sulle classi con support > 0.
    """

    # Metriche Task Binario
    print("Binary Task")

    prec_b_all, rec_b_all, f1_b_all, support_b = precision_recall_fscore_support(
        bin_labels, bin_preds, average=None, zero_division=0
    )

    for i, label in enumerate(["Normale (0)", "Attacco (1)"]):
        print(f"{label:12s} | P: {prec_b_all[i]:.4f} – R: {rec_b_all[i]:.4f} – F1: {f1_b_all[i]:.4f} – sup: {support_b[i]}")


    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
        bin_labels, bin_preds, average="weighted", zero_division=0
    )
    print(f"\nMacro – Precision: {prec_b:.4f} – Recall: {rec_b:.4f} – F1: {f1_b:.4f}\n")


    # Metriche Task per i tipi di attacchi
    print("Multiclass Task")
    K = len(label_encoder_multi.classes_)

    prec_m, rec_m, f1_m, support_m = precision_recall_fscore_support(
        multi_labels, multi_preds,
        labels=list(range(K)),
        average=None,
        zero_division=0
    )

    # 2) stampa per‐classe
    for i, class_name in enumerate(label_encoder_multi.classes_):
        print(f"{class_name:25s} | "
              f"P: {prec_m[i]:.4f} – "
              f"R: {rec_m[i]:.4f} – "
              f"F1: {f1_m[i]:.4f} – "
              f"sup: {support_m[i]}\n")
    print()

   # calcola macro‐media solo per gli atatcch ipresenti nel dataset TON
    mask = support_m > 0 
    prec_macro_present = np.mean(prec_m[mask])
    rec_macro_present  = np.mean(rec_m[mask])
    f1_macro_present   = np.mean(f1_m[mask])


    print("Macro (solo classi con support>0): "
          f"P: {prec_macro_present:.4f} – "
          f"R: {rec_macro_present:.4f} – "
          f"F1: {f1_macro_present:.4f}\n")


# Validazione sul test set
test_loss, test_metrics,  test_bin_labels, test_bin_preds, test_multi_labels, test_multi_preds  = evaluate_model(test_loader, model, device)

print(f"\nRis per tipo il riconoscimento binario | \n"
    f"Test Loss: {test_loss:.4f} | \n"
    f"Binary Accuracy: {test_metrics['binary']['accuracy']:.4f} | \n"
    f"Binary Precision: {test_metrics['binary']['precision']:.4f} | \n"
    f"Binary Recall: {test_metrics['binary']['recall']:.4f} | \n"
    f"Binary F1: {test_metrics['binary']['f1']:.4f} | \n"
    f"\nRis per tipo di attacco | \n"
    f"Multi Accuracy: {test_metrics['multi']['accuracy']:.4f} | \n"
    f"Multi Precision: {test_metrics['multi']['precision']:.4f} | \n"
    f"Multi Recall: {test_metrics['multi']['recall']:.4f} | \n"
    f"Multi F1: {test_metrics['multi']['f1']:.4f}"
)
    #PER TON 11.35

metrics_per_task_EDGE(
    test_bin_labels.cpu().numpy(),
    test_bin_preds.cpu().numpy(),
    test_multi_labels.cpu().numpy(),
    test_multi_preds.cpu().numpy(),
    label_encoders["Attack_type"]
)

#metrics_per_task_TON(
    #test_bin_labels.cpu().numpy(),
    #test_bin_preds.cpu().numpy(),
    #test_multi_labels.cpu().numpy(),
    #test_multi_preds.cpu().numpy(),
    #label_encoders["Attack_type"]
#)




















