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
        self.binary_classifier = nn.Linear(config.hidden_size, 1)
        self.multi_classifier  = nn.Linear(config.hidden_size, num_attack_types)
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

        # Pool sul token [CLS]
        pooled = bert_outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        logits_bin   = self.binary_classifier(pooled).squeeze(-1)
        logits_multi = self.multi_classifier(pooled)

        loss = None
        if label_bin is not None and label_multi is not None:
            loss_fn_bin   = nn.BCEWithLogitsLoss()
            loss_fn_multi = nn.CrossEntropyLoss()
            loss_bin      = loss_fn_bin(logits_bin, label_bin.float())
            loss_multi    = loss_fn_multi(logits_multi, label_multi.long())
            loss = loss_bin + loss_multi

        return {"loss": loss, "logits_bin": logits_bin, "logits_multi": logits_multi}



























