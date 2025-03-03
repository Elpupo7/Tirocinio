# Fase 1: download del set di dati Edge-IIoTset dalla piattaforma Kaggle
from google.colab import files

!pip install -q kaggle

files.upload()

!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

!unzip DNN-EdgeIIoT-dataset.csv.zip

!rm DNN-EdgeIIoT-dataset.csv.zip

# Fase2: lettura del file CSV dei set di dati in un Pandas DataFrame:
import pandas as pd
import numpy as np
df = pd.read_csv('DNN-EdgeIIoT-dataset.csv', low_memory=False)

# Fase 3: Eliminazione dei dati (colonne, righe duplicate, NAN, Null...):
from sklearn.utils import shuffle

drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4",

         "http.file_data","http.request.full_uri","icmp.transmit_timestamp",

         "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",

         "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)

df.dropna(axis=0, how='any', inplace=True)

df.drop_duplicates(subset=None, keep="first", inplace=True)

df = shuffle(df)

df.isna().sum()

print(df['Attack_type'].value_counts())

# Fase 4: Codifica dei dati categoriali 
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)

encode_text_dummy(df,'http.request.method')
encode_text_dummy(df,'http.referer')
encode_text_dummy(df,"http.request.version")
encode_text_dummy(df,"dns.qry.name.len")
encode_text_dummy(df,"mqtt.conack.flags")
encode_text_dummy(df,"mqtt.protoname")
encode_text_dummy(df,"mqtt.topic")

# Fase 5:  Creazione del set di dati preelaborato
df.to_csv('preprocessed_DNN.csv', codifica='utf-8')

!pip install transformers
!pip install tensorflow
!pip install datasets

import os
import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, TFBertForSequenceClassification

df_full = pd.read_csv("preprocessed_DNN.csv", delimiter=",")

# Suddivisione per attacco
attack_dfs = {attack_type: df_full[df_full["Attack_type"] == attack_type]
              for attack_type in df_full["Attack_type"].unique()}

# Scelta di 1000 campioni (eccetto Fingerprinting e MITM avendone di meno
attack_samples = {
    attack_type: (
        attack_dfs["Fingerprinting"] if attack_type == "Fingerprinting"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else df.sample(n=1000, random_state=42)
    )
    for attack_type, df in attack_dfs.items()
}

# Creo il dataframe da usare per l'addestramento e il test del modello
df1 = pd.concat(list(attack_samples.values())).sample(frac=1, random_state=42)

print("Dimensione del nuovo DataFrame:", df1.shape)
print("Distribuzione di Attack_label:")
print(df1["Attack_label"].value_counts())
print('\n')
print(df1["Attack_type"].value_counts())

#Mapping degli attacchi
#possibili classi (15 etichette)
attack_types = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'DDoS_HTTP', 'Port_Scanning',
    'SQL_injection', 'Password', 'Backdoor', 'Uploading', 'Vulnerability_scanner',
    'XSS', 'Ransomware', 'DDoS_TCP', 'Fingerprinting', 'MITM'
]

# Dizionari di mapping
label2id = {label: idx for idx, label in enumerate(attack_types)}
id2label = {idx: label for label, idx in label2id.items()}

# Conversione della colonna "Attack_type" in una colonna numerica "label"
df1["label"] = df1["Attack_type"].map(label2id)
dataset1 = Dataset.from_pandas(df1)
dataset = DatasetDict({"train": dataset1})  # Assegna la chiave "train"

# Funzione per "testualizzare" ciascuna riga (senza le colonne non utili per l'addestramento)
def combine_features(example):
    exclude_keys = ['Unnamed: 0', 'Attack_label', 'Attack_type']
    text_parts = []
    for key, value in example.items():
        if key not in exclude_keys:
            text_parts.append(f"{key}: {value}")
    example["text"] = " ".join(text_parts)
    return example

dataset = dataset["train"].map(combine_features)
#Divisione in train e test (80%-20%)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Distribuzione delle etichette nei sottoinsiemi
from collections import Counter
print("Distribuzione nel train set:")
print(Counter(dataset["train"]["label"]))
print("Distribuzione nel test set:")
print(Counter(dataset["test"]["label"]))

# Tokenizer BERT pre-addestrato
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Funzione per tokenizzare il campo "text"
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

batchN = 10
train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"], # Input al modello
    label_cols=["label"],  # Output del modello 
    shuffle=True,
    batch_size=batchN,
)

test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=batchN,
)


# Modello BERT per la classificazione testuale con num_labels=15 (le nostre etichette)
model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(attack_types),
    id2label=id2label,
    label2id=label2id
)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(train_dataset, validation_data=test_dataset, epochs=3)

from transformers import BertTokenizer

# Tokenizer usato per l'addestramento
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict_attack(model, sample_row, id2label):
    # Applica la funzione di combine_features per creare il campo "text", all'esempio estratto dal CSV
    sample_text = combine_features(sample_row)["text"]

    # Tokenizza il testo come fatto durante il training
    inputs = tokenizer(sample_text, padding="max_length", truncation=True, max_length=500, return_tensors="tf")

    # Predizione con il modello
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits

    # Estrazione classe predetta (il numero intero corrispondente all'indice della classe)
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]

    # Mapping tra numero predetto e la corrispondente etichetta testuale
    predicted_label = id2label[predicted_class]

    return predicted_label


# Dataset originale dal file CSV
df = pd.read_csv("preprocessed_DNN.csv")

# Filtra solo le righe in cui è presente un attacco
attack_df = df[df["Attack_label"] == 1]

# Per estarre un esempio casuale
sample_row = attack_df.sample(n=1, random_state=None).iloc[0]

# Estrazione di una riga casuale (tipo Pandas Series)
sample_row = attack_df.sample(n=1, random_state=None).iloc[0]

# Predizione del tipo di attacco sulla riga casuale estratta
predicted_label = predict_attack(model, sample_row, id2label)
print(sample_row['Attack_label'])
print(sample_row['Attack_type'])
print("Predicted label:", predicted_label)



# PER PROVARE UNA RIGA CASUALE
#df = pd.read_csv("preprocessed_DNN.csv")
# Per estarre un esempio casuale
#sample_row = attack_df.sample(n=1, random_state=None).iloc[0]

# Estrazione di una riga casuale (tipo Pandas Series)
#sample_row = df.sample(n=1, random_state=None).iloc[0]

# Predizione del tipo di attacco sulla riga casuale estratta
#predicted_label = predict_attack(model, sample_row, id2label)
#print(sample_row['Attack_label'])
#print(sample_row['Attack_type'])
#print("Predicted label:", predicted_label)
