import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, TFT5ForConditionalGeneration, T5Config
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np


DATASET = 'EdgeIoT_250_rows.csv'
#DATASET = 'TON_350_rows.csv'
LABEL_COLUMNS = ["Attack_type","Attack_label"]  # List of target columns
EPOCHS = 3
batch_size = 32
MAX_LENGTH = 128
MODEL_NAME = 'google/t5-efficient-tiny'  # Use a valid T5 model
OUTPUT_MODEL = f'models/t5_trained.pth'

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


classi = [] # Usato per il calcolo dell emetriche precision recall f1 per ogni tipo in Attack Label e Attack Type
for col, encoder in label_encoders.items():
    print(f"\n{col} - Classi encodate:")
    for i, label in enumerate(encoder.classes_):
        print(f"{label} -> {i}")
        classi.append(label)






















