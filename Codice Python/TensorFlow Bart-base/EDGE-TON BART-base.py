import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import TFBartForSequenceClassification, BartTokenizerFast
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm



LABEL_COLUMNS = ["Attack_type", "Attack_label"]
MAX_LENGTH = 128
batch_size = 20  # Altrimenti ottengo un ResourceExhaustedError
EPOCHS = 3
DATASE = 
DATASET = 
MODEL_NAME = "facebook/bart-base"
tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
print(f"[+] Loading dataset {file_path}...")
df1 = pd.read_csv(DATASET)












































