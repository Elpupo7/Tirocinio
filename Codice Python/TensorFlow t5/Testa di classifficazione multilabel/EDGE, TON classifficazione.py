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







