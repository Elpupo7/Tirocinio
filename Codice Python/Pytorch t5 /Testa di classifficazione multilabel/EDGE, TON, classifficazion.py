import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from tqdm import tqdm
import argparse
from peft import PeftModel
from peft import LoraConfig, get_peft_model


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = 'EdgeIoT_250_rows.csv'
DATASET = 'TON_350_rows.csv'
LABEL_COLUMNS = ["Attack_type", "Attack_label"]
EPOCHS = 2
BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = 'google/t5-efficient-tiny'  # Use a valid T5 model
OUTPUT_MODEL = f'models/t5_trained.pth'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)



  def calculate_accuracy(labels, preds, threshold=0.5):
      preds_binary = (preds > threshold).astype(int)
      correct = np.sum(labels == preds_binary)
      total = labels.size
      accuracy = correct / total
      return accuracy * 100


# Load dataset
print(f"[+] Loading dataset {DATASET}...")
df1 = pd.read_csv(DATASET, delimiter=",")






















