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
MODEL_NAME = 'google/t5-efficient-tiny'
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

num_attack_type = len(label_encoders["Attack_type"].classes_)
num_attack_label = len(label_encoders["Attack_label"].classes_)
num_classes_split = [num_attack_type, num_attack_label]

classi = [] # Usato per il calcolo dell emetriche precision recall f1 per ogni tipo in Attack Label e Attack Type
for col, encoder in label_encoders.items():
    print(f"\n{col} - Classi encodate:")
    for i, label in enumerate(encoder.classes_):
        print(f"{label} -> {i}")
        classi.append(label)

  num_classes = sum(len(label_encoders[col].classes_) for col in LABEL_COLUMNS)
  print(f"[INFO] Numero totale di classi: {num_classes}")


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df1["text"], df1["multilabel_target"], test_size=0.4, random_state=42
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Funzione di encoding per TensorFlow usando py_function
def encode_text(text, label_attack_type, label_attack_label):
    text = text.numpy().decode('utf-8')

    # Le label sono già intere
    label_attack_type = label_attack_type.numpy()
    label_attack_label = label_attack_label.numpy()

    encodings = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    input_ids = np.array(encodings['input_ids'], dtype=np.int32)
    attention_mask = np.array(encodings['attention_mask'], dtype=np.int32)

    # Costruisci il vettore one-hot per Attack_type
    one_hot_attack_type = np.zeros(num_attack_type, dtype=np.float32)
    one_hot_attack_type[int(label_attack_type)] = 1.0

    # Costruisci il vettore one-hot per Attack_label
    one_hot_attack_label = np.zeros(num_attack_label, dtype=np.float32)
    one_hot_attack_label[int(label_attack_label)] = 1.0

    # Concatenare i due vettori in un unico vettore one-hot
    one_hot = np.concatenate([one_hot_attack_type, one_hot_attack_label])
    return input_ids, attention_mask, one_hot


def tf_encode(text, label_attack_type, label_attack_label):
    input_ids, attention_mask, one_hot = tf.py_function(
        encode_text, inp=[text, label_attack_type, label_attack_label],
        Tout=(tf.int32, tf.int32, tf.float32)
    )
    input_ids.set_shape([MAX_LENGTH])
    attention_mask.set_shape([MAX_LENGTH])
    one_hot.set_shape([num_classes])
    return {"input_ids": input_ids, "attention_mask": attention_mask}, one_hot



# Creazione delle etichette separate
train_labels_attack_type = [label[0] for label in train_labels.tolist()]
train_labels_attack_label = [label[1] for label in train_labels.tolist()]

val_labels_attack_type = [label[0] for label in val_labels.tolist()]
val_labels_attack_label = [label[1] for label in val_labels.tolist()]

test_labels_attack_type = [label[0] for label in test_labels.tolist()]
test_labels_attack_label = [label[1] for label in test_labels.tolist()]


train_dataset = tf.data.Dataset.from_tensor_slices((
    train_texts.tolist(),
    train_labels_attack_type,
    train_labels_attack_label
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    val_texts.tolist(),
    val_labels_attack_type,
    val_labels_attack_label
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    test_texts.tolist(),
    test_labels_attack_type,
    test_labels_attack_label
))


train_dataset = train_dataset.shuffle(buffer_size=len(train_texts))  # Shuffle
train_dataset = train_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.shuffle(buffer_size=len(val_texts))  # Shuffle
val_dataset = val_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.shuffle(buffer_size=len(test_texts))  # Shuffle
test_dataset = test_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class T5ForMultiLabelClassificationTF(tf.keras.Model):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        # 1) Tokenizer e config
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.config    = T5Config.from_pretrained(model_name)
        # 2) Intero modello T5 (encoder+decoder)
        self.t5 = TFT5ForConditionalGeneration.from_pretrained(
            model_name,
            from_pt=False,
            config=self.config,
        )
        # 3) Testa di classificazione
        self.classifier = tf.keras.layers.Dense(num_labels, activation=None)

    def call(self, inputs, training=False):
        """
        inputs: dict con chiavi "input_ids" e "attention_mask"
        """
        # 1) Encoder forward (come in PyTorch: self.t5.encoder)
        encoder_outputs = self.t5.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            training=training
        )
        # 2) Pooling CLS (token 0)
        hidden_states = encoder_outputs.last_hidden_state  # [B, seq_len, d_model]
        cls_repr = hidden_states[:, 0, :]                  # [B, d_model]
        # 3) Calcolo logits
        logits = self.classifier(cls_repr)                 # [B, num_labels]
        return logits


print(f"[+] Initializing T5 tokenizer and model: {MODEL_NAME}...")
model = T5ForMultiLabelClassificationTF(MODEL_NAME, num_classes)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)



optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn   = tf.keras.losses.BinaryCrossentropy(from_logits=True)
THRESHOLD = 0.5

def calculate_accuracy(labels, preds, threshold=THRESHOLD):
    preds_binary = (preds > threshold).astype(int)
    return (labels == preds_binary).mean() * 100


def train_model(model, train_dataset):
    total_loss = 0.0
    all_labels = []
    all_preds  = []

    for batch in tqdm(train_dataset, desc="Train"):
        inputs, labels = batch

        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)    # -> [B, num_labels]
            loss   = loss_fn(labels, logits)         # BCEWithLogits
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss += loss.numpy()
        all_labels.append(labels.numpy())
        all_preds .append(tf.sigmoid(logits).numpy())

    all_labels = np.vstack(all_labels)
    all_preds  = np.vstack(all_preds)
    all_preds_binary = (all_preds > THRESHOLD).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds_binary, average="macro", zero_division=0
    )
    accuracy = calculate_accuracy(all_labels, all_preds)

    print(f"Train — Loss: {total_loss/len(train_dataset):.4f} "
          f"Acc: {accuracy:.2f}%  Prec: {precision:.4f}  Rec: {recall:.4f}  F1: {f1:.4f}")
    return total_loss/len(train_dataset)

def evaluate_model(model, val_dataset):
    total_loss = 0.0
    all_labels = []
    all_preds  = []

    for batch in tqdm(val_dataset, desc="Val"):
        inputs, labels = batch

        logits = model(inputs, training=False)
        loss   = loss_fn(labels, logits)

        total_loss += loss.numpy()
        all_labels.append(labels.numpy())
        all_preds .append(tf.sigmoid(logits).numpy())

    all_labels = np.vstack(all_labels)
    all_preds  = np.vstack(all_preds)
    all_preds_binary = (all_preds > THRESHOLD).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds_binary, average="macro", zero_division=0
    )
    accuracy = calculate_accuracy(all_labels, all_preds)

    print(f"Val   — Loss: {total_loss/len(val_dataset):.4f} "
          f"Acc: {accuracy:.2f}%  Prec: {precision:.4f}  Rec: {recall:.4f}  F1: {f1:.4f}")
    return total_loss/len(val_dataset)



train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    train_loss = train_model(model, train_dataset)
    val_loss   = evaluate_model(model, val_dataset)


plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


def metrics(labels, pred):
    preds_binary = (pred > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_binary, average=None)
    for i, class_name in enumerate(classi):
        print(f"Class {class_name} | Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}\n")

# PER EDGE
# Calcola la recall precision f1 macro per i tipi Attack Type e Attack Label
def metrics1(labels, pred, class_splits):
    preds_binary = (pred > 0.5).astype(int)

    start = 0
    for i, class_name in enumerate(LABEL_COLUMNS):
        n_classes = class_splits[i]  # Numero di classi nel gruppo
        end = start + n_classes

        # Estrai solo le classi corrispondenti
        labels_group = labels[:, start:end]
        preds_group = preds_binary[:, start:end]

        # Calcola le metriche con average='macro'
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_group, preds_group, average="macro", zero_division=0
        )

        print(f"Group {class_name} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        start = end  # Aggiorna "start" per la prossima iterazione

# PER TON
#CONSENTE DI PRENDERE E CALCOLARE LE MACRO METRICHE SOLO SUGLI EFFETTIVI ATTACK TYPE PRESENTI DENTRO AL DATASET TON
#def metrics1(labels, pred, class_splits):
    #preds_binary = (pred > 0.5).astype(int)

    #start = 0
    #for i, class_name in enumerate(LABEL_COLUMNS):
        #n_classes = class_splits[i]  # Numero di classi nel gruppo corrente
        #end = start + n_classes

        # Estrai le colonne per il gruppo corrente
        #labels_group = labels[:, start:end]
        #preds_group = preds_binary[:, start:end]

        # Se il gruppo corrente è "Attack_type", seleziona solo le classi presenti in Dataset 2
        #if class_name == "Attack_type":
            # Indici relativi all'interno del gruppo "Attack_type" di Dataset 1 per le classi di Dataset 2:
            #selected_indices = [3, 4, 6, 7, 8, 9, 11, 14]
            #labels_group = labels_group[:, selected_indices]
            #preds_group = preds_group[:, selected_indices]


        # Calcola le metriche per il gruppo corrente
        #precision, recall, f1, _ = precision_recall_fscore_support(
            #labels_group, preds_group, average="macro", zero_division=0
        #)

        #print(f"Group {class_name} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        #start = end  # Aggiorna "start" per il prossimo gruppo

y_trues, y_preds = [], []

for batch in test_dataset:
    inputs, labels = batch
    preds = model.predict(inputs, verbose=0)
    y_trues.append(labels.numpy())
    y_preds.append(tf.sigmoid(preds.logits).numpy())

# Concatena tutti i batch
y_true = np.concatenate(y_trues, axis=0)
y_pred = np.concatenate(y_preds, axis=0)


print('Metriche per ogni classe')
metrics(y_true, y_pred)
print('\nMacro:')
metrics1(y_true, y_pred, num_classes_split)









