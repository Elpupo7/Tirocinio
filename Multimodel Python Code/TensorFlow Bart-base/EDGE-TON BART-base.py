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
DATASET = 'EdgeIoT_250_rows.csv' # Per usare EDGE
#DATASET = 'TON_350_rows.csv' # Per Validare su TON
MODEL_NAME = "facebook/bart-base"
tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
print(f"[+] Loading dataset {file_path}...")
df1 = pd.read_csv(DATASET)


  def calculate_accuracy(labels, preds, threshold=0.5):
      preds_binary = (preds > threshold).astype(int)
      correct = np.sum(labels == preds_binary)
      total = labels.size
      accuracy = correct / total
      return accuracy * 100

print(f"[+] Loading dataset {DATASET}...")
df1 = pd.read_csv(DATASET)

# MULTILABEL per EDGE
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

  num_classes = sum(len(label_encoders[col].classes_) for col in LABEL_COLUMNS)
  print(f"[INFO] Numero totale di classi: {num_classes}")

num_attack_type = len(label_encoders["Attack_type"].classes_)
num_attack_label = len(label_encoders["Attack_label"].classes_)
num_classes_split = [num_attack_type, num_attack_label]


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df1["text"], df1["multilabel_target"], test_size=0.4, random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)


def make_tf_dataset(texts, labels_attack_type, labels_attack_label,
                    tokenizer, max_length, batch_size, shuffle=False):
    # Tokenizzazione
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf"
    )
    # Definizione one_hot multi‐label
    # labels_attack_type e labels_attack_label sono liste di int
    one_hot = []
    for at, al in zip(labels_attack_type, labels_attack_label):
        v = tf.concat([
            tf.one_hot(at, depth=num_attack_type),
            tf.one_hot(al, depth=num_attack_label)
        ], axis=0)
        one_hot.append(v)
    one_hot = tf.stack(one_hot)  # shape (N, num_classes)

    # Dataset
    ds = tf.data.Dataset.from_tensor_slices((
        {"input_ids": encodings["input_ids"],
         "attention_mask": encodings["attention_mask"]},
        one_hot
    ))

    # Shuffle
    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# Creazione delle etichette separate
train_labels_attack_type = [label[0] for label in train_labels.tolist()]
train_labels_attack_label = [label[1] for label in train_labels.tolist()]

val_labels_attack_type = [label[0] for label in val_labels.tolist()]
val_labels_attack_label = [label[1] for label in val_labels.tolist()]

test_labels_attack_type = [label[0] for label in test_labels.tolist()]
test_labels_attack_label = [label[1] for label in test_labels.tolist()]


train_dataset = make_tf_dataset(train_texts,
                                train_labels_attack_type,
                                train_labels_attack_label,
                                tokenizer,
                                max_length=MAX_LENGTH,
                                batch_size=batch_size,
                                shuffle=True)

val_dataset = make_tf_dataset(val_texts,
                              val_labels_attack_type,
                              val_labels_attack_label,
                              tokenizer,
                              max_length=MAX_LENGTH,
                              batch_size=batch_size,
                              shuffle=False)

test_dataset = make_tf_dataset(test_texts,
                               test_labels_attack_type,
                               test_labels_attack_label,
                               tokenizer,
                               max_length=MAX_LENGTH,
                               batch_size=batch_size,
                               shuffle=False)


model = TFBartForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    problem_type="multi_label_classification" 
)


# optimizer e loss
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss_fn   = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Soglia per binarizzare
THRESHOLD = 0.5

def calculate_accuracy(labels, preds, threshold=THRESHOLD):
    preds_binary = (preds > threshold).astype(int)
    correct = np.sum(labels == preds_binary)
    total = labels.size
    return correct / total * 100


def traning_loop(model, train_dataset):
    total_loss = 0.0
    all_labels = []
    all_preds  = []

    for batch in tqdm(train_dataset, desc="Train"):
        inputs, labels = batch
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        decoder_input_ids = input_ids

        with tf.GradientTape() as tape:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                training=True
            )
            logits = outputs.logits  # (batch_size, num_classes)
            loss = loss_fn(labels, logits)

        # Calcolo gradienti e update dei pesi
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss += loss.numpy()
        # Salvataggio dei risultati per metriche
        all_labels.append(labels.numpy())
        all_preds.append(tf.sigmoid(logits).numpy())

    # Concatenazione per calcolo metriche finali
    all_labels = np.vstack(all_labels)
    all_preds  = np.vstack(all_preds)
    all_preds_binary = (all_preds > THRESHOLD).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds_binary, average="macro", zero_division=0
    )
    accuracy = calculate_accuracy(all_labels, all_preds)

    print(f"Train — Loss: {total_loss/len(train_dataset):.4f} "
          f"Acc: {accuracy:.2f}%  Prec: {precision:.4f}  Rec: {recall:.4f}  F1: {f1:.4f}")
    return total_loss / len(train_dataset)


def evaluate_loop(model, val_dataset):
    total_loss = 0.0
    all_labels = []
    all_preds  = []

    for batch in tqdm(val_dataset, desc="Val"):
        inputs, labels = batch
        # Estrai input_ids e attention_mask decoder_input_ids
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # decoder_input_ids uguali agli input_ids
        decoder_input_ids = input_ids

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            training=False
        )
        logits = outputs.logits
        loss = loss_fn(labels, logits)

        total_loss += loss.numpy()
        all_labels.append(labels.numpy())
        all_preds.append(tf.sigmoid(logits).numpy())

    all_labels = np.vstack(all_labels)
    all_preds  = np.vstack(all_preds)
    all_preds_binary = (all_preds > THRESHOLD).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds_binary, average="macro", zero_division=0
    )
    accuracy = calculate_accuracy(all_labels, all_preds)

    print(f"Val   — Loss: {total_loss/len(val_dataset):.4f} "
          f"Acc: {accuracy:.2f}%  Prec: {precision:.4f}  Rec: {recall:.4f}  F1: {f1:.4f}")
    return total_loss / len(val_dataset)


for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    train_loss = train_one_epoch(model, train_dataset)
    val_loss   = evaluate_loop(model, val_dataset)


evaluate_loop(model, test_dataset)

# Calcola la recall precision f1 per ogni tipo in Attack Type e Attack Label
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





























