
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support 


# Per usare il dataset con le SOLE COLONNE RILEVANTI
df_full = pd.read_csv('preprocessed_DNN.csv', delimiter=",")
print(len(df_full.columns))


colonne_rilevanti = [
    # TCP
    'tcp.ack', 'tcp.ack_raw', 'tcp.checksum', 'tcp.connection.syn',
    'tcp.connection.synack', 'tcp.flags', 'tcp.flags.ack', 'tcp.len', 'tcp.seq',

    # UDP
    'udp.stream', 'udp.time_delta',

    # DNS
    'dns.qry.name', 'dns.qry.qu', 'dns.qry.type', 'dns.retransmission',
    'dns.retransmit_request', 'dns.retransmit_request_in',

    # HTTP
    'http.content_length', 'http.response', 'http.tls_port',
    'http.request.method-GET', 'http.request.method-POST',
    'http.request.method-OPTIONS', 'http.request.method-PROPFIND',
    'http.request.method-PUT', 'http.request.method-SEARCH',
    'http.request.method-TRACE',

    # MQTT
    'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len',
    'mqtt.msg_decoded_as', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.topic_len', 'mqtt.ver',

    # Modbus/TCP
    'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id',

    # Target / Etichetta
    'Attack_label', 'Attack_type'
]

# Creazione del nuovo DataFrame con solo le colonne selezionate
df = df_full[colonne_rilevanti]
print(len(df.columns))


LABEL_COLUMNS = ["Attack_type", "Attack_label"]
MAX_LENGTH = 128
batch_size = 32 
EPOCHS = 3
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")



df = pd.read_csv(file_path, delimiter=",") # Per usare il dataset con le FEATURES PREDEFINITE

print("Dimensione del nuovo DataFrame:", df.shape)
print("Distribuzione di Attack_label:")
print(df["Attack_label"].value_counts())
print('\n')
print(df["Attack_type"].value_counts())



attack_dfs = {attack_type: df[df["Attack_type"] == attack_type]
              for attack_type in df["Attack_type"].unique()}

attack_samples = {
    attack_type: (
        attack_dfs["Fingerprinting"] if attack_type == "Fingerprinting"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else df.sample(n=2000)
    )
    for attack_type, df in attack_dfs.items()
}


df1 = pd.concat(list(attack_samples.values())).sample(frac=1, random_state=42)
print("Dimensione del nuovo DataFrame:", df1.shape)
print("Distribuzione di Attack_label:")
print(df1["Attack_label"].value_counts())
print('\n')
print(df1["Attack_type"].value_counts())


label_encoders = {}
for col in LABEL_COLUMNS:
    label_encoders[col] = LabelEncoder()
    df1[col] = label_encoders[col].fit_transform(df1[col])

# Combine encoded label columns into a list of indices for each sample
df1["multilabel_target"] = df1.apply(lambda row: [row[col] for col in LABEL_COLUMNS], axis=1)

# Concatenate relevant feature columns into a single 'text' column for input to the model
df1["text"] = df1.apply(lambda row: " ".join([str(row[col]) for col in df1.columns if col not in LABEL_COLUMNS + ["multilabel_target"]]), axis=1)

print(df1["multilabel_target"].value_counts())


num_attack_type = len(label_encoders["Attack_type"].classes_)
num_attack_label = len(label_encoders["Attack_label"].classes_)
num_classes_split = [num_attack_type, num_attack_label] # Usato per il calcolo dell emetriche precision recall f1 macro



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


def encode_text(text, label_attack_type, label_attack_label):
    text = text.numpy().decode('utf-8')
  
    label_attack_type = label_attack_type.numpy()
    label_attack_label = label_attack_label.numpy()

    encodings = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    input_ids = np.array(encodings['input_ids'], dtype=np.int32)
    attention_mask = np.array(encodings['attention_mask'], dtype=np.int32)

    # Crea il vettore one-hot per Attack_type
    one_hot_attack_type = np.zeros(num_attack_type, dtype=np.float32)
    one_hot_attack_type[int(label_attack_type)] = 1.0

    # Crea il vettore one-hot per Attack_label
    one_hot_attack_label = np.zeros(num_attack_label, dtype=np.float32)
    one_hot_attack_label[int(label_attack_label)] = 1.0

    # Creo un unico vettore one-hot
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

# Etichette separate per Attack Type e Attack Label 
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


train_dataset = train_dataset.shuffle(buffer_size=len(train_texts))  # Shuffle per il train dataset
train_dataset = train_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.shuffle(buffer_size=len(val_texts))  # Shuffle per il validation dataset
val_dataset = val_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.shuffle(buffer_size=len(test_texts))  # Aggiunge lo shuffle
test_dataset = test_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)


class CustomF1Metric(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", threshold=0.5, **kwargs):
        super(CustomF1Metric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=self.threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=self.threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(tf.sigmoid(y_pred), self.threshold), tf.float32)
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


model.compile( 
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision', thresholds=0.5),
        tf.keras.metrics.Recall(name='recall', thresholds=0.5),
        CustomF1Metric(name='f1_score')
    ]
)


history  = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

train_loss_history = history.history['loss']
val_loss_history = history.history['val_loss']

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid(alpha=0.3)

# Calcola la recall precision f1 per ogni tipo in Attack Type e Attack Label
def metrics(labels, pred):
    preds_binary = (pred > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_binary, average=None)
    for i, class_name in enumerate(classi):
        print(f"Class {class_name} | Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}\n")


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


y_trues, y_preds = [], []

for batch in test_dataset:
    inputs, labels = batch 
    preds = model.predict(inputs, verbose=0)
    y_trues.append(labels.numpy())
    y_preds.append(tf.sigmoid(preds.logits).numpy())

# Concatena tutti i batc
y_true = np.concatenate(y_trues, axis=0)
y_pred = np.concatenate(y_preds, axis=0)


print('Metriche per ogni classe')
metrics(y_true, y_pred)
print('\nMacro:')
metrics1(y_true, y_pred, num_classes_split)


