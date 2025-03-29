
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
df_full = pd.read_csv(file_path, delimiter=",")
df_full = df_full.rename(columns={'type': 'Attack_type'})
df_full = df_full.rename(columns={'label': 'Attack_label'})
print(len(df_full.columns))


# Considero solo le colonne più rilevanti per l'addestramento del modello
import pandas as pd

# Lista delle colonne da tenere
colonne_da_tenere = [
    'Network_I(Intel R _82574L_GNC)TCP_APS',
    'Network_I(Intel R _82574L_GNC) Packets Received Unknown',
    'Network_I(Intel R _82574L_GNC) Bytes Received sec',
    'Network_I(Intel R _82574L_GNC) Bytes Sent sec',
    'Network_I(Intel R _82574L_GNC) Packets Outbound Errors',
    'Network_I(Intel R _82574L_GNC) Packets Received Discarded',
    'Network_I(Intel R _82574L_GNC) Bytes Total sec',
    'Network_I(Intel R _82574L_GNC) Packets Outbound Discarded',
    'Network_I(Intel R _82574L_GNC) TCP RSC Exceptions sec',
    'Network_I(Intel R _82574L_GNC) Packets Sent Unicast sec',
    'Network_I(Intel R _82574L_GNC) Output Queue Length',
    'Network_I(Intel R _82574L_GNC) Packets Received sec',
    'Network_I(Intel R _82574L_GNC) Current Bandwidth',
    'Network_I(Intel R _82574L_GNC) Packets sec',
    'Network_I(Intel R _82574L_GNC) TCP Active RSC Connections',
    'Network_I(Intel R _82574L_GNC) Packets Sent sec',
    'Network_I(Intel R _82574L_GNC) Packets Received Unicast sec',
    'Network_I(Intel R _82574L_GNC) Packets Sent Non-Unicast sec',
    'Network_I(Intel R _82574L_GNC) Packets Received Non-Unicast sec',
    'Network_I(Intel R _82574L_GNC) TCP RSC Coalesced Packets sec',
    'Network_I(Intel R _82574L_GNC) Offloaded Connections',
    'Network_I(Intel R _82574L_GNC) Packets Received Errors',
    'Attack_label', 
    'Attack_type'
]

# Creazione del nuovo DataFrame con solo le colonne selezionate
df = df_full[colonne_da_tenere]

print(len(df.columns))

# Creazione del nuovo DataFrame con solo le colonne selezionate
df = df_full[colonne_rilevanti]
print(len(df.columns))


LABEL_COLUMNS = ["Attack_type", "Attack_label"]
MAX_LENGTH = 128
batch_size = 32 
EPOCHS = 3
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# Per usare il dataset con le FEATURES PREDEFINITE
#df = pd.read_csv(file_path, delimiter=",")
#df = df.rename(columns={'type': 'Attack_type'})
#df = df.rename(columns={'label': 'Attack_label'})
print("Dimensione del nuovo DataFrame:", df.shape)
print("Distribuzione di Attack_label:")
print(df["Attack_label"].value_counts())
print('\n')
print(df["Attack_type"].value_counts())


# Formattazione attacchi di TON in modo da mappare correttamente le occorenze con i dizionari usati per addestrare il modello DistilBert di EDGE IIoT
df["Attack_type"] = df["Attack_type"].replace({
    "normal": "Normal",
    "ddos": "DDoS_UDP",
    "password": "Password",
    "xss": "XSS",
    "injection": "SQL_injection",
    "dos": "DDoS_TCP",
    "scanning": "Port_Scanning",
    "mitm": "MITM"
})

print(df["Attack_type"].value_counts())
print(df['Attack_label'].value_counts())


attack_dfs = {attack_type: df[df["Attack_type"] == attack_type]
              for attack_type in df["Attack_type"].unique()}

attack_samples = {
    attack_type: (
        attack_dfs["SQL_injection"] if attack_type == "SQL_injection"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else attack_dfs["DDoS_TCP"] if attack_type == "DDoS_TCP"
        else attack_dfs["Port_Scanning"] if attack_type == "Port_Scanning"
        else attack_dfs["XSS"] if attack_type == "XSS"
        else attack_dfs["Password"] if attack_type == "Password"
        else attack_dfs["DDoS_UDP"] if attack_type == "DDoS_UDP"
        else df.sample(n=5000, random_state=42)  # Spostato random_state all'interno di sample()
    )
    for attack_type, df in attack_dfs.items()
}


df1 = pd.concat(list(attack_samples.values())).sample(frac=1, random_state=42)
print("Dimensione del nuovo DataFrame:", df1.shape)
print("Distribuzione di Attack_label:")
print(df1["Attack_label"].value_counts())
print('\n')
print(df1["Attack_type"].value_counts())


# Ricarico gli encoder usati per EDGE
from google.colab import drive
import pickle

load_path = "/content/drive/My Drive/label_encoders/label_encoders.pkl"

with open(load_path, "rb") as f:
    label_encoders = pickle.load(f)

df1["Attack_type_encoded"] = label_encoders["Attack_type"].transform(df1["Attack_type"])
df1["Attack_label_encoded"] = label_encoders["Attack_label"].transform(df1["Attack_label"])

# Crea la colonna multilabel_target con gli stessi criteri del dataset di training
df1["multilabel_target"] = df1.apply(lambda row: [row["Attack_type_encoded"], row["Attack_label_encoded"]], axis=1)

df1["text"] = df1.apply(lambda row: " ".join([str(row[col]) for col in df1.columns if col not in LABEL_COLUMNS + ["multilabel_target"]]), axis=1)

print(df1["multilabel_target"].value_counts())  # Controllo finale


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

# Per addestrare il modello su TON
#model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
# Per testare il modello EDGE su TON
model_path = "/content/drive/My Drive/tensorFlow_model_checkpointEDGE_DistilBert_2000ES_MULTILABEL_SOLO_COL_RILEVANTI/tf_model.h5"
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
model.load_weights(model_path)

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

#CONSENTE DI PRENDERE E CALCOLARE LE MACRO METRICHE SOLO SUGLI EFFETTIVI ATTACK TYPE PRESENTI DENTRO AL DATASET TON
def metrics1(labels, pred, class_splits):
    preds_binary = (pred > 0.5).astype(int)

    start = 0
    for i, class_name in enumerate(LABEL_COLUMNS):
        n_classes = class_splits[i]  # Numero di classi nel gruppo corrente
        end = start + n_classes

        # Estrai le colonne per il gruppo corrente
        labels_group = labels[:, start:end]
        preds_group = preds_binary[:, start:end]

        # Se il gruppo corrente è "Attack_type", seleziona solo le classi presenti in Dataset 2
        if class_name == "Attack_type":
            # Indici relativi all'interno del gruppo "Attack_type" di Dataset 1 per le classi di Dataset 2:
            selected_indices = [3, 4, 6, 7, 8, 9, 11, 14]
            # Nota: Poiché in questo gruppo gli indici partono da 0, questi numeri sono validi se il gruppo è completo
            labels_group = labels_group[:, selected_indices]
            preds_group = preds_group[:, selected_indices]


        # Calcola le metriche per il gruppo corrente
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_group, preds_group, average="macro", zero_division=0
        )

        print(f"Group {class_name} | Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        start = end  # Aggiorna "start" per il prossimo gruppo


# Validazione sul test set 
results = model.evaluate(test_dataset)
print(f"Loss: {results[0]}")
print(f"Accuracy: {results[1]}")
print(f"Precision: {results[2]}")
print(f"Recall: {results[3]}")
print(f"F1-Score: {results[4]}")


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


