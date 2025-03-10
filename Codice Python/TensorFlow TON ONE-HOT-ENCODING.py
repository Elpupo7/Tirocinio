
import os
import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
df = pd.read_csv('windows10_dataset.csv')
df = df.rename(columns={'label': 'Attack_label'}) # Uso label già per il mapping delgi attacchi SOLO SE STAI SCARICANDO IL CSV widows10

# Formattazione attacchi di TON in modo da mappare correttamente le occorenze con i dizionari usati per addestrare il modello DistilBert di EDGE IIoT
# Per addestrare solo il modello su tensorflow non è propriamente necessario, ma il modello l'ho addestrato così io
df["type"] = df["type"].replace({
    "normal": "Normal",
    "ddos": "DDoS_UDP",
    "password": "Password",
    "xss": "XSS",
    "injection": "SQL_injection",
    "dos": "DDoS_TCP",
    "scanning": "Port_Scanning",
    "mitm": "MITM"
})

print(df["type"].value_counts())
print(df['Attack_label'].value_counts())

attack_dfs = {attack_type: df[df["type"] == attack_type]
              for attack_type in df["type"].unique()}
# Tutti quelli che hanno 1000 campioni
attack_samples = {
    attack_type: (
         attack_dfs["SQL_injection"] if attack_type == "SQL_injection"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else attack_dfs["DDoS_TCP"] if attack_type == "DDoS_TCP"
        else attack_dfs["Port_Scanning"] if attack_type == "Port_Scanning"
        else attack_dfs["XSS"] if attack_type == "XSS"
        else attack_dfs["Password"] if attack_type == "Password"
        else attack_dfs["DDoS_UDP"] if attack_type == "DDoS_UDP"
        else df.sample(n=5000, random_state=42)
    )                                          
    for attack_type, df in attack_dfs.items()
}

df1 = pd.concat(list(attack_samples.values())).sample(frac=1, random_state=42)
print("Dimensione del nuovo DataFrame:", df1.shape)
print("Distribuzione di Attack_label:")
print(df1["Attack_label"].value_counts())
print('\n')
print(df1["type"].value_counts())

# Mapping
attack_types = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'DDoS_HTTP', 'Port_Scanning',
    'SQL_injection', 'Password', 'Backdoor', 'Uploading', 'Vulnerability_scanner',
    'XSS', 'Ransomware', 'DDoS_TCP', 'Fingerprinting', 'MITM'
]

# Dizionari di mapping
label2id = {label: idx for idx, label in enumerate(attack_types)}
id2label = {idx: label for label, idx in label2id.items()}
df1["label"] = df1["type"].map(label2id)

# Dataset di Hugging face
import pandas as pd
from datasets import Dataset, DatasetDict
dataset1 = Dataset.from_pandas(df1)
dataset = DatasetDict({"train": dataset1}) 

# "Testualizzo" ogni riga
def combine_features(example):
    exclude_keys = ['Unnamed: 0', 'Attack_label', 'type']
    text_parts = []
    for key, value in example.items():
        if key not in exclude_keys:
            text_parts.append(f"{key}: {value}")
    example["text"] = " ".join(text_parts)
    return example
  
dataset = dataset["train"].map(combine_features)
# Dividi in train e test (80%-20%)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

from collections import Counter
print("Distribuzione nel train set:")
print(Counter(dataset["train"]["label"]))
print("Distribuzione nel test set:")
print(Counter(dataset["test"]["label"]))

# Tokenizzazione
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

dim_batch = 20

train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],  # Usa la colonna "label" (numerica) creata in precedenza
    shuffle=True,
    batch_size= dim_batch,
)

test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size= dim_batch,
)


# CONVERSINE IN ONE HOT ENCODE IL CAMPO LABEL (chiesto a chatgpt)

def one_hot_encode_labels(batch, num_classes=15):
    inputs, labels = batch  # Estrae inputs e labels dal batch
    labels_int = tf.cast(labels, tf.int32)  # Converte le etichette in int32
    one_hot_labels = tf.one_hot(labels_int, depth=num_classes)  # One-hot encoding
    return inputs, one_hot_labels
  

train_dataset = train_dataset.map(lambda x, y: one_hot_encode_labels((x, y), num_classes=len(attack_types)))
test_dataset = test_dataset.map(lambda x, y: one_hot_encode_labels((x, y), num_classes=len(attack_types)))



model = TFDistilBertForSequenceClassification.from_pretrained(  
    "distilbert-base-uncased",
    num_labels=len(attack_types),
    id2label=id2label,
    label2id=label2id
)

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-5, decay_steps=1000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Metrica FScore (chiesta a chatgpt)
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Converti le predizioni in one-hot (se necessario)
        y_pred = tf.nn.softmax(y_pred)  # Applica softmax ai logits
        y_pred = tf.argmax(y_pred, axis=1)  # Ottieni la classe predetta
        y_true = tf.argmax(y_true, axis=1)  # Converti le etichette da one-hot a classi

        # Aggiorna Precision e Recall
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        F1Score(name="f1_score")
    ]
)

model.fit(train_dataset, validation_data=test_dataset, epochs=3)




# Prova del modello
def predict_attack(model, sample_row, id2label):
  
    # Creo il campo "text" per l'esempio che devo predire
    sample_text = combine_features(sample_row)["text"]

    inputs = tokenizer(sample_text, padding="max_length", truncation=True, max_length=500, return_tensors="tf")

    # Predizione
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    predicted_label = id2label[predicted_class]

    return predicted_label




#attack_df = df[df["Attack_label"] == 1]
# Estrai una riga casuale (ottenendo una Pandas Series)
#sample_row = attack_df.sample(n=1, random_state=None).iloc[0]

# Estrazione di una riga casuale (tipo Pandas Series)
sample_row = df.sample(n=1, random_state=None).iloc[0]

# Esegui la predizione
predicted_label = predict_attack(model, sample_row, id2label)
print(sample_row['Attack_label'])
print(sample_row['type'])
print("Predicted label:", predicted_label)


# Per provare più predizioni di uno stesso tipo di attacco assieme:
attack_type = 'Normal'
attack_df = df[df["type"] == attack_type]
# Selezione casuale
attack_df_limited = attack_df.sample(n=10)

print('Esempi totali presenti nel dataset del tipo scelto: ',len(attack_df))

for index, row in attack_df_limited.iterrows():
    predicted_label = predict_attack(model, row, id2label)
    print("Attack_label:", row['Attack_label'])
    print("Attack_type:", row['type'])
    print("Predicted label:", predicted_label)
    print('index: ', index, '\n')












