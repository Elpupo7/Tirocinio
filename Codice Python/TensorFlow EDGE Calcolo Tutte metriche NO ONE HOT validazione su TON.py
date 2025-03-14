import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

df = pd.read_csv("windows10_dataset.csv")
df = df.rename(columns={'label': 'Attack_label'})

# Formattazione attacchi di TON in modo da mappare correttamente le occorenze con i dizionari usati per addestrare il modello DistilBert di EDGE IIoT
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
attack_samples = {
    attack_type: (
         attack_dfs["SQL_injection"] if attack_type == "SQL_injection"
        else attack_dfs["MITM"] if attack_type == "MITM"
        else attack_dfs["DDoS_TCP"] if attack_type == "DDoS_TCP"
        else attack_dfs["Port_Scanning"] if attack_type == "Port_Scanning"
        else attack_dfs["XSS"] if attack_type == "XSS"
        else df.sample(n=2000, random_state=42)
    )                                           
    for attack_type, df in attack_dfs.items()


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



dataset1 = Dataset.from_pandas(df1)

# "Testualizzazione"
def combine_features(example):
    exclude_keys = ['Unnamed: 0', 'Attack_label', 'Attack_type']
    text_parts = []
    for key, value in example.items():
        if key not in exclude_keys:
            text_parts.append(f"{key}: {value}")
    example["text"] = " ".join(text_parts)
    return example

dataset1 = dataset1.map(combine_features)
# Train e test (80%-20%)
dataset = dataset1.train_test_split(test_size=0.2, seed=42)
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


test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=20,
)


class MacroPrecision(tf.keras.metrics.Metric):
    def __init__(self, name="macro_precision", num_classes=15, **kwargs):
        super(MacroPrecision, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precisions = [tf.keras.metrics.Precision() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)  # Converti le predizioni in etichette intere
        y_true = tf.cast(y_true, tf.int32)  # Assicura che le etichette siano interi

        for i in range(self.num_classes):
            y_true_i = tf.cast(y_true == i, tf.int32)
            y_pred_i = tf.cast(y_pred == i, tf.int32)
            self.precisions[i].update_state(y_true_i, y_pred_i, sample_weight)

    def result(self):
        precisions = [p.result() for p in self.precisions]
        return tf.reduce_mean(tf.stack(precisions))  # Media delle precisioni di tutte le classi

    def reset_state(self):
        for p in self.precisions:
            p.reset_state()


class MacroRecall(tf.keras.metrics.Metric):
    def __init__(self, name="macro_recall", num_classes=15, **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.recalls = [tf.keras.metrics.Recall() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int32)

        for i in range(self.num_classes):
            y_true_i = tf.cast(y_true == i, tf.int32)
            y_pred_i = tf.cast(y_pred == i, tf.int32)
            self.recalls[i].update_state(y_true_i, y_pred_i, sample_weight)

    def result(self):
        recalls = [r.result() for r in self.recalls]
        return tf.reduce_mean(tf.stack(recalls))  # Media dei recall di tutte le classi

    def reset_state(self):
        for r in self.recalls:
            r.reset_state()

class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="macro_f1", num_classes=15, **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision_metric = MacroPrecision(num_classes=num_classes)
        self.recall_metric = MacroRecall(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()

model_path = "/content/drive/My Drive/tensorFlow_model_checkpointEDGE_Distill_1000ES_NO_ONE_HOT_2/tf_model.h5"

model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(attack_types),
    id2label=id2label,
    label2id=label2id
)

model.load_weights(model_path)
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-5, decay_steps=1000, decay_rate=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
        MacroPrecision(num_classes=len(attack_types)),
        MacroRecall(num_classes=len(attack_types)),
        MacroF1Score(num_classes=len(attack_types))
    ]
)


results = model.evaluate(test_dataset)

print(f"Loss: {results[0]}")
print(f"Accuracy: {results[1]}")
print(f"Precision: {results[2]}")
print(f"Recall: {results[3]}")
print(f"F1-Score: {results[4]}")









