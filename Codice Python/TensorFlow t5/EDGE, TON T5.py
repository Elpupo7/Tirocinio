import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# Configuration
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = 'EdgeIoT_250_rows.csv'
#DATASET = 'TON_350_rows.csv' # Per Ton 
TARGET_COLUMNS = ["Attack_type","Attack_label"]  # List of target columns
EPOCHS = 80
BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = 'google/t5-efficient-tiny'
OUTPUT_MODEL = f'models/t5_trained.pth'

# Load dataset
print(f"[+] Loading dataset {DATASET}...")
df1 = pd.read_csv(DATASET, delimiter=",")

# Combine all non-target columns into a single text column
text_columns = [col for col in df1.columns if col not in TARGET_COLUMNS]
df1["text"] = df1[text_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

# Encode labels as strings (T5 expects text outputs)
unique_labels = sorted(df1[TARGET_COLUMNS[0]].unique())  # ordinato in modo deterministico
label_to_id = {label: str(i) for i, label in enumerate(unique_labels)}
id_to_label = {str(i): label for label, i in label_to_id.items()}
labels = [label_to_id[label] for label in df1[TARGET_COLUMNS[0]]]  # Assuming single target column

train_texts, test_texts, train_labels, test_labels = train_test_split(df1["text"], labels, test_size=0.4, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

print(f"[+] Initializing T5 tokenizer and model: {MODEL_NAME}...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = TFT5ForConditionalGeneration.from_pretrained(MODEL_NAME)


# Pre-tokenizzazione train set
train_input_texts = ["classify: " + text for text in train_texts]
train_encodings = tokenizer(
    train_input_texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="np"
)
train_labels_encodings = tokenizer(
    train_labels,
    truncation=True,
    padding="max_length",
    max_length=10,
    return_tensors="np"
)

# Pre-tokenizzazione val set
val_input_texts = ["classify: " + text for text in val_texts]
val_encodings = tokenizer(
    val_input_texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="np"
)
val_labels_encodings = tokenizer(
    val_labels,
    truncation=True,
    padding="max_length",
    max_length=10,
    return_tensors="np"
)

# Pre-tokenizzazione test set
test_input_texts = ["classify: " + text for text in test_texts]
test_encodings = tokenizer(
    test_input_texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="np"
)
test_labels_encodings = tokenizer(
    test_labels,
    truncation=True,
    padding="max_length",
    max_length=10,
    return_tensors="np"
)


# Dataset train
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"]
    },
    train_labels_encodings["input_ids"]
))
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Dataset validazione
val_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"]
    },
    val_labels_encodings["input_ids"]
))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Dataset test
test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"]
    },
    test_labels_encodings["input_ids"]
))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

@tf.function
def train_step(batch_inputs, batch_labels):
    with tf.GradientTape() as tape:
        outputs = model(batch_inputs, labels=batch_labels, training=True)
        # La losscalcolata dal modello
        loss = outputs.loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_model(dataset):
    total_loss = 0.0
    steps = 0
    all_preds = []
    all_labels = []
    for batch, (inputs, labels) in enumerate(tqdm(dataset)):
        loss = train_step(inputs, labels)
        total_loss += loss
        steps += 1

        preds_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=10)
        preds = [tokenizer.decode(ids, skip_special_tokens=True).strip().lower() for ids in preds_ids]
        labels_decoded = [tokenizer.decode(ids, skip_special_tokens=True).strip().lower() for ids in labels]
        all_preds.extend(preds)
        all_labels.extend(labels_decoded)

    avg_loss = total_loss / steps
  
    avg_loss_value = float(avg_loss.numpy()) if hasattr(avg_loss, "numpy") else float(avg_loss)

    correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
    accuracy = (correct / len(all_labels)) * 100 if all_labels else 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)

    print(f"Train Loss: {avg_loss_value:.4f}, Train Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return avg_loss_value, accuracy


@tf.function
def val_step(batch_inputs, batch_labels):
    outputs = model(batch_inputs, labels=batch_labels, training=False)
    return outputs.loss

def evaluate_model(dataset):
    total_loss = 0.0
    steps = 0
    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(dataset):
        loss = val_step(inputs, labels)
        total_loss += loss
        steps += 1

        preds_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=10)
        preds = [tokenizer.decode(ids, skip_special_tokens=True).strip().lower() for ids in preds_ids]
        labels_decoded = [tokenizer.decode(ids, skip_special_tokens=True).strip().lower() for ids in labels]
        all_preds.extend(preds)
        all_labels.extend(labels_decoded)

    avg_loss = total_loss / steps
    avg_loss_value = float(avg_loss.numpy()) if hasattr(avg_loss, "numpy") else float(avg_loss)
    correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
    accuracy = (correct / len(all_labels)) * 100 if all_labels else 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    print(f"Val Loss: {avg_loss_value:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return avg_loss_value, accuracy, all_preds, all_labels


train_loss_history = []
val_loss_history = []

print(f"[+] Training the model for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss, train_accuracy = train_model(train_dataset)
    val_loss, val_accuracy, all_preds_val, all_labels_val = evaluate_model(val_dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    #val_accuracy_history.append(val_accuracy)

plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid(alpha=0.3)
plt.show()



test_loss, test_accuracy, all_preds_test, all_labels_test = evaluate_model(test_dataset)


ordered_ids = sorted(id_to_label.keys(), key=int)

# Metriche per Attack type
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels_test,
    all_preds_test,
    labels=ordered_ids,
    average=None
)

for idx, id_str in enumerate(ordered_ids):
    attack_name = id_to_label[id_str]
    print(f"Tipo di attacco: {attack_name}")
    print(f"  Precision: {precision[idx]:.4f}")
    print(f"  Recall:    {recall[idx]:.4f}")
    print(f"  F1-score:  {f1[idx]:.4f}")
    print("-----------------------------")











