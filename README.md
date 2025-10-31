
# Models and Datasets used in this repository
- [Models](#models)  
- [Datasets](#datasets)

## Models
---
### Encoder-only
[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)

### Decoder-only
[distilGPT2](https://huggingface.co/distilbert/distilgpt2)  
  
[GPT-Neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125m)

### Encoder-Decoder
[t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)  
  
[bart-base](https://huggingface.co/facebook/bart-base)

---
## Datasets
[EDGE-IIoTset – A New Comprehensive Realistic Cyber Security Dataset for IoT and IIoT Applications](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)  
  
[TON](https://ieee-dataport.org/documents/toniot-datasets#files), For download [(TON dataset)](https://research.unsw.edu.au/projects/toniot-datasets)  



# Incremental Malware Detector (Detailed)

> **Based on the source file**  
> `Incremental Malware Detector.py`

---

## Short description
This project implements an experimental **incremental learning** system for classifying types of attacks (malware / network attacks) using **DistilBERT** as the backbone and **LoRA** (via `peft`) to add new adapters in successive rounds.  
The architecture saves for each round a LoRA adapter (if present) and the corresponding classification head in `round_<n>/`.

---

## Index
- [Main features](#main-features)  
- [Getting started](#getting-started)
  - [Requirements](#requirements)
  - [Main dependencies](#main-dependencies)
  - [Quick installation](#quick-installation)
  - [Download the EDGE-IIoTset Dataset](#download-the-edge-iiotset-dataset)  
  
- [Main file structure](#main-file-structure)  
- [Where to modify the file path](#where-to-modify-the-file-path)  
- [Saved files](#saved-files)  
- [Required dataset format](#required-dataset-format)  

---

# Main features
- Initial training of a **classification head** on an initial subset of classes (round 0).  
- Mechanism to add new rounds with **LoRA**: each subsequent round creates a new adapter trained only on the new examples.  
- Separate saving of **heads (head.pt)** and **adapters (round_<n>/adapter)** for reuse or evaluation.  
- Functions to simulate federated rounds (`simulate_federated_rounds`) and evaluate performance on test DataFrames (`evaluate_on_dataframe`).  
- Summary function (`get_model_summary`) to view known classes, rounds, and the mapping `class → round`.

---

# Getting started
This section explains how to set up the environment, prepare the data, and start incremental training.
The typical workflow consists of four main phases: setup, dataset preparation, initial round training, and addition of new rounds.

## Requirements
Recommended **Python ≥ 3.9** and GPU (optional, the code automatically detects `cuda` if available).

## Main dependencies
```bash
python -m pip install torch transformers peft datasets pandas scikit-learn tqdm numpy
```
> **Note:** select the version of torch that is compatible with your CUDA (if you are using a GPU).

## Quick installation
1. Download the file `Incremental Malware Detector.py'.

2. Set the path to your CSV dataset (edit the variable `file_path` inside the script).

3. Run: 
```bash
python "Incremental Malware Detector.py"
```

---
### Download the EDGE-IIoTset Dataset
The dataset is available on IEEE Dataport:
[EDGE-IIoTset – A New Comprehensive Realistic Cyber Security Dataset for IoT and IIoT Applications](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)
Extract the `.csv` file into a local directory.
Set the path to your CSV dataset (edit the variable file_path inside the script).

---
> **Note:** The model was trained and tested using a sub-dataset derived from the `train.csv` file.

## Main file structure
File **`Incremental Malware Detector.py`**

Contains the class `IncrementalMalwareDetector` with the main methods:

- `initialize_base_model(initial_classes, training_data)`  
- `add_new_malware_round(new_classes, training_data)`  
- `_create_lora_for_round(new_classes)`  
- `_train_base_model(model, training_data, classes)`  
- `_train_lora_round(lora_model, training_data, target_classes)`  
- `_create_dataset_for_classes(data, target_classes)` – returns a custom `torch.utils.data.Dataset` personalizzata  
- `simulate_federated_rounds(data, rounds_config)`  
- `get_model_summary()`  
- `evaluate_on_dataframe(df)`  

Also includes function `setup_incremental_experiment(df)` which creates an example configuration and launches the simulation.

---

## Where to modify the file path

```python
file_path = ''        # <- Insert here the path to the CSV, es. '/path/to/dataset.csv'
DATASET = file_path
df = pd.read_csv(DATASET)

train_set, test_set = train_test_split(df, test_size=0.4, random_state=42)

detector = setup_incremental_experiment(train_set)

filtered_data_test = test_set[test_set['Attack_type'].isin(detector.known_classes)]
detector.evaluate_on_dataframe(filtered_data_test)
```
---

## Saved files
For each round `n` a folder `round_n/` is created containing:
- `head.pt` → classification head state
- `adapter/` → LoRA adapter (only if present)
```bash
round_0/
 ├── head.pt
round_1/
 ├── adapter/
 └── head.pt
...
```

---

## Required dataset format
Il codice richiede un `pandas.DataFrame` con almeno:
- `Attack_type` (string) — class label
- `Attack_label` (int) — binary value (0 = normal, 1 = attack)
- All other columns are concatenated into a single string for model input:
```python
text = " ".join([str(row[col]) for col in self.data.columns if col not in ['Attack_type', 'Attack_label']])
```

Example:
```csv
Attack_type,Attack_label,src_ip,dst_ip,protocol,length
DDoS_UDP,1,192.168.1.10,172.16.0.4,UDP,512
Normal,0,10.0.0.2,10.0.0.3,TCP,60
```

---

© 2025 — Incremental Malware Detector — Demonstration project for academic research.



