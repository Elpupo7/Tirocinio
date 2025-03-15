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










