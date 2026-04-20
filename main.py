import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer

# specify GPU
device = torch.device("cuda")

# Now, we need a way that I can create training, testing, and validation
true_data = pd.read_csv("archive/True.csv")
fake_data = pd.read_csv("archive/Fake.csv")
# print(true_data.head())
# print(fake_data.head())
# print(torch.cuda.is_available())

# Instantiate the tokenizer for the BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


true_data["label"] = 1
fake_data["label"] = 0

df = pd.concat([true_data, fake_data], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# print(df.head(), "\n\n")
# print(df["label"].value_counts())
#
import pandas as pd
import re


def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove specialA characters (keep basic punctuation)
    text = re.sub(r"[^A-Za-z0-9.,!?;\'\"\-\s]", "", text)
    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text).strip()
    # Optional: lowercase (BERT tokenizer lowercases automatically if using bert-base-uncased)
    text = text.lower()
    return text


df["title"] = df["title"].apply(clean_text)
df["text"] = df["text"].apply(clean_text)

# Concatenate (title + separator + text)
df["content"] = df["title"] + ". " + df["text"]

# Keep onlly what's needed
df = df[["content", "label"]]

train_text, temp_text, train_labels, temp_labels = train_test_split(
    df["content"], df["label"], random_state=2018, test_size=0.3, stratify=df["label"]
)


val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels
)
