#!/usr/bin/env python
# coding: utf-8

# In[1]:


# FROM: https://www.kaggle.com/code/aryand03/nlp-assignment
# Import required libraries for modifying and fine-tuning BERT model
import shutil
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from datasets import load_dataset, Dataset

# Clear stale checkpoints before each run. Old transformers versions (v4) saved
# LayerNorm params as gamma/beta; v5 expects weight/bias. Stale files cause a
# hard mismatch when load_best_model_at_end=True tries to reload the checkpoint.
for stale_dir in ("./results", "./logs"):
    if os.path.exists(stale_dir):
        shutil.rmtree(stale_dir)


# In[2]:


# FROM https://www.kaggle.com/code/aryand03/nlp-assignment
# Manipulate data to use datasets library - 
true_df = pd.read_csv("archive/True.csv")
fake_df = pd.read_csv("archive/Fake.csv")

# Add labels: 1 = True, 0 = Fake
true_df['label'] = 1
fake_df['label'] = 0

# Combine
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# # combine title and text into one single field.
# df['combined_text'] = df['title'] + " " + df['text']

# Only include the title, text, and label columns in final dataframe
df = df[['title','text', 'label']]


# In[3]:


# FROM - https://huggingface.co/docs/datasets/use_with_pandas
# Convert from pandas to Dataset 
ds = Dataset.from_pandas(df)

# Convert the integer column to a ClassLabel
ds = ds.class_encode_column("label")

# First split: 80% Train, 20% "Remainder" (Val + Test)
train_remainder_split = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

# Second split: Split that 20% remainder into half (10% Val, 10% Test)
test_val_split = train_remainder_split['test'].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

# Combine into a single DatasetDict
from datasets import DatasetDict

split_dataset = DatasetDict({
    'train': train_remainder_split['train'],
    'test': test_val_split['test'],
    'validation': test_val_split['train']
})

print(split_dataset)


# In[4]:


# FROM - https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
# BERT Tokenization
from transformers import AutoTokenizer

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["title"], 
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )

# Apply the tokenizer to the dataset
tokenized_datasets = split_dataset.map(tokenize_function, batched=True)

# Inspect tokenized samples
print(tokenized_datasets["train"][0])


# In[5]:


# FROM - https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
# Model selection

from transformers import AutoModelForSequenceClassification

# Initialize a BERT model for binary classification
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

print(model.config)


# In[6]:


# FROM - https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
# Freeze layers

# Freeze all layers except the classifier
for param in model.bert.parameters():
    param.requires_grad = False

# Keep only the classification head trainable
for param in model.classifier.parameters():
    param.requires_grad = True

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# In[15]:


# FROM - training pipeline https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c

from transformers import TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    # Save and select the best checkpoint by F1, not loss. A model can have
    # lower loss but worse real-world classification — F1 is the right signal.
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    fp16=True,
    save_strategy="best",
    # logging_dir is deprecated in v5.2+; use the env var TENSORBOARD_LOGGING_DIR
    # instead if you want TensorBoard logs.
)

print(training_args)


# In[9]:


# FROM - https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
# Define a custom metric
from transformers import Trainer
from evaluate import load

# Load both metrics; each is a separate stateful object
metric_f1  = load("f1")
metric_acc = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    f1  = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    acc = metric_acc.compute(predictions=predictions, references=labels)
    # Merge both dicts so the Trainer logs eval_f1 and eval_accuracy each epoch
    return {**f1, **acc}


# In[11]:


# FROM - https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
# Data collation
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[16]:


# FROM - https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
# Setup for Trainer
trainer = Trainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=data_collator,        # Efficient batching
    compute_metrics=compute_metrics     # Custom metric
)

# Start training
trainer.train()

# In[ ]:

import matplotlib.pyplot as plt

# trainer.state.log_history is a flat list of dicts; eval entries are
# distinguished by having "eval_f1" present.
eval_logs = [e for e in trainer.state.log_history if "eval_f1" in e]

epochs    = [e["epoch"]    for e in eval_logs]
f1_scores = [e["eval_f1"]  for e in eval_logs]
acc_scores = [e["eval_accuracy"] for e in eval_logs]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, f1_scores,  marker="o", linewidth=2, label="Weighted F1")
ax.plot(epochs, acc_scores, marker="s", linewidth=2, label="Accuracy")

ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_title("F1 and Accuracy per Epoch")
ax.set_xticks(epochs)
ax.set_xticklabels([int(e) for e in epochs])
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig("metrics_per_epoch.png", dpi=150)
plt.show()

# In[ ]:

# ── Qualitative Analysis ──────────────────────────────────────────────────────
# Pull raw (un-tokenized) test rows so we can display human-readable text.
# split_dataset["test"] still has the original title/text columns because
# tokenized_datasets is a separate object; the raw split is unchanged.

import torch
import torch.nn.functional as F

LABEL_NAMES = {0: "FAKE", 1: "REAL"}
CORRECT_MARKER = "✓"
WRONG_MARKER   = "✗"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(title: str, text: str) -> tuple[str, float]:
    """Return predicted label name and the confidence (probability) for it."""
    inputs = tokenizer(
        title,
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits          # shape: (1, 2)
    probs = F.softmax(logits, dim=-1)[0]         # convert raw scores → probabilities
    pred_idx = probs.argmax().item()
    return LABEL_NAMES[pred_idx], probs[pred_idx].item()


# Sample a fixed set: 5 correctly classified + 5 misclassified, for balance.
# We iterate the raw test split and collect examples until we have enough of each.
correct_samples, wrong_samples = [], []

for row in split_dataset["test"]:
    if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
        break
    pred_label, confidence = predict(row["title"], row["text"])
    true_label = LABEL_NAMES[row["label"]]
    entry = {
        "title":      row["title"],
        "text":       row["text"][:300],   # truncate long articles for display
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": confidence,
        "correct":    pred_label == true_label,
    }
    if entry["correct"] and len(correct_samples) < 5:
        correct_samples.append(entry)
    elif not entry["correct"] and len(wrong_samples) < 5:
        wrong_samples.append(entry)

# Print a formatted qualitative report.
def print_sample(idx: int, entry: dict) -> None:
    marker = CORRECT_MARKER if entry["correct"] else WRONG_MARKER
    print(f"  [{marker}] Sample {idx + 1}")
    print(f"      Title      : {entry['title'][:120]}")
    print(f"      Text (clip): {entry['text'][:120]}...")
    print(f"      True label : {entry['true_label']}")
    print(f"      Predicted  : {entry['pred_label']}  (confidence: {entry['confidence']:.1%})")
    print()

print("\n" + "═" * 70)
print("  QUALITATIVE ANALYSIS — Sample Predictions")
print("═" * 70)

print("\n── Correctly Classified ─────────────────────────────────────────────\n")
for i, entry in enumerate(correct_samples):
    print_sample(i, entry)

print("── Misclassified ────────────────────────────────────────────────────\n")
for i, entry in enumerate(wrong_samples):
    print_sample(i, entry)

print("═" * 70)
