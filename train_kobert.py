import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import os, pickle

# 🔧 DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔧 CONFIG
MODEL_NAME = 'monologg/kobert'
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
MODEL_PATH = "saved_model"

# 🔧 STEP 1 – LOAD CSV
train_df = pd.read_csv("kobert_train.csv")
val_df = pd.read_csv("kobert_val.csv")
test_df = pd.read_csv("kobert_test.csv")

le = LabelEncoder()
train_df['label_id'] = le.fit_transform(train_df['label'])
val_df['label_id'] = le.transform(val_df['label'])
test_df['label_id'] = le.transform(test_df['label'])

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# 🔧 STEP 2 – DATASET CLASS
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_loader = DataLoader(TextDataset(train_df['text'], train_df['label_id']), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TextDataset(val_df['text'], val_df['label_id']), batch_size=BATCH_SIZE)
test_loader = DataLoader(TextDataset(test_df['text'], test_df['label_id']), batch_size=BATCH_SIZE)

# 🔧 STEP 3 – MODEL INIT
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_))
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 🔧 STEP 4 – TRAINING LOOP
EPOCHS = 10  # Number of epochs to train
for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Training Loss: {total_loss:.4f}")

    # VALIDATION
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1)

            val_preds.extend(pred.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {acc:.4f}")

# 🔧 STEP 5 – EVALUATE ON TEST
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1)
        test_preds.extend(pred.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

print("\n📊 TEST RESULT")
print("Accuracy:", accuracy_score(test_labels, test_preds))
print(classification_report(test_labels, test_preds, target_names=[str(c) for c in le.classes_]))

# 💾 SAVE MODEL & TOKENIZER
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")