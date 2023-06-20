
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import os

# --- Configuration ---
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

# --- Dummy Data Generation (Replace with actual dataset loading) ---
def load_dummy_data(num_samples=1000):
    texts = ["This movie was fantastic!", "I hated every minute of it.", "It was okay, nothing special."] * (num_samples // 3)
    labels = [1, 0, 1] * (num_samples // 3) # 1 for positive, 0 for negative
    # Ensure lists have exactly num_samples elements
    texts = texts[:num_samples]
    labels = labels[:num_samples]
    return pd.DataFrame({'text': texts, 'label': labels})

df = load_dummy_data()

# --- Tokenization and Data Preparation ---
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

def tokenize_data(dataframe, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for text in dataframe['text']:
        encoded_dict = tokenizer.encode_plus(
            text,                       # Sentence to encode.
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
            max_length=max_len,         # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True, # Construct attn. masks.
            return_tensors='pt',        # Return pytorch tensors.
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(dataframe['label'].values)

    return input_ids, attention_masks, labels

input_ids, attention_masks, labels = tokenize_data(df, tokenizer, MAX_LEN)

# Split data into train and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, labels, random_state=42, test_size=0.1
)
train_masks, validation_masks, _, _ = train_test_split(
    attention_masks, labels, random_state=42, test_size=0.1
)

# Create DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Initialization ---
model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, 
    num_labels=2, 
    output_attentions=False, 
    output_hidden_states=False
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Training Loop ---
print("Starting BERT model training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        
        model.zero_grad()
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"  Epoch {epoch+1} | Average training loss: {avg_train_loss:.2f}")

    # Validation
    model.eval()
    predictions, true_labels = [], []
    for batch in validation_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_labels.flatten()
    
    acc = accuracy_score(labels_flat, preds_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='binary')
    
    print(f"  Validation Accuracy: {acc:.2f}")
    print(f"  Validation Precision: {precision:.2f}")
    print(f"  Validation Recall: {recall:.2f}")
    print(f"  Validation F1-Score: {f1:.2f}")

print("BERT model training complete.")

# --- Save Model (Optional) ---
# output_dir = './model_save/'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
# model_to_save.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# Commit 1 marker: 2023-03-15 10:30:00

# Commit 2 marker: 2023-06-20 14:00:00
