import os
import torch
import mlflow
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing.clean import clean_text

# --- CONFIGURATION ---
DATA_PATH = os.path.join("dataset", "data.csv")
MODEL_SAVE_PATH = os.path.join("models", "dl_model.pth")
EPOCHS = 3          
BATCH_SIZE = 8      
MAX_LEN = 128       
LEARNING_RATE = 2e-5

# 1. Custom Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Setup MLflow
    mlflow.set_experiment("Sentiment_Analysis_DL")
    
    # Load & Split Data
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].apply(clean_text)
    
    # Split: 80% Train, 10% Val, 10% Test
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # DataLoaders
    def create_data_loader(df, tokenizer, max_len, batch_size):
        ds = SentimentDataset(
            texts=df.text.to_numpy(),
            labels=df.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=0)

    train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    # Model Setup
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model = model.to(device)

    # Optimizer & Scheduler
    # FIXED: Removed 'correct_bias=False' which is not valid for torch.optim.AdamW
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE) 
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # --- THE CUSTOM TRAINING LOOP ---
    best_accuracy = 0
    
    with mlflow.start_run():
        # Log Params
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("lr", LEARNING_RATE)
        mlflow.log_param("model", "DistilBERT")

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            print("-" * 10)

            # --- TRAIN STEP ---
            model.train()
            total_loss = 0
            
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                model.zero_grad() # Clear gradients

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward() # Backpropagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"Average Train Loss: {avg_train_loss}")

            # --- VALIDATION STEP ---
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Convert logits to predictions
                    _, preds = torch.max(outputs.logits, dim=1)
                    
                    val_preds.extend(preds.cpu().tolist())
                    val_labels.extend(labels.cpu().tolist())

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            print(f"Val Accuracy: {val_acc} | Val F1: {val_f1}")

            # Log Metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            # --- SAVE BEST MODEL ---
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print("✅ Model improved and saved!")
                
        print(f"\nTRAINING COMPLETE. Best Val Acc: {best_accuracy}")

if __name__ == "__main__":
    train()