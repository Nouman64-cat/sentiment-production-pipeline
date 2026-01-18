import os
import yaml
import torch
import mlflow
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing.clean import clean_text
from src.config import DL_CONFIG_PATH

# Load configuration from YAML
with open(DL_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# Custom Dataset Class
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
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Load & Split Data
    df = pd.read_csv(config["paths"]["data"])
    df['text'] = df['text'].apply(clean_text)
    
    # Split: 80% Train, 10% Val, 10% Test
    df_train, df_temp = train_test_split(
        df, 
        test_size=config["data"]["test_size"], 
        random_state=config["data"]["random_state"]
    )
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=config["data"]["val_test_split"], 
        random_state=config["data"]["random_state"]
    )

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config["model"]["name"])

    # DataLoaders
    def create_data_loader(df, tokenizer, max_len, batch_size):
        ds = SentimentDataset(
            texts=df.text.to_numpy(),
            labels=df.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=0)

    max_len = config["training"]["max_len"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    
    train_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)

    # Model Setup
    model = DistilBertForSequenceClassification.from_pretrained(
        config["model"]["name"], 
        num_labels=config["model"]["num_labels"]
    )
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"]) 
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["optimizer"]["num_warmup_steps"],
        num_training_steps=total_steps
    )

    # Training Loop
    best_accuracy = 0
    counter = 0
    patience = config["early_stopping"]["patience"]
    max_grad_norm = config["regularization"]["max_grad_norm"]
    model_save_path = config["paths"]["model"]
    
    with mlflow.start_run():
        # Log Params
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", config["training"]["learning_rate"])
        mlflow.log_param("max_len", max_len)
        mlflow.log_param("model", config["model"]["name"])
        mlflow.log_param("patience", patience)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 10)

            # --- TRAIN STEP ---
            model.train()
            total_loss = 0
            
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                model.zero_grad()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm) 
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
                    
                    _, preds = torch.max(outputs.logits, dim=1)
                    
                    val_preds.extend(preds.cpu().tolist())
                    val_labels.extend(labels.cpu().tolist())

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            print(f"Val Accuracy: {val_acc} | Val F1: {val_f1}")

            # Log Metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                counter = 0 
                torch.save(model.state_dict(), model_save_path)
                print(f"Model improved! Saved to {model_save_path}")
            else:
                counter += 1
                print(f"No improvement for {counter} epochs.")

            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
                
        print(f"\nTRAINING COMPLETE. Best Val Acc: {best_accuracy}")


if __name__ == "__main__":
    train()