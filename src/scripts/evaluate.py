import os
import argparse
import yaml
import joblib
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from src.preprocessing.clean import clean_text
from src.evaluation.metrics import calculate_metrics, get_classification_report, plot_confusion_matrix
from src.config import ML_CONFIG_PATH, DL_CONFIG_PATH

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """Simple Dataset for DL Inference"""
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------------------------------------------------------
# EVALUATION FUNCTIONS
# -----------------------------------------------------------------------------

def evaluate_ml():
    """Evaluate the Classical ML model"""
    print("Evaluating Classical ML Model...")
    config = load_config(ML_CONFIG_PATH)
    
    # 1. Load Data
    df = pd.read_csv(config["paths"]["data"])
    df['text'] = df['text'].apply(clean_text)
    
    # Use split to simulate "Test Set" (ensure random_state matches training!)
    _, X_test, _, y_test = train_test_split(
        df['text'], 
        df['sentiment'], 
        test_size=config["data"]["test_size"], 
        random_state=config["data"]["random_state"]
    )
    
    # 2. Load Model
    model_path = config["paths"]["model"]
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run training first.")
        return

    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # 3. Predict
    y_pred = model.predict(X_test)
    
    # 4. Report
    metrics = calculate_metrics(y_test, y_pred)
    print(f"\nResults:\nAccuracy: {metrics['accuracy']:.4f}\nF1 Score: {metrics['f1_score']:.4f}")
    print("\nDetailed Report:\n", get_classification_report(y_test, y_pred))
    
    # 5. Confusion Matrix
    cm_path = "models/ml_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, output_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")


def evaluate_dl():
    """Evaluate the Deep Learning model"""
    print("Evaluating Deep Learning Model...")
    config = load_config(DL_CONFIG_PATH)
    
    # 1. Load Data
    df = pd.read_csv(config["paths"]["data"])
    df['text'] = df['text'].apply(clean_text)
    
    # Split to get proper test set (must match training split logic)
    # Train/Temp Split
    _, df_temp = train_test_split(
        df, 
        test_size=config["data"]["test_size"], 
        random_state=config["data"]["random_state"]
    )
    # Val/Test Split
    _, df_test = train_test_split(
        df_temp, 
        test_size=config["data"]["val_test_split"], 
        random_state=config["data"]["random_state"]
    )
    
    texts = df_test['text'].to_numpy()
    labels = df_test['sentiment'].to_numpy()
    
    # 2. Load Model Components
    model_path = config["paths"]["model"]
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run training first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(config["model"]["name"])
    model = DistilBertForSequenceClassification.from_pretrained(
        config["model"]["name"], 
        num_labels=config["model"]["num_labels"]
    )
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Predict
    dataset = SentimentDataset(texts, tokenizer, max_len=config["training"]["max_len"])
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], num_workers=0)
    
    preds = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)
            preds.extend(predicted.cpu().tolist())
            
    # 4. Report
    metrics = calculate_metrics(labels, preds)
    print(f"\nResults:\nAccuracy: {metrics['accuracy']:.4f}\nF1 Score: {metrics['f1_score']:.4f}")
    print("\nDetailed Report:\n", get_classification_report(labels, preds))

    # 5. Confusion Matrix
    cm_path = "models/dl_confusion_matrix.png"
    plot_confusion_matrix(labels, preds, output_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test set")
    parser.add_argument("--model", type=str, choices=["ml", "dl"], required=True, help="Model to evaluate: 'ml' or 'dl'")
    args = parser.parse_args()
    
    if args.model == "ml":
        evaluate_ml()
    elif args.model == "dl":
        evaluate_dl()
