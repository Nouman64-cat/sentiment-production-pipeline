import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.preprocessing.clean import clean_text

# Paths
DATA_PATH = os.path.join("dataset", "data.csv")
MODEL_PATH = os.path.join("models", "ml_model.joblib")

def train():
    # 1. Setup MLflow
    mlflow.set_experiment("Sentiment_Analysis_ML")
    
    # 2. Load & Clean
    if not os.path.exists("models"):
        os.makedirs("models")
        
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].apply(clean_text)
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42
    )

    # 4. Define Hyperparameters
    MAX_FEATURES = 5000
    
    # 5. Start Run
    with mlflow.start_run():
        print("Training Model A (Classical ML)...")
        
        # Build Pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=MAX_FEATURES)),
            ('clf', LogisticRegression())
        ])

        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # --- LOGGING TO MLFLOW ---
        # Log Params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", MAX_FEATURES)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log Model (This saves the pickle file for us automatically)
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Also save locally for API usage later
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Model saved locally to {MODEL_PATH}")

if __name__ == "__main__":
    train()