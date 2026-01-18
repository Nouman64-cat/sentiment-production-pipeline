import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
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
        print("Training Model A (Classical ML) with GridSearchCV...")
        
        # 1. Define the base pipeline (no hardcoded params yet)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])

        # 2. Define the parameter grid
        # Note: Use 'stepname__parameter' syntax
        param_grid = {
            'tfidf__max_features': [3000, 5000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigrams vs Bigrams
            'clf__C': [0.1, 1.0, 10.0]               # Regularization strength
        }

        # 3. Setup GridSearchCV
        # cv=3 is sufficient for a small dataset; n_jobs=-1 uses all cores
        search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        
        # 4. Fit the Search
        search.fit(X_train, y_train)
        
        # 5. Extract the best model
        best_model = search.best_estimator_
        
        # --- LOGGING ---
        # Evaluate using the best model
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Best Params: {search.best_params_}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Log Best Params to MLflow
        mlflow.log_params(search.best_params_)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log the BEST model (not just the last one)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save locally
        joblib.dump(best_model, MODEL_PATH)
        print(f"Best model saved locally to {MODEL_PATH}")

if __name__ == "__main__":
    train()