import os
import yaml
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing.clean import clean_text
from src.config import ML_CONFIG_PATH

# Load configuration from YAML
with open(ML_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def train():
    # 1. Setup MLflow
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # 2. Load & Clean
    if not os.path.exists("models"):
        os.makedirs("models")
        
    print("Loading data...")
    df = pd.read_csv(config["paths"]["data"])
    df['text'] = df['text'].apply(clean_text)
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['sentiment'], 
        test_size=config["data"]["test_size"], 
        random_state=config["data"]["random_state"]
    )

    # 4. Build parameter grid from config
    param_grid = {
        'tfidf__max_features': config["tfidf"]["max_features_options"],
        'tfidf__ngram_range': [tuple(r) for r in config["tfidf"]["ngram_range_options"]],
        'clf__C': config["logistic_regression"]["C_options"]
    }

    # 5. Start Run
    with mlflow.start_run():
        print("Training Model A (Classical ML) with GridSearchCV...")
        
        # Define the base pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])

        # Setup GridSearchCV with config parameters
        search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=config["grid_search"]["cv_folds"], 
            scoring=config["grid_search"]["scoring"], 
            n_jobs=config["grid_search"]["n_jobs"]
        )
        
        # Fit the Search
        search.fit(X_train, y_train)
        
        # Extract the best model
        best_model = search.best_estimator_
        
        # --- LOGGING ---
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Best Params: {search.best_params_}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Log to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save locally
        model_path = config["paths"]["model"]
        joblib.dump(best_model, model_path)
        print(f"Best model saved locally to {model_path}")


if __name__ == "__main__":
    train()