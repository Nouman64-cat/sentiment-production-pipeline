import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Dict, Any, List, Union

def calculate_metrics(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> Dict[str, float]:
    """
    Calculate standard binary classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing accuracy and f1_score
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average='binary')) # ‘binary’ for binary classification
    }

def get_classification_report(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> str:
    """
    Wrapper for sklearn classification_report.
    """
    return classification_report(y_true, y_pred)

def plot_confusion_matrix(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], output_path: str = None):
    """
    Generate and optionally save a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return None
    
    return fig
