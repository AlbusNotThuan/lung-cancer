"""
Evaluator Module
Calculate metrics for binary and multiclass classification
"""
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score,
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)
import numpy as np


class Evaluator:
    """Evaluation metrics for binary and multiclass classification"""
    
    def __init__(self, thresholds=[0.5, 0.75, 0.9], task_type="binary", average="macro"):
        """
        Initialize evaluator
        
        Args:
            thresholds: List of thresholds for binary classification
            task_type: "binary" or "multiclass"
            average: Averaging strategy for multiclass (macro, micro, weighted)
        """
        self.thresholds = thresholds
        self.task_type = task_type
        self.average = average
    
    def calculate_metrics(self, y_true, y_pred_proba, threshold=0.5):
        """
        Calculate all metrics at given threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold (binary only)
        
        Returns:
            Dictionary of metrics
        """
        if self.task_type == "binary":
            # Binary classification
            y_pred = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            return {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "sensitivity": recall_score(y_true, y_pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "auc": roc_auc_score(y_true, y_pred_proba),
                "tp": int(tp), 
                "fp": int(fp), 
                "tn": int(tn), 
                "fn": int(fn)
            }
        else:
            # Multiclass classification
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            return {
                "threshold": None,  # Not applicable for multiclass
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average=self.average, zero_division=0),
                "recall": recall_score(y_true, y_pred, average=self.average, zero_division=0),
                "f1": f1_score(y_true, y_pred, average=self.average, zero_division=0),
                "auc": roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=self.average),
            }
    
    def evaluate_model(self, y_true, y_pred_proba, model_name, imbalance_method, train_time):
        """
        Evaluate at all thresholds (binary) or once (multiclass)
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            imbalance_method: Imbalance handling method used
            train_time: Training time in seconds
        
        Returns:
            List of metric dictionaries
        """
        results = []
        
        if self.task_type == "binary":
            for threshold in self.thresholds:
                metrics = self.calculate_metrics(y_true, y_pred_proba, threshold)
                metrics["model"] = model_name
                metrics["imbalance_method"] = imbalance_method
                metrics["train_time"] = train_time
                results.append(metrics)
        else:
            metrics = self.calculate_metrics(y_true, y_pred_proba)
            metrics["model"] = model_name
            metrics["imbalance_method"] = imbalance_method
            metrics["train_time"] = train_time
            results.append(metrics)
        
        return results
