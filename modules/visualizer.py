"""
Visualizer Module
Create plots and visualizations for model evaluation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import pandas as pd


class Visualizer:
    """Visualization tools for model evaluation"""
    
    @staticmethod
    def plot_roc_curves(results_dict, y_test, save_path=None):
        """
        Plot ROC curves for all models (binary classification)
        
        Args:
            results_dict: Dictionary {job_name: y_pred_proba}
            y_test: True labels
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        for job_name, y_pred_proba in results_dict.items():
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{job_name} (AUC={auc:.3f})", lw=2)
        
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - All Models", fontsize=14)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_confusion_matrices(results_dict, y_test, threshold=0.5, save_path=None):
        """
        Plot confusion matrices for all models
        
        Args:
            results_dict: Dictionary {job_name: y_pred_proba}
            y_test: True labels
            threshold: Classification threshold
            save_path: Optional path to save figure
        """
        n_models = len(results_dict)
        n_cols = min(4, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (job_name, y_pred_proba) in enumerate(results_dict.items()):
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
            axes[idx].set_title(f"{job_name}\nThreshold={threshold}", fontsize=10)
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("True")
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(metrics_df, save_path=None):
        """
        Plot comparison of key metrics across models
        
        Args:
            metrics_df: DataFrame with metrics
            save_path: Optional path to save figure
        """
        # Filter for threshold 0.5 (or first row for multiclass)
        if 'threshold' in metrics_df.columns:
            data = metrics_df[metrics_df['threshold'] == 0.5].copy()
        else:
            data = metrics_df.copy()
        
        if len(data) == 0:
            print("No data to plot")
            return
        
        # Create job name column for plotting
        data['job'] = data['model'] + '_' + data['imbalance_method']
        
        # Key metrics to plot
        key_metrics = ['accuracy', 'f1', 'auc']
        if 'sensitivity' in data.columns:
            key_metrics.extend(['sensitivity', 'specificity'])
        
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(5 * len(key_metrics), 5))
        
        if len(key_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(key_metrics):
            if metric in data.columns:
                data_sorted = data.sort_values(metric, ascending=False)
                axes[idx].barh(range(len(data_sorted)), data_sorted[metric])
                axes[idx].set_yticks(range(len(data_sorted)))
                axes[idx].set_yticklabels(data_sorted['job'], fontsize=8)
                axes[idx].set_xlabel(metric.upper(), fontsize=12)
                axes[idx].set_xlim([0, 1])
                axes[idx].grid(axis='x', alpha=0.3)
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_multiclass_roc(results_dict, y_test, save_path=None):
        """
        Plot ROC curves for multiclass classification
        
        Args:
            results_dict: Dictionary {job_name: y_pred_proba_matrix}
            y_test: True labels
            save_path: Optional path to save figure
        """
        n_classes = len(np.unique(y_test))
        
        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
        if n_classes == 1:
            axes = [axes]
        
        for class_idx in range(n_classes):
            for job_name, y_pred_proba in results_dict.items():
                # One-vs-Rest for each class
                y_true_binary = (y_test == class_idx).astype(int)
                y_score = y_pred_proba[:, class_idx]
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                auc = roc_auc_score(y_true_binary, y_score)
                
                axes[class_idx].plot(fpr, tpr, label=f"{job_name} (AUC={auc:.3f})", lw=2)
            
            axes[class_idx].plot([0, 1], [0, 1], "k--", lw=1)
            axes[class_idx].set_xlabel("False Positive Rate")
            axes[class_idx].set_ylabel("True Positive Rate")
            axes[class_idx].set_title(f"Class {class_idx} vs Rest")
            axes[class_idx].legend(loc="lower right", fontsize=8)
            axes[class_idx].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
