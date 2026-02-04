# Modular Training Pipeline - Implementation Plan (Simplified)

## 🎯 Objective
Create a simple, modular training pipeline for Google Colab that allows:
- Easy dataset swapping
- Multiple model selection
- Configurable imbalance handling techniques
- Flexible visualization options
- Reproducible experiments with automatic result tracking

---

## 📁 Project Structure

```
lung/
├── config.yaml                       # Single configuration file
│
├── datasets/                         # Dataset-specific loaders
│   ├── __init__.py
│   ├── lung_cancer.py                # Lung cancer dataset loader
│   └── [other_dataset].py            # Add more dataset loaders as needed
│
├── modules/
│   ├── __init__.py
│   ├── imbalance_handler.py          # Imbalance handling methods
│   ├── models.py                     # Model factory
│   ├── trainer.py                    # Training orchestration
│   ├── evaluator.py                  # Metrics calculation
│   └── visualizer.py                 # Plotting functions
│
├── data/                             # Raw data files only
│   └── lung_cancer/
│       ├── lung_train.csv
│       └── lung_test.csv
│
├── results/                          # Auto-generated run folders
│   ├── run_20260203_143052/          # Timestamp-based folders
│   │   ├── summary.json              # Quick summary (params + results)
│   │   ├── metrics.csv               # Detailed metrics per model
│   │   ├── training.log              # Raw training logs
│   │   ├── models/                   # Model files (optional, default: false)
│   │   └── plots/                    # Visualizations
│   └── run_20260203_150122/
│       └── ...
│
└── train.ipynb                       # Main training notebook
```

---

## 🧩 Module Breakdown

### 1. **Single Configuration File** (`config.yaml`)

**Purpose**: Centralize all parameters in one simple file

```yaml
# Experiment Info
experiment:
  name: "baseline_all_models"
  description: "Testing all models without imbalance handling"

# Dataset Selection
dataset:
  name: "lung_cancer"  # Corresponds to datasets/lung_cancer.py
  # Optional: override default paths
  # train_path: "custom/path/train.csv"
  # test_path: "custom/path/test.csv"

# Model Selection
models:
  active:
    - "logistic_regression"
    - "random_forest"
    - "gradient_boosting"
    - "xgboost"
  
  # Model-specific parameters (optional overrides)
  params:
    logistic_regression:
      max_iter: 2000
      C: 1.0
    random_forest:
      n_estimators: 500
      max_depth: null
    xgboost:
      n_estimators: 500
      learning_rate: 0.01

# Imbalance Handling Methods to Test
imbalance:
  methods:
    - "none"  # Baseline
    - "smote"
    # - "adasyn"
    # - "svmsmote"
    # - "kmeans_smote"
    # - "borderline_smote"
  
  params:
    k_neighbors: 5
    sampling_strategy: "auto"

# Training Configuration
training:
  n_workers: 4  # Number of parallel workers (models trained simultaneously)
  # Will train all combinations of models x imbalance_methods in parallel

# Evaluation Settings
evaluation:
  thresholds: [0.5, 0.75, 0.9]  # For binary classification
  metrics: ["accuracy", "sensitivity", "specificity", "f1", "auc", "precision", "recall"]
  
  # Multi-class settings
  multiclass:
    average: "macro"  # Options: macro, micro, weighted
    roc_strategy: "ovr"  # One-vs-Rest for multi-class ROC

# Output Settings
output:
  save_models: false  # Set to true to save model files
  save_plots: true

# Global Settings
random_seed: 42
```

---

### 2. **Dataset Loaders** (`datasets/`)

**Purpose**: Each dataset has its own loader module with built-in paths and metadata

#### `datasets/lung_cancer.py`
```python
import pandas as pd
import os

class LungCancerDataset:
    """Lung cancer survival prediction dataset"""
    
    def __init__(self, train_path=None, test_path=None):
        # Default paths
        self.default_train = "data/lung_cancer/lung_train.csv"
        self.default_test = "data/lung_cancer/lung_test.csv"
        
        # Allow custom paths
        self.train_path = train_path or self.default_train
        self.test_path = test_path or self.default_test
        
        # Dataset metadata
        self.target_column = "dead"
        self.name = "lung_cancer"
        self.description = "Lung cancer survival prediction (Hospitals w+t vs s)"
        self.task_type = "binary"  # Options: binary, multiclass
    
    def load(self):
        """Load train and test data"""
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # Separate features and target
        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]
        
        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]
        
        return X_train, y_train, X_test, y_test
    
    def get_info(self):
        """Return dataset information"""
        return {
            "name": self.name,
            "description": self.description,
            "train_path": self.train_path,
            "test_path": self.test_path,
            "target": self.target_column
        }
```

**Adding new datasets**: Simply create a new file like `datasets/custom_dataset.py` with the same interface.

---

### 3. **Imbalance Handler** (`modules/imbalance_handler.py`)

**Purpose**: Apply imbalance handling techniques (data is already preprocessed)

**Supported methods**:
- `none` - No oversampling (baseline)
- `smote` - Synthetic Minority Over-sampling Technique
- `adasyn` - Adaptive Synthetic Sampling
- `svmsmote` - SVM-SMOTE
- `borderline_smote` - Borderline-SMOTE (1 and 2)
- `kmeans_smote` - KMeans-SMOTE

```python
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE

class ImbalanceHandler:
    def __init__(self, method="none", **kwargs):
        self.method = method
        self.kwargs = kwargs
    
    def apply(self, X_train, y_train, task_type="binary"):
        """Apply selected imbalance handling method"""
        if self.method == "none":
            return X_train, y_train
        
        # For multi-class, set sampling_strategy='auto' or 'not majority'
        if task_type == "multiclass" and 'sampling_strategy' in self.kwargs:
            self.kwargs['sampling_strategy'] = 'not majority'
        
        try:
            if self.method == "smote":
                sampler = SMOTE(**self.kwargs)
            
            elif self.method == "adasyn":
                sampler = ADASYN(**self.kwargs)
            
            elif self.method == "svmsmote":
                sampler = SVMSMOTE(**self.kwargs)
            
            elif self.method == "borderline_smote":
                sampler = BorderlineSMOTE(**self.kwargs)
            
            elif self.method == "kmeans_smote":
                sampler = KMeansSMOTE(**self.kwargs)
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            return sampler.fit_resample(X_train, y_train)
        
        except Exception as e:
            print(f"Warning: {self.method} failed ({str(e)}). Using original data.")
            return X_train, y_train
```

---

### 4. **Model Factory** (`modules/models.py`)

**Purpose**: Create and configure models from config

**Supported Models**:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Decision Tree
- KNN (K-Nearest Neighbors)
- SVM (Support Vector Machine)
- SGD (Stochastic Gradient Descent)
- AdaBoost
- Extra Trees

```python
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

class ModelFactory:
    # Default parameters for each model
    DEFAULT_PARAMS = {
        "logistic_regression": {"max_iter": 2000, "random_state": 42},
        "random_forest": {"n_estimators": 500, "random_state": 42},
        "gradient_boosting": {"n_estimators": 500, "random_state": 42},
        "xgboost": {"n_estimators": 500, "learning_rate": 0.01, "random_state": 42},
        "lightgbm": {"n_estimators": 500, "learning_rate": 0.01, "random_state": 42},
        "decision_tree": {"random_state": 42},
        "knn": {"n_neighbors": 5},
        "svm": {"kernel": "rbf", "random_state": 42},
        "sgd": {"loss": "log_loss", "random_state": 42},
        "adaboost": {"n_estimators": 500, "random_state": 42},
        "extra_trees": {"n_estimators": 500, "random_state": 42},
    }
    
    # Models that require feature scaling
    REQUIRES_SCALING = ["logistic_regression", "knn", "svm", "sgd"]
    
    @staticmethod
    def create_model(model_name, params=None):
        """Create model instance"""
        params = params or ModelFactory.DEFAULT_PARAMS.get(model_name, {})
        
        models_map = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "xgboost": XGBClassifier,
            "lightgbm": LGBMClassifier,
            "decision_tree": DecisionTreeClassifier,
            "knn": KNeighborsClassifier,
            "svm": SVC,
            "sgd": SGDClassifier,
            "adaboost": AdaBoostClassifier,
            "extra_trees": ExtraTreesClassifier,
        }
        
        if model_name not in models_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Special handling for SVM probability
        if model_name == "svm":
            params["probability"] = True
        
        return models_map[model_name](**params)
    
    @staticmethod
    def requires_scaling(model_name):
        """Check if model needs scaling"""
        return model_name in ModelFactory.REQUIRES_SCALING
```

---

### 5. **Trainer** (`modules/trainer.py`)

**Purpose**: Training orchestration with automatic result tracking

```python
class Trainer:
    def __init__(self, model, model_name, scaler=None):
        self.model = model
        self.model_name = model_name
        self.scaler = scaler
        self.train_time = None
    
    def train(self, X_train, y_train):
        """Train the model and track time"""
        import time
        start_time = time.time()
        
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        """Predict on test set"""
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Predict probabilities"""
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test)[:, 1]
    
    def save_model(self, filepath):
        """Save trained model"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
```

---

### 6. **Evaluator** (`modules/evaluator.py`)

**Purpose**: Calculate metrics and generate reports

```python
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, confusion_matrix)
import numpy as np

class Evaluator:
    def __init__(self, thresholds=[0.5, 0.75, 0.9], task_type="binary", average="macro"):
        self.thresholds = thresholds
        self.task_type = task_type
        self.average = average  # For multi-class: macro, micro, weighted
    
    def calculate_metrics(self, y_true, y_pred_proba, threshold=0.5):
        """Calculate all metrics at given threshold"""
        if self.task_type == "binary":
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
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            }
        else:  # multiclass
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
        """Evaluate at all thresholds (binary) or once (multiclass)"""
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
```

---

### 7. **Visualizer** (`modules/visualizer.py`)

**Purpose**: Create plots and visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

class Visualizer:
    @staticmethod
    def plot_roc_curves(results_dict, y_test, save_path=None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred_proba in results_dict.items():
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", lw=2)
        
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - All Models")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrices(results_dict, y_test, threshold=0.5, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = len(results_dict)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 10))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, y_pred_proba) in enumerate(results_dict.items()):
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f"{model_name}\nThreshold={threshold}")
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("True")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_df, save_path=None):
        """Plot comparison of key metrics across models"""
        key_metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
        
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(20, 4))
        
        for idx, metric in enumerate(key_metrics):
            data = metrics_df[metrics_df['threshold'] == 0.5]
            data.plot(x='model', y=metric, kind='bar', ax=axes[idx], legend=False)
            axes[idx].set_title(metric.upper())
            axes[idx].set_xlabel('')
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

---

### 8. **Result Manager** (Built into `train.ipynb`)

**Purpose**: Automatically save results for each run

```python
import json
import os
from datetime import datetime

def create_run_folder():
    """Create timestamped run folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"results/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(f"{run_folder}/plots", exist_ok=True)
    if config['output']['save_models']:
        os.makedirs(f"{run_folder}/models", exist_ok=True)
    return run_folder

def save_summary(run_folder, config, metrics_df, dataset_info):
    """Save quick summary JSON"""
    summary = {
        "experiment": config['experiment'],
        "dataset": dataset_info,
        "models": config['models']['active'],
        "imbalance_method": config['imbalance']['method'],
        "timestamp": datetime.now().isoformat(),
        "best_model": {
            "name": metrics_df.loc[metrics_df['auc'].idxmax(), 'model'],
            "auc": float(metrics_df['auc'].max()),
            "accuracy": float(metrics_df.loc[metrics_df['auc'].idxmax(), 'accuracy'])
        },
        "config": config
    }
    
    with open(f"{run_folder}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def save_metrics(run_folder, metrics_df):
    """Save detailed metrics CSV"""
    metrics_df.to_csv(f"{run_folder}/metrics.csv", index=False)

def save_training_log(run_folder, log_messages):
    """Save raw training logs"""
    with open(f"{run_folder}/training.log", 'w') as f:
        f.write('\n'.join(log_messages))
```

---

## 🔄 Workflow in Main Notebook

```python
# train.ipynb - Complete workflow with parallel training

# 1. Setup
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datasets import *
from modules import *
from joblib import Parallel, delayed
import numpy as np

# 2. Load Configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 3. Create Run Folder
run_folder = create_run_folder()
log = []

# 4. Load Dataset
dataset_class = eval(f"{config['dataset']['name'].title().replace('_', '')}Dataset")
dataset = dataset_class(
    train_path=config['dataset'].get('train_path'),
    test_path=config['dataset'].get('test_path')
)
X_train, y_train, X_test, y_test = dataset.load()
log.append(f"Loaded dataset: {dataset.name}")
log.append(f"Task type: {dataset.task_type}")
log.append(f"Classes: {np.unique(y_train)}")

# 5. Create Training Jobs (all combinations of models x imbalance methods)
def train_single_job(model_name, imbalance_method, X_train, y_train, X_test, y_test, config, dataset):
    """Train a single model with given imbalance method"""
    job_log = []
    job_name = f"{model_name}_{imbalance_method}"
    
    # Apply imbalance handling
    imbalance_handler = ImbalanceHandler(
        method=imbalance_method,
        **config['imbalance']['params']
    )
    X_train_balanced, y_train_balanced = imbalance_handler.apply(
        X_train.copy(), y_train.copy(), task_type=dataset.task_type
    )
    job_log.append(f"Samples: {len(X_train)} -> {len(X_train_balanced)}")
    
    # Create model
    params = config['models'].get('params', {}).get(model_name, {})
    model = ModelFactory.create_model(model_name, params)
    
    # Add scaler if needed
    scaler = StandardScaler() if ModelFactory.requires_scaling(model_name) else None
    
    # Train
    trainer = Trainer(model, model_name, scaler)
    trainer.train(X_train_balanced, y_train_balanced)
    job_log.append(f"Training time: {trainer.train_time:.2f}s")
    
    # Predict
    if dataset.task_type == "binary":
        y_pred_proba = trainer.predict_proba(X_test)
    else:
        y_pred_proba = trainer.model.predict_proba(X_test)  # Full probability matrix
    
    # Evaluate
    eval_config = config['evaluation']
    evaluator = Evaluator(
        thresholds=eval_config['thresholds'],
        task_type=dataset.task_type,
        average=eval_config.get('multiclass', {}).get('average', 'macro')
    )
    metrics = evaluator.evaluate_model(y_test, y_pred_proba, model_name, imbalance_method, trainer.train_time)
    
    return {
        'job_name': job_name,
        'model_name': model_name,
        'imbalance_method': imbalance_method,
        'metrics': metrics,
        'y_pred_proba': y_pred_proba,
        'trainer': trainer,
        'log': job_log
    }

# 6. Generate all training jobs
training_jobs = []
for model_name in config['models']['active']:
    for imbalance_method in config['imbalance']['methods']:
        training_jobs.append((model_name, imbalance_method))

log.append(f"\nTotal training jobs: {len(training_jobs)}")
log.append(f"Parallel workers: {config['training']['n_workers']}")
log.append(f"\nStarting parallel training...")

# 7. Train all jobs in parallel
results = Parallel(n_jobs=config['training']['n_workers'], verbose=10)(
    delayed(train_single_job)(
        model_name, imbalance_method, 
        X_train, y_train, X_test, y_test, 
        config, dataset
    )
    for model_name, imbalance_method in training_jobs
)

# 8. Collect results
results_dict = {}  # For visualization: {job_name: y_pred_proba}
all_metrics = []

for result in results:
    job_name = result['job_name']
    results_dict[job_name] = result['y_pred_proba']
    all_metrics.extend(result['metrics'])
    
    log.append(f"\n{job_name}:")
    for msg in result['log']:
        log.append(f"  {msg}")
    
    # Save model if configured
    if config['output']['save_models']:
        result['trainer'].save_model(f"{run_folder}/models/{job_name}.pkl")

# 9. Create metrics DataFrame
metrics_df = pd.DataFrame(all_metrics)

# 10. Visualize
visualizer = Visualizer()
if config['output']['save_plots']:
    if dataset.task_type == "binary":
        visualizer.plot_roc_curves(results_dict, y_test, f"{run_folder}/plots/roc_curves.png")
        visualizer.plot_confusion_matrices(results_dict, y_test, 0.5, f"{run_folder}/plots/confusion_matrices.png")
    else:
        # Multi-class visualizations
        visualizer.plot_multiclass_roc(results_dict, y_test, f"{run_folder}/plots/roc_curves_multiclass.png")
    
    visualizer.plot_metrics_comparison(metrics_df, f"{run_folder}/plots/metrics_comparison.png")

# 11. Save Results
save_summary(run_folder, config, metrics_df, dataset.get_info())
save_metrics(run_folder, metrics_df)
save_training_log(run_folder, log)

print(f"\n✅ Run complete! Results saved to: {run_folder}")
print(f"\nBest model: {metrics_df.loc[metrics_df['auc'].idxmax(), 'model']}")
print(f"Best AUC: {metrics_df['auc'].max():.3f}")
```

---

## 🎨 Usage Examples

### Example 1: Switch Dataset
```yaml
# In config.yaml:
dataset:
  name: "custom_dataset"  # Uses datasets/custom_dataset.py
```

### Example 2: Run Specific Models
```yaml
# In config.yaml:
models:
  active:
    - "xgboost"
    - "lightgbm"
    - "random_forest"
```

### Example 3: Apply SMOTE
```yaml
# In config.yaml:
imbalance:
  method: "smote"
  params:
    k_neighbors: 5
    random_state: 42
```

### Example 4: Save Model Files
```yaml
# In config.yaml:
output:
  save_models: true  # Models will be saved in results/run_*/models/
```

---

## 📊 Benefits of This Structure

✅ **Simple**: Single config file, clear structure  
✅ **Modular**: Easy to add new datasets and models  
✅ **Traceable**: Every run saved in separate folder with full details  
✅ **Reproducible**: summary.json contains all parameters  
✅ **Flexible**: Easy to swap datasets/models/methods  
✅ **Google Colab Ready**: Works seamlessly in Colab environment  

---

## 🚀 Implementation Phases

### Phase 1: Core Setup (Day 1)
- [ ] Create directory structure
- [ ] Create config.yaml template
- [ ] Implement ModelFactory with all 11 models
- [ ] Test model creation

### Phase 2: Data & Imbalance (Day 2)
- [ ] Create datasets/lung_cancer.py
- [ ] Implement ImbalanceHandler with 6 methods
- [ ] Test data loading and imbalance handling

### Phase 3: Training & Evaluation (Day 3)
- [ ] Implement Trainer class
- [ ] Implement Evaluator class
- [ ] Implement result saving functions
- [ ] Test end-to-end training

### Phase 4: Visualization & Polish (Day 4)
- [ ] Implement Visualizer class
- [ ] Create train.ipynb with complete workflow
- [ ] Test full pipeline
- [ ] Documentation

---

## ⚠️ Potential Issues & Solutions

### Issue 1: SVM Training Time
**Problem**: SVM can be very slow on large datasets  
**Solution**: 
- Add timeout parameter in config
- Consider using SGD as faster alternative
- Document expected training times

### Issue 2: Model File Size
**Problem**: Saving all models can consume significant disk space  
**Solution**: 
- Default `save_models: false`
- Only save best model option
- Compress model files

### Issue 3: Parallel Training Resource Management
**Problem**: Too many parallel workers may cause issues  
**Solution**: 
- Configurable n_workers (default: 4)
- joblib handles resource management automatically
- User can adjust based on available resources

### Issue 4: Different Model Interfaces
**Problem**: Not all models have predict_proba (e.g., SVM without probability=True)  
**Solution**: 
- Always set probability=True for SVM (already in ModelFactory)
- Add error handling for models without predict_proba
- Fallback to decision_function if needed

### Issue 5: Imbalance Methods Compatibility
**Problem**: Some imbalance methods may fail with certain data distributions  
**Solution**: 
- Add try-except wrapper in ImbalanceHandler
- Log warning and skip if method fails
- Continue with original data

---

## 📝 Notes

### Data Assumptions
- Data is already preprocessed (encoded, imputed, cleaned)
- Features are numeric
- Target can be binary (0/1) or multi-class (0, 1, 2, ...)
- No missing values

### Model Notes
- **Logistic Regression, KNN, SVM, SGD**: Require scaling (handled automatically)
- **XGBoost, LightGBM**: Handle missing values internally
- **SVM**: Probability enabled by default for ROC curves
- **All models**: random_state=42 for reproducibility

### Imbalance Methods
- **None**: Baseline (no resampling)
- **SMOTE**: Standard synthetic oversampling
- **ADASYN**: Adaptive synthetic sampling (focuses on harder examples)
- **SVMSMOTE**: Uses SVM to generate samples
- **BorderlineSMOTE**: Only oversamples borderline examples
- **KMeansSMOTE**: Uses clustering before SMOTE

---

## 🔧 Next Steps

1. **Review this simplified plan** ✓
2. **Approve structure and address any concerns**
3. **Begin Phase 1 implementation**

---

## 🎯 Quick Summary

**What's Included**:
- ✅ Single config.yaml file
- ✅ 11 ML models (sklearn, xgboost, lightgbm)
- ✅ 6 imbalance handling methods
- ✅ Automatic result tracking (timestamped folders)
- ✅ Dataset modules for easy swapping
- ✅ Comprehensive metrics and visualizations

**What's Excluded** (Keeping it simple):
- ❌ Preprocessing module (data already processed)
- ❌ Hyperparameter tuning
- ❌ Experiment tracking (MLflow/W&B)
- ❌ Automated reporting (PDF/HTML)
- ❌ Cross-validation (can add later if needed)

**Result Folder Contents**:
```
results/run_20260203_143052/
├── summary.json              # Quick overview (params + best results)
├── metrics.csv               # All metrics for all models/thresholds
├── training.log              # Raw logs from training process
├── plots/                    # All visualizations
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── metrics_comparison.png
└── models/                   # Optional (default: false)
    ├── xgboost.pkl
    ├── random_forest.pkl
    └── ...
```
