# Auto-Labeling ML Pipeline

A modular machine learning pipeline for supervised training with multiple models and imbalance handling methods. This repository provides an easy-to-use framework for training and evaluating classification models on custom datasets.

## 📋 Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Configuration Guide](#configuration-guide)
- [Running the Training Pipeline](#running-the-training-pipeline)
- [Supported Models and Methods](#supported-models-and-methods)
- [Output Structure](#output-structure)
- [Adding Custom Datasets](#adding-custom-datasets)

---

## ✨ Features

- **11 ML Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, Decision Tree, KNN, SVM, SGD, AdaBoost, Extra Trees
- **6 Imbalance Handling Methods**: None (baseline), SMOTE, ADASYN, SVMSMOTE, KMeans-SMOTE, Borderline-SMOTE
- **Parallel Training**: Configure multiple workers for efficient training
- **Binary & Multiclass Classification**: Automatic task detection
- **Comprehensive Evaluation**: Multiple metrics (Accuracy, F1, AUC, Precision, Recall)
- **Automatic Result Tracking**: Timestamped folders with metrics, plots, and logs

---

## 🚀 Getting Started

### 1. Setup in Google Drive

1. **Upload the repository** to your Google Drive in a folder structure like:
   ```
   MyDrive/
   └── projects/
       └── auto-labeling/
           ├── train.ipynb
           ├── config.yaml
           ├── setup.py
           ├── data/
           ├── datasets/
           ├── modules/
           └── results/
   ```

2. **Open Google Colab** and navigate to your uploaded `train.ipynb` notebook

3. **Mount Google Drive** by running the first cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/projects/auto-labeling
   ```

4. **Install the package** in development mode:
   ```python
   !pip install -e .
   ```

### 2. Prepare Your Data

Place your dataset files in the `data/` folder:

- **For Lung Cancer dataset** (binary classification):
  ```
  data/lung_cancer/
  ├── lung_train.csv
  └── lung_test.csv
  ```

- **For Call Quality dataset** (multiclass classification):
  ```
  data/call-quality/
  └── Filtered_Call_Records-cut (1).csv
  ```

---

## ⚙️ Configuration Guide

Before running training, configure your experiment in `config.yaml`. Here's what each section does:

### 1. Experiment Settings

```yaml
experiment:
  name: "baseline_all_models"
  description: "Testing all models with multiple imbalance handling methods"
```

**Purpose**: Name and describe your experiment for tracking purposes.

### 2. Dataset Selection

```yaml
dataset:
  name: "lung_cancer"  # Options: 'lung_cancer', 'call_quality'
```

**Available Datasets**:
- `lung_cancer`: Binary classification for 2-year survival prediction
- `call_quality`: Multiclass classification for call quality prediction

**Optional Parameters** (uncomment to customize):
```yaml
# For lung_cancer:
#   train_path: "custom/path/train.csv"
#   test_path: "custom/path/test.csv"

# For call_quality:
#   file_path: "custom/path/calls.csv"
#   test_size: 0.2
#   random_state: 42
```

### 3. Model Selection

**Choose which models to train**:

```yaml
models:
  active:
    - "logistic_regression"
    - "random_forest"
    - "gradient_boosting"
    - "xgboost"
    - "lightgbm"
    - "decision_tree"
    - "knn"
    - "svm"
    - "sgd"
    - "adaboost"
    - "extra_trees"
```

**Tip**: Start with 2-3 models for quick testing, then expand to all models for comprehensive comparison.

**Model-Specific Parameters** (optional):

```yaml
params:
  logistic_regression:
    max_iter: 2000
    C: 1.0
  random_forest:
    n_estimators: 5000
    max_depth: null
  # ... add parameters for other models
```

### 4. Imbalance Handling Methods

**Select which methods to test**:

```yaml
imbalance:
  methods:
    - "none"          # Baseline without imbalance handling
    - "smote"         # Synthetic Minority Over-sampling
    - "adasyn"        # Adaptive Synthetic Sampling
    - "svmsmote"      # SVM-based SMOTE
    - "kmeans_smote"  # KMeans clustering with SMOTE
    - "borderline_smote"  # Borderline cases SMOTE
```

**Recommended**: Always include `"none"` as a baseline for comparison.

### 5. Training Configuration

```yaml
training:
  n_workers: 4  # Number of parallel workers (adjust based on your resources)
```

**Tip**: 
- On Google Colab Free: Use 1-2 workers
- On Google Colab Pro: Use 4-8 workers

### 6. Evaluation Settings

```yaml
evaluation:
  thresholds: [0.5, 0.75, 0.9]  # For binary classification
  metrics: ["accuracy", "macro_f1", "weighted_f1", "f1", "auc", "precision", "recall"]
  
  multiclass:
    average: "macro"     # Options: macro, micro, weighted
    roc_strategy: "ovr"  # One-vs-Rest for multi-class ROC
```

### 7. Output Settings

```yaml
output:
  save_models: false  # Set to true to save trained model files
  save_plots: true    # Save visualization plots

random_seed: 42  # For reproducibility
```

---

## 🏃 Running the Training Pipeline

### Main Notebook: `train.ipynb`

This is your **primary notebook** for all supervised training needs.

**Step-by-Step Workflow**:

1. **Mount Drive and Install** (if in Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/projects/auto-labeling
   !pip install -e .
   ```

2. **Run all cells sequentially** or use "Runtime > Run all"

3. **The notebook will**:
   - Load your configuration from `config.yaml`
   - Load and preprocess your selected dataset
   - Generate training jobs (model × imbalance_method combinations)
   - Train all models in parallel
   - Evaluate and save results
   - Generate visualizations

4. **Monitor Progress**:
   - Training progress is displayed in real-time
   - Each job shows: `[model_name]_[imbalance_method]`

5. **Check Results**:
   - Results are automatically saved in `results/run_YYYYMMDD_HHMMSS/`

---

## 📊 Supported Models and Methods

### Models

| Model | Type | Best For |
|-------|------|----------|
| Logistic Regression | Linear | Fast baseline, interpretable |
| Random Forest | Ensemble | General purpose, robust |
| Gradient Boosting | Ensemble | High accuracy, feature importance |
| XGBoost | Ensemble | Competition-grade performance |
| LightGBM | Ensemble | Fast training, large datasets |
| Decision Tree | Tree | Interpretable, simple patterns |
| KNN | Instance-based | Local patterns |
| SVM | Kernel | Non-linear boundaries |
| SGD | Linear | Large-scale data |
| AdaBoost | Ensemble | Weak learners combination |
| Extra Trees | Ensemble | Fast, random splits |

### Imbalance Handling Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| None | No balancing | Balanced datasets |
| SMOTE | Synthetic minority oversampling | Most imbalanced datasets |
| ADASYN | Adaptive synthetic sampling | Focus on harder cases |
| SVMSMOTE | SVM-based SMOTE | Better decision boundaries |
| KMeans-SMOTE | Clustering + SMOTE | Complex distributions |
| Borderline-SMOTE | Focus on borderline cases | Challenging boundaries |

---

## 📁 Output Structure

After training, results are saved in a timestamped folder:

```
results/run_20260208_223601/
├── metrics.csv              # Detailed metrics for all model-method combinations
├── summary.json             # Quick summary with best results
├── training.log             # Raw training logs
└── plots/
    ├── model_comparison.png # Performance comparison across models
    └── [additional plots]
```

### Interpreting Results

**metrics.csv** contains:
- `model`: Model name
- `imbalance_method`: Balancing method used
- `accuracy`, `precision`, `recall`, `f1`, `auc`: Performance metrics
- `train_time`: Training duration in seconds

**summary.json** includes:
- Experiment configuration
- Dataset information
- Best performing model-method combination
- Complete config snapshot

---

## 🔧 Adding Custom Datasets

### 1. Create Dataset Loader

Create a new file in `datasets/` folder (e.g., `my_dataset.py`):

```python
"""
My Custom Dataset Loader
"""
import pandas as pd
import os

class MyDataset:
    def __init__(self, train_path=None, test_path=None):
        self.train_path = train_path or "data/my_dataset/train.csv"
        self.test_path = test_path or "data/my_dataset/test.csv"
        
        self.target_column = "target"  # Your target column name
        self.name = "my_dataset"
        self.description = "Description of your dataset"
        self.task_type = "binary"  # or "multiclass"
    
    def load(self):
        """Load train and test data"""
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
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
            "target": self.target_column,
            "task_type": self.task_type
        }
```

### 2. Register in DatasetFactory

Add your dataset to `datasets/__init__.py`:

```python
from .my_dataset import MyDataset

class DatasetFactory:
    @staticmethod
    def load_dataset(dataset_name, **kwargs):
        if dataset_name == "my_dataset":
            dataset = MyDataset(**kwargs)
        # ... existing datasets
```

### 3. Update config.yaml

```yaml
dataset:
  name: "my_dataset"
  # Optional: train_path, test_path
```


## 🤝 Support

For issues or questions:
1. Check that your `config.yaml` is properly formatted (YAML syntax)
2. Verify your data files are in the correct location
3. Ensure all dependencies are installed (`pip install -e .`)
4. Review the `training.log` in results folder for error details

---