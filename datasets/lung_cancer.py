"""
Lung Cancer Survival Prediction Dataset Loader
"""
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
        self.description = "Lung cancer 2-year survival prediction from index_age"
        self.task_type = "binary"  # Options: binary, multiclass
    
    def load(self):
        """Load train and test data"""
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training data not found: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test data not found: {self.test_path}")
        
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # Check if target column exists
        if self.target_column not in train_df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in training data")
        
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
            "target": self.target_column,
            "task_type": self.task_type
        }
