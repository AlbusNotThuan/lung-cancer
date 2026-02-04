"""
Trainer Module
Handles model training with optional scaling
"""
import time
import pickle
import numpy as np


class Trainer:
    """Training orchestration with timing and model management"""
    
    def __init__(self, model, model_name, scaler=None):
        """
        Initialize trainer
        
        Args:
            model: ML model instance
            model_name: Name of the model
            scaler: Optional scaler (e.g., StandardScaler)
        """
        self.model = model
        self.model_name = model_name
        self.scaler = scaler
        self.train_time = None
    
    def train(self, X_train, y_train):
        """
        Train the model and track time
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        start_time = time.time()
        
        # Apply scaling if scaler provided
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        self.train_time = time.time() - start_time
    
    def predict(self, X_test):
        """
        Predict on test set
        
        Args:
            X_test: Test features
        
        Returns:
            Predictions
        """
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities
        
        Args:
            X_test: Test features
        
        Returns:
            Probability estimates (for binary: positive class probabilities)
        """
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        
        proba = self.model.predict_proba(X_test)
        
        # For binary classification, return probability of positive class
        if proba.shape[1] == 2:
            return proba[:, 1]
        
        # For multiclass, return full probability matrix
        return proba
    
    def save_model(self, filepath):
        """
        Save trained model and scaler
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_name': self.model_name
            }, f)
    
    @staticmethod
    def load_model(filepath):
        """
        Load trained model
        
        Args:
            filepath: Path to saved model
        
        Returns:
            Trainer instance with loaded model
        """
        with open(filepath, 'rb') as f:
            saved = pickle.load(f)
        
        trainer = Trainer(
            model=saved['model'],
            model_name=saved['model_name'],
            scaler=saved.get('scaler')
        )
        return trainer
