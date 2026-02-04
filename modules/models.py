"""
Model Factory Module
Creates and configures all supported ML models
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier, 
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelFactory:
    """Factory for creating ML models"""
    
    # Default parameters for each model
    DEFAULT_PARAMS = {
        "logistic_regression": {
            "max_iter": 2000, 
            "random_state": 42
        },
        "random_forest": {
            "n_estimators": 500, 
            "random_state": 42
        },
        "gradient_boosting": {
            "n_estimators": 500, 
            "learning_rate": 0.01,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 500, 
            "learning_rate": 0.01,
            "max_depth": 10,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        },
        "lightgbm": {
            "n_estimators": 500, 
            "learning_rate": 0.01,
            "random_state": 42,
            "verbose": -1
        },
        "decision_tree": {
            "random_state": 42
        },
        "knn": {
            "n_neighbors": 5
        },
        "svm": {
            "kernel": "rbf",
            "C": 1.0,
            "random_state": 42,
            "probability": True  # Enable probability estimates for ROC
        },
        "sgd": {
            "loss": "log_loss",
            "max_iter": 2000,
            "random_state": 42
        },
        "adaboost": {
            "n_estimators": 500,
            "random_state": 42
        },
        "extra_trees": {
            "n_estimators": 500,
            "random_state": 42
        },
    }
    
    # Models that require feature scaling
    REQUIRES_SCALING = ["logistic_regression", "knn", "svm", "sgd"]
    
    @staticmethod
    def create_model(model_name, params=None):
        """
        Create model instance
        
        Args:
            model_name: Name of the model
            params: Optional parameter overrides
        
        Returns:
            Model instance
        """
        # Merge default params with custom params
        model_params = ModelFactory.DEFAULT_PARAMS.get(model_name, {}).copy()
        if params:
            model_params.update(params)
        
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
        
        return models_map[model_name](**model_params)
    
    @staticmethod
    def requires_scaling(model_name):
        """Check if model requires feature scaling"""
        return model_name in ModelFactory.REQUIRES_SCALING
    
    @staticmethod
    def get_available_models():
        """Return list of available model names"""
        return list(ModelFactory.DEFAULT_PARAMS.keys())
