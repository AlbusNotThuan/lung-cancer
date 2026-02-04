"""
Imbalance Handling Module
Supports multiple oversampling techniques for class imbalance
"""
from imblearn.over_sampling import (
    SMOTE, 
    ADASYN, 
    SVMSMOTE, 
    BorderlineSMOTE, 
    KMeansSMOTE
)


class ImbalanceHandler:
    """Handle class imbalance with various oversampling methods"""
    
    def __init__(self, method="none", **kwargs):
        """
        Initialize imbalance handler
        
        Args:
            method: Oversampling method (none, smote, adasyn, svmsmote, 
                    borderline_smote, kmeans_smote)
            **kwargs: Additional parameters for the sampling method
        """
        self.method = method.lower()
        self.kwargs = kwargs
    
    def apply(self, X_train, y_train, task_type="binary"):
        """
        Apply selected imbalance handling method
        
        Args:
            X_train: Training features
            y_train: Training labels
            task_type: "binary" or "multiclass"
        
        Returns:
            X_resampled, y_resampled: Resampled data
        """
        if self.method == "none":
            return X_train, y_train
        
        # For multi-class, adjust sampling_strategy
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
            
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        
        except Exception as e:
            print(f"⚠️ Warning: {self.method} failed ({str(e)}). Using original data.")
            return X_train, y_train
