"""
Dataset loaders for the training pipeline.
Each dataset should have its own class following the template.
"""

from .lung_cancer import LungCancerDataset

__all__ = ['LungCancerDataset','DatasetFactory']


class DatasetFactory:
    """Factory class to create dataset instances based on name"""
    
    _datasets = {
        'lung_cancer': LungCancerDataset,
    }
    
    @staticmethod
    def create(dataset_name, **kwargs):
        """
        Create and return a dataset instance
        
        Args:
            dataset_name: Name of the dataset ('lung_cancer' or 'call_quality')
            **kwargs: Additional parameters to pass to the dataset constructor
        
        Returns:
            Dataset instance
        
        Raises:
            ValueError: If dataset_name is not recognized
        """
        if dataset_name not in DatasetFactory._datasets:
            available = ', '.join(DatasetFactory._datasets.keys())
            raise ValueError(f"Unknown dataset: '{dataset_name}'. Available: {available}")
        
        dataset_class = DatasetFactory._datasets[dataset_name]
        return dataset_class(**kwargs)
    
    @staticmethod
    def get_available_datasets():
        """Return list of available dataset names"""
        return list(DatasetFactory._datasets.keys())
    
    @staticmethod
    def load_dataset(dataset_name, **kwargs):
        """
        Convenience method to create dataset and load data in one call
        
        Args:
            dataset_name: Name of the dataset
            **kwargs: Additional parameters for dataset constructor
        
        Returns:
            tuple: (dataset_instance, X_train, y_train, X_test, y_test, dataset_info)
        """
        dataset = DatasetFactory.create(dataset_name, **kwargs)
        X_train, y_train, X_test, y_test = dataset.load()
        dataset_info = dataset.get_info()
        return dataset, X_train, y_train, X_test, y_test, dataset_info
