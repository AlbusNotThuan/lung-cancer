"""
Dataset loaders for the training pipeline.
Each dataset should have its own class following the template.
"""

from .lung_cancer import LungCancerDataset

__all__ = ['LungCancerDataset']
