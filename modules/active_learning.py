"""
Active Learning Module
Implements pseudo-labeling with confidence-based sample selection
"""
from sklearn.metrics import log_loss
from scipy.stats import entropy
import numpy as np
import pandas as pd


def compute_dynamic_threshold(epoch, total_epochs, tau_min=0.7, tau_max=0.95):
    """
    Compute progressively increasing confidence threshold
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        tau_min: Minimum threshold (starting point)
        tau_max: Maximum threshold (ending point)
    
    Returns:
        threshold: Confidence threshold for current epoch
    """
    if total_epochs <= 1:
        return tau_max
    return tau_min + (tau_max - tau_min) * (epoch - 1) / (total_epochs - 1)


def active_learning_cycle(model, X_labeled, y_labeled,
                          X_unlabeled, y_unlabeled,
                          X_val, y_val,
                          epoch, total_epochs,
                          use_dynamic_threshold=False,
                          confidence_threshold=0.8,
                          tau_min=0.7, tau_max=0.95,
                          verbose=True):
    """
    Perform one cycle of active learning with pseudo-labeling
    
    Args:
        model: Trained model with predict_proba method
        X_labeled: Current labeled features
        y_labeled: Current labeled targets
        X_unlabeled: Unlabeled pool features
        y_unlabeled: True labels (for monitoring only, not used in training)
        X_val: Validation features
        y_val: Validation targets
        epoch: Current epoch number
        total_epochs: Total number of epochs
        use_dynamic_threshold: Whether to use progressive threshold
        confidence_threshold: Static threshold (used if use_dynamic_threshold=False)
        tau_min: Minimum threshold for dynamic mode
        tau_max: Maximum threshold for dynamic mode
        verbose: Print progress messages
    
    Returns:
        model: Retrained model
        X_labeled_new: Updated labeled features
        y_labeled_new: Updated labeled targets
        X_unlabeled_new: Remaining unlabeled features
        y_unlabeled_new: Remaining unlabeled true labels
        metrics: Dictionary with training/validation metrics
    """
    
    # Check if unlabeled pool is empty
    if len(X_unlabeled) == 0:
        if verbose:
            print(f"Epoch {epoch}: No unlabeled data remaining.")
        metrics = {
            'epoch': epoch,
            'threshold': 0.0,
            'pseudo_labeled': 0,
            'unlabeled_remaining': 0,
            'train_acc': model.score(X_labeled, y_labeled),
            'val_acc': model.score(X_val, y_val),
            'train_loss': 0.0,
            'val_loss': 0.0
        }
        return model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, metrics
    
    # Determine confidence threshold
    if use_dynamic_threshold:
        threshold = compute_dynamic_threshold(epoch, total_epochs, tau_min, tau_max)
    else:
        threshold = confidence_threshold
    
    # Predict on unlabeled pool
    proba = model.predict_proba(X_unlabeled)
    predictions = np.argmax(proba, axis=1)
    
    # Calculate confidence using entropy
    entropy_scores = entropy(proba.T)
    entropy_conf = 1 - entropy_scores / np.log(proba.shape[1])
    confidences = entropy_conf
    
    # Select high-confidence samples
    high_conf_mask = confidences >= threshold
    high_conf_indices = np.where(high_conf_mask)[0]
    
    X_pseudo = X_unlabeled[high_conf_mask]
    y_pseudo = predictions[high_conf_mask]
    conf_pseudo = confidences[high_conf_mask]
    
    if verbose:
        print(f"Epoch {epoch}: Pseudo-labels added: {len(X_pseudo)} at threshold {threshold:.4f}")
    
    # If no pseudo-labels, return without retraining
    if len(X_pseudo) == 0:
        metrics = {
            'epoch': epoch,
            'threshold': threshold,
            'pseudo_labeled': 0,
            'unlabeled_remaining': len(X_unlabeled),
            'train_acc': model.score(X_labeled, y_labeled),
            'val_acc': model.score(X_val, y_val),
            'train_loss': 0.0,
            'val_loss': 0.0
        }
        return model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, metrics
    
    # Combine labeled and pseudo-labeled data
    X_combined = np.vstack([X_labeled, X_pseudo])
    y_combined = np.hstack([y_labeled, y_pseudo])
    
    # Retrain model
    model.fit(X_combined, y_combined)
    
    # Calculate training metrics
    train_acc = model.score(X_combined, y_combined)
    try:
        proba_train = model.predict_proba(X_combined)
        train_loss = log_loss(y_combined, proba_train)
    except (ValueError, AttributeError):
        train_loss = 1.0 - train_acc
    
    # Calculate validation metrics
    val_acc = model.score(X_val, y_val)
    try:
        proba_val = model.predict_proba(X_val)
        val_loss = log_loss(y_val, proba_val)
    except (ValueError, AttributeError):
        val_loss = 1.0 - val_acc
    
    if verbose:
        print(f"Epoch {epoch} — Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
    
    # Remove pseudo-labeled samples from unlabeled pool
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[high_conf_indices] = False
    X_unlabeled_new = X_unlabeled[mask]
    y_unlabeled_new = y_unlabeled[mask]
    
    # Prepare metrics
    metrics = {
        'epoch': epoch,
        'threshold': threshold,
        'pseudo_labeled': len(X_pseudo),
        'unlabeled_remaining': len(X_unlabeled_new),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    return model, X_combined, y_combined, X_unlabeled_new, y_unlabeled_new, metrics


def split_data_combined(X_train, y_train, X_test, y_test, labeled_ratio=0.1, validation_ratio=0.2, random_state=42):
    """
    Approach 1: Combine train and test, then split into labeled/unlabeled
    
    Args:
        X_train, y_train: Original training data
        X_test, y_test: Original test data
        labeled_ratio: Ratio of data to use as initial labeled set
        validation_ratio: Ratio of labeled data to use for validation
        random_state: Random seed
    
    Returns:
        X_labeled, y_labeled: Initial labeled training data
        X_val, y_val: Validation data
        X_unlabeled, y_unlabeled: Unlabeled pool (with true labels for monitoring)
    """
    np.random.seed(random_state)
    
    # Combine all data
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([y_train, y_test])
    
    # Shuffle
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    # Split into labeled and unlabeled
    n_labeled = int(len(X_combined) * labeled_ratio)
    X_labeled_all = X_combined[:n_labeled]
    y_labeled_all = y_combined[:n_labeled]
    X_unlabeled = X_combined[n_labeled:]
    y_unlabeled = y_combined[n_labeled:]
    
    # Split labeled into train and validation
    n_val = int(len(X_labeled_all) * validation_ratio)
    X_val = X_labeled_all[:n_val]
    y_val = y_labeled_all[:n_val]
    X_labeled = X_labeled_all[n_val:]
    y_labeled = y_labeled_all[n_val:]
    
    return X_labeled, y_labeled, X_val, y_val, X_unlabeled, y_unlabeled


def split_data_train_only(X_train, y_train, X_test, y_test, labeled_ratio=0.1, validation_ratio=0.2, random_state=42):
    """
    Approach 2: Split only training data, keep test set separate
    
    Args:
        X_train, y_train: Original training data
        X_test, y_test: Original test data (kept as final test set)
        labeled_ratio: Ratio of training data to use as initial labeled set
        validation_ratio: Ratio of labeled data to use for validation
        random_state: Random seed
    
    Returns:
        X_labeled, y_labeled: Initial labeled training data
        X_val, y_val: Validation data
        X_unlabeled, y_unlabeled: Unlabeled pool from training data
        X_test, y_test: Unchanged test set
    """
    np.random.seed(random_state)
    
    # Shuffle training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # Split training data into labeled and unlabeled
    n_labeled = int(len(X_train) * labeled_ratio)
    X_labeled_all = X_train[:n_labeled]
    y_labeled_all = y_train[:n_labeled]
    X_unlabeled = X_train[n_labeled:]
    y_unlabeled = y_train[n_labeled:]
    
    # Split labeled into train and validation
    n_val = int(len(X_labeled_all) * validation_ratio)
    X_val = X_labeled_all[:n_val]
    y_val = y_labeled_all[:n_val]
    X_labeled = X_labeled_all[n_val:]
    y_labeled = y_labeled_all[n_val:]
    
    return X_labeled, y_labeled, X_val, y_val, X_unlabeled, y_unlabeled, X_test, y_test
