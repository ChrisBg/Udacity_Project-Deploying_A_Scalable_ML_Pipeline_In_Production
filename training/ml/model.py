"""
This script is used to train the model.
Author : Christophe Bourgoin
Date : 2024-12-28
"""

# Importing libraries
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from typing import List, Dict, Any
import logging  

logger = logging.getLogger(__name__)

# Training a model histgradientboosting based on hyper-parameters tuning with RandomizedSearchCV
def train_model(X_train, y_train):
    """
    Trains a machine learning model from a randomizedSearchCV and returns the best model.
    

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Initialize the model
    logger.info("Initializing model")
    model = HistGradientBoostingClassifier()

    # Define parameter distributions
    logger.info("Defining parameter distributions")
    param_distributions = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 3, 5, 10],
        'min_samples_leaf': [10, 20, 30],
        'max_iter': [100, 200, 300],
        'l2_regularization': [0.0, 1.0, 10.0],
        'max_features': [0.5, 0.75, 1.0],
        'early_stopping': [True, False],
        'validation_fraction': [0.1, 0.2, 0.3],
        'n_iter_no_change': [10, 20, 30],
          }

    # Initialize RandomizedSearchCV
    logger.info("Initializing RandomizedSearchCV")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='f1',
        n_jobs=-1,
        cv=5,
        verbose=2,
        random_state=42
    )

    # Fit the model
    logger.info("Fitting model")
    random_search.fit(X_train, y_train) 

    # Output best parameters and evaluate
    logger.info("Outputting best parameters and evaluating")
    logger.info("Best Hyperparameters: %s", random_search.best_params_)
    
    # Keep the best model
    logger.info("Keeping the best model")
    best_model = random_search.best_estimator_

    return best_model


# Computing model metrics : precision, recall, fbeta
def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


# Inference function
def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

# Prepare DataFrame for slice analysis by adding true and predicted labels
def prepare_analysis_df(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Prepare DataFrame for slice analysis by adding true and predicted labels.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame with features
    y_true : numpy.ndarray
        Array of true labels
    y_pred : numpy.ndarray
        Array of predicted labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added y_true and y_pred columns
    """
    # Create a copy to avoid modifying the original DataFrame
    analysis_df = df.copy()
    
    # Add true and predicted labels as new columns
    analysis_df['y_true'] = y_true
    analysis_df['y_pred'] = y_pred
    
    return analysis_df

# Model Evaluation & metrics on a given slice
def compute_slice_metrics(
    df: pd.DataFrame,
    categorical_features: List[str],
    y_true: str,
    y_pred: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute classification metrics for each unique value in specified categorical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the features and predictions
    categorical_features : List[str]
        List of categorical feature column names to slice by
    y_true : str
        Column name for true labels
    y_pred : str
        Column name for predicted labels
        
    Returns:
    --------
    Dict[str, Dict[str, Dict[str, float]]]
        Nested dictionary with metrics for each feature slice
        Structure: {feature: {value: {metric: score}}}
    """
    metrics = {}
    
    for feature in categorical_features:
        metrics[feature] = {}
        
        # Get unique values for the feature
        unique_values = df[feature].unique()
        
        for value in unique_values:
            # Create slice for current feature value
            slice_mask = df[feature] == value
            slice_size = slice_mask.sum()
            
            # Skip if slice is too small
            if slice_size < 10:
                continue
                
            # Compute metrics for this slice
            slice_metrics = {
                'size': int(slice_size),
                'proportion': float(slice_size / len(df)),
                'fbeta': fbeta_score(
                    df.loc[slice_mask, y_true],
                    df.loc[slice_mask, y_pred],
                    beta=1,
                    zero_division=1
                ),
                'precision': precision_score(
                    df.loc[slice_mask, y_true],
                    df.loc[slice_mask, y_pred],
                    zero_division=1
                ),
                'recall': recall_score(
                    df.loc[slice_mask, y_true],
                    df.loc[slice_mask, y_pred],
                    zero_division=1
                )
            }
            
            metrics[feature][str(value)] = slice_metrics
            
    return metrics


def print_slice_metrics(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Pretty print the slice metrics in a readable format.
    
    Parameters:
    -----------
    metrics : Dict[str, Dict[str, Dict[str, float]]]
        The metrics dictionary returned by compute_slice_metrics
    """
    for feature, feature_metrics in metrics.items():
        print(f"\n=== Metrics for {feature} ===")
        
        # Create a DataFrame for easy viewing
        rows = []
        for value, value_metrics in feature_metrics.items():
            row = {'value': value}
            row.update(value_metrics)
            rows.append(row)
            
        df_metrics = pd.DataFrame(rows)
        df_metrics.set_index('value', inplace=True)
        
        # Format numeric columns
        for col in df_metrics.columns:
            if col in ['size']:
                continue
            df_metrics[col] = df_metrics[col].map('{:.3f}'.format)
            
        print(df_metrics)