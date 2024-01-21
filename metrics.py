from typing import Union
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error, mean_absolute_error

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    if y_hat is None:
        print("Warning: y_hat is None. Returning NaN.")
        return np.nan

    assert y_hat.size == y.size
    return accuracy_score(y, y_hat)

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    if y_hat is None:
        print("Warning: y_hat is None. Returning NaN.")
        return np.nan

    assert y_hat.size == y.size
    return precision_score(y, y_hat, pos_label=cls)

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    if y_hat is None:
        print("Warning: y_hat is None. Returning NaN.")
        return np.nan

    assert y_hat.size == y.size
    return recall_score(y, y_hat, pos_label=cls)

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    if y_hat is None:
        print("Warning: y_hat is None. Returning NaN.")
        return np.nan

    assert y_hat.size == y.size
    return np.sqrt(mean_squared_error(y, y_hat))

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    if y_hat is None:
        print("Warning: y_hat is None. Returning NaN.")
        return np.nan

    assert y_hat.size == y.size
    return mean_absolute_error(y, y_hat)
