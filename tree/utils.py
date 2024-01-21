"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    unique_values = y.unique()

    # Check if the data type is numeric (real values)
    if pd.api.types.is_numeric_dtype(y):
        return True

    # Check if the unique values are not integers (considered as discrete)
    elif not all(isinstance(value, (int, np.integer)) for value in unique_values):
        return True

    # If neither condition is met, it's considered discrete
    return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    if isinstance(Y, pd.Series):
        a = Y.value_counts() / Y.shape[0]
        entropy_val = np.sum(-a * np.log2(a + 1e-9))
        return entropy_val
    else:
        raise ValueError('Object must be a Pandas Series.')


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    if isinstance(Y, pd.Series):
        p = Y.value_counts() / Y.shape[0]
        gini = 1 - np.sum(p**2)
        return gini
    else:
        raise ValueError('Object must be a Pandas Series.')


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    a = sum(attr)
    b = attr.shape[0] - a

    if a == 0 or b == 0:
        ig = 0
    else:
        if Y.dtypes != 'O':
            ig = variance(Y) - (a / (a + b) * variance(Y[attr])) - (b / (a + b) * variance(Y[~attr]))
        else:
            ig = func(Y) - a / (a + b) * func(Y[attr]) - b / (a + b) * func(Y[~attr])

    return ig


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    best_attribute = None
    best_info_gain = -float('inf')

    for attribute in features:
        if criterion == 'information_gain':
            info_gain = information_gain(y, X[attribute])
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attribute = attribute
        elif criterion == 'gini_index':
            gini = gini_index(y, X[attribute])
            if gini < best_info_gain:
                best_info_gain = gini
                best_attribute = attribute
        elif criterion == 'mse':
            mse = mean_squared_error(y, X[attribute])
            if mse < best_info_gain:
                best_info_gain = mse
                best_attribute = attribute
        elif criterion == 'mae':
            mae = mean_absolute_error(y, X[attribute])
            if mae < best_info_gain:
                best_info_gain = mae
                best_attribute = attribute

    return best_attribute


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):  #edit this function if problem occurs
    """
    Function to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real-valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data (Input and output)
    """
    if attribute in X.columns:
        # For discrete features
        mask = X[attribute] == value
        X_split = X[mask]
        y_split = y[mask]
    else:
        # For real-valued features
        mask = X[attribute] <= value
        X_split = X[mask]
        y_split = y[mask]

    return X_split, y_split
