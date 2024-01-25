"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np


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
    attr = attr.astype(bool)  # convert attr to a boolean mask
    a = sum(attr)
    b = attr.shape[0] - a
    ig = np.var(Y) - (a / (a + b) * np.var(Y[attr])) - (b / (a + b) * np.var(Y[~attr]))
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

def variance_reduction(Y: pd.Series, attr: pd.Series , threshold = None , input_type = 'r' ) -> float:
    """
    Calculates reduction in variance for the passed attribute
    """
    data = pd.DataFrame()
    data['attr'] = attr
    data['y'] = Y

    # When Input is Real
    if input_type == 'r':

        # Segregating data for variance calculation
        left_data = data[data['attr']<=threshold]['y']
        right_data = data[data['attr']>threshold]['y']

        # Split made should have some samples...they cant be zero
        if (len(right_data) > 0) and (len(left_data) >0) :

            #Calculating weights of left and right sub-tree
            left_wt = len(left_data)/len(data)
            right_wt = len(right_data)/len(data)

            var_red = np.var(data['y']) - (left_wt*np.var(left_data) + right_wt*np.var(right_data))
            return var_red

        # If the either of the split data has size zero then variance reduction becomes zero
        return 0

    # When Input is Discrete
    else:

        # Calculating variance of entire set
        var_red = np.var(Y)

        #Toggling through the unique values of given attribute
        for val in data['attr'].unique():

            # Creating a Data Split based on the current value
            Data_split = data[data['attr']==val]

            # Weight of the Split
            val_wt = len(Data_split)/len(data)
            var_red = var_red - val_wt*np.var(Data_split['y'])

        return var_red




def BestSplit( Dataset : pd.DataFrame , output_type = 'r' , input_type = 'r', criteria = 'information_gain') -> dict:
    """
    Determines the best split that could be made in the given Dataset
    """
    #input_type = 'r'

    #initializing variable
    split = {}
    max = -np.inf

    # When input is Real
    if input_type == 'r':

        # Loop through all the available columns
        for features in Dataset.iloc[:,:-1].columns:

            # Toggling through all the possible splits.
            for possible_split in pd.unique(Dataset.iloc[:,:-1][features]):

                if output_type == 'r':
                    # For real output we do variance reduction calculation
                    curr_var = variance_reduction(Dataset.iloc[:,-1] , Dataset.iloc[:,:-1][features] , possible_split , input_type )
                else:
                    # For discrete output we do information gain calculation
                    curr_var = information_gain(Dataset.iloc[:,-1] , Dataset.iloc[:,:-1][features] , criteria , possible_split, input_type )

                # Building split dict values
                if curr_var  > max:
                    split["feature"] = features
                    split["split"] = possible_split
                    split["val"] = curr_var
                    max = curr_var

        return split

    # When input is discrete
    else:

        split = {}
        max = -np.inf

        # Loop through all the available columns
        for features in Dataset.iloc[:,:-1].columns:

            if output_type == 'r':
                # For real output we do variance reduction calculation
                curr_var = variance_reduction(Dataset.iloc[:,-1] , Dataset.iloc[:,:-1][features] , None , input_type )
            else:
                # For discrete output we do information gain calculation
                curr_var = information_gain(Dataset.iloc[:,-1] , Dataset.iloc[:,:-1][features] , criteria , None, input_type )

            # Building split dict values
            if curr_var  > max:
                split["feature"] = features
                split["split"] = None
                split["val"] = curr_var
                max = curr_var

        return split

