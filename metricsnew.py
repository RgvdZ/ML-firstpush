from typing import Union
import pandas as pd
import numpy as np



def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
        y_hat : Predicted Values
        y : Ground Truths
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """

    assert y_hat.size == y.size , "Size of Predicted values and Ground truth needs to be Same"
    assert y_hat.size != 0, "Size of Predicted Value and Ground Truth can't be zero" #checks if the size of y_hat and y are not zero
    # TODO: Write here

    # Assigning 1 when the predicted value is same as ground truth else zero
    trues = pd.DataFrame(np.where(y_hat == y ,1,0))  #creates a new DataFrame trues where each element is 1 if the corresponding elements in y_hat and y are equal (i.e., the prediction is correct), and 0 otherwise. np.where(condition, [x, y]): This is a function from the NumPy library. It returns elements chosen from x or y depending on the condition. If the condition is true, it returns an element from x, otherwise it returns an element from y.

    # Sum up the whole dataframe
    Total_Value = len(trues)   #calculates the total number of predictions by getting the length of the trues DataFrame.

    #Sum up the True Positives in the dataframe
    No_of_trues = int(trues.sum())    #calculates the total number of correct predictions (i.e., the sum of the trues DataFrame).

    if Total_Value !=0 :
        return No_of_trues/Total_Value
    else:
        return None



def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
        y_hat : Predicted Values
        y : Ground Truths
        cls : Class for which Precision needs to calculated
    """

    # The following assert checks if sizes of y_hat and y are equal and they are not equal to zero
    assert y_hat.size == y.size , "Size of Predicted values and Ground truth needs to be Same"
    assert y_hat.size != 0, "Size of Predicted Value and Ground Truth can't be zero"

    #Calculating Confusion matrix parameters
    c = confusion_matrix( y_hat , y , cls )   #calculates the confusion matrix for the given class cls. The confusion matrix is a table that is often used to describe the performance of a classification model.

    if (c['TP'].sum() + c['FP'].sum()) !=0 :    # This is denominator for precision calculation which needs to be nonzero. checks if the sum of True Positives (TP) and False Positives (FP) is not zero. This is the denominator for the precision calculation which needs to be nonzero to avoid division by zero.
        prec = c['TP'].sum() / (c['TP'].sum() + c['FP'].sum())  #If the denominator is not zero, this line calculates the precision as the ratio of True Positives (TP) to the sum of True Positives (TP) and False Positives (FP)
    else :
        prec = 0 #If the denominator is zero, this line assigns zero to the precision.

    return prec   #returns the calculated precision.



def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
        y_hat : Predicted Values
        y : Ground Truths
        cls : Class for which Recall needs to calculated
    """

    # The following assert checks if sizes of y_hat and y are equal and they are not equal to zero
    assert y_hat.size == y.size , "Size of Predicted values and Ground truth needs to be Same"
    assert y_hat.size != 0, "Size of Predicted Value and Ground Truth can't be zero"

    #Calculating Confusion matrix parameters
    c = confusion_matrix( y_hat , y , cls )   #calculates the confusion matrix for the given class cls. The confusion matrix is a table that is often used to describe the performance of a classification model.

    if (c['TP'].sum() + c['FN'].sum()) != 0 :   # This is denominator for Recall calculation which needs to be nonzero.checks if the sum of True Positives (TP) and False Negatives (FN) is not zero. This is the denominator for the recall calculation which needs to be nonzero to avoid division by zero.
        recall = c['TP'].sum() / (c['TP'].sum() + c['FN'].sum())    # If the denominator is not zero, this line calculates the recall as the ratio of True Positives (TP) to the sum of True Positives (TP) and False Negatives (FN).
    else:
        recall = 0

    return recall



def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
        y_hat : Predicted Values
        y : Ground Truths
    """

    # The following assert checks if sizes of y_hat and y are equal and they are not equal to zero
    assert y_hat.size == y.size , "Size of Predicted values and Ground truth needs to be Same"
    assert y_hat.size != 0, "Size of Predicted Value and Ground Truth can't be zero"

    Squared_Error = (y_hat - y)**2
    Mean_Sq_Err = float(Squared_Error.mean())
    Root_Mn_Sq_Err = Mean_Sq_Err**0.5

    return Root_Mn_Sq_Err



def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
        y_hat : Predicted Values
        y : Ground Truths
    """

    # The following assert checks if sizes of y_hat and y are equal and they are not equal to zero
    assert y_hat.size == y.size , "Size of Predicted values and Ground truth needs to be Same"
    assert y_hat.size != 0, "Size of Predicted Value and Ground Truth can't be zero"

    # Calculating Absolute Difference between Predicted values and Ground Truth
    Absolute_Error = abs(y_hat - y)

    #Calculating Mean of the Absoulte values obtained
    Mean_Abs_Err = Absolute_Error.mean()

    return Mean_Abs_Err



def confusion_matrix( Pred : pd.Series, Truth : pd.Series, cls : Union[int, str] ) -> pd.DataFrame :
    """Gives out Confusion DataFrame"""

    #Creating a Confusion Dataframe
    cm=pd.DataFrame()   #creates an empty DataFrame cm which will be used to store the confusion matrix.

    #Assigning Prediction and Ground_Truth values
    cm['P'] = Pred   #assigns the predicted values to the column 'P' in the DataFrame cm.
    cm['GT'] = Truth  #assigns the ground truth values to the column 'GT' in the DataFrame cm

    #Calculating Confusion matrix Parameters
    cm['TP'] = pd.DataFrame( np.where( (cm['P'] == cls ) & (cm['GT'] == cls ), 1,0))   #calculates the True Positives (TP) - where the predicted class and the actual class are both cls, and assigns it to the column 'TP' in the DataFrame cm.
    cm['TN'] = pd.DataFrame( np.where( (cm['P'] != cls ) & (cm['GT'] != cls ), 1,0))   #calculates the True Negatives (TN) - where the predicted class and the actual class are both not cls, and assigns it to the column 'TN' in the DataFrame cm
    cm['FP'] = pd.DataFrame( np.where( (cm['P'] == cls ) & (cm['GT'] != cls ), 1,0))
    cm['FN'] = pd.DataFrame( np.where( (cm['P'] != cls ) & (cm['GT'] == cls ), 1,0))

    #Returning Confusion Matrix
    return cm
