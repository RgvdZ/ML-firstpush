import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)  #Generate a synthetic(artificial) classification dataset with 2 informative features, no redundant features, 2 clusters per class, and a class separation of 0.5.


split_range= int(0.7*len(X))   #Split the data into a training set (70% of the data) and a testing set (30% of the data).

X_train= pd.DataFrame(X[:split_range])
X_test = pd.DataFrame(X[split_range:])
y_train= pd.Series(y[:split_range],dtype=y.dtype)
y_test = pd.Series(y[split_range:],dtype=y.dtype)

clf= DecisionTree(criterion='information_gain',max_depth=10)     #Train a decision tree on the training data. The tree is initialized with the criterion of 'information_gain' and a maximum depth of 10.
clf.fit(X_train,y_train)
y_hat = clf.predict(X_test)     #Predict the class labels for the test data using the trained decision tree.

print("=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=")
print("Metrics Obtained after performing a 70-30 Train-Test split")     #Print the accuracy, precision, and recall of the predictions for each class. These are common metrics for evaluating classification models.
print("Accuracy : ", accuracy(y_hat, y_test))
for classs in np.unique(y):
    print("Precision for class %s is %f " %(str(classs), precision(y_hat,y_test,classs)))
    print("Recall for class %s is %f " %(str(classs), recall(y_hat,y_test,classs)))



def KFold_CV(X_train: pd.DataFrame, y_train:pd.Series, k) -> DecisionTree :   #Define a function for performing k-fold cross-validation on the training data. This function splits the training data into k folds, trains a decision tree on k-1 folds and validates it on the remaining fold. This process is repeated k times, each time with a different fold used for validation. The function returns the decision tree that achieved the highest average accuracy across the k folds.
    depth = [1,2,3,4,5]

    len_train = len(X_train)
    k_folds = len_train//k
    start = 0
    end = k_folds

    acc_dict = {}
    for i in range(k):
        if start>0 and end<=len_train:
            trainX = pd.concat([X_train.iloc[0:start-1,:], X_train.iloc[end+1:,:] ])
            trainY = pd.concat([y_train.iloc[0:start-1], y_train.iloc[end+1:]])
            validation_dataX = X_train.iloc[start:end, :]
            validation_dataY = y_train.iloc[start:end]
        else:
            trainX = X_train.iloc[end:, :]
            trainY = y_train.iloc[end:]
            validation_dataX = X_train.iloc[start:end-1, :]
            validation_dataY = y_train.iloc[start:end-1]

        start = end
        end += k_folds

        ########################
        depth_list = []
        for j in range(len(depth)):
            tree = DecisionTree(criterion="information_gain", max_depth = depth[j]) #Split based on Inf. Gain
            tree.fit(trainX, trainY)
            y_hat = tree.predict(validation_dataX)

            y_hat = pd.Series(y_hat, dtype=np.int64, name='0')
            validation_dataY = pd.Series(validation_dataY, dtype=np.int64, name='0')
            validation_dataY.index = [i for i in range(len(validation_dataY))]

            depth_list.append(accuracy(y_hat, validation_dataY))
        acc_dict[i] = depth_list
    df = pd.DataFrame(acc_dict)
    df['Avg_acc'] = (df.loc[:].sum())/k
    optimum_depth = df['Avg_acc'].argmax()+1
    tree = DecisionTree(criterion="information_gain",max_depth=optimum_depth)
    tree.fit(X_train, y_train)
    tree.max_depth = optimum_depth
    return tree


def Nested_CV(X, y, k = 5):   #Define a function for performing nested cross-validation. This function splits the entire dataset into k folds, and for each fold, it performs k-fold cross-validation on the remaining k-1 folds. The function returns the depths of the decision trees that achieved the highest average accuracy in each round of cross-validation.
    X = pd.DataFrame(X)
    y = pd.Series(y)

    length = len(X)
    fold = length//k

    start = 0
    end = fold
    acc = []
    depth = []
    for i in range(k):
        X_copy = X.copy()
        y_copy = y.copy()
        X_test = X_copy[start:end]
        X_train = X_copy.drop(X_test.index)
        y_test = y_copy[start:end]
        y_train = y_copy.drop(y_test.index)

        start = end
        end += fold
        #################################
        X_test.index = [i for i in range(len(X_test))]
        X_train.index = [i for i in range(len(X_train))]
        y_test.index = [i for i in range(len(y_test))]
        y_train.index = [i for i in range(len(y_train))]
        #################################
        tree = KFold_CV(X_train, y_train,5)
        depth.append(tree.max_depth)
    #print(depth)
    return depth

print("=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=")
print("Depth obtained at each fold is given below")
depth = Nested_CV(X,y,5)    #Call the nested cross-validation function and print the depths of the decision trees that achieved the highest average accuracy in each round of cross-validation.
for i in range(len(depth)):
    print("     Depth at fold ",i+1,":",depth[i])
print("=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=")
#nested cross-validation, you have a double loop, an outer loop (that will serve for assessing the quality of the model), and an inner loop (that will serve for model/parameter selection). It's very important that those loops are independent, so each step or layer of cross-validation does one and only one thing
