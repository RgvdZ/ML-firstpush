"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index, variance_reduction,BestSplit

np.random.seed(42)


class Node():
    def __init__(self, feature=None, threshold=None, children =[], tuning=None, value=None):

        # When it is a  decision node
        self.feature = feature      #Contains the feature for which the split/decision has been made
        self.threshold = threshold  # Contains the value of feature for which the split has been made
        self.children = children    # list that contains the children of the node
        self.tuning = tuning        # Value of Information Gain/ Variance reduction

        # When it is a leaf node
        self.value = value          # Value of the leaf



@dataclass
class DecisionTree:
    root = None     #variable root initiated to none
    criterion: Literal["information_gain", "gini_index"]  # criterion for Classification Probelms
    max_depth: int = 4  # The maximum depth the tree can grow to

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        input_type = None
        output_type = None

        # Identifying the Input Type
        if ('category' in list(X.dtypes)) or ('object' in list(X.dtypes)):    #If any column in X has a datatype of 'category' or 'object',
            input_type = 'd'
        else:
            input_type = 'r'

        # Identifying the Output type
        output_type = None
        if ('category' in list(pd.DataFrame(y).dtypes)) or ('object' in list(pd.DataFrame(y).dtypes)):  #based on whether the data is categorical or continuous
            output_type = 'd'
        else:
            output_type = 'r'

        # Build tree rooted at the root of the class.
        self.root = self.build_tree(X,y,0,input_type,output_type) #0 is the depth of the tree initial, self root is the root of the tree


    def build_tree(self, X: pd.DataFrame, y: pd.Series , depth : int , input_type , output_type ) -> Node:

        Dataset = pd.concat([X,y],axis=1)

        # For discrete data input
        if input_type == 'd' :
            # if depth has not yet reached the max depth, we keep building
            if depth < self.max_depth:   #checks if the current depth of the tree is less than the maximum depth allowed.

                # Acquire the best split
                best_split = BestSplit( Dataset , output_type, input_type , self.criterion)

                children = []   #initializes an empty list to store the child nodes of the current node.
                if best_split["val"] > 0:   #If best_split["val"] is greater than 0, it means that the proposed split improves the purity of the child nodes compared to the parent node, and thus the split is considered beneficial
                    feature = best_split["feature"]
                    # Building tree for all the classes of value the feature can take
                    for f in pd.unique(Dataset.iloc[:,:-1][feature]):   #iterates over the unique values of the best split feature. -1 is labelled column, feature = column, unique - this is done because each unique value of the feature can potentially form a split or a branch in the tree.

                        #Splitting data based on attributes
                        Datasplit = Dataset[Dataset.iloc[:,:-1][feature] == f]  #iterates over the unique values of the best split feature.

                        # Building children Trees/Nodes
                        child = self.build_tree(Datasplit.iloc[:,:-1],Datasplit.iloc[:,-1], depth + 1,input_type,output_type)     #This line recursively calls the build_tree function to build the child nodes. [:,:-1]: This is selecting all rows (:) and all columns except the last one (:-1) from the Datasplit DataFrame. This represents the features of the dataset after it has been split based on a specific attribute value. :,-1]: This is selecting all rows (:) and only the last column (-1) from the Datasplit DataFrame. This represents the labels or target values of the dataset after it has been split.
                        children.append({ 'threshold': f,
                                      'tree':child })

                    # When it is a decision node it will exit from here
                    #print('decision')
                    return Node(best_split['feature'],None,children,best_split['val'],None)  #returns a new Node with the best split feature, the children, and the best split value.

            # When it is a leaf node it will exit from here
            if output_type == 'd':
                # For Discrete output we will take the most occuring class in the remaining  DataSet
                leaf_value =  y.mode().iloc[0]   # sets the leaf value to the mode of the output labels y.
            else:
                # For Real output we will take the mean of the remaining DataSet
                leaf_value =  np.mean(y)   # sets the leaf value to the mean of the output labels y.
            #print('leaf')
            return Node(value=leaf_value)   #returns a new Node with the leaf value.

        # For real data input
        else:

            if depth < self.max_depth:
                # Acquire the best split
                best_split = BestSplit( Dataset , output_type, input_type , self.criterion)  #calls the BestSplit function to find the best feature to split the data on.
                children = []
                #print(best_split)
                if best_split["val"] > 0:
                    feature = best_split["feature"]
                    threshold = best_split["split"]  #gets the threshold value for the best split.

                    # Building trees for the threshold split
                    left_split = Dataset[Dataset.iloc[:,:-1][feature] <= threshold]    #splits the data into a left subset where the values of the best feature are less than or equal to the threshold. Compares feature and threshold.  It returns a Series of boolean values (True or False) where each value indicates whether the corresponding value in the feature column is less than or equal to threshold. this is used to split the data at each node of the tree based on whether the feature value is less than or equal to a threshold.
                    right_split = Dataset[Dataset.iloc[:,:-1][feature] > threshold]

                    # Building Left tree
                    left_tree = self.build_tree(left_split.iloc[:,:-1],left_split.iloc[:,-1], depth + 1,input_type,output_type)   #input features [:,:-1], output labels []:,-1], depth + 1
                    children.append({'threshold': threshold,
                                      'tree':left_tree})

                    # Building Right tree
                    right_tree = self.build_tree(right_split.iloc[:,:-1],right_split.iloc[:,-1], depth + 1,input_type,output_type)
                    children.append({'threshold': threshold,
                                      'tree':right_tree})

                    # When it is a decision node it will exit from here
                    return Node( best_split['feature'], threshold , children , best_split['val'],None )

            # When it is a leaf node it will exit from here
            if output_type == 'd':
                # For Discrete output we will take the most occuring class in the remaining  DataSet
                leaf_value =  y.mode().iloc[0]
            else:
                # For Real output we will take the mean of the remaining DataSet
                leaf_value =  np.mean(y)
            return Node(value=leaf_value)



    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Identifying the input type
        if ('category' in list(X.dtypes)) or ('object' in list(X.dtypes)):
            input_type = 'd'    # Discrete Input
        else:
            input_type = 'r'    # Real Input

        y_hat = []   # to store predictions

        # Toggling through all the input records, making prediction and storing them in a list
        for i in range(len(X)):   #starts a loop that iterates over each row in the DataFrame X.
            predition = self.make_prediction( X[i:i+1] , self.root , input_type)   #calls the make_prediction method to make a prediction for the current row of data. The make_prediction method takes three arguments: a slice of the DataFrame X containing the current row of data, the root of the decision tree, and the type of the input data.
            y_hat.append(predition)

        return pd.Series(y_hat)     # Returning the Prediction



    def make_prediction(self, y , tree : Node , input_type):   # y -> datapoint
        """ function to predict new dataset """

        # Only leaf nodes will have values. If you have found a value that means its a leaf node. leaf node has decision or prediction of the tree for a certain path of conditions
        if tree.value!=None:   #checks if the current node in the decision tree is a leaf node.
            return tree.value  #returns the value of the leaf node. leaf node = base case

        # feature value of the node
        feature_val = y[tree.feature].values  #gets the value of the feature for the current node.This value will be used to decide which branch of the tree to follow next.

        # When input id Discrete
        if input_type == 'd':

            # Toggling through all the children nodes to identify the tree we need to move to make a prediction.
            for child in tree.children:                 # Each child represents a possible next step in the decision tree. This loop iterates over each child of the current node.
                if child['threshold'] == feature_val:    #checks if the value of the feature for the current node is equal to the threshold value of the child node.
                    return self.make_prediction(y, child['tree'],input_type)    #If the condition in the previous line is true, this line makes a recursive call to make_prediction to continue traversing the decision tree from the current child node.

        # When input is Real
        else:

            # Values less than or equal to threshold are kept at left tree
            if feature_val <= tree.threshold:
                return self.make_prediction(y, tree.children[0]['tree'],input_type)
            else:
            # Values greater threshold are kept at right tree
                return self.make_prediction(y, tree.children[1]['tree'],input_type)



    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        # Plotting the Tree using the helper function
        self.Tree_Plot()



    def Tree_Plot(self,tree = None,indent = "| ") -> None:
        """
        Helper function to plot the tree
         """

        # if tree is not provided take it from the root.
        if not tree :
            tree = self.root

        # If we encounter a value at any node it will be a leaf node.
        if tree.value is not None :
            print(indent,tree.value)

        else:
            # Recurse for the children and print the Tree.
            for i in tree.children:
                print( indent , tree.feature , ' : ' , i['threshold'] , '?' )
                self.Tree_Plot(i['tree'],indent + "| ")

#The function takes two parameters: tree and indent. tree is the current node in the tree that we want to print, and indent is a string used to control the indentation of the printed tree. If no tree is provided, the function uses self.root as the starting point, which is the root of the tree.

#The function first checks if the value attribute of the current node (tree) is not None. If it's not None, it means the node is a leaf node (a node with no children), so it prints the value of the node with the current indentation.

#If the value attribute of the node is None, it means the node is not a leaf node but an internal node. In this case, the function iterates over the children of the node. For each child, it prints the feature of the current node, the threshold of the child, and a question mark. The question mark indicates that this is a decision node, where the feature of the data is compared with the threshold to decide which child node to go to next.

#After printing the information for a child, the function calls itself recursively with the tree parameter set to the child's tree and the indent parameter increased by one level. This is to print the subtree rooted at the child node. The recursion continues until all nodes in the tree have been visited and printed.
