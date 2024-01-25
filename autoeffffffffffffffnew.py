import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeRegressor
from metrics import *
from sklearn.metrics import median_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])           #mpg = milles per gallon



#Train test split
train_data = data[:276] #70% train data           395/276
test_data = data[276:] #30% test data
test_data = test_data.reset_index(drop = True)

# #Training DecisionTree
tree = DecisionTree(criterion='information_gain')  # Split based on Inf. Gain
tree.fit(train_data[["cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin"]], train_data['mpg'])    #Train a decision tree on the training data. The tree is initialized with the criterion of 'information_gain', meaning it will split nodes based on which split results in the greatest information gain

# #Testing
y_hat = tree.predict(test_data[["cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin"]])

print("Decision Tree from scratch")    #Predict the mpg for the test data using the trained decision tree.
print("Criteria :information_gain")

print("RMSE: ", rmse(y_hat, test_data['mpg']))
print("MAE: ", mae(y_hat, test_data['mpg']))       #Predict the mpg for the test data using the trained decision tree.

#DecisionTree from sklearn
print('DecisionTree from sklearn\n')

data = data.replace(r'\S+$', np.nan, regex = True)     #Print a separator line for output clarity.
for attribute in data.columns:
       val_list = []                        #Replace all non-numeric values in the data with NaN
       for value in data[attribute]:
              val_list.append(float(value))     #Convert all values in the data to floats
       data[attribute] = val_list

data.drop(['horsepower'],axis = 1)   #Drop the 'horsepower' column from the data.
X_train, X_test, y_train, y_test = train_test_split(data[["cylinders", "displacement", "weight",
                        "acceleration", "model year", "origin"]],data['mpg'] , test_size = 0.20)

clf = DecisionTreeRegressor(max_depth = 5)       #nitialize and train a decision tree regressor from sklearn. The maximum depth of the tree is set to 5. Used to create a decision tree regression model.
clf = clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

print("Criteria :information_gain")
print('RMSE: ',sqrt(mean_squared_error(y_hat, y_test)))
print('MAE: ',median_absolute_error(y_hat, y_test))
