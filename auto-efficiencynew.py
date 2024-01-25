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
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn


# print('Top 5 values\n',data.head(5))

# for attribute in data.columns:
#        print('Attribute:', attribute,' data type:',data[attribute].dtype)

# for attribute in data.columns:
#        print('No of unique values in ',attribute,'are ',len(data[attribute].unique()))

# print('Total number of records:', len(data))

#Drop column car name
# data = data.drop(['car name'], axis = 1)
# print('Top 5 values\n',data.head(5))

#Train test split
train_data = data[:276] #70% train data
test_data = data[276:] #30% test data
test_data = test_data.reset_index(drop = True)

# #Training DecisionTree
tree = DecisionTree(criterion='information_gain')  # Split based on Inf. Gain
tree.fit(train_data[["cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin"]], train_data['mpg'])

# #Testing
y_hat = tree.predict(test_data[["cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin"]])

print("Decision Tree from scratch")
print("Criteria :information_gain")

print("RMSE: ", rmse(y_hat, test_data['mpg']))
print("MAE: ", mae(y_hat, test_data['mpg']))

#DecisionTree from sklearn
print('DecisionTree from sklearn\n')

data = data.replace(r'\S+$', np.nan, regex = True)
for attribute in data.columns:
       val_list = []
       for value in data[attribute]:
              val_list.append(float(value))
       data[attribute] = val_list

data.drop(['horsepower'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(data[["cylinders", "displacement", "weight",
                        "acceleration", "model year", "origin"]],data['mpg'] , test_size = 0.20)

clf = DecisionTreeRegressor(max_depth = 5)
clf = clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

print("Criteria :information_gain")
print('RMSE: ',sqrt(mean_squared_error(y_hat, y_test)))
print('MAE: ',median_absolute_error(y_hat, y_test))
