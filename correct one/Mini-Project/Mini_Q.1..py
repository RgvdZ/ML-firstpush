# Library imports
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


import MakeDataset
from MakeDataset import folders,dataset_dir
from MakeDataset import X_train,y_train

classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}

# Define the classes and corresponding colors
activity_classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
colors = ['#FF7675', '#74B9FF', '#FFD700']  # Elegant colors

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Plot the waveform for each activity class
for i, activity in enumerate(activity_classes):
    # Get indices corresponding to the current activity in the training set
    indices = np.where(y_train == classes[activity])[0][0]

    # Plot acceleration data for each axis on the same graph
    for j, axis in enumerate(['accx', 'accy', 'accz']):
        row = i // 3
        col = i % 3
        axes[row, col].plot(X_train[indices, :, j].flatten(), color=colors[j], label=f"{axis} - {activity}")

    axes[row, col].set_title(f"{activity}")
    axes[row, col].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()



















