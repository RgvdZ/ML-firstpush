import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# # ... (rest of the code)

train_path = r"C:\Users\rutuj\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train"
test_path = r"C:\Users\rutuj\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test"

ACTIVITIES = {
    1: 'WALKING'            ,
    2: 'WALKING_UPSTAIRS'   ,
    3: 'WALKING_DOWNSTAIRS' ,
    4: 'SITTING'            ,
    5: 'STANDING'           ,
    6: 'LAYING'             ,
}

# Function to calculate magnitude of linear acceleration
def calculate_magnitude(accx, accy, accz):
    return np.sqrt(accx**2 + accy**2 + accz**2)

# Load all the accelerometer data
total_acc_x = pd.read_csv(os.path.join(train_path, "Inertial Signals", "total_acc_x_train.txt"), delim_whitespace=True, header=None)
total_acc_y = pd.read_csv(os.path.join(train_path, "Inertial Signals", "total_acc_y_train.txt"), delim_whitespace=True, header=None)
total_acc_z = pd.read_csv(os.path.join(train_path, "Inertial Signals", "total_acc_z_train.txt"), delim_whitespace=True, header=None)

# Read the labels
y = pd.read_csv(os.path.join(train_path, "y_train.txt"), delim_whitespace=True, header=None)

# Lists to store linear acceleration magnitudes for static and dynamic activities
static_magnitudes = []
dynamic_magnitudes = []

# Toggle through all the labels.
for label in np.unique(y.values):
    label_idxs = y[y.iloc[:, 0] == label].index

    magnitudes = []
    for idx in label_idxs:
        magnitude = calculate_magnitude(total_acc_x.loc[idx][64:], total_acc_y.loc[idx][64:], total_acc_z.loc[idx][64:])
        magnitudes.extend(magnitude)

    avg_magnitude = np.mean(magnitudes)

    if label in [4, 5, 6]:  # Labels for static activities (sitting, standing, laying)
        static_magnitudes.extend(magnitudes)
    else:
        dynamic_magnitudes.extend(magnitudes)

    plt.figure(figsize=(10, 5))
    plt.title(f'Linear Acceleration Magnitude for Activity {ACTIVITIES[label]}')
    plt.xlabel('Data Samples')
    plt.ylabel('Magnitude')
    plt.plot(magnitudes, label=f'Activity {ACTIVITIES[label]}')
    plt.legend()
    plt.show()

# Calculate and compare average magnitudes for static and dynamic activities
avg_static_magnitude = np.mean(static_magnitudes)
avg_dynamic_magnitude = np.mean(dynamic_magnitudes)

print(f'Average Magnitude for Static Activities: {avg_static_magnitude}')
print(f'Average Magnitude for Dynamic Activities: {avg_dynamic_magnitude}')








   

