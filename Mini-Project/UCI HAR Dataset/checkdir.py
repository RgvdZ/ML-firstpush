import os

# Define the path to the directory containing the file
dir_path = "/mnt/c/Users/rugve/OneDrive/Pictures/Project/Machine Learning IITGN/es335-assignment1/Mini-Project/UCI HAR Dataset/Combined"

# Correct the file path
file_path = os.path.join(dir_path, "Train", "LAYING", "LAYING_1.csv")

# Replace backslashes with forward slashes (for Unix-based systems)
file_path = file_path.replace("\\", "/")

# Check if the file exists at the specified path
if os.path.isfile(file_path):
    print(f"File found: {file_path}")
else:
    print(f"File not found: {file_path}")
