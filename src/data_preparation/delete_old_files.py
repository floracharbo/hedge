import datetime
import os

folder_path = 'data/other_outputs'  # Replace with the actual folder path

# Get the current date
current_date = datetime.datetime.now()

# Calculate the date one month ago
one_month_ago = current_date - datetime.timedelta(days=30*3)

# Recursively iterate over all files and subdirectories in the folder
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        file_path = os.path.join(root, filename)

        # Check if the file is older than one month
        modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        if modification_time < one_month_ago and root[-4:] != 'outs':
            # Delete the file
            os.remove(file_path)
            # print(f"Delete file: {file_path} modification_time {modification_time}")
