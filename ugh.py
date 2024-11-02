import os

# Define the directory where the files are located
eod_data_dir = 'data/eod_data/'

# Define the warning message to search for
warning_message = "Data is limited by one year as you have free subscription"

# Loop through each file in the eod_data directory
for filename in os.listdir(eod_data_dir):
    file_path = os.path.join(eod_data_dir, filename)
    
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        # Open and read the file
        with open(file_path, 'r') as file:
            file_contents = file.read()
            
            # Check if the warning message is in the file
            if warning_message in file_contents:
                # If warning is found, delete the file
                os.remove(file_path)
                print(f"Deleted: {filename}")

print("Check and deletion process complete.")
