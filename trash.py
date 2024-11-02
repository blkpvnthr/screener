import os
import shutil

# List of tickers from your missing data log
missing_tickers = [
    'FRC', 'SIVB', 'ABMD', 'FBHS', 'TWTR', 'NLSN', 'CTXS', 'DRE', 'CERN', 'PBCT', 
    'INFO', 'XLNX', 'KSU', 'CDAY', 'MXIM', 'ALXN', 'HFC', 'FLIR', 'VAR', 'CXO', 
    'TIF', 'NBL', 'ETFC', 'ADS', 'AGN', 'RTN', 'ARNC', 'XEC', 'WCG', 'VIAB', 'CELG', 
    'TSS', 'APC', 'RHT', 'LLL', 'DWDP', 'CA', 'XL', 'GGP', 'DPS', 'MON', 'WYN', 'CHK', 
    'BCR', 'LVLT', 'SPLS', 'WFM', 'Q', 'BBBY', 'MNK', 'RAI', 'YHOO', 'MJN', 'RE', 
    'FTR', 'LLTC', 'ENDP', 'STJ', 'LM', 'TE', 'CVC', 'BXLT', 'CCE', 'ARG', 'TWC', 
    'SNDK', 'CAM', 'ESV', 'GMCR', 'PCP', 'BRCM', 'ACE', 'CMCSK', 'SIAL', 'HCBK', 
    'JOY', 'HSP', 'DTV', 'FDO', 'KRFT', 'TEG', 'QEP', 'LO', 'WIN', 'DNR', 'AVP', 
    'CFN', 'PETM', 'SWY', 'RDC', 'DISCK', 'FRX', 'LSI', 'WPX', 'JDSU', 'FB', 'JCP', 
    'NYX', 'KORS', 'APOL', 'DF', 'CVH', 'PCS', 'FII'
]

# Path to Trash folder on macOS
trash_path = os.path.expanduser('~/.Trash')

# Destination folder where missing files should be restored
destination_folder = 'data/historical_data/'

# Function to search for a file in the Trash
def search_trash_for_file(filename):
    for root, dirs, files in os.walk(trash_path):
        for file in files:
            if filename.lower() in file.lower():
                print(f"Found {file} in {root}")
                return os.path.join(root, file)
    return None

# Create destination directory if it doesn't exist
#os.makedirs(destination_folder, exist_ok=True)
destination_folder = 'data/historical_data/'
# Iterate through each missing ticker
for ticker in missing_tickers:
    # Create the expected filename (e.g., "FRC_eod_data.csv")
    expected_filename = f"{ticker}_historical_data.csv"
    
    # Search the Trash for the file
    found_file = search_trash_for_file(expected_filename)
    
    if found_file:
        # File found, attempt to restore it
        try:
            destination_file = os.path.join(destination_folder, expected_filename)
            shutil.move(found_file, destination_file)
            print(f"Restored {expected_filename} to {destination_file}")
        except Exception as e:
            print(f"Error moving {expected_filename}: {e}")
    else:
        print(f"{expected_filename} not found in Trash.")

print("Search and restore complete.")

# Define source and destination directories
source_dir = 'data/historical_data/'
destination_dir = 'data/eod_data/'

# Check if the destination directory exists, if not, create it
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Loop through files in the source directory
for filename in os.listdir(source_dir):
    # Check if the file ends with '_eod_data.csv'
    if filename.endswith('_eod_data.csv'):
        # Construct full file path
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        
        # Move the file to the destination directory
        shutil.move(source_file, destination_file)
        print(f"Moved: {filename}")

print("File transfer complete.")