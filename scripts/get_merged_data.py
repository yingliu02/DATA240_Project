import os
import zipfile
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

def unzip_data():
    # Define the directory and file paths
    directory = "Datasets"
    zip_file_path = os.path.join("data.zip")

    # Check if the zip file exists
    if os.path.exists(zip_file_path):
        # Check if it's the only file in the directory
        if len(os.listdir(directory)) == 1:
            # Unzip the file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
                logging.info("Data unzipped successfully.")
        else:
            logging.warning("There are other files in the directory.")
    else:
        logging.error("The file 'data.zip' does not exist.")

def get_merged_data():
    # Check that data is unzipped and in the correct location
    unzip_data()
    
    # Load datasets locally
    is_churn_df = pd.read_csv("Datasets/is_churn.csv")
    members_df = pd.read_csv("Datasets/members.csv")
    transactions_df = pd.read_csv("Datasets/transactions.csv")

    # Merge all datasets on 'msno' column
    merged_data = is_churn_df.merge(members_df, on='msno').merge(transactions_df, on='msno')
    # Save the merged dataframe to a CSV file
    merged_data.to_csv('Datasets/merged_data.csv', index=False)
    logging.info("Data merged and saved successfully.")
    
    return merged_data

if __name__ == "__main__":
    get_merged_data()
