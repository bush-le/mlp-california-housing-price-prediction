import pandas as pd
import os

def load_raw_data(filename='housing.csv'):
    # Get the absolute path to this file (data_loader.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up to the project root, then into data/raw
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, 'data', 'raw', filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}. Please check the data/raw directory.")
        
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    return df