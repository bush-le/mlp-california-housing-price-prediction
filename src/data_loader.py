import pandas as pd
import os

def load_raw_data(filename='housing.csv'):
    # Lấy đường dẫn tuyệt đối đến file này (data_loader.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Đi ngược ra thư mục gốc dự án, rồi vào data/raw
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, 'data', 'raw', filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}. Hãy kiểm tra lại thư mục data/raw/")
        
    print(f"Đang đọc dữ liệu từ: {file_path}")
    df = pd.read_csv(file_path)
    return df