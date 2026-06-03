import json
import os

# 1. Update 02_data_preprocessing.ipynb
with open('notebooks/02_data_preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb02 = json.load(f)

for cell in nb02['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'X_test = scaler.transform(X_test_raw)' in source and 'pickle.dump(scaler' not in source:
            cell['source'].append('\n# LƯU SCALER ĐỂ SỬ DỤNG CHO DỰ ĐOÁN & PHÂN TÍCH\n')
            cell['source'].append('import pickle\n')
            cell['source'].append('import os\n')
            cell['source'].append('models_dir = os.path.join(\'..\', \'models\')\n')
            cell['source'].append('os.makedirs(models_dir, exist_ok=True)\n')
            cell['source'].append('with open(os.path.join(models_dir, \'scaler.pkl\'), \'wb\') as f:\n')
            cell['source'].append('    pickle.dump(scaler, f)\n')
            cell['source'].append('print("Đã lưu scaler vào models/scaler.pkl")\n')

with open('notebooks/02_data_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb02, f, indent=1, ensure_ascii=False)

# 2. Update 04_model_evaluation.ipynb
with open('notebooks/04_model_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb04 = json.load(f)

for cell in nb04['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'california_map' in source and 'abs_residuals' in source:
            new_source = """import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib.request
import matplotlib.image as mpimg

try:
    # =====================================================================
    # 0. CHUẨN BỊ DỮ LIỆU & TÍNH TOÁN SAI SỐ (BỔ SUNG ĐỂ TRÁNH LỖI)
    # =====================================================================
    processed_dir = os.path.join('..', 'data', 'processed')
    models_dir = os.path.join('..', 'models')
    
    # 1. Load X_test và y_test từ file đã lưu
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    # 2. Load model và tính abs_residuals
    with open(os.path.join(models_dir, 'mlp_model.pkl'), 'rb') as f:
        model = pickle.load(f)
        
    y_pred = model.predict(X_test)
    abs_residuals = np.abs(y_test.flatten() - y_pred.flatten())

    # =====================================================================
    # 1. KHÔI PHỤC TỌA ĐỘ GỐC
    # =====================================================================
    # Load scaler đã lưu từ file 02 lên đây:
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    X_test_raw = scaler.inverse_transform(X_test)
    
    # Cột 0 là Kinh độ (Longitude), Cột 1 là Vĩ độ (Latitude)
    longitude_real = X_test_raw[:, 0]
    latitude_real = X_test_raw[:, 1]
    
    # =====================================================================
    # 2. TẢI VÀ CHUẨN BỊ BẢN ĐỒ NỀN (BASEMAP)
    # =====================================================================
    map_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/images/end_to_end_project/california.png"
    urllib.request.urlretrieve(map_url, "california_map.png")
    california_img = mpimg.imread("california_map.png")
    
    # =====================================================================
    # 3. VẼ BIỂU ĐỒ LỒNG GHÉP
    # =====================================================================
    plt.figure(figsize=(10, 8))
    
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)

    scatter = plt.scatter(longitude_real, latitude_real, c=abs_residuals, cmap='YlOrRd',
                          alpha=0.7, s=30, edgecolors='none')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Độ lớn Sai số (USD)', fontsize=12)
    formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    cbar.ax.yaxis.set_major_formatter(formatter)
    
    plt.title('Bản đồ Sai số Địa lý lồng ghép Thực tế', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Kinh độ thực tế (Longitude)', fontsize=12)
    plt.ylabel('Vĩ độ thực tế (Latitude)', fontsize=12)
    
    plt.style.use('default') 
    plt.grid(False) 
    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(f"❌ LỖI KHÔNG TÌM THẤY FILE: {e.filename}")
    print("👉 Code không tìm thấy biến `scaler`. Bạn cần chạy lại ô có chứa khai báo StandardScaler ở file Notebook 02 để sinh ra file scaler.pkl.")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")
"""
            # Need to format it as list of strings with newlines like standard ipynb format
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip('\n') # remove last newline

with open('notebooks/04_model_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb04, f, indent=1, ensure_ascii=False)

print("Notebooks updated successfully!")
