import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Tính toán mean và standard deviation dọc theo từng cột (feature).
        """
        # Trích xuất values nếu đầu vào là Pandas DataFrame để đồng bộ tính toán
        if hasattr(X, 'values'):
            X_array = np.asarray(X.values, dtype=float)
        else:
            X_array = np.asarray(X, dtype=float)
            
        self.mean_ = np.mean(X_array, axis=0)
        self.scale_ = np.std(X_array, axis=0)
        return self

    def transform(self, X):
        """
        Thực hiện chuẩn hóa dữ liệu: (X - mean) / std
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler chưa được fit. Hãy gọi hàm fit() trước.")
        
        if hasattr(X, 'values'):
            X_array = np.asarray(X.values, dtype=float)
        else:
            X_array = np.asarray(X, dtype=float)
        
        # Xử lý an toàn tránh lỗi chia cho 0 (DivideByZero)
        scale_safe = np.copy(self.scale_)
        scale_safe[scale_safe == 0.0] = 1.0
        
        X_scaled = (X_array - self.mean_) / scale_safe
        
        # Nếu đầu vào ban đầu là Pandas DataFrame, trả về dạng DataFrame kèm đặt lại index/columns ban đầu
        if hasattr(X, 'columns'):
            import pandas as pd
            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled

    def fit_transform(self, X):
        """
        Tối ưu hóa tốc độ: fit và transform đồng bộ
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Khôi phục lại dữ liệu gốc từ dữ liệu đã scale: (X_scaled * std) + mean
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler chưa được fit.")
            
        if hasattr(X_scaled, 'values'):
            X_scaled_array = np.asarray(X_scaled.values, dtype=float)
        else:
            X_scaled_array = np.asarray(X_scaled, dtype=float)
            
        X_inv = (X_scaled_array * self.scale_) + self.mean_
        
        if hasattr(X_scaled, 'columns'):
            import pandas as pd
            return pd.DataFrame(X_inv, columns=X_scaled.columns, index=X_scaled.index)
            
        return X_inv


def train_test_split(*arrays, test_size=0.25, random_state=None, shuffle=True):
    """
    Hàm chia train/test từ scratch hỗ trợ mượt mà cả Pandas và NumPy.
    
    Tính năng nâng cấp:
    - Sử dụng bộ sinh số ngẫu nhiên cục bộ Generator (np.random.default_rng).
    - Hỗ trợ test_size dưới dạng tỉ lệ (float) hoặc số lượng phần tử cố định (int).
    """
    if len(arrays) == 0:
        raise ValueError("Phải truyền ít nhất một mảng dữ liệu vào hàm.")
        
    n_samples = len(arrays[0])
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("Tất cả các mảng truyền vào phải có độ dài bằng nhau.")

    indices = np.arange(n_samples)

    if shuffle:
        # Sử dụng Generator cục bộ (Modern NumPy) thay vì làm thay đổi seed global
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Xử lý linh hoạt kiểu dữ liệu của test_size
    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("Nếu test_size là float, giá trị phải nằm trong khoảng (0.0, 1.0)")
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        if test_size <= 0 or test_size >= n_samples:
            raise ValueError(f"Nếu test_size là int, giá trị phải lớn hơn 0 và nhỏ hơn tổng số mẫu ({n_samples})")
        n_test = test_size
    else:
        raise ValueError("test_size phải là kiểu dữ liệu float hoặc int.")

    test_indices = indices[:n_test] 
    train_indices = indices[n_test:]

    result = []
    for arr in arrays:
        # Kiểm tra nếu cấu trúc dữ liệu có hỗ trợ iloc (Pandas DataFrame hoặc Series)
        if hasattr(arr, 'iloc'):
            arr_train = arr.iloc[train_indices]
            arr_test = arr.iloc[test_indices]
        else:
            # Nếu là NumPy array hoặc danh sách List thô thông thường
            arr_np = np.asarray(arr)
            arr_train = arr_np[train_indices]
            arr_test = arr_np[test_indices]
            
        result.append(arr_train)
        result.append(arr_test)

    return result
