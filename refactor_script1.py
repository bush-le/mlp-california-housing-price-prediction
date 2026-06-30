import os

base_dir = "/home/bush/Desktop/mlp-california-housing-price-prediction/pipeline"

S4 = """
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_PATH, LOGS_DIR, PLOTS_DIR, TEST_SIZE, RANDOM_SEED, TARGET_COL
from src.preprocessing import train_test_split
np.random.seed(RANDOM_SEED)

def split_data(df, log_fn=print):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train
    
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    
    log_fn(f"  Train shape: {train_df.shape}")
    log_fn(f"  Test shape:  {test_df.shape}")
    
    return train_df, test_df

def main():
    df = pd.read_csv(RAW_DATA_PATH)
    log_path = os.path.join(LOGS_DIR, '04_split.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\\n')
    p("=" * 60)
    p("  STAGE 04 — TRAIN/TEST SPLIT (LEAKAGE BOUNDARY)")
    p("=" * 60)
    train_df, test_df = split_data(df, log_fn=p)
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(train_df[TARGET_COL].dropna(), bins=50, alpha=0.5, label='Train', density=True)
    ax.hist(test_df[TARGET_COL].dropna(), bins=50, alpha=0.5, label='Test', density=True)
    ax.set_title('Target Distribution (Train vs Test)')
    ax.legend()
    plot_path = os.path.join(PLOTS_DIR, '04_split_balance.png')
    fig.savefig(plot_path)
    plt.close(fig)
    p(f"\\n  [✓] Split distribution plot saved -> {plot_path}")
    log.close()
    return train_df, test_df

if __name__ == '__main__':
    main()
"""

S5 = """
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, RANDOM_SEED
np.random.seed(RANDOM_SEED)

def handle_missing_values(train_df, test_df, log_fn=print):
    train_df = train_df.copy()
    test_df = test_df.copy()
    missing = train_df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    
    if len(cols_with_missing) == 0:
        log_fn("No missing values found in train set.")
        return train_df, test_df

    log_fn(f"Columns with missing values (based on train set):")
    for col, n in cols_with_missing.items():
        dtype = train_df[col].dtype
        pct = n / len(train_df) * 100
        if np.issubdtype(dtype, np.number):
            skew = train_df[col].dropna().skew()
            if abs(skew) < 0.5:
                strategy = 'mean'
                fill_val = train_df[col].mean()
            else:
                strategy = 'median'
                fill_val = train_df[col].median()
            log_fn(f"  {col:25s}  {n:5d} missing ({pct:.2f}%)  skew={skew:+.2f} → {strategy} (fill={fill_val:.2f})")
        else:
            mode_val = train_df[col].mode()[0]
            fill_val = mode_val
            log_fn(f"  {col:25s}  {n:5d} missing ({pct:.2f}%)  → mode (fill='{mode_val}')")
            
        train_df[col] = train_df[col].fillna(fill_val)
        test_df[col] = test_df[col].fillna(fill_val)
        
    return train_df, test_df

def main():
    from stage_04_split import main as run_split
    train_df, test_df = run_split()
    log_path = os.path.join(LOGS_DIR, '05_imputation.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\\n')
    p("=" * 60)
    p("  STAGE 05 — MISSING VALUE HANDLING")
    p("=" * 60)
    train_df, test_df = handle_missing_values(train_df, test_df, log_fn=p)
    log.close()
    return train_df, test_df

if __name__ == '__main__':
    main()
"""

S6 = """
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, NUM_FEATURES, CAP_VALUE, IQR_MULTIPLIER, RANDOM_SEED
np.random.seed(RANDOM_SEED)

def iqr_bounds(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - multiplier * IQR, Q3 + multiplier * IQR

def detect_outliers(df, num_cols, multiplier=1.5):
    rows = []
    for col in num_cols:
        data = df[col].dropna()
        lower, upper = iqr_bounds(data, multiplier)
        n_low = int((data < lower).sum())
        n_high = int((data > upper).sum())
        rows.append({
            'Feature': col, 'Min': data.min(), 'Q1': data.quantile(0.25),
            'Median': data.median(), 'Q3': data.quantile(0.75), 'Max': data.max(),
            'Lower Fence': lower, 'Upper Fence': upper,
            'Total Outliers': n_low + n_high, '% Outliers': round((n_low + n_high) / len(data) * 100, 2)
        })
    return pd.DataFrame(rows).set_index('Feature')

def plot_boxplots(df, num_cols, summary, save_path):
    n = len(num_cols)
    ncols_grid = 3
    nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, 4 * nrows_grid))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors
    for i, col in enumerate(num_cols):
        ax = axes[i]
        bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor=colors[i%10], alpha=0.6))
        n_out = int(summary.loc[col, 'Total Outliers'])
        pct = summary.loc[col, '% Outliers']
        ax.set_title(f'{col}\\n({n_out:,} outliers, {pct:.1f}%)', fontsize=9)
        ax.set_xticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Box Plots', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def treat_outliers(train_df, test_df, log_fn=print):
    n_before = len(train_df)
    train_df = train_df[train_df['median_house_value'] < CAP_VALUE].copy()
    n_removed_train = n_before - len(train_df)
    
    n_before_test = len(test_df)
    test_df = test_df[test_df['median_house_value'] < CAP_VALUE].copy()
    n_removed_test = n_before_test - len(test_df)
    
    log_fn(f"  Removed {n_removed_train} censored from train, {n_removed_test} from test (target >= {CAP_VALUE})")
    return train_df, test_df

def main():
    from stage_05_missing_values import main as run_stage_05
    train_df, test_df = run_stage_05()
    log_path = os.path.join(LOGS_DIR, '06_outliers.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\\n')
    p("=" * 60)
    p("  STAGE 06 — OUTLIER DETECTION AND TREATMENT")
    p("=" * 60)
    
    num_cols = [c for c in NUM_FEATURES if c in train_df.columns]
    summary = detect_outliers(train_df, num_cols, IQR_MULTIPLIER) # Train only bounds
    
    before_path = os.path.join(PLOTS_DIR, '06_outliers_before.png')
    plot_boxplots(train_df, num_cols, summary, before_path)
    
    train_df, test_df = treat_outliers(train_df, test_df, log_fn=p)
    
    summary_after = detect_outliers(train_df, num_cols, IQR_MULTIPLIER)
    after_path = os.path.join(PLOTS_DIR, '06_outliers_after.png')
    plot_boxplots(train_df, num_cols, summary_after, after_path)
    
    log.close()
    return train_df, test_df

if __name__ == '__main__':
    main()
"""

S7 = """
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, FEATURE_CLIP_RANGE, TARGET_SCALE_FACTOR, TARGET_COL, RANDOM_SEED
from src.preprocessing import StandardScaler
np.random.seed(RANDOM_SEED)

def scale_features(X_train, X_test, clip_range=(-5, 5), log_fn=print):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check if they are DataFrames
    if hasattr(X_train_scaled, 'values'): X_train_scaled = X_train_scaled.values
    if hasattr(X_test_scaled, 'values'): X_test_scaled = X_test_scaled.values
    
    X_train_scaled = np.clip(X_train_scaled, clip_range[0], clip_range[1])
    X_test_scaled = np.clip(X_test_scaled, clip_range[0], clip_range[1])
    log_fn(f"  Scaled features (StandardScaler, clip {clip_range})")
    return X_train_scaled, X_test_scaled, scaler

def scale_target(y_train, y_test, scale_factor, log_fn=print):
    y_train_scaled = y_train / scale_factor
    y_test_scaled = y_test / scale_factor
    log_fn(f"  Target scaled by / {scale_factor}")
    return y_train_scaled, y_test_scaled

def main():
    log_path = os.path.join(LOGS_DIR, '07_scaling.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\\n')
    p("=" * 60)
    p("  STAGE 07 — FEATURE SCALING")
    p("=" * 60)
    p("  This stage is integrated into prepare_data.py pipeline.")
    log.close()

if __name__ == '__main__':
    main()
"""

S8 = """
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, CATEGORICAL_COL, RANDOM_SEED
np.random.seed(RANDOM_SEED)

def one_hot_encode(X_train_df, X_test_df, cat_col, log_fn=print):
    if cat_col not in X_train_df.columns:
        return X_train_df.values, X_test_df.values, []
    known_categories = sorted(X_train_df[cat_col].dropna().unique().tolist())
    log_fn(f"  Learned categories from train: {known_categories}")
    
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    X_train_df[cat_col] = pd.Categorical(X_train_df[cat_col], categories=known_categories)
    X_test_df[cat_col] = pd.Categorical(X_test_df[cat_col], categories=known_categories)
    
    X_train_encoded = pd.get_dummies(X_train_df, columns=[cat_col], dtype=float)
    X_test_encoded = pd.get_dummies(X_test_df, columns=[cat_col], dtype=float)
    
    return X_train_encoded, X_test_encoded, known_categories

def main():
    log_path = os.path.join(LOGS_DIR, '08_encoding.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\\n')
    p("=" * 60)
    p("  STAGE 08 — ENCODING")
    p("=" * 60)
    p("  This stage is integrated into prepare_data.py pipeline.")
    log.close()

if __name__ == '__main__':
    main()
"""

S10 = """
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, RANDOM_SEED, TARGET_COL
np.random.seed(RANDOM_SEED)

def engineer_features(df, log_fn=print):
    df = df.copy()
    df['rooms_per_household'] = np.where(df['households'] > 0, df['total_rooms'] / df['households'], 0.0)
    df['bedrooms_per_room'] = np.where(df['total_rooms'] > 0, df['total_bedrooms'] / df['total_rooms'], 0.0)
    df['population_per_household'] = np.where(df['households'] > 0, df['population'] / df['households'], 0.0)
    log_fn(f"  Created engineered features.")
    return df

def main():
    from stage_06_outliers import main as run_stage_06
    train_df, test_df = run_stage_06()
    log_path = os.path.join(LOGS_DIR, '10_feature_engineering.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\\n')
    p("=" * 60)
    p("  STAGE 10 — FEATURE ENGINEERING")
    p("=" * 60)
    train_df = engineer_features(train_df, log_fn=p)
    test_df = engineer_features(test_df, log_fn=p)
    log.close()
    return train_df, test_df

if __name__ == '__main__':
    main()
"""

P_DATA = """
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_COL, CATEGORICAL_COL, TARGET_SCALE_FACTOR, FEATURE_CLIP_RANGE
from stage_04_split import split_data
from stage_05_missing_values import handle_missing_values
from stage_06_outliers import treat_outliers
from stage_10_feature_engineering import engineer_features
from stage_08_encoding import one_hot_encode
from stage_07_scaling import scale_features, scale_target

def get_prepared_data():
    from stage_02_data_loading import load_raw_data
    df = load_raw_data()
    # 4. Split FIRST
    train_df, test_df = split_data(df, log_fn=lambda x: None)
    # 5. Missing values
    train_df, test_df = handle_missing_values(train_df, test_df, log_fn=lambda x: None)
    # 6. Outliers
    train_df, test_df = treat_outliers(train_df, test_df, log_fn=lambda x: None)
    # 10. Engineering
    train_df = engineer_features(train_df, log_fn=lambda x: None)
    test_df = engineer_features(test_df, log_fn=lambda x: None)
    
    # 8. Encoding
    X_train_df = train_df.drop(columns=[TARGET_COL])
    X_test_df = test_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].values.reshape(-1, 1)
    y_test = test_df[TARGET_COL].values.reshape(-1, 1)
    
    X_train_enc, X_test_enc, _ = one_hot_encode(X_train_df, X_test_df, CATEGORICAL_COL, log_fn=lambda x: None)
    
    # 7. Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_enc, X_test_enc, clip_range=FEATURE_CLIP_RANGE, log_fn=lambda x: None)
    y_train_scaled, y_test_scaled = scale_target(y_train, y_test, TARGET_SCALE_FACTOR, log_fn=lambda x: None)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler
"""

S11 = """
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, TARGET_SCALE_FACTOR, METRICS_DIR
from prepare_data import get_prepared_data

def main():
    log_path = os.path.join(LOGS_DIR, '11_baseline.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\\n')
    p("=" * 60)
    p("  STAGE 11 — BASELINE MODEL")
    p("=" * 60)
    
    X_train, X_test, y_train, y_test, _ = get_prepared_data()
    mean_pred = np.mean(y_train)
    y_pred = np.full_like(y_test, fill_value=mean_pred)
    
    y_pred_unscaled = y_pred * TARGET_SCALE_FACTOR
    y_test_unscaled = y_test * TARGET_SCALE_FACTOR
    
    rmse = np.sqrt(np.mean((y_test_unscaled - y_pred_unscaled)**2))
    p(f"  Baseline (Mean Prediction) RMSE: ${rmse:,.2f}")
    
    with open(os.path.join(METRICS_DIR, 'baseline_metrics.txt'), 'w') as f:
        f.write(f"RMSE: {rmse}\\n")
    log.close()

if __name__ == '__main__':
    main()
"""

for name, content in [('stage_04_split.py', S4), ('stage_05_missing_values.py', S5), ('stage_06_outliers.py', S6), ('stage_07_scaling.py', S7), ('stage_08_encoding.py', S8), ('stage_10_feature_engineering.py', S10), ('prepare_data.py', P_DATA), ('stage_11_baseline.py', S11)]:
    with open(os.path.join(base_dir, name), 'w') as f:
        f.write(content.strip())
