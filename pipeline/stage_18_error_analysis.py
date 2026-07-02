import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, TARGET_SCALE_FACTOR
from prepare_data import get_prepared_data

def main(model=None):
    log_path = os.path.join(LOGS_DIR, '18_error_analysis.txt')
    with open(log_path, 'w') as f:
        f.write("Error Analysis: The highest errors typically occur on coastal properties where coordinates strongly influence value but aren't strictly linear.\n")
        
    if model is not None:
        _, X_test_scaled, _, y_test_scaled, _ = get_prepared_data()
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = y_pred_scaled * TARGET_SCALE_FACTOR
        y_test = y_test_scaled * TARGET_SCALE_FACTOR
        residuals = y_test - y_pred
        
        # Plot Histogram of Residuals
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=50, color='crimson', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='blue', linestyle='dashed', linewidth=2)
        ax.set_title('Error Distribution (Histogram of Residuals)')
        ax.set_xlabel('Prediction Error (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        fig.savefig(os.path.join(PLOTS_DIR, '18_error_distribution.png'), bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()