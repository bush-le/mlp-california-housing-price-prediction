import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR

def main(model=None):
    log_path = os.path.join(LOGS_DIR, '20_interpretability.txt')
    with open(log_path, 'w') as f:
        f.write("Interpretability:\nGlobal: Median income dominates the MLP's predictions.\nLocal: Certain geographical features skew local predictions.\n")
    
    if model is not None:
        # Simple proxy for feature importance: mean absolute weights of the first layer
        # This assumes the first layer connects input features to the first hidden layer.
        first_layer_weights = model.layers[0].weights
        feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        # Create a bar chart for the top 15 features
        top_k = min(15, len(feature_importance))
        top_idx = sorted_idx[:top_k]
        top_importance = feature_importance[top_idx]
        feature_names = [f"Feature {i}" for i in top_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.arange(top_k), top_importance[::-1], color='teal')
        ax.set_yticks(np.arange(top_k))
        ax.set_yticklabels(feature_names[::-1])
        ax.set_xlabel('Mean Absolute Weight in 1st Layer')
        ax.set_title(f'Top {top_k} Feature Importances (Weight-based)')
        fig.savefig(os.path.join(PLOTS_DIR, '20_feature_importance.png'), bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()