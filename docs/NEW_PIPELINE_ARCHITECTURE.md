# California Housing Price Prediction — Pipeline Architecture

## Overview
This project predicts housing prices in California using a custom-built Multi-Layer Perceptron (MLP). The entire machine learning pipeline—from data loading to neural network backpropagation—is implemented from scratch using **pure NumPy** and **Pandas**. No high-level machine learning libraries (like `scikit-learn`, `PyTorch`, or `TensorFlow`) are used for training or preprocessing, making this a strictly foundational implementation.

---

## How to Run

To execute the entire end-to-end pipeline, simply run the master orchestrator script from the root directory:

```bash
python3 run_all.py
```

This script will automatically sequence through data processing, exploratory analysis, model training, and evaluation, populating the `results/` folder with detailed logs, metrics, plots, and saved model weights.

---

## Directory Structure

### `src/` — The Core Machine Learning Engine
Contains the mathematical foundation of the Neural Network built from scratch.
- `activations.py` — Activation functions (`ReLU`, `Sigmoid`, `Linear`) and their derivatives.
- `layers.py` — Neural network layer structures (e.g., `Dense` layer with He initialization and gradient computation).
- `losses.py` — Loss functions (e.g., `MSE` for calculating Mean Squared Error and gradients).
- `optimizer.py` — Optimization algorithms (e.g., `SGD` for updating weights via gradient descent).
- `model.py` — The `MLP` class that pieces together layers, performs forward propagation, and handles the backward pass via chain rule.

### Root Directory — Modular Pipeline Stages
The pipeline is broken down into modular stages corresponding to best-practice ML workflows.

- **`config.py`**  
  Central configuration file containing all magic numbers, hyperparameters (epochs, learning rate), and reproducible random seeds. All scripts import from here.

- **`prepare_data.py`**  
  A central orchestrator for preprocessing. It chains Stages 02 through 09 to output a fully clean, scaled, and encoded dataset ready for the MLP.

- **`stage_02_data_loading.py`**  
  Loads `housing.csv`, logs shape, data types, and missing value counts.

- **`stage_03_eda.py`**  
  Generates univariate histograms, categorical bar charts, correlation heatmaps, and spatial geographic mapping to understand underlying distributions.

- **`stage_04_missing_values.py`**  
  Imputes missing values. E.g., applies Median Imputation to `total_bedrooms` because EDA showed its distribution was highly right-skewed.

- **`stage_05_outliers.py`**  
  Handles outliers. Crucially, it removes censored houses (where price was artificially capped at $500,001) as this invalidates the regression target.

- **`stage_06_scaling.py`**  
  Implements a custom `StandardScaler`. Applies feature standardization and **target scaling** (dividing the target variable by 100,000) to prevent exploding gradients during network training. 

- **`stage_07_encoding.py`**  
  Implements One-Hot Encoding for the `ocean_proximity` feature. Categories are strictly learned from the training set to prevent data leakage.

- **`stage_08_imbalance.py`**  
  Documents that since this is a regression task, categorical class imbalance strategies (like SMOTE) do not apply.

- **`stage_09_feature_engineering.py`**  
  Engineers domain-specific ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) to capture housing density signals better than raw counts.

- **`stage_10_mlp_training.py`**  
  Trains the custom `MLP` model on the processed data, logging validation metrics per epoch and saving the loss curve.

- **`stage_12_evaluation.py`**  
  Computes robust regression metrics (RMSE, MAE, R²) and additionally computes binarized classification metrics (Accuracy, F1) to evaluate discriminatory power at a specific price threshold (e.g. > $250k).

- **`stage_13_visualization.py`**  
  Compiles a `README.txt` in the `plots/` directory to document and interpret every generated visualization.

- **`run_all.py`**  
  The main entry point that executes the pipeline stages sequentially.

---

## Output Architecture (`results/`)

Running the pipeline will generate the following structured folders:

1. **`results/logs/`**  
   Contains text files tracking the detailed console output of every stage (e.g., imputation choices, epoch losses).
2. **`results/plots/`**  
   Contains all generated matplotlib figures. Includes a `README.txt` interpreting each plot.
3. **`results/metrics/`**  
   Stores the final computed accuracy and regression scores for easy review.
4. **`results/models/`**  
   Saves the resulting model parameters (NumPy weights/biases) for future inference or analysis.
