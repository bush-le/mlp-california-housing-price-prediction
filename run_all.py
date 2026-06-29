"""
Execute the entire modular ML Pipeline sequentially.
"""
import os
import sys

from pipeline.stage_02_data_loading import main as run_stage_02
from pipeline.stage_03_eda import main as run_stage_03
from pipeline.stage_04_missing_values import main as run_stage_04
from pipeline.stage_05_outliers import main as run_stage_05
from pipeline.stage_06_scaling import main as run_stage_06
from pipeline.stage_07_encoding import main as run_stage_07
from pipeline.stage_08_imbalance import main as run_stage_08
from pipeline.stage_09_feature_engineering import main as run_stage_09
from pipeline.stage_10_mlp_training import main as run_stage_10
from pipeline.stage_12_evaluation import main as run_stage_12
from pipeline.stage_13_visualization import main as run_stage_13

def run_pipeline():
    print("\n" + "="*80)
    print("  CALIFORNIA HOUSING PRICE PREDICTION PIPELINE (NUMPY SCRATCH)")
    print("="*80)
    
    # Preprocessing (run independently to generate logs and plots)
    run_stage_02()
    run_stage_03()
    run_stage_04()
    run_stage_05()
    run_stage_06()
    run_stage_07()
    run_stage_08()
    run_stage_09()
    
    # Models
    mlp_model = run_stage_10()
    
    # Evaluation & Viz
    run_stage_12(mlp_model=mlp_model, gmm_model=None)
    run_stage_13()
    
    print("\n" + "="*80)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("  Check the results/ directory for logs, plots, metrics, and models.")
    print("="*80)

if __name__ == "__main__":
    run_pipeline()
