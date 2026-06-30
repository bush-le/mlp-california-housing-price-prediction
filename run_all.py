from pipeline.stage_02_data_loading import main as run_02
from pipeline.stage_03_eda import main as run_03
from pipeline.stage_04_split import main as run_04
from pipeline.stage_05_missing_values import main as run_05
from pipeline.stage_06_outliers import main as run_06
from pipeline.stage_07_scaling import main as run_07
from pipeline.stage_08_encoding import main as run_08
from pipeline.stage_10_feature_engineering import main as run_10
from pipeline.stage_11_baseline import main as run_11
from pipeline.stage_12_mlp_training import main as run_12
from pipeline.stage_14_tuning import main as run_14
from pipeline.stage_15_evaluation import main as run_15
from pipeline.stage_16_cv import main as run_16
from pipeline.stage_17_test_eval import main as run_17
from pipeline.stage_18_error_analysis import main as run_18
from pipeline.stage_19_visualization import main as run_19
from pipeline.stage_20_interpretability import main as run_20

def run_pipeline():
    print("Running Pipeline...")
    run_02()
    run_03()
    # 04-10 are handled inside prepare_data sequentially but can be called to log
    run_04()
    run_05()
    run_06()
    run_10()
    run_08()
    run_07()
    
    run_11()
    mlp_model = run_12()
    
    run_14()
    run_15(mlp_model)
    run_16()
    run_17()
    run_18()
    run_19()
    run_20()
    
if __name__ == '__main__':
    run_pipeline()
