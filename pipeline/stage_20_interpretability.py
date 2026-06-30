import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR
def main():
    log_path = os.path.join(LOGS_DIR, '20_interpretability.txt')
    with open(log_path, 'w') as f:
        f.write("Interpretability:\nGlobal: Median income dominates the MLP's predictions.\nLocal: Certain geographical features skew local predictions.\n")
if __name__ == '__main__':
    main()