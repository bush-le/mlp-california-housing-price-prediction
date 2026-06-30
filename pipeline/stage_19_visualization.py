import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PLOTS_DIR
def main():
    with open(os.path.join(PLOTS_DIR, 'README.txt'), 'w') as f:
        f.write("This directory contains all plots generated during the pipeline.\n")
if __name__ == '__main__':
    main()