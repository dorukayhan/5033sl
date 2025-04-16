import numpy as np
import pandas as pd
import util
from sklearn.model_selection import train_test_split

def main(argv: list[str]):
    # get and 3-way split dataset
    mushrooms, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test = util.data_prep(int(argv[1]) if len(argv) > 1 else None)
    

if __name__ == "__main__":
    import sys
    main(sys.argv)