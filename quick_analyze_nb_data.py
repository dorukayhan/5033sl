import pandas as pd
import util
from sys import argv

if __name__ == "__main__":
    results: pd.DataFrame = pd.read_csv(argv[1])
    print(results.describe()[["ACC", "TPR", "FPR"]])
    util.plot_nb_results(results)
else:
    print("there's nothing to import in here!")