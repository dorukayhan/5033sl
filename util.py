import pandas as pd
import math
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Optional
from ucimlrepo import fetch_ucirepo, dotdict

# THESE MUST MATCH https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset ADDITIONAL VARIABLE INFORMATION
category_feats: dict[str, set[str]] = {
    "cap-shape": set(["b", "c", "x", "f", "s", "p", "o"]),
    "cap-surface": set(["i", "g", "y", "s", "h", "l", "k", "t", "w", "e"]),
    "cap-color": set(["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k"]),
    "does-bruise-or-bleed": set(["t", "f"]),
    "gill-attachment": set(["a", "x", "d", "e", "s", "p", "f"]), # i'm assuming "unknown" is just nan since there's not a single question mark in the dataset
    "gill-spacing": set(["c", "d", "f"]),
    "gill-color": set(["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k", "f"]),
    "stem-root": set(["b", "s", "c", "u", "e", "z", "r"]),
    "stem-surface": set(["i", "g", "y", "s", "h", "l", "k", "t", "w", "e", "f"]),
    "stem-color": set(["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k", "f"]),
    "veil-type": set(["p", "u"]),
    "veil-color": set(["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k", "f"]),
    "has-ring": set(["t", "f"]),
    "ring-type": set(["c", "e", "r", "g", "l", "p", "s", "z", "y", "m", "f"]),
    "spore-print-color": set(["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k"]),
    "habitat": set(["g", "l", "m", "p", "h", "u", "w", "d"]),
    "season": set(["s", "u", "a", "w"])
}
continuous_feats: set[str] = set(["cap-diameter", "stem-height", "stem-width"])
labels: set[str] = set(["e", "p"])

def get_dataset() -> dotdict:
    """downloads the mushroom dataset from uci ml repo"""
    return fetch_ucirepo(id=848)

def split_dataset(dataset: dotdict, tts_seed: Optional[int]=None) -> tuple[dotdict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    3-way splits the given uci ml repo dataset from fetch_ucirepo, returning
    (raw fetch_ucirepo return, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test)
    can reproduce a specific split by specifying tts_seed
    """
    X: pd.DataFrame = dataset.data.features
    y: pd.DataFrame = dataset.data.targets
    tts_seed_real: int = tts_seed if tts_seed else random.randint(0, 2**32 - 1)
    # print(f"splitting dataset, seed={tts_seed_real}")
    # 60% training 20% validation 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=tts_seed_real)
    X_train_real, X_validate, y_train_real, y_validate = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    return dataset, X, y, X_train_real, X_validate, X_test, y_train_real, y_validate, y_test

#                                      get_dataset()        X             y          X_train     X_validate      X_test        y_train     y_validate      y_test
def data_prep(tts_seed: Optional[int]=None) -> tuple[dotdict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    downloads the dataset from uci ml repo and 3-way splits it, returning
    (raw fetch_ucirepo return, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test)
    can reproduce a specific split by specifying tts_seed
    """
    return split_dataset(get_dataset(), tts_seed)
    # print("getting dataset")
    # dataset = get_dataset()
    # X: pd.DataFrame = dataset.data.features
    # y: pd.DataFrame = dataset.data.targets
    # tts_seed_real: int = tts_seed if tts_seed else random.randint(0, 2**32 - 1)
    # print(f"splitting dataset, seed={tts_seed_real}")
    # # 60% training 20% validation 20% testing
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=tts_seed_real)
    # X_train_real, X_validate, y_train_real, y_validate = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    # return dataset, X, y, X_train_real, X_validate, X_test, y_train_real, y_validate, y_test

def gauss(a: float, mean: float, var: float) -> float:
    """gaussian probability evaluator"""
    pay: float = math.exp(-((a - mean) ** 2) / (2 * var))
    payda: float = math.sqrt(2 * math.pi * var)
    return pay / payda

def plot_nb_results(results: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.boxplot(results[["ACC", "TPR", "FPR"]], label=["Accuracy", "TPR", "FPR"])
    ax.set_ylim(0, 1)
    ax.set_xticklabels(["Accuracy", "TPR", "FPR"])
    ax.set(title=f"Metrics of {len(results)} naive Bayes runs with different train-test shuffles", ylabel="Metric value")
    plt.show()

class ConfusionMatrix():
    def __init__(self, TP: int, TN: int, FP: int, FN: int):
        self.data: tuple[int, int, int, int] = (TP, TN, FP, FN)
    
    # HELP I ACCIDENTALLY WROTE ARM ASSEMBLY
    def accuracy(self) -> float:
        TP, TN, FP, FN = self.data
        return (TP + TN) / sum(self.data)
    
    def TPR(self) -> float:
        TP, TN, FP, FN = self.data
        return TP / (TP + FN)
    
    def FPR(self) -> float:
        TP, TN, FP, FN = self.data
        return FP / (FP + TN)
    
    def recall(self) -> float:
        return self.TPR()
    
    def everythingdict(self) -> dict[str, int | float]:
        TP, TN, FP, FN = self.data
        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "ACC": self.accuracy(), "TPR": self.TPR(), "FPR": self.FPR()}