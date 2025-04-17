import pandas as pd
import math
import random
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
    """downloads the dataset from uci ml repo"""
    return fetch_ucirepo(id=848)

#                                      get_dataset()        X             y          X_train     X_validate      X_test        y_train     y_validate      y_test
def data_prep(tts_seed: Optional[int]=None) -> tuple[dotdict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    downloads the dataset from uci ml repo and 3-way splits it, returning
    (raw fetch_ucirepo return, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test)
    can reproduce a specific split by specifying tts_seed
    """
    print("getting dataset")
    dataset = get_dataset()
    X: pd.DataFrame = dataset.data.features
    y: pd.DataFrame = dataset.data.targets
    tts_seed_real: int = tts_seed if tts_seed else random.randint(0, 2**32 - 1)
    print(f"splitting dataset, seed={tts_seed_real}")
    # 60% training 20% validation 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=tts_seed_real)
    X_train_real, X_validate, y_train_real, y_validate = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    return dataset, X, y, X_train_real, X_validate, X_test, y_train_real, y_validate, y_test

def gauss(a: float, mean: float, variance: float) -> float:
    """gaussian probability evaluator"""
    pay: float = math.exp(-((a - mean) ** 2) / (2 * variance))
    payda: float = math.sqrt(2 * math.pi * variance)
    return pay / payda