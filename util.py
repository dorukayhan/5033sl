import pandas as pd
import random
from sklearn.model_selection import train_test_split
from typing import Optional
from ucimlrepo import fetch_ucirepo, dotdict

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