import math
import numpy as np
import pandas as pd
import random
import util
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from tqdm.auto import tqdm
from util import ConfusionMatrix

# tqdm.pandas()

def main(argv: list[str]=[]):
    # get and 3-way split dataset
    print("getting dataset")
    mushrooms = util.get_dataset()
    if len(argv) > 1:
        random.seed(int(argv[1]))
    results: list[dict[str, int | float]] = []
    run_count: int = 50
    print(f"splitting dataset {run_count} different ways to train nb")
    with ProcessPoolExecutor() as executor:
        results = [
            ConfusionMatrix(TP, TN, FP, FN).everythingdict() for TP, TN, FP, FN
            in tqdm(executor.map(NB, [mushrooms]*run_count, random.sample(range(2**32), run_count)), total=run_count)
        ]
    result_df: pd.DataFrame = pd.DataFrame(results)
    result_df.to_csv("naive_bayes.csv", mode="w", index=False)
    util.plot_nb_results(result_df)
    print(result_df)
    print(result_df.describe()[["ACC", "TPR", "FPR"]])

def NB(mushrooms: util.dotdict, tts_seed: int) -> tuple[int, int, int, int]:
    # 3-way split dataset
    _, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test = util.split_dataset(mushrooms, tts_seed)
    # go back to 2-way split because naive bayes doesn't have any hyperparameters???
    X_train_full: pd.DataFrame = pd.concat([X_train, X_validate])
    y_train_full: pd.DataFrame = pd.concat([y_train, y_validate])
    trainset: pd.DataFrame = pd.concat([X_train_full, y_train_full], axis="columns", join="inner") # match instances w/ labels in one df
    if len(X_train_full) != len(y_train_full) != len(trainset):
        raise AssertionError(f"training dataframes have different lengths? xtf: {len(X_train_full)}, ytf: {len(y_train_full)}, combined: {len(trainset)}")
    # now we begin
    # figure out label probabilities. labels are balanced-ish so no need to smooth them
    label_counts: pd.Series = y_train_full.groupby("class").size()
    p_label: dict[str, Fraction] = {label: Fraction(label_counts[label], label_counts.sum()) for label in util.labels}
    # print(f"P(label) for each label: {p_label} (lc.sum() = {label_counts.sum()})")
    # figure out features' frequencies per label
    p_feat: dict[str, dict[str, dict[str, float]]] = {} # p_feat[label][feature][value] is P(feature=value|label). value can't be nan
    for label in util.labels:
        p_feat[label] = {} # dict[str, dict[str, float]]
        train_given_label: pd.DataFrame = trainset.loc[trainset["class"] == label]
        # category features - find P by just counting
        for feature in util.category_feats:
            # count the categories (except nans and values not listed in the dataset's data sheet) and their occurrences
            value_counts: pd.Series = train_given_label.groupby(feature).size().filter(items=util.category_feats[feature])
            p_feat[label][feature] = { # yay collection comprehensions!
                # apply laplace smoothing here
                # denom is value_counts.sum() instead of len(train_given_label) to ignore missing values
                value: (value_counts.get(value, default=0) + (1/len(util.category_feats[feature]))) / (value_counts.sum() + 1)
                for value in util.category_feats[feature]
            }
            # print(f"{value_counts.sum()} instances w/ {feature}")
            # print(f"P({feature}|{label}) = {p_feat[label][feature]}")
        # continuous features - assume a normal distribution to find P
        for feature in util.continuous_feats:
            # ignore missing values again
            p_feat[label][feature] = { # dict[str, float]
                "mean": train_given_label[feature].mean(),
                "var": train_given_label[feature].var(),
                "n": len(train_given_label[feature].dropna())
            }
            # print(f"params for P({feature}|{label}): {p_feat[label][feature]}")
    # "training" done
    # print("training done, testing...")
    def nbeval(instance: pd.Series) -> str:
        logscores: dict[str, float] = {label: math.log(p_label[label]) for label in util.labels}
        notna: pd.Series = instance.notna() # quite slow, avoid spamming in the continuous feature loop
        for label in logscores:
            # category features: P is right there in p_feat
            for feature in util.category_feats:
                value = instance[feature]
                if value in util.category_feats[feature]:
                    logscores[label] += math.log(p_feat[label][feature][value])
            # continuous features: util.gauss gives P
            for feature in util.continuous_feats:
                if notna[feature]:
                    logscores[label] += math.log(util.gauss(instance[feature], p_feat[label][feature]["mean"], p_feat[label][feature]["var"]))
        # argmax
        return max(logscores.keys(), key=logscores.get)
    def random_nbeval(instance: pd.Series) -> str: # intentionally left from testing
        return random.choice(list(util.labels))
    y_pred: pd.Series = X_test.apply(nbeval, axis="columns", result_type="reduce")
    result: pd.DataFrame = y_pred.compare(y_test["class"], keep_shape=True, keep_equal=True)
    TP: int = len(result.loc[(result["self"] == "e") & (result["other"] == "e")])
    TN: int = len(result.loc[(result["self"] == "p") & (result["other"] == "p")])
    FP: int = len(result.loc[(result["self"] == "e") & (result["other"] == "p")])
    FN: int = len(result.loc[(result["self"] == "p") & (result["other"] == "e")])
    return TP, TN, FP, FN

if __name__ == "__main__":
    import sys
    main(sys.argv)