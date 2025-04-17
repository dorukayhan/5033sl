import numpy as np
import pandas as pd
import util
from fractions import Fraction

def main(argv: list[str]):
    # get and 3-way split dataset
    mushrooms, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test = util.data_prep(int(argv[1]) if len(argv) > 1 else None)
    # go back to 2-way split because naive bayes doesn't have any hyperparameters???
    X_train_full: pd.DataFrame = pd.concat([X_train, X_validate])
    y_train_full: pd.DataFrame = pd.concat([y_train, y_validate])
    trainset: pd.DataFrame = pd.concat([X_train_full, y_train_full], axis="columns", join="inner") # match instances w/ labels in one df
    if len(X_train_full) != len(y_train_full) != len(trainset):
        raise AssertionError(f"training dataframes have different lengths? xtf: {len(X_train_full)}, ytf: {len(y_train_full)}, combined: {len(trainset)}")
    # now we begin
    # figure out label probabilities
    p_label: dict[str, Fraction] = {}
    label_counts: pd.Series = y_train_full.groupby("class").size()
    for label in util.labels:
        # labels are balanced-ish, no need to smooth them
        p_label[label] = Fraction(label_counts[label], label_counts.sum())
    print(f"P(label) for each label: {p_label}")
    # figure out features' frequencies per label
    p_feat: dict[str, dict[str, dict[str, float]]] = {} # p_feat[label][feature][value] is P(feature=value|label). value can't be nan
    for label in util.labels:
        p_feat[label] = {} # dict[str, dict[str, float]]
        train_given_label: pd.DataFrame = trainset.loc[trainset["class"] == label]
        # category features - find P by just counting
        for feature in util.category_feats.keys():
            p_feat[label][feature] = {} # dict[str, float]
            # count the categories (except nans) and their occurrences
            value_counts: pd.Series = train_given_label.groupby(feature).size()
            for value in util.category_feats[feature]:
                # apply laplace smoothing here
                # denom is value_counts.sum() instead of len(train_given_label) to ignore missing values
                p_feat[label][feature][value] = (value_counts.get(value, default=0) + (1/len(util.category_feats[feature]))) / (value_counts.sum() + 1)
                print(f"P({feature}={value}|{label}) = {p_feat[label][feature][value]}")
            print(f"{value_counts.sum()} instances w/ {feature}")
        # continuous features - assume a normal distribution to find P
        for feature in util.continuous_feats:
            # ignore missing values again
            p_feat[label][feature] = { # dict[str, float]
                "mean": train_given_label[feature].mean(),
                "variance": train_given_label[feature].var(),
                "n": len(train_given_label[feature].dropna())
            }
            print(f"params for P({feature}|{label}): {p_feat[label][feature]}")
    # "training" done, time to evaluate
    def nbeval(instance: pd.Series) -> str:
        import random
        return "e" if random.random() < 0.5 else "p"
    y_pred: pd.Series = X_test.apply(nbeval, axis="columns", result_type="reduce")
    result: pd.DataFrame = y_pred.compare(y_test["class"], keep_shape=True, keep_equal=True)
    TP: int = len(result.loc[(result["self"] == "e") & (result["other"] == "e")])
    TN: int = len(result.loc[(result["self"] == "p") & (result["other"] == "p")])
    FP: int = len(result.loc[(result["self"] == "e") & (result["other"] == "p")])
    FN: int = len(result.loc[(result["self"] == "p") & (result["other"] == "e")])
    print(f"{len(result)} evals, {TP} TPs, {TN} TNs, {FP} FPs, {FN} FNs")
    print(f"accuracy {(TP+TN)/len(result)}, recall/TPR {TP/(TP+FN)}, FPR {FP/(FP+TN)}")
    return p_feat

if __name__ == "__main__":
    import sys
    main(sys.argv)