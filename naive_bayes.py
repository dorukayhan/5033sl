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
    # figure out label probabilities
    p_label: dict[str, Fraction] = {}
    label_counts: pd.Series = y_train_full.groupby("class").size()
    for label in label_counts.index:
        # p_label[label] = int(label_counts[label]) / len(y_train_full)
        p_label[label] = Fraction(label_counts[label], label_counts.sum())
    print(f"P(label) for each label: {p_label}")
    # figure out probabilities of category features given each label
    # continuous features (cap-diameter, stem-height, stem-width) show up as float64s and should be treated differently from category features
    category_feats: pd.Index = X_train_full.select_dtypes(exclude=[np.number]).columns
    continuous_feats: pd.Index = X_train_full.select_dtypes(include=[np.number]).columns
    p_feat: dict[str, dict[str, dict[str, Fraction | float]]] = {} # p_feat[label][feature][value] is P(feature=value|label). value can't be nan
    for label in label_counts.index:
        p_feat[label] = {} # dict[str, dict[str, Fraction]]
        train_given_label: pd.DataFrame = trainset.loc[trainset["class"] == label]
        # category features - find P by just counting
        for feature in category_feats:
            p_feat[label][feature] = {} # dict[str, Fraction]
            # count the categories (except nans) and their occurrences
            value_counts: pd.Series = train_given_label.groupby(feature).size()
            for value in value_counts.index:
                p_feat[label][feature][value] = Fraction(value_counts[value], value_counts.sum())
                print(f"P({feature}={value}|{label}) = {p_feat[label][feature][value]}")
        # continuous features - assume a normal distribution to find P
        for feature in continuous_feats:
            p_feat[label][feature] = { # dict[str, float]
                "mean": train_given_label[feature].mean(),
                "stdev": train_given_label[feature].std()
            }
            print(f"params for P({feature}|{label}): {p_feat[label][feature]}")
    return p_feat

if __name__ == "__main__":
    import sys
    main(sys.argv)