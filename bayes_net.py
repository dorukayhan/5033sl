import math
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import util
from typing import Callable

tqdm.pandas()

def unnan(maybenan: str | float) -> float:
    if isinstance(maybenan, float) and math.isnan(maybenan):
        return "nan"
    else:
        return maybenan

def main(argv):
    # get and 3-way split dataset
    print("getting dataset")
    if len(argv) > 1:
        random.seed(int(argv[1]))
    mushrooms, X, y, X_train, X_validate, X_test, y_train, y_validate, y_test = util.data_prep()
    correlant: str = random.choice(list(util.category_feats))
    print(f"testing BN w/ {correlant} as correlant")
    bn, y_pred, cm = try_bn(correlant, X_train, y_train, X_validate, y_validate)
    print(cm.everythingdict())

class CPT:
    """conditional probability table for category target"""
    def __init__(self, data: pd.DataFrame, target: tuple[str, set[str]], evidence: dict[str, set[str]]={}):
        self.evidence_names: list[str] = list(evidence.keys())
        # first count everything
        self.counts: pd.Series = data.groupby(self.evidence_names + [target[0]], dropna=False).size()
        # then figure out the actual domains of things
        self.target: str = target[0]
        self.domains: dict[str, set] = {
            attribute: set(self.counts.index.get_level_values(attribute)).union(evidence[attribute])
            for attribute in self.evidence_names
        }
        self.domains[target[0]] = set(self.counts.index.get_level_values(target[0])).union(target[1])
    
    def probs(self, evidence: dict[str, str]={}) -> dict[str, float]:
        """returns P(target|the given evidence) for every known value of target"""
        query: list[str] = [evidence[name] for name in self.evidence_names]
        def get_count(target: str) -> int:
            try:
                # to query a series with a multiindex you pass each index's value in index level order
                # which is easily done using one (1) character
                # argument expansion is an unhinged feature thanks python
                # but it breaks if query is empty due to this cpt not being conditional on anything so we check that with "if query"
                return self.counts[*(query + [target])] if query else self.counts[target]
            except KeyError: # count not recorded, assume 0
                return 0
        # other than that the math is the same as naive bayes training
        denom: int = (self.counts.get(tuple(query), default=pd.Series(dtype=int)).sum() if query else self.counts.sum()) + 1
        return {
            unnan(tgt): (get_count(tgt) + (1/len(self.domains[self.target]))) / denom
            for tgt in self.domains[self.target]
        }

class ContinuousCPT(CPT):
    """conditional probability table for continuous target"""
    def __init__(self, data: pd.DataFrame, target: str, evidence: dict[str, set[str]]):
        self.evidence_names: list[str] = list(evidence.keys())
        # first get the gaussian stats
        self.stats: pd.DataFrame = data.pivot_table(values=target, index=self.evidence_names, aggfunc=["mean", "var"], dropna=False)
        # then get the domains
        self.target: str = target
        self.domains: dict[str, set] = {
            attribute: set(self.stats.index.get_level_values(attribute)).union(evidence[attribute])
            for attribute in self.evidence_names
        }
    
    def probs(self, evidence: dict[str, str]) -> dict[str, float]:
        query: list[str] = [evidence[name] for name in self.evidence_names]
        return {
            stat: self.stats[stat][self.target][query[0] if len(query) == 1 else tuple(query)]
            for stat in ("mean", "var")
        }

class TreeAugmentedNB:
    def __init__(self, data: pd.DataFrame, klass: tuple[str, set[str]], category_feats: dict[str, set[str]], continuous_feats: set[str], correlant: str):
        self.data: pd.DataFrame = data # might need it idk
        self.klass: tuple[str, set[str]] = klass
        self.correlant: str = correlant # every other feature depends on label AND this
        self.continuous_cpts: dict[str, ContinuousCPT] = {
            feature: ContinuousCPT(data, feature, {klass[0]: klass[1], correlant: category_feats[correlant]})
            for feature in continuous_feats
        }
        self.category_cpts: dict[str, CPT] = {} # includes correlant and class
        self.category_cpts[klass[0]] = CPT(data, klass)
        self.category_cpts[correlant] = CPT(data, (correlant, category_feats[correlant]), {klass[0]: klass[1]})
        evidence = {
            klass[0]: klass[1],
            correlant: category_feats[correlant]
        }
        for feature, values in category_feats.items(): # everything other than the correlant
            if feature != correlant:
                self.category_cpts[feature] = CPT(data, (feature, values), evidence)
        self.category_domains: dict[str, set[str]] = category_feats # might need this too
    
    def create_bneval(self) -> Callable[[pd.Series], str]:
        """returns function to pass to X_test.apply to get predictions"""
        category_cpts_class_only: dict[str, CPT] = {
            feat: CPT(self.data, (feat, self.category_domains[feat]), {self.klass[0]: self.klass[1]})
            for feat in self.category_domains if feat != self.correlant
        }
        continuous_cpts_class_only: dict[str, ContinuousCPT] = {
            feat: ContinuousCPT(self.data, feat, {self.klass[0]: self.klass[1]})
            for feat in self.continuous_cpts.keys()
        }
        def bneval(instance: pd.Series) -> str:
            # TODO do naive bayes twice
            # once for correlant, once for class
            # though for correlant we just find the probability of its given value instead of its most likely value
            C: str = self.klass[0] # self.klass[0] is too long to type out constantly
            CORR: str = self.correlant # so is self.correlant
            class_evidence: dict = instance.to_dict()
            correlant_evidence: dict = {k: v for k, v in class_evidence.items() if k != CORR}
            correlant_cpt: CPT = self.category_cpts[CORR]
            # correlant nb - finding P(correlant|C) w/ unknown C so need to do it for every label
            logp_correlant: dict[str, float] = {
                label: math.log(correlant_cpt.probs({C: label})[unnan(class_evidence[CORR])])
                for label in self.klass[1]
            }
            for label in self.klass[1]:
                feat_evidence: dict[str, str] = {C: label, CORR: class_evidence[CORR]}
                # the numerator
                for feature, value in correlant_evidence.items():
                    if feature in self.category_cpts:
                        # print(self.category_cpts[feature].probs(feat_evidence))
                        logp_correlant[label] += math.log(self.category_cpts[feature].probs(feat_evidence)[unnan(value)])
                    else:
                        dist_props: dict[str, float] = self.continuous_cpts[feature].probs(feat_evidence)
                        logp_correlant[label] += math.log(util.gauss(unnan(value), **dist_props))
                # the denominator - divide P(correlant|C) by every P(feature|C)
                for feature, value in correlant_evidence.items():
                    if feature in self.category_cpts:
                        cpt = category_cpts_class_only[feature]
                        logp_correlant[label] -= math.log(cpt.probs({C: label})[unnan(value)])
                    else:
                        cpt = continuous_cpts_class_only[feature]
                        dist_props: dict[str, float] = cpt.probs({C: label})
                        logp_correlant[label] -= math.log(util.gauss(unnan(value), **dist_props))
            # class nb
            logscores: dict[str, float] = {
                label: math.log(self.category_cpts[C].probs()[label]) + logp_correlant[label]
                for label in self.klass[1]
            }
            for label in self.klass[1]:
                feat_evidence: dict[str, str] = {C: label, CORR: class_evidence[CORR]}
                # i'm pretty sure we're supposed to use probabilities given class and correlant?
                # if we summed out the correlant we'd just be canceling out logp_correlant's denominator
                for feature, value in correlant_evidence.items():
                    if feature in self.category_cpts:
                        logscores[label] += math.log(self.category_cpts[feature].probs(feat_evidence)[unnan(value)])
                    else:
                        dist_props: dict[str, float] = self.continuous_cpts[feature].probs(feat_evidence)
                        logscores[label] += math.log(util.gauss(unnan(value), **dist_props))
            # and finally do the prediction with argmax
            return max(logscores.keys(), key=logscores.get)
        return bneval

def try_bn(correlant: str, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[TreeAugmentedNB, pd.Series, util.ConfusionMatrix]:
    data: pd.DataFrame = pd.concat([X_train, y_train], axis="columns", join="inner").fillna("nan")
    bn = TreeAugmentedNB(data, ("class", util.labels), util.category_feats, util.continuous_feats, correlant)
    y_pred: pd.Series = X_test.fillna("nan").progress_apply(bn.create_bneval(), axis="columns", result_type="reduce")
    result: pd.DataFrame = y_pred.compare(y_test["class"], keep_shape=True, keep_equal=True)
    TP: int = len(result.loc[(result["self"] == "e") & (result["other"] == "e")])
    TN: int = len(result.loc[(result["self"] == "p") & (result["other"] == "p")])
    FP: int = len(result.loc[(result["self"] == "e") & (result["other"] == "p")])
    FN: int = len(result.loc[(result["self"] == "p") & (result["other"] == "e")])
    return bn, y_pred, util.ConfusionMatrix(TP, TN, FP, FN)

if __name__ == "__main__":
    import sys
    main(sys.argv)