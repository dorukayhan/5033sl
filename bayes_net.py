import numpy as np
import pandas as pd
import util
from math import prod
from typing import Callable

def main(argv):
    # TODO
    pass

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
        denom: int = (self.counts[*query].sum() if query else self.counts.sum()) + 1
        return {
            tgt: (get_count(tgt) + (1/len(self.domains[self.target]))) / denom
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
            stat: self.stats[stat][self.target][*query]
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
    
    def create_nbeval(self, **kwargs) -> Callable[[pd.Series], str]:
        """returns function to pass to X_test.apply to get predictions"""
        def nbeval(instance: pd.Series) -> str:
            # TODO
            pass
        return nbeval

if __name__ == "__main__":
    import sys
    main(sys.argv)