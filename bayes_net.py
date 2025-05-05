import numpy as np
import pandas as pd
import util
from math import prod

def main(argv):
    # TODO
    pass

class CPT:
    """conditional probability table for category target"""
    def __init__(self, data: pd.DataFrame, target: tuple[str, set[str]], evidence: dict[str, set[str]]):
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
    
    def probs(self, evidence: dict[str, str]) -> dict[str, float]:
        """returns P(target|the given evidence) for every known value of target"""
        query: list[str] = [evidence[name] for name in self.evidence_names]
        def get_count(target: str) -> int:
            try:
                # to query a series with a multiindex you pass each index's value in index level order
                # which is easily done using one (1) character
                # argument expansion is an unhinged feature thanks python
                return self.counts[*(query + [target])]
            except KeyError: # count not recorded, assume 0
                return 0
        return {
            # other than that the math is the same as naive bayes training
            tgt: (get_count(tgt) + (1/len(self.domains[self.target]))) / (self.counts[*query].sum() + 1)
            for tgt in self.domains[self.target]
        }

class BNNode:
    pass

class BayesNet:
    pass

if __name__ == "__main__":
    import sys
    main(sys.argv)