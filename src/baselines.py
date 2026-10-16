"""
Investigations without a language model, to compare the agent against.

  one signal       flag every account in any cluster of that signal (what a single
                   off-the-shelf detector reports)
  all signals      flag every account in any content-signal cluster
  scripted analyst a fixed rule an analyst might write: keep a content cluster only if it is
                   corroborated (at least half its members also sit in a cluster of another
                   content signal) and has at least 5 members
"""
import os
import pandas as pd
from config import RES
from cases import load, case_ids
from context import sif_tau
from tools import Investigation, score, SIGNALS

CONTENT = ["exact_text", "same_link", "near_duplicate", "same_meaning"]


def members(inv, signals):
    out = set()
    for s in signals:
        for c in inv.clusters(s):
            out |= set(c)
    return out


def scripted(inv, min_size=5, corroboration=0.5):
    flagged = set()
    for s in CONTENT:
        others = members(inv, [o for o in CONTENT if o != s])
        for c in inv.clusters(s):
            if len(c) >= min_size and len(set(c) & others) / len(c) >= corroboration:
                flagged |= set(c)
    return flagged


def strategies(inv):
    out = {f"one signal: {s}": members(inv, [s]) for s in SIGNALS}
    out["all content signals"] = members(inv, CONTENT)
    out["scripted analyst"] = scripted(inv)
    return out


def oracle(inv, truth):
    """Ceiling for cluster-level reasoning: keep exactly the content clusters whose members are
    mostly coordinated (uses the ground truth, so it is a reference, not a method)."""
    t, flagged = set(truth), set()
    for s in CONTENT:
        for c in inv.clusters(s):
            if len(set(c) & t) / len(c) > 0.5:
                flagged |= set(c)
    return flagged


def main():
    sif, tau = sif_tau()
    rows = []
    for cid in case_ids():
        df, truth = load(cid)
        inv = Investigation(df, sif, tau)
        strat = strategies(inv)
        strat["oracle cluster choice (ceiling)"] = oracle(inv, truth["coordinated"])
        for name, flagged in strat.items():
            rows.append(dict(case=cid, kind=truth["kind"], method=name, **score(flagged, truth["coordinated"])))
        print(cid)
    r = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    r.to_csv(os.path.join(RES, "baselines.csv"), index=False)
    print(r.groupby(["method", "kind"])[["f1", "false_accusations"]].mean().round(2).unstack())


if __name__ == "__main__":
    main()
