"""Combine baseline and agent results into one table and per-kind summaries."""
import os, glob
import pandas as pd
from config import RES

ORDER = ["one signal: exact_text", "one signal: same_link", "one signal: near_duplicate",
         "one signal: same_meaning", "one signal: synchronized_timing", "all content signals",
         "scripted analyst"]


def load_all():
    b = pd.read_csv(os.path.join(RES, "baselines.csv"))
    parts = [b]
    for f in sorted(glob.glob(os.path.join(RES, "agent_*.csv"))):
        a = pd.read_csv(f)
        a["method"] = "agent: " + a.model.str.split(":", n=1).str[-1]
        parts.append(a)
    return pd.concat(parts, ignore_index=True)


def main():
    d = load_all()
    overall = d.groupby("method").agg(f1=("f1", "mean"), false_accusations=("false_accusations", "mean"),
                                      missed=("missed", "mean")).round(3)
    by_kind = d.pivot_table(index="method", columns="kind", values=["f1", "false_accusations"], aggfunc="mean").round(3)
    overall.to_csv(os.path.join(RES, "summary_overall.csv"))
    by_kind.to_csv(os.path.join(RES, "summary_by_kind.csv"))
    print(overall.to_string()); print(by_kind.to_string())


if __name__ == "__main__":
    main()
