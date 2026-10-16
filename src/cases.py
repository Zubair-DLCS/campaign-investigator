"""
Investigation cases built with the Beyond Copy-Paste benchmark: real text (human Reddit
comments, news headlines, IRA seed messages and AI paraphrases of them), simulated behaviour,
known ground truth.

Each case is a week of posts from about 600 ordinary accounts, with zero, one or two hidden
campaigns of 20 accounts. Campaign content comes only from campaigns 4-7 of the benchmark.
Four kinds of case, three of each:
  quiet        no campaign at all (anything flagged is a false accusation)
  copy-paste   one campaign posting identical text and links in bursts
  ai-rewrite   one campaign using AI paraphrases, unique short links and 60-min timing jitter
  mixed        one copy-paste and one ai-rewrite campaign, both hiding in cover traffic
Account names are replaced by random handles, so names reveal nothing.
"""
import os, json
import numpy as np
import pandas as pd
import bcp  # noqa: F401  (makes the benchmark importable)
from benchmark import Adversary, simulate
from config import CASES, SEED

KINDS = {
    "quiet": [],
    "copy-paste": [("naive", Adversary())],
    "ai-rewrite": [("ai", Adversary(p_para=1, p_edit=1, p_url=1, jitter=60))],
    "mixed": [("naive", Adversary(cover=1)), ("ai", Adversary(p_para=1, p_edit=1, p_url=1, jitter=60, cover=1))],
}
N_ORG, BOTS, DAYS = 600, 20, 7


def handles(n, rng):
    syll = ["ka", "lo", "mi", "ra", "ze", "tu", "no", "vi", "sa", "de", "po", "ri", "ga", "bel", "an", "or"]
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(syll, rng.integers(2, 4))) + str(rng.integers(10, 9999)))
    return list(out)


def build_case(kind, rep):
    rng = np.random.default_rng([SEED, list(KINDS).index(kind), rep])
    camps = rng.choice([4, 5, 6, 7], size=len(KINDS[kind]), replace=False).tolist()
    seed = int(rng.integers(1_000_000))
    org = simulate(Adversary(), campaigns=[], seed=seed, n_org=N_ORG, days=DAYS)
    parts = [org]
    for (tag, adv), c in zip(KINDS[kind], camps):
        d = simulate(adv, campaigns=[c], seed=seed + 1 + c, n_org=0, bots_per_campaign=BOTS, days=DAYS)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).sort_values("ts").reset_index(drop=True)
    names = dict(zip(df.account.unique(), handles(df.account.nunique(), rng)))
    truth = sorted({names[a] for a in df.account[df.campaign >= 0].unique()})
    df["account"] = df.account.map(names)
    case_id = f"{kind}-{rep}"
    return case_id, df[["account", "ts", "text", "url"]], dict(case=case_id, kind=kind,
                                                               coordinated=truth, campaigns=camps)


def build_all(reps=3):
    os.makedirs(CASES, exist_ok=True)
    index = []
    for kind in KINDS:
        for rep in range(reps):
            cid, df, truth = build_case(kind, rep)
            df.to_parquet(os.path.join(CASES, f"{cid}.parquet"), index=False)
            json.dump(truth, open(os.path.join(CASES, f"{cid}.truth.json"), "w"))
            index.append(dict(case=cid, kind=kind, posts=len(df), accounts=df.account.nunique(),
                              coordinated=len(truth["coordinated"])))
            print(index[-1])
    pd.DataFrame(index).to_csv(os.path.join(CASES, "index.csv"), index=False)


def load(case_id):
    df = pd.read_parquet(os.path.join(CASES, f"{case_id}.parquet"))
    truth = json.load(open(os.path.join(CASES, f"{case_id}.truth.json")))
    return df, truth


def case_ids():
    return pd.read_csv(os.path.join(CASES, "index.csv")).case.tolist()


if __name__ == "__main__":
    build_all()
