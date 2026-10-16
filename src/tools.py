"""
The investigator's toolbox. Each tool runs a real detector from Beyond Copy-Paste on the case
and returns a short JSON summary an analyst (human or model) can read.

  overview()                       size of the case, busiest texts and links
  find_clusters(signal)            groups of 3+ accounts linked by repeated co-posting
       signal: exact_text | same_link | near_duplicate | same_meaning | synchronized_timing
  inspect_cluster(signal, id)      members, sample posts, overlap with other signals, and the
                                   share of members who also post exact copies
  inspect_account(handle)          activity, sample posts, which clusters it belongs to
  report(clusters, add, remove, explanation)
                                   the final answer: the accounts judged coordinated

Two accounts are linked when they post the same thing (by the signal's definition) within
15 minutes of each other at least twice; one shared viral story is not coordination.
"""
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import bcp  # noqa: F401
from detectors import (layer_exact, layer_url, layer_tfidf, layer_semantic, activity_matrix,
                       timing_from_matrix, MIN_REPEAT)
from common import norm_text

SIGNALS = {"exact_text": "identical text", "same_link": "identical link",
           "near_duplicate": "near-identical wording (character n-grams)",
           "same_meaning": "same meaning in different words (sentence embeddings)",
           "synchronized_timing": "near-identical activity rhythm over the week"}
MAX_LIST = 25


class Investigation:
    def __init__(self, df, sif, tau):
        self.df, self.sif, self.tau = df.reset_index(drop=True), sif, tau
        self.accts = np.array(sorted(df.account.unique()))
        self._clusters, self.calls = {}, []

    # -------------------------------------------------- internals
    def _pairs(self, signal):
        df, a = self.df, self.accts
        if signal == "exact_text":
            return layer_exact(df, a)[1]
        if signal == "same_link":
            return layer_url(df, a)[1]
        if signal == "near_duplicate":
            return layer_tfidf(df, a)[1]
        if signal == "same_meaning":
            return layer_semantic(df, a, sif=self.sif, tau=self.tau)[1]
        if signal == "synchronized_timing":
            M, s = activity_matrix(df, list(a))
            return timing_from_matrix(M, s, list(a))[1]
        raise ValueError(f"unknown signal {signal}")

    def clusters(self, signal):
        if signal not in self._clusters:
            G = nx.Graph()
            for (x, y), c in self._pairs(signal).items():
                if c >= MIN_REPEAT:
                    G.add_edge(x, y, w=c)
            comps = sorted((sorted(c) for c in nx.connected_components(G) if len(c) >= 3),
                           key=len, reverse=True)
            self._clusters[signal] = comps
        return self._clusters[signal]

    def _copy_share(self):
        if not hasattr(self, "_cs"):
            k = self.df.text.map(norm_text)
            n_acc = self.df.assign(k=k).groupby("k").account.nunique()
            dup = k.map(n_acc) > 1
            self._cs = dup.groupby(self.df.account).mean()
        return self._cs

    # -------------------------------------------------- tools
    def overview(self):
        df = self.df
        top_text = (df.assign(k=df.text.map(norm_text)).groupby("k")
                    .agg(text=("text", "first"), accounts=("account", "nunique"))
                    .sort_values("accounts", ascending=False).head(5))
        top_url = (df[df.url != ""].groupby("url").account.nunique().sort_values(ascending=False).head(5))
        return {"posts": len(df), "accounts": len(self.accts), "days": round(df.ts.max() / 86400, 1),
                "posts_with_links": round(float((df.url != "").mean()), 2),
                "most_shared_texts": [{"text": r.text[:120], "accounts": int(r.accounts)} for r in top_text.itertuples()],
                "most_shared_links": [{"link": u, "accounts": int(n)} for u, n in top_url.items()],
                "signals_available": SIGNALS}

    def find_clusters(self, signal):
        cl = self.clusters(signal)
        cs = self._copy_share()
        return {"signal": signal, "meaning": SIGNALS[signal], "n_clusters": len(cl),
                "accounts_in_clusters": int(sum(map(len, cl))),
                "clusters": [{"id": i, "size": len(c),
                              "share_of_members_posting_exact_copies": round(float((cs[c] > 0.2).mean()), 2)}
                             for i, c in enumerate(cl[:15])]}

    def inspect_cluster(self, signal, cluster_id):
        cl = self.clusters(signal)
        if not 0 <= int(cluster_id) < len(cl):
            return {"error": f"{signal} has clusters 0..{len(cl) - 1}"}
        mem = cl[int(cluster_id)]
        d = self.df[self.df.account.isin(mem)]
        other = {}
        for s in SIGNALS:
            if s != signal and s in self._clusters:
                inside = set().union(*map(set, self._clusters[s])) if self._clusters[s] else set()
                other[s] = round(len(set(mem) & inside) / len(mem), 2)
        cs = self._copy_share()
        hours = (d.ts // 3600 % 24).value_counts(normalize=True).sort_index()
        return {"signal": signal, "id": int(cluster_id), "size": len(mem),
                "members": mem[:MAX_LIST] + ([f"... {len(mem) - MAX_LIST} more"] if len(mem) > MAX_LIST else []),
                "posts_by_members": len(d), "posts_per_member": round(len(d) / len(mem), 1),
                "share_of_members_posting_exact_copies": round(float((cs[mem] > 0.2).mean()), 2),
                "members_also_in_clusters_of": other or "run find_clusters on other signals to compare",
                "busiest_hour_share": round(float(hours.max()), 2),
                "sample_posts": d.sample(min(6, len(d)), random_state=0).text.str[:140].tolist()}

    def inspect_account(self, handle):
        d = self.df[self.df.account == handle]
        if d.empty:
            return {"error": "no such account"}
        member_of = [f"{s}#{i}" for s, cl in self._clusters.items() for i, c in enumerate(cl) if handle in c]
        return {"account": handle, "posts": len(d), "with_links": int((d.url != "").sum()),
                "share_of_posts_that_are_exact_copies": round(float(self._copy_share()[handle]), 2),
                "member_of_clusters": member_of or "none found so far",
                "sample_posts": d.head(6).text.str[:140].tolist()}

    def resolve(self, clusters=(), add=(), remove=()):
        flagged = set()
        for item in clusters:
            s, i = (item.split("#") if isinstance(item, str) else item)
            cl = self.clusters(s)
            if 0 <= int(i) < len(cl):
                flagged |= set(cl[int(i)])
        flagged |= {a for a in add if a in set(self.accts)}
        return sorted(flagged - set(remove))

    TOOLS = ["overview", "find_clusters", "inspect_cluster", "inspect_account"]

    def call(self, name, args):
        self.calls.append(name)
        if name not in self.TOOLS:
            return {"error": f"unknown tool {name}; tools: {self.TOOLS + ['report']}"}
        try:
            return getattr(self, name)(**(args or {}))
        except (TypeError, ValueError, KeyError) as e:
            return {"error": str(e)}


def score(flagged, truth):
    f, t = set(flagged), set(truth)
    tp = len(f & t)
    return dict(flagged=len(f), true_positives=tp, false_accusations=len(f - t),
                missed=len(t - f), precision=tp / len(f) if f else (1.0 if not t else 0.0),
                recall=tp / len(t) if t else np.nan,
                f1=(2 * tp / (len(f) + len(t))) if (f or t) else 1.0)
