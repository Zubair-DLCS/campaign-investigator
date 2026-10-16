"""Sentence-embedding model and the label-free semantic threshold, built once per session.
The threshold is calibrated on a simulation with no campaigns at all (as in Beyond Copy-Paste),
so no ground truth is used."""
import os, json
import numpy as np
import bcp  # noqa: F401
from config import DATA

_CTX = {}


def sif_tau():
    if "sif" not in _CTX:
        from benchmark import NAIVE, simulate, _pools
        from embed import SIF
        from detectors import calibrate_semantic
        chat, heads, camps = _pools()
        msgs = [m["seed"] for c in camps for m in c["messages"]] + \
               [p for c in camps for m in c["messages"] for p in m["para"]]
        sif = SIF(chat + heads + msgs)
        p = os.path.join(DATA, "semantic_threshold.json")
        if os.path.exists(p):
            tau = json.load(open(p))["tau"]
        else:
            org = simulate(NAIVE, campaigns=[], seed=99)
            tau = calibrate_semantic(org.text.tolist(), sif, np.random.default_rng(0))
            json.dump({"tau": tau}, open(p, "w"))
        _CTX.update(sif=sif, tau=tau)
    return _CTX["sif"], _CTX["tau"]
