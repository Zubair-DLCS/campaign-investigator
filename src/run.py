"""
Run the agent on every case for one model.

  python src/run.py --provider ollama --model qwen2.5:7b
  python src/run.py --provider groq --model <model>

Writes results/agent_<model>.csv (scores per case) and results/trace_<model>.jsonl (every
action and the final explanation). Model replies are cached, so a rerun resumes.
"""
import os, re, json, argparse, time
import pandas as pd
from config import RES
from cases import load, case_ids
from context import sif_tau
from agent import investigate
from tools import score
from llm import make_llm


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def run(provider=None, model=None, budget=15, cases=None, res_dir=RES, llm=None):
    os.makedirs(res_dir, exist_ok=True)
    if llm is None:
        tag = slug(f"{provider}:{model}")
        llm = make_llm(provider, model, cache_path=os.path.join(res_dir, f"cache_{tag}.jsonl"))
    tag = slug(llm.name)
    sif, tau = sif_tau()
    rows = []
    with open(os.path.join(res_dir, f"trace_{tag}.jsonl"), "w", encoding="utf8") as tr:
        for cid in cases or case_ids():
            t0 = time.time()
            df, truth = load(cid)
            out = investigate(llm, df, sif, tau, budget)
            s = score(out["flagged"], truth["coordinated"])
            rows.append(dict(model=llm.name, case=cid, kind=truth["kind"], steps=out["steps"],
                             finished=out["finished"], n_tool_calls=len(out["tool_calls"]),
                             signals_used=len(out["signals"]),
                             cited_unseen=len(out["cited_unseen"]), **s))
            tr.write(json.dumps(dict(case=cid, **{k: out[k] for k in ("trace", "cited", "explanation", "flagged")})) + "\n")
            print(cid, {k: rows[-1][k] for k in ("steps", "flagged", "false_accusations", "missed", "f1")},
                  f"{time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(res_dir, f"agent_{tag}.csv"), index=False)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider"); ap.add_argument("--model")
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--cases", nargs="+")
    a = ap.parse_args()
    run(a.provider, a.model, a.budget, a.cases)
