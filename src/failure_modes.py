"""How the agents went wrong: reads results/agent_*.csv and results/trace_*.jsonl."""
import os, glob, json
import pandas as pd
from config import RES


def signals_consulted(trace):
    out = set()
    for t in trace:
        s = (t.get("args") or {}).get("signal") if isinstance(t.get("args"), dict) else None
        if s:
            out.add(s)
    return out


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "agent_*.csv"))):
        a = pd.read_csv(f)
        model = a.model.iloc[0].split(":", 1)[-1]
        tag = os.path.basename(f)[len("agent_"):-4]
        traces = {json.loads(l)["case"]: json.loads(l) for l in open(os.path.join(RES, f"trace_{tag}.jsonl"))}
        used = {c: signals_consulted(t["trace"]) for c, t in traces.items()}
        q, cp, ai, mx = (a[a.kind == k] for k in ("quiet", "copy-paste", "ai-rewrite", "mixed"))
        rows.append({
            "model": model,
            "mean F1": round(a.f1.mean(), 2),
            "runs that ran out of turns without a report": f"{(~a.finished).sum()} of {len(a)}",
            "no-campaign cases where it accused someone": f"{(q.flagged > 0).sum()} of {len(q)}",
            "ordinary accounts accused per no-campaign case": round(q.false_accusations.mean(), 1),
            "runs that ever consulted the 'same meaning' signal": f"{sum('same_meaning' in used[c] for c in a.case)} of {len(a)}",
            "AI-rewritten campaigns found (of 6: 3 alone, 3 in mixed cases)": int(
                (ai.true_positives >= 10).sum() + (mx.true_positives >= 30).sum()),
            "mixed cases where both campaigns were found": f"{(mx.missed < 10).sum()} of {len(mx)}",
            "mean turns used": round(a.steps.mean(), 1),
        })
    out = pd.DataFrame(rows).set_index("model").T
    out.to_csv(os.path.join(RES, "failure_modes.csv"))
    print(out.to_string())


if __name__ == "__main__":
    main()
