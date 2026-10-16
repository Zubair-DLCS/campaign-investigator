"""
The Campaign Investigator: a language-model agent that decides which detectors to run, reads
their output, inspects suspicious clusters and accounts, and files a report naming the
accounts it judges coordinated, with an explanation.

One JSON action per turn:
  {"thought": "...", "tool": "find_clusters", "args": {"signal": "same_meaning"}}
  {"thought": "...", "report": {"clusters": ["same_meaning#0"], "add": [], "remove": [],
                                "explanation": "..."}}
A step budget caps the investigation; an agent that runs out has flagged nothing.
"""
import json, re
from tools import Investigation, SIGNALS

SYSTEM = """You are an analyst investigating coordinated inauthentic behaviour on a social platform.
You have one week of posts from several hundred accounts. There may be zero, one or several
coordinated campaigns: groups of accounts that push the same messages at the same time,
possibly rewritten in different words, with different links and spread-out timing.
Ordinary people also share the same popular news stories, which is NOT coordination.
Accusing an ordinary account is a serious error; so is missing a campaign.

Tools (call exactly one per turn):
- overview(): size of the case, most-shared texts and links.
- find_clusters(signal): groups of 3+ accounts that repeatedly posted the same thing within
  15 minutes of each other. signal is one of: {signals}.
- inspect_cluster(signal, cluster_id): members, sample posts, overlap with other signals'
  clusters, and the share of members who also post exact copies of others' posts.
- inspect_account(handle): one account's activity and sample posts.

Reply with ONE JSON object and nothing else, either
{{"thought": "...", "tool": "<name>", "args": {{...}}}}
or, when you have decided,
{{"thought": "...", "report": {{"clusters": ["<signal>#<id>", ...], "add": ["<handle>", ...],
  "remove": ["<handle>", ...], "explanation": "<why these accounts are coordinated>"}}}}
Report an empty cluster list if you find no campaign. You have at most {budget} turns."""


def parse_action(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    raw = m.group(0)
    for cut in range(len(raw), 0, -1):             # tolerate trailing text after the JSON
        if raw[cut - 1] != "}":
            continue
        try:
            return json.loads(raw[:cut])
        except json.JSONDecodeError:
            continue
    return None


def investigate(llm, df, sif, tau, budget=15):
    inv = Investigation(df, sif, tau)
    messages = [{"role": "system", "content": SYSTEM.format(signals=", ".join(SIGNALS), budget=budget)},
                {"role": "user", "content": "Begin the investigation."}]
    trace, seen = [], set()
    for step in range(budget):
        reply = llm.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        act = parse_action(reply)
        if act is None:
            obs = {"error": "reply was not a JSON object; reply with exactly one JSON object"}
        elif "report" in act:
            rep = act["report"] or {}
            cited = [c for c in rep.get("clusters", []) if isinstance(c, str) and "#" in c]
            flagged = inv.resolve(cited, rep.get("add", []), rep.get("remove", []))
            trace.append(dict(step=step, action="report", detail=rep))
            return dict(flagged=flagged, steps=step + 1, tool_calls=list(inv.calls), signals=sorted(seen),
                        cited=cited, cited_unseen=[c for c in cited if c.split("#")[0] not in seen],
                        explanation=str(rep.get("explanation", "")), trace=trace, finished=True)
        else:
            name, args = act.get("tool"), act.get("args") or {}
            obs = inv.call(name, args)
            if name == "find_clusters" and "signal" in args:
                seen.add(args["signal"])
            trace.append(dict(step=step, action=name, args=args))
        messages.append({"role": "user", "content": json.dumps(obs)[:6000]})
    return dict(flagged=[], steps=budget, tool_calls=list(inv.calls), signals=sorted(seen), cited=[], cited_unseen=[],
                explanation="", trace=trace, finished=False)
