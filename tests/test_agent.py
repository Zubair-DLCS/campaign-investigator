"""End-to-end check of the agent loop with a scripted stand-in for the language model."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent import parse_action
from run import run


class ScriptedLLM:
    """Runs overview, the four content signals, then reports every cluster of 5+ members."""
    name = "scripted:analyst"
    PLAN = ["overview", "exact_text", "same_link", "near_duplicate", "same_meaning"]

    def chat(self, messages):
        k = sum(m["role"] == "assistant" for m in messages)
        if k == 0:
            return json.dumps({"thought": "start", "tool": "overview", "args": {}})
        if k < len(self.PLAN):
            return 'Next: {"thought": "look", "tool": "find_clusters", "args": {"signal": "%s"}}' % self.PLAN[k]
        cites = []
        for m in messages:
            if m["role"] == "user" and m["content"].startswith("{"):
                o = json.loads(m["content"])
                if "clusters" in o:
                    cites += [f'{o["signal"]}#{c["id"]}' for c in o["clusters"] if c["size"] >= 5]
        return json.dumps({"thought": "done", "report": {"clusters": cites, "explanation": "large clusters"}})


def test_parse_action():
    assert parse_action('x {"tool": "overview", "args": {}} trailing }')["tool"] == "overview"
    assert parse_action("no json") is None


def test_agent_loop():
    with tempfile.TemporaryDirectory() as tmp:
        r = run(llm=ScriptedLLM(), cases=["copy-paste-0", "quiet-0"], res_dir=tmp)
        assert r.finished.all() and (r.steps == 6).all()
        cp = r[r.case == "copy-paste-0"].iloc[0]
        assert cp.true_positives == 20


if __name__ == "__main__":
    test_parse_action(); print("ok parse")
    test_agent_loop(); print("ok loop")
