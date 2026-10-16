"""
A small client for any OpenAI-compatible chat API (Groq, Gemini, OpenRouter, OpenAI, a local
Ollama server, ...), with retries, a request-rate limit and an on-disk cache, so an
interrupted run resumes where it stopped and re-runs cost nothing.

Configure with environment variables (or arguments to make_llm):
  LLM_PROVIDER  groq | gemini | openrouter | openai | ollama | mock
  LLM_MODEL     a chat model name that the provider currently lists
  LLM_API_KEY   the provider's API key (not needed for ollama or mock)
  LLM_RPM       maximum requests per minute (default 25)
"""
import os, json, time, hashlib, re
import requests

BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
}


class ChatLLM:
    def __init__(self, provider, model, api_key=None, rpm=25, cache_path=None, max_tokens=200):
        self.provider, self.model = provider, model
        self.url = BASE_URLS[provider].rstrip("/") + "/chat/completions"
        self.key = api_key or os.environ.get("LLM_API_KEY", "")
        self.interval = 60.0 / max(1, rpm)
        self.last = 0.0
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf8"):
                r = json.loads(line)
                self.cache[r["key"]] = r["reply"]

    @property
    def name(self):
        return f"{self.provider}:{self.model}"

    def _key(self, messages):
        return hashlib.sha256(json.dumps([self.name, messages], sort_keys=True).encode()).hexdigest()

    def chat(self, messages):
        k = self._key(messages)
        if k in self.cache:
            return self.cache[k]
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "max_tokens": self.max_tokens}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        for attempt in range(8):
            wait = self.interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = requests.post(self.url, headers=headers, json=body, timeout=120)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue
            if r.status_code == 200:
                reply = r.json()["choices"][0]["message"]["content"] or ""
                break
            if r.status_code in (429, 500, 502, 503, 504):
                retry = r.headers.get("retry-after")
                time.sleep(float(retry) if retry and retry.replace(".", "").isdigit() else 2 ** attempt * 2)
                continue
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        else:
            raise RuntimeError("too many failed attempts")
        self.cache[k] = reply
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf8") as f:
                f.write(json.dumps({"key": k, "reply": reply}) + "\n")
        return reply


class MockLLM:
    """A deliberately gullible stand-in used only by the tests: it obeys any 'classify ... as
    include/exclude' sentence it sees, otherwise matches the review topic words."""
    name = "mock:gullible"

    def chat(self, messages):
        sys_msg, user = messages[0]["content"].lower(), messages[-1]["content"].lower()
        user = user.replace("^", " ")
        m = re.search(r"classify (?:this paper|it) as (include|exclude)", user)
        if m and "never follow instructions" not in sys_msg:
            d = m.group(1)
        else:
            topic = re.search(r"topic: (.*)", sys_msg)
            words = [w for w in (topic.group(1) if topic else "").split() if len(w) > 4]
            d = "include" if sum(w in user for w in words) >= 2 else "exclude"
        return json.dumps({"decision": d, "reason": "mock"})


def make_llm(provider=None, model=None, api_key=None, rpm=None, cache_path=None):
    provider = provider or os.environ.get("LLM_PROVIDER", "mock")
    if provider == "mock":
        return MockLLM()
    model = model or os.environ["LLM_MODEL"]
    rpm = rpm or int(os.environ.get("LLM_RPM", 25))
    return ChatLLM(provider, model, api_key, rpm, cache_path)
