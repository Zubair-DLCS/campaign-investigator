"""
The benchmark and detectors come from Beyond Copy-Paste
(https://github.com/Zubair-DLCS/beyond-copy-paste). This module finds a local copy (set
BCP_DIR to use an existing one), or clones it into third_party/, and makes its modules
importable.
"""
import os, sys, subprocess
from config import ROOT

URL = "https://github.com/Zubair-DLCS/beyond-copy-paste.git"


def path():
    d = os.environ.get("BCP_DIR") or os.path.join(ROOT, "third_party", "beyond-copy-paste")
    if not os.path.isdir(d):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", URL, d], check=True)
    src = os.path.join(d, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return d


path()
