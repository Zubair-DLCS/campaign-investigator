import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RES, FIG
from evaluate import load_all

BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8f8e89"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False})
LABEL = {"one signal: exact_text": "Identical text", "one signal: same_link": "Identical link",
         "one signal: near_duplicate": "Near-identical wording", "one signal: same_meaning": "Same meaning",
         "one signal: synchronized_timing": "Synchronised timing", "all content signals": "All content signals",
         "scripted analyst": "Scripted analyst rule", "oracle cluster choice (ceiling)": "Perfect choice of clusters\n(ceiling, uses ground truth)"}
KINDS = ["quiet", "copy-paste", "ai-rewrite", "mixed"]
KIND_LABEL = {"quiet": "No campaign", "copy-paste": "Copy-paste\ncampaign", "ai-rewrite": "AI-rewritten\ncampaign",
              "mixed": "Two campaigns\n+ cover traffic"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


def method_order(d):
    base = [m for m in LABEL if m in set(d.method) and not m.startswith("oracle")]
    agents = sorted(m for m in set(d.method) if m.startswith("agent"))
    return base + agents + ["oracle cluster choice (ceiling)"]


def fig_overview():
    d = load_all()
    order = method_order(d)
    g = d.groupby("method").agg(f1=("f1", "mean"), fa=("false_accusations", "mean")).loc[order]
    labels = [LABEL.get(m, m.replace("agent: ", "Agent: ")) for m in order]
    cols = [BLUE if m.startswith("one") else VIOLET if m in ("all content signals", "scripted analyst")
            else ORANGE if m.startswith("agent") else AQUA for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))
    axes[0].barh(y, g.f1, color=cols, height=0.62)
    for yi, v in zip(y, g.f1):
        axes[0].text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=9.5)
    axes[0].set_xlim(0, 1.12); axes[0].set_xlabel("F1 over 12 cases (higher is better)")
    axes[0].set_title("Finding the campaign accounts", fontsize=10.5)
    axes[1].barh(y, g.fa, color=cols, height=0.62)
    for yi, v in zip(y, g.fa):
        axes[1].text(v * 1.08 + 0.05, yi, f"{v:.1f}", va="center", fontsize=9.5)
    axes[1].set_xscale("symlog", linthresh=1); axes[1].set_xlim(0, 400)
    axes[1].set_xticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    axes[1].set_xlabel("Ordinary accounts wrongly accused, per case")
    axes[1].set_title("False accusations", fontsize=10.5)
    axes[0].set_yticks(y, labels, fontsize=9); axes[0].invert_yaxis()
    fig.suptitle("Every detector catches campaigns; the hard part is deciding which clusters are campaigns",
                 x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig1_overview.png")


def fig_by_kind():
    d = load_all()
    order = method_order(d)
    M = d.pivot_table(index="method", columns="kind", values="f1", aggfunc="mean").reindex(index=order, columns=KINDS)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#fbeee8", "#eef4fc", "#9cc1ee", BLUE])
    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(order) + 1.6))
    ax.imshow(M.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if v > 0.8 else INK)
    ax.set_xticks(range(len(KINDS)), [KIND_LABEL[k] for k in KINDS], fontsize=9)
    ax.set_yticks(range(len(order)), [LABEL.get(m, m.replace("agent: ", "Agent: ")).replace("\n", " ") for m in order], fontsize=9)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("F1 by kind of case (on a case with no campaign, 1 = nobody accused, 0 = someone accused)",
                 fontsize=10)
    save(fig, "fig2_by_kind.png")


if __name__ == "__main__":
    fig_overview(); fig_by_kind()
