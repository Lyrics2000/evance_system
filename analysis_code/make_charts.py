"""
Comparison visuals for the KSL model study. Follows the dataviz method:
single-hue for magnitude, fixed-order categorical hues, direct value labels,
recessive grid, colorblind-safe validated palette.
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = json.load(open("/home/claude/ksl/results.json"))
L = json.load(open("/home/claude/ksl/lang_results.json"))
OOF = json.load(open("/home/claude/ksl/oof_preds.json"))
import os; os.makedirs("/home/claude/ksl/figs", exist_ok=True)

# validated palette
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; YELLOW="#eda100"
MAG="#e87ba4"; GREEN="#008300"; VIOLET="#4a3aa7"; RED="#e34948"
INK="#0b0b0b"; SUB="#52514e"; GRID="#e7e7e3"; SURF="#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "font.size": 11, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": SUB, "ytick.color": SUB,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
})
classes = R["meta"]["classes"]
base = R["meta"]["majority_baseline"]
M = R["models"]
ORDER = ["HMM (naive)", "Random Forest", "HMM (corrected)", "BiLSTM", "ST-GCN",
         "Transformer", "Hybrid (Soft-Vote)", "DTW 1-NN", "Hybrid (Two-Stream)"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---- 1. Master accuracy ranking (horizontal, single hue, CV std whiskers) ----
def fig_ranking():
    proper = [m for m in ORDER if m != "HMM (naive)"]
    order = sorted(proper, key=lambda m: M[m]["accuracy"])
    accs = [M[m]["accuracy"] for m in order]
    sds = [M[m]["fold_acc_std"] for m in order]
    colors = []
    for m in order:
        if "Hybrid" in m: colors.append(ORANGE)
        elif m == "DTW 1-NN": colors.append(AQUA)
        else: colors.append(BLUE)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    y = np.arange(len(order))
    ax.barh(y, accs, color=colors, height=0.66, zorder=3,
            xerr=sds, error_kw=dict(ecolor=SUB, lw=1.1, capsize=3, alpha=0.7))
    ax.axvline(base, color=RED, lw=1.4, ls="--", zorder=2)
    ax.text(base, len(order)-0.3, f" majority baseline {base:.2f}",
            color=RED, fontsize=9, va="center")
    ax.set_yticks(y); ax.set_yticklabels(order)
    for i, a in enumerate(accs):
        ax.text(a + 0.012, i, f"{a:.3f}", va="center", ha="left",
                color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, max(accs)+0.12); ax.set_xlabel("Out-of-fold accuracy (5-fold CV)")
    ax.set_title("Sign-recognition accuracy on WLASL (5 classes, 67 videos)",
                 fontweight="bold", color=INK, pad=12)
    leg = [Patch(color=BLUE, label="Individual model"),
           Patch(color=AQUA, label="Non-parametric baseline"),
           Patch(color=ORANGE, label="Hybrid model")]
    ax.legend(handles=leg, loc="lower right", frameon=False, fontsize=9)
    style(ax); fig.tight_layout(); fig.savefig("/home/claude/ksl/figs/1_ranking.png", dpi=150)
    plt.close(fig)


# ---- 2. Reported single-holdout vs CV accuracy ----
def fig_holdout_vs_cv():
    rep = R["meta"]["reported_single_holdout"]
    names = ["HMM (naive)", "Random Forest", "BiLSTM", "ST-GCN", "Transformer"]
    r = [rep[n]["accuracy"] for n in names]
    c = [M[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x-w/2, r, w, color=YELLOW, zorder=3, label="Reported (single 17-sample holdout)")
    ax.bar(x+w/2, c, w, color=BLUE, zorder=3, label="This study (5-fold CV, 67 predictions)")
    for i in range(len(names)):
        ax.text(x[i]-w/2, r[i]+0.01, f"{r[i]:.2f}", ha="center", fontsize=8.5, color=INK)
        ax.text(x[i]+w/2, c[i]+0.01, f"{c[i]:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(base, color=RED, lw=1.2, ls="--", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (naive)","") for n in names], fontsize=9.5)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 0.72)
    ax.set_title("Why evaluation method matters: single holdout vs cross-validation",
                 fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=8.7, loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig("/home/claude/ksl/figs/2_holdout_vs_cv.png", dpi=150)
    plt.close(fig)


# ---- 3. Per-class F1 heatmap ----
def fig_perclass():
    order = [m for m in ORDER]
    data = np.array([[M[m]["per_class_f1"][c] for c in classes] for m in order])
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    im = ax.imshow(data, cmap="BuGn", vmin=0, vmax=max(0.7, data.max()), aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9.5)
    for i in range(len(order)):
        for j in range(len(classes)):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.42 else INK)
    ax.set_title("Per-class F1 by model", fontweight="bold", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("F1", color=SUB)
    cb.outline.set_edgecolor(GRID)
    fig.tight_layout(); fig.savefig("/home/claude/ksl/figs/3_perclass_f1.png", dpi=150)
    plt.close(fig)


# ---- 4. Hybrid vs its components ----
def fig_hybrid():
    names = ["DTW 1-NN", "Transformer", "ST-GCN", "BiLSTM",
             "Hybrid (Soft-Vote)", "Hybrid (Two-Stream)"]
    acc = [M[n]["accuracy"] for n in names]
    f1 = [M[n]["weighted_f1"] for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(names)); w = 0.38
    cols = [AQUA if "Hybrid" not in n else ORANGE for n in names]
    ax.bar(x-w/2, acc, w, color=cols, zorder=3, label="Accuracy")
    ax.bar(x+w/2, f1, w, color=[BLUE if "Hybrid" not in n else YELLOW for n in names],
           zorder=3, label="Weighted F1")
    for i in range(len(names)):
        ax.text(x[i]-w/2, acc[i]+0.008, f"{acc[i]:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(x[i]+w/2, f1[i]+0.008, f"{f1[i]:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("Hybrid ","Hyb\n") for n in names], fontsize=8.6)
    ax.set_ylabel("Score"); ax.set_ylim(0, max(max(acc),max(f1))+0.1)
    ax.set_title("Hybrid models vs their strongest components", fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig("/home/claude/ksl/figs/4_hybrid.png", dpi=150)
    plt.close(fig)


# ---- 5. Language metrics ----
def fig_language():
    conds = [("Exact rule-based\n(clean)", L["clean"][0]),
             ("Fuzzy\n(clean)", L["clean"][1]),
             ("Exact rule-based\n(noisy)", L["noisy"][0]),
             ("Fuzzy\n(noisy)", L["noisy"][1]),
             ("Fuzzy nearest-NN\n(generalization)", L["generalization"][0])]
    metrics = ["BLEU", "ROUGEL", "METEOR", "WER"]
    mcol = {"BLEU": BLUE, "ROUGEL": AQUA, "METEOR": VIOLET, "WER": RED}
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(conds)); w = 0.2
    for k, met in enumerate(metrics):
        vals = [c[1][met] for c in conds]
        ax.bar(x + (k-1.5)*w, vals, w, color=mcol[met], zorder=3,
               label=("ROUGE-L" if met=="ROUGEL" else met))
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in conds], fontsize=8.4)
    ax.set_ylabel("Score (0–100)"); ax.set_ylim(0, 108)
    ax.set_title("Gloss→healthcare-text translation quality (BLEU / ROUGE-L / METEOR / WER)",
                 fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig("/home/claude/ksl/figs/5_language.png", dpi=150)
    plt.close(fig)


# ---- 6. Confusion matrices (naive HMM collapse, Transformer, DTW, Hybrid) ----
def fig_confusion():
    picks = ["HMM (corrected)", "ST-GCN", "Transformer", "Hybrid (Two-Stream)"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7))
    for ax, name in zip(axes, picks):
        cm = np.array(M[name]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_title(f"{name}\nacc {M[name]['accuracy']:.2f}", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, fontsize=7, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i,j] > cm.max()/2 else INK)
        if ax is axes[0]: ax.set_ylabel("True", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
    fig.suptitle("Confusion matrices under 5-fold CV — weakest (HMM) to strongest (Hybrid)",
                 fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig("/home/claude/ksl/figs/6_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


for f in [fig_ranking, fig_holdout_vs_cv, fig_perclass, fig_hybrid, fig_language, fig_confusion]:
    f(); print("ok", f.__name__)
print("charts written")
