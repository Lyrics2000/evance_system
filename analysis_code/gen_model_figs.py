import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("results.json")); M = R["models"]; classes = R["meta"]["classes"]
BLUE="#2a78d6"; INK="#0b0b0b"; SUB="#52514e"; GRID="#e7e7e3"; SURF="#fcfcfb"; ORANGE="#eb6834"
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"font.family":"DejaVu Sans",
                     "text.color":INK,"xtick.color":SUB,"ytick.color":SUB,"axes.edgecolor":GRID})
key2slug = {"Random Forest":"rf","HMM (corrected)":"hmm","BiLSTM":"lstm","ST-GCN":"stgcn",
            "Transformer":"transformer","DTW 1-NN":"dtw","Hybrid (Two-Stream)":"hybrid"}
for key, slug in key2slug.items():
    d = M[key]; cm = np.array(d["confusion_matrix"])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios":[1,1.1]})
    im = ax[0].imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax[0].set_title(f"Confusion matrix (acc {d['accuracy']:.2f})", fontsize=10, fontweight="bold")
    ax[0].set_xticks(range(len(classes))); ax[0].set_xticklabels(classes, rotation=45, fontsize=8, ha="right")
    ax[0].set_yticks(range(len(classes))); ax[0].set_yticklabels(classes, fontsize=8)
    ax[0].set_xlabel("Predicted", fontsize=8.5); ax[0].set_ylabel("True", fontsize=8.5)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax[0].text(j,i,cm[i,j],ha="center",va="center",fontsize=8.5,
                       color="white" if cm[i,j]>cm.max()/2 else INK)
    # per-class F1
    f1s = [d["per_class_f1"][c] for c in classes]
    ax[1].barh(range(len(classes)), f1s, color=BLUE, zorder=3, height=0.6)
    ax[1].set_yticks(range(len(classes))); ax[1].set_yticklabels(classes, fontsize=9)
    ax[1].invert_yaxis(); ax[1].set_xlim(0,1); ax[1].set_xlabel("F1", fontsize=8.5)
    ax[1].set_title("Per-class F1", fontsize=10, fontweight="bold")
    for i,v in enumerate(f1s):
        ax[1].text(v+0.02,i,f"{v:.2f}",va="center",fontsize=8.5,color=INK)
    ax[1].spines[["top","right"]].set_visible(False); ax[1].grid(axis="x",color=GRID,lw=0.7); ax[1].set_axisbelow(True)
    # per-fold accuracy line
    fig.tight_layout(); fig.savefig(f"figs/m_{slug}.png", dpi=150); plt.close(fig)
    print("ok", slug)

# fold-accuracy comparison strip for appendix (all models)
fig, ax = plt.subplots(figsize=(8.4,3.6))
names=["HMM (corrected)","BiLSTM","ST-GCN","Random Forest","Transformer","DTW 1-NN","Hybrid (Two-Stream)"]
for i,n in enumerate(names):
    fa=M[n]["fold_accuracies"]
    ax.scatter([i]*5, fa, color=BLUE, alpha=0.55, s=42, zorder=3)
    ax.scatter([i],[M[n]["accuracy"]], color=ORANGE, marker="D", s=60, zorder=4)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" (corrected)","").replace(" (Two-Stream)","\nhybrid") for n in names], fontsize=8)
ax.set_ylabel("Accuracy"); ax.axhline(R["meta"]["majority_baseline"], color="#e34948", ls="--", lw=1.1)
ax.set_title("Per-fold accuracy spread (blue) vs pooled OOF accuracy (orange ◆)", fontsize=10, fontweight="bold")
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",color=GRID,lw=0.7); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figs/m_foldspread.png", dpi=150); plt.close(fig)
print("ok foldspread")
