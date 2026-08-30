"""
Unified 5-fold stratified cross-validation for every model on identical folds.
Produces real, comparable metrics + out-of-fold predictions for confusion
matrices, and trains the hybrid model. Results -> results.json
"""
import json, time, numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import common, models as M

t0 = time.time()
X, y, classes = common.load_sequences()          # (67,60,1629)
Xg, yg, _ = common.load_graph()                   # (67,60,75,3)
assert np.array_equal(y, yg), "sequence/graph label order mismatch"
lens = common.seq_lengths(X)
N, T, D = X.shape
cls_idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([cls_idx[c] for c in y])
A = torch.tensor(M.build_adjacency(75))
print(f"N={N} T={T} D={D} classes={classes}")
vals, counts = np.unique(y, return_counts=True)
print("class counts:", dict(zip(vals.tolist(), counts.tolist())))
majority = counts.max() / N
print(f"majority-class baseline = {majority:.4f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=common.SEED)

MODEL_NAMES = ["Random Forest", "HMM (naive)", "HMM (corrected)", "DTW 1-NN",
               "BiLSTM", "ST-GCN", "Transformer",
               "Hybrid (Two-Stream)", "Hybrid (Soft-Vote)"]
oof = {m: np.empty(N, dtype=object) for m in MODEL_NAMES}
fold_acc = {m: [] for m in MODEL_NAMES}

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    ft = time.time()
    Xtr, Xte = X[tr], X[te]
    ltr, lte = lens[tr], lens[te]
    ytr, yte = y[tr], y[te]
    ytr_i = y_idx[tr]

    # ---- shared scaling for deep landmark models ----
    Xtr_s, Xte_s = M._scale_seq(Xtr, Xte)
    seq_tr = torch.tensor(Xtr_s); seq_te = torch.tensor(Xte_s)
    len_tr = torch.tensor(ltr); len_te = torch.tensor(lte)
    mask_tr = M._mask(ltr, T); mask_te = M._mask(lte, T)
    land_tr = seq_tr[:, :, :225]; land_te = seq_te[:, :, :225]   # face-dropped

    # ---- graph scaling ----
    gsc = StandardScaler().fit(Xg[tr].reshape(-1, 75 * 3))
    gtf = lambda Z: gsc.transform(Z.reshape(-1, 75 * 3)).reshape(Z.shape[0], T, 75, 3).astype(np.float32)
    g_tr = torch.tensor(gtf(Xg[tr])).permute(0, 3, 1, 2)   # (N,3,T,75)
    g_te = torch.tensor(gtf(Xg[te])).permute(0, 3, 1, 2)

    # ---------- classical models ----------
    oof["Random Forest"][te] = M.run_random_forest(Xtr, ytr, Xte, ltr, lte)
    oof["HMM (naive)"][te]   = M.run_hmm_naive(Xtr, ytr, Xte, ltr, lte, classes)
    oof["HMM (corrected)"][te] = M.run_hmm(Xtr, ytr, Xte, ltr, lte, classes)
    oof["DTW 1-NN"][te]      = M.run_dtw(Xtr, ytr, Xte, ltr, lte, classes)

    # ---------- deep models ----------
    torch.manual_seed(common.SEED)
    lstm = M.BiLSTM(D, len(classes))
    M._train_torch(lstm, lambda m: m(seq_tr, len_tr), ytr_i, n_epochs=35)
    lstm.eval()
    with torch.no_grad():
        p_lstm = torch.softmax(lstm(seq_te, len_te), 1).numpy()

    torch.manual_seed(common.SEED)
    stg = M.STGCN(len(classes), A)
    M._train_torch(stg, lambda m: m(g_tr), ytr_i, n_epochs=35)
    stg.eval()
    with torch.no_grad():
        p_stg = torch.softmax(stg(g_te), 1).numpy()

    torch.manual_seed(common.SEED)
    trf = M.TransformerEnc(D, len(classes))
    M._train_torch(trf, lambda m: m(seq_tr, mask_tr), ytr_i)
    trf.eval()
    with torch.no_grad():
        p_trf = torch.softmax(trf(seq_te, mask_te), 1).numpy()

    torch.manual_seed(common.SEED)
    hyb = M.HybridNet(len(classes), A)
    M._train_torch(hyb, lambda m: m(g_tr, land_tr, mask_tr), ytr_i, n_epochs=45)
    hyb.eval()
    with torch.no_grad():
        p_hyb = torch.softmax(hyb(g_te, land_te, mask_te), 1).numpy()

    p_ens = (p_lstm + p_stg + p_trf) / 3.0

    inv = np.array(classes)
    oof["BiLSTM"][te]              = inv[p_lstm.argmax(1)]
    oof["ST-GCN"][te]             = inv[p_stg.argmax(1)]
    oof["Transformer"][te]        = inv[p_trf.argmax(1)]
    oof["Hybrid (Two-Stream)"][te] = inv[p_hyb.argmax(1)]
    oof["Hybrid (Soft-Vote)"][te]  = inv[p_ens.argmax(1)]

    line = []
    for m in MODEL_NAMES:
        acc = (oof[m][te] == yte).mean()
        fold_acc[m].append(float(acc))
        line.append(f"{m[:11]}={acc:.2f}")
    print(f"fold {fold}/5 tr={len(tr)} te={len(te)} [{time.time()-ft:.0f}s] " + " ".join(line))

# ---------- aggregate OOF metrics ----------
results = {"meta": {"n_samples": int(N), "classes": classes,
                    "majority_baseline": float(majority),
                    "cv": "5-fold stratified, seed 42",
                    "reported_single_holdout": {
                        "Random Forest": {"accuracy": 0.350, "weighted_f1": 0.312},
                        "HMM (naive)": {"accuracy": 0.235, "weighted_f1": 0.090},
                        "BiLSTM": {"accuracy": 0.353, "weighted_f1": 0.297},
                        "ST-GCN": {"accuracy": 0.412, "weighted_f1": 0.293},
                        "Transformer": {"accuracy": 0.529, "weighted_f1": 0.482}}},
           "models": {}}
for m in MODEL_NAMES:
    met = common.classification_metrics(y, oof[m].astype(str), classes)
    met["fold_accuracies"] = fold_acc[m]
    met["fold_acc_mean"] = float(np.mean(fold_acc[m]))
    met["fold_acc_std"] = float(np.std(fold_acc[m]))
    results["models"][m] = met
    print(f"{m:22s} acc={met['accuracy']:.4f} f1={met['weighted_f1']:.4f} "
          f"wer={met['wer']:.4f} classes_pred={met['distinct_classes_predicted']} "
          f"foldSD={met['fold_acc_std']:.3f}")

json.dump({"oof": {m: oof[m].astype(str).tolist() for m in MODEL_NAMES},
           "y_true": y.tolist()},
          open("/home/claude/ksl/oof_preds.json", "w"))
json.dump(results, open("/home/claude/ksl/results.json", "w"), indent=2)
print(f"\nTotal time {time.time()-t0:.0f}s -> results.json")
