"""
Model definitions for the KSL comparison. Every model exposes fit(...) /
predict(...) on the SAME cached features, evaluated under one shared 5-fold CV.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
import warnings
from common import POSE, LH, RH, FACE, L_SHOULDER, R_SHOULDER, SEED

warnings.filterwarnings("ignore")
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cpu"


# ===========================================================================
# 1. RANDOM FOREST  — non-temporal statistical baseline
# ===========================================================================
def rf_pool(X, lens):
    """Collapse each variable-length sequence to mean/std/min/max -> 6516-D."""
    feats = []
    for i in range(X.shape[0]):
        s = X[i, :lens[i]]                     # (T,1629)
        feats.append(np.concatenate([s.mean(0), s.std(0), s.min(0), s.max(0)]))
    return np.asarray(feats, dtype=np.float32)


def run_random_forest(Xtr, ytr, Xte, ltr, lte):
    Ftr, Fte = rf_pool(Xtr, ltr), rf_pool(Xte, lte)
    sc = StandardScaler().fit(Ftr)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    clf.fit(sc.transform(Ftr), ytr)
    return clf.predict(sc.transform(Fte))


# ===========================================================================
# 2. HIDDEN MARKOV MODEL — corrected pipeline (face dropped, body-centred,
#    velocity, left-to-right Bakis topology, length-normalised scoring)
# ===========================================================================
def hmm_features(X, lens):
    """1629-D raw frames -> body-centred hands+pose + presence flags + velocity."""
    out = []
    for i in range(X.shape[0]):
        seq = X[i, :lens[i]]                    # (T,1629)
        pose = seq[:, POSE].reshape(-1, 33, 3)
        lh   = seq[:, LH].reshape(-1, 21, 3)
        rh   = seq[:, RH].reshape(-1, 21, 3)
        # body reference frame: mid-shoulder origin, shoulder-width scale
        mid = (pose[:, L_SHOULDER, :2] + pose[:, R_SHOULDER, :2]) / 2.0
        width = np.linalg.norm(pose[:, L_SHOULDER, :2] - pose[:, R_SHOULDER, :2],
                               axis=1, keepdims=True)
        width = np.where(width < 1e-3, 1.0, width)
        def norm(block):                        # block (T,K,3) -> (T,K,2) centred
            xy = block[:, :, :2]
            return (xy - mid[:, None, :]) / width[:, None, :]
        # upper-body pose subset (0..22): head, shoulders, elbows, wrists
        pose_n = norm(pose[:, 0:23, :])         # (T,23,2)
        lh_present = (np.abs(lh).sum(axis=(1, 2)) > 1e-6).astype(np.float32)
        rh_present = (np.abs(rh).sum(axis=(1, 2)) > 1e-6).astype(np.float32)
        lh_n = norm(lh)                          # (T,21,2)
        rh_n = norm(rh)
        static = np.concatenate([pose_n.reshape(len(seq), -1),
                                 lh_n.reshape(len(seq), -1),
                                 rh_n.reshape(len(seq), -1),
                                 lh_present[:, None], rh_present[:, None]], axis=1)
        vel = np.zeros_like(static)
        vel[1:] = np.diff(static, axis=0)
        out.append(np.concatenate([static, vel], axis=1).astype(np.float32))
    return out                                   # list of (T,D)


def bakis_transmat(n, self_loop):
    t = np.zeros((n, n))
    for i in range(n - 1):
        t[i, i], t[i, i + 1] = self_loop, 1 - self_loop
    t[-1, -1] = 1.0
    return t


def run_hmm(Xtr, ytr, Xte, ltr, lte, classes, n_states=3, n_pca=16):
    tr_seqs = hmm_features(Xtr, ltr)
    te_seqs = hmm_features(Xte, lte)
    scaler = StandardScaler().fit(np.vstack(tr_seqs))
    n_comp = min(n_pca, np.vstack(tr_seqs).shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED).fit(scaler.transform(np.vstack(tr_seqs)))
    tf = lambda s: pca.transform(scaler.transform(s))
    tr_t = [tf(s) for s in tr_seqs]
    te_t = [tf(s) for s in te_seqs]

    models = {}
    for c in classes:
        seqs = [tr_t[i] for i in range(len(tr_t)) if ytr[i] == c]
        if not seqs:
            continue
        mean_len = int(np.median([len(s) for s in seqs]))
        self_loop = 0.9 if mean_len <= n_states else 1.0 - n_states / float(mean_len)
        m = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                            min_covar=1e-2, n_iter=100, tol=1e-3,
                            init_params="mc", params="mc", random_state=SEED)
        m.startprob_ = np.eye(n_states)[0]
        m.transmat_ = bakis_transmat(n_states, self_loop)
        try:
            m.fit(np.vstack(seqs), lengths=[len(s) for s in seqs])
            models[c] = m
        except Exception:
            pass

    preds = []
    for s in te_t:
        best, best_score = None, -np.inf
        T = max(len(s), 1)
        for c, m in models.items():
            try:
                sc = m.score(s) / T
                if np.isfinite(sc) and sc > best_score:
                    best_score, best = sc, c
            except Exception:
                pass
        preds.append(best if best is not None else classes[0])
    return np.asarray(preds)


def run_hmm_naive(Xtr, ytr, Xte, ltr, lte, classes, n_states=5, n_pca=30):
    """The ORIGINAL broken configuration: full 1629-D incl. face, 5-state
    ergodic, argmax over possibly -inf scores. Reproduces the collapse."""
    def raw(X, lens):
        return [X[i, :lens[i]].astype(np.float32) for i in range(X.shape[0])]
    tr_seqs, te_seqs = raw(Xtr, ltr), raw(Xte, lte)
    scaler = StandardScaler().fit(np.vstack(tr_seqs))
    pca = PCA(n_components=n_pca, random_state=SEED).fit(scaler.transform(np.vstack(tr_seqs)))
    tf = lambda s: pca.transform(scaler.transform(s))
    tr_t = [tf(s) for s in tr_seqs]; te_t = [tf(s) for s in te_seqs]
    models = {}
    for c in classes:
        seqs = [tr_t[i] for i in range(len(tr_t)) if ytr[i] == c]
        if not seqs:
            continue
        m = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=100, tol=1e-2, random_state=SEED)
        try:
            m.fit(np.vstack(seqs), lengths=[len(s) for s in seqs]); models[c] = m
        except Exception:
            pass
    preds = []
    for s in te_t:
        scores = {}
        for c in classes:
            if c in models:
                try:
                    scores[c] = models[c].score(s)
                except Exception:
                    scores[c] = -np.inf
            else:
                scores[c] = -np.inf
        preds.append(max(scores, key=scores.get))   # naive argmax -> class 0 on ties
    return np.asarray(preds)


# ===========================================================================
# DTW + 1-NN — non-parametric reference
# ===========================================================================
def _dtw(a, b):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        d = np.linalg.norm(b - a[i - 1], axis=1)         # (m,)
        for j in range(1, m + 1):
            D[i, j] = d[j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def run_dtw(Xtr, ytr, Xte, ltr, lte, classes, n_pca=16):
    tr_seqs = hmm_features(Xtr, ltr); te_seqs = hmm_features(Xte, lte)
    scaler = StandardScaler().fit(np.vstack(tr_seqs))
    pca = PCA(n_components=min(n_pca, np.vstack(tr_seqs).shape[1]),
              random_state=SEED).fit(scaler.transform(np.vstack(tr_seqs)))
    tf = lambda s: pca.transform(scaler.transform(s)).astype(np.float32)
    tr_t = [tf(s) for s in tr_seqs]; te_t = [tf(s) for s in te_seqs]
    preds = []
    for s in te_t:
        dists = [_dtw(s, t) for t in tr_t]
        preds.append(ytr[int(np.argmin(dists))])
    return np.asarray(preds)


# ===========================================================================
# Deep models (BiLSTM, Transformer, ST-GCN) + Hybrid
# ===========================================================================
def _scale_seq(Xtr, Xte):
    """Fit StandardScaler on train frames, apply to both (keeps padding at ~0)."""
    N, T, D = Xtr.shape
    sc = StandardScaler().fit(Xtr.reshape(-1, D))
    f = lambda X: sc.transform(X.reshape(-1, D)).reshape(X.shape[0], T, D).astype(np.float32)
    return f(Xtr), f(Xte)


def _mask(lens, T):
    m = torch.zeros(len(lens), T)
    for i, l in enumerate(lens):
        m[i, :l] = 1.0
    return m


class BiLSTM(nn.Module):
    """Bidirectional 2-layer LSTM over landmark sequences. A linear input
    projection (1629 -> proj) precedes the recurrence: it makes the recurrent
    cost tractable on CPU and acts as a learned feature compressor, standard for
    high-dimensional landmark streams."""
    def __init__(self, d_in, n_cls, hid=128, proj=128):
        super().__init__()
        self.inproj = nn.Sequential(nn.Linear(d_in, proj), nn.ReLU())
        self.lstm = nn.LSTM(proj, hid, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(hid * 2, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x, lens):
        x = self.inproj(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(h)


class PosEnc(nn.Module):
    def __init__(self, d, T=60):
        super().__init__()
        pe = torch.zeros(T, d)
        pos = torch.arange(T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEnc(nn.Module):
    def __init__(self, d_in, n_cls, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pe = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                           dropout=0.3, activation="gelu",
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

    def embed(self, x, mask):
        h = self.pe(self.proj(x))
        pad = (mask == 0)
        h = self.enc(h, src_key_padding_mask=pad)
        msk = mask.unsqueeze(-1)
        pooled = (h * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.norm(pooled)

    def forward(self, x, mask):
        return self.head(self.embed(x, mask))


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, A, k_t=5):
        super().__init__()
        self.register_buffer("A", A)
        self.gc = nn.Conv2d(c_in, c_out, 1)
        self.tc = nn.Conv2d(c_out, c_out, (k_t, 1), padding=(k_t // 2, 0))
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.res = (nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity())

    def forward(self, x):                        # x: (N,C,T,V)
        r = self.res(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        x = self.gc(x); x = self.tc(x); x = self.bn(x)
        return self.drop(self.relu(x + r))


def build_adjacency(V=75):
    """pose(33)+lh(21)+rh(21). Hand-built edges + self loops, symmetric norm."""
    A = np.eye(V)
    pose_edges = [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),
                  (0,11),(0,12)]
    for a, b in pose_edges:
        A[a, b] = A[b, a] = 1
    def hand(off):
        base = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
                (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),
                (18,19),(19,20)]
        for a, b in base:
            A[off+a, off+b] = A[off+b, off+a] = 1
    hand(33); hand(54)
    A[15, 33] = A[33, 15] = 1                    # wrist->left hand root
    A[16, 54] = A[54, 16] = 1                    # wrist->right hand root
    d = A.sum(1); Dm = np.diag(1.0 / np.sqrt(np.where(d > 0, d, 1)))
    return (Dm @ A @ Dm).astype(np.float32)


class STGCN(nn.Module):
    def __init__(self, n_cls, A, c_in=3):
        super().__init__()
        self.b1 = STGCNBlock(c_in, 32, A)
        self.b2 = STGCNBlock(32, 64, A)
        self.b3 = STGCNBlock(64, 64, A)
        self.head = nn.Linear(64, n_cls)

    def embed(self, x):                          # x: (N,C,T,V)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        return x.mean(dim=[2, 3])                 # global pool -> (N,128)

    def forward(self, x):
        return self.head(self.embed(x))


class HybridNet(nn.Module):
    """Hybrid Multimodal Transformer: a graph spatial-temporal stream (ST-GCN
    on the skeleton) fused with a landmark attention stream (Transformer on the
    face-dropped 225-D hands+pose vector). Late-concatenated embeddings -> head.
    """
    def __init__(self, n_cls, A, d_land=225, d_model=96):
        super().__init__()
        self.g1 = STGCNBlock(3, 32, A)
        self.g2 = STGCNBlock(32, 64, A)
        self.proj = nn.Linear(d_land, d_model)
        self.pe = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=192,
                                           dropout=0.3, activation="gelu",
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.fuse = nn.Sequential(nn.Linear(64 + d_model, 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, n_cls))

    def forward(self, xg, xl, mask):
        g = self.g2(self.g1(xg)).mean(dim=[2, 3])            # (N,64)
        h = self.pe(self.proj(xl))
        h = self.enc(h, src_key_padding_mask=(mask == 0))
        msk = mask.unsqueeze(-1)
        l = (h * msk).sum(1) / msk.sum(1).clamp(min=1)       # (N,d_model)
        return self.fuse(torch.cat([g, l], dim=1))


def _train_torch(model, forward_fn, ytr_idx, n_epochs=45, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    yt = torch.tensor(ytr_idx)
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        out = forward_fn(model)
        loss = lossf(out, yt)
        loss.backward(); opt.step()
    return model
