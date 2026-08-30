"""
Real BLEU / WER / ROUGE / METEOR for the gloss -> healthcare-text layer.
Evaluates the two baseline translators (exact rule-based, fuzzy token-overlap)
on (a) clean gloss input and (b) simulated noisy recogniser output.
"""
import csv, json, re, itertools, numpy as np
import sacrebleu, jiwer
from nltk.translate.meteor_score import meteor_score

BASE = "/mnt/user-data/uploads/evanceaiproject/dataset/ksl_project_data"
random_seed = 42
rng = np.random.default_rng(random_seed)


def norm(g):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", g.lower()).split())


# ---------- load 20-phrase dataset ----------
rows = list(csv.DictReader(open(f"{BASE}/language_translation/healthcare_gloss_phrase_dataset.csv")))
glosses = [norm(r["gloss"]) for r in rows]
targets = [r["healthcare_text"] for r in rows]
gloss2text = {g: t for g, t in zip(glosses, targets)}


# ---------- translators ----------
class ExactRuleBasedTranslator:
    def __init__(self, mapping): self.m = mapping
    def translate(self, g):
        return self.m.get(norm(g), "UNKNOWN_PHRASE")


class FuzzyTokenTranslator:
    def __init__(self, mapping, threshold=0.5):
        self.m = mapping; self.th = threshold
        self.keys = [(k, set(k.split())) for k in mapping]
    def translate(self, g):
        toks = set(norm(g).split())
        best, best_s = None, 0.0
        for k, ks in self.keys:
            j = len(toks & ks) / len(toks | ks) if (toks | ks) else 0.0
            if j > best_s:
                best_s, best = j, k
        if best is not None and best_s >= self.th:
            return self.m[best]
        return "UNKNOWN_PHRASE"


exact = ExactRuleBasedTranslator(gloss2text)
fuzzy = FuzzyTokenTranslator(gloss2text, 0.5)


# ---------- simulate noisy recogniser output ----------
def perturb(gloss):
    toks = gloss.split()
    t = list(toks)
    op = rng.integers(0, 3)
    if op == 0 and len(t) > 1:                       # reorder
        rng.shuffle(t)
    elif op == 1 and len(t) > 2:                     # drop one
        t.pop(int(rng.integers(0, len(t))))
    elif op == 2 and len(t) > 1:                     # reorder + drop
        rng.shuffle(t); t.pop()
    return " ".join(t)


noisy_inputs = [perturb(g) for g in glosses]


# ---------- ROUGE (1,2,L) implemented directly ----------
def _ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))


def _f(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def rouge_n(hyp, ref, n):
    h, r = hyp.split(), ref.split()
    hg, rg = _ngrams(h, n), _ngrams(r, n)
    if not hg or not rg:
        return 0.0
    from collections import Counter
    ch, cr = Counter(hg), Counter(rg)
    overlap = sum((ch & cr).values())
    p = overlap / len(hg); rec = overlap / len(rg)
    return _f(p, rec)


def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(hyp, ref):
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    l = lcs(h, r)
    return _f(l / len(h), l / len(r))


def meteor(hyp, ref):
    try:
        return meteor_score([ref.split()], hyp.split())
    except Exception:
        # exact-unigram fallback (no WordNet synonyms)
        h, r = hyp.split(), ref.split()
        if not h or not r:
            return 0.0
        common = sum((__import__("collections").Counter(h) &
                      __import__("collections").Counter(r)).values())
        if common == 0:
            return 0.0
        p, rec = common / len(h), common / len(r)
        fmean = (10 * p * rec) / (rec + 9 * p)
        chunks = 1
        pen = 0.5 * (chunks / common) ** 3
        return fmean * (1 - pen)


def evaluate(name, translator, inputs, refs):
    preds = [translator.translate(g) for g in inputs]
    hyps = [p if p != "UNKNOWN_PHRASE" else "" for p in preds]
    exact_match = np.mean([1.0 if p == t else 0.0 for p, t in zip(preds, refs)])
    coverage = np.mean([1.0 if p != "UNKNOWN_PHRASE" else 0.0 for p in preds])
    # corpus BLEU (sacrebleu expects list-of-references)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    r1 = np.mean([rouge_n(h, t, 1) for h, t in zip(hyps, refs)]) * 100
    r2 = np.mean([rouge_n(h, t, 2) for h, t in zip(hyps, refs)]) * 100
    rl = np.mean([rouge_l(h, t) for h, t in zip(hyps, refs)]) * 100
    met = np.mean([meteor(h, t) for h, t in zip(hyps, refs)]) * 100
    # WER: reference vs hypothesis over the whole corpus
    wer = jiwer.wer(refs, [h if h else "@" for h in hyps]) * 100
    return {"model": name, "exact_match": round(float(exact_match)*100, 2),
            "coverage": round(float(coverage)*100, 2),
            "BLEU": round(bleu, 2), "chrF": round(chrf, 2),
            "ROUGE1": round(float(r1), 2), "ROUGE2": round(float(r2), 2),
            "ROUGEL": round(float(rl), 2), "METEOR": round(float(met), 2),
            "WER": round(float(wer), 2)}


# ---------- leave-one-out generalisation ----------
# Remove each phrase's own key, force the fuzzy translator to retrieve the
# NEAREST REMAINING phrase. Predictions now differ from the reference, so
# BLEU/ROUGE/METEOR measure semantic closeness rather than exact hit/miss.
def loo_fuzzy_predict():
    preds = []
    for i, g in enumerate(glosses):
        reduced = {k: v for j, (k, v) in enumerate(gloss2text.items()) if j != i}
        preds.append(FuzzyTokenTranslator(reduced, threshold=0.0).translate(g))
    return preds


class _Fixed:
    def __init__(self, preds): self.preds = preds; self.i = 0
    def translate(self, g):
        p = self.preds[self.i]; self.i += 1; return p


out = {"clean": [], "noisy": [], "generalization": [], "n_phrases": len(glosses)}
for name, tr in [("Exact rule-based", exact), ("Fuzzy token-overlap", fuzzy)]:
    out["clean"].append(evaluate(name, tr, glosses, targets))
    out["noisy"].append(evaluate(name, tr, noisy_inputs, targets))
out["generalization"].append(
    evaluate("Fuzzy token-overlap (nearest-neighbour retrieval)",
             _Fixed(loo_fuzzy_predict()), glosses, targets))

json.dump(out, open("/home/claude/ksl/lang_results.json", "w"), indent=2)
print("CLEAN input (exact gloss):")
for r in out["clean"]:
    print(" ", r)
print("NOISY input (simulated recogniser):")
for r in out["noisy"]:
    print(" ", r)
print("GENERALIZATION (leave-one-out nearest retrieval):")
for r in out["generalization"]:
    print(" ", r)
