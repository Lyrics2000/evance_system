## What the numbers actually say

Your model is predicting **`before` for all 17 test samples**. Nothing else. Proof:

- `before`: recall = 1.00 (caught all 4), precision = 0.24 ≈ 4/17 → it was predicted 17 times
- Every other class: precision = recall = 0.00 → predicted zero times
- Accuracy 0.2353 = 4/17 = **exactly the majority-class baseline**

So the classifier has learned nothing. This is not "weak performance," it's a **degenerate collapse**. Random guessing over 5 classes would give ~0.20; you're at 0.235 only because `before` happens to have 4 samples instead of 3.

## The #1 suspect: `argmax` over a broken score vector

`before` is **alphabetically first**. If your per-class scoring loop produces `-inf` or `NaN` for every model, `np.argmax` silently returns index 0 — which is `before`. That single fact explains the exact pattern you're seeing better than any "the features are weak" story.

`hmmlearn`'s `.score()` returns `-inf` when a sequence has zero probability under the model (hard zeros in `startprob_`/`transmat_`, unseen symbols in `MultinomialHMM`) and `NaN` when Baum-Welch diverged or a covariance went singular.

## Diagnostic — run this before changing anything

```python
import numpy as np

# models: dict {label: fitted hmmlearn model}
# X_test, lengths_test, y_test as you already have them

labels = sorted(models.keys())
rows = []
for seq, true in zip(test_sequences, y_test):     # seq: (T, D) ndarray
    scores = []
    for lab in labels:
        try:
            s = models[lab].score(seq)
        except Exception as e:
            s = float("nan")
        scores.append(s)
    rows.append(scores)
    print(f"true={true:10s} " + " ".join(f"{l}={s:12.3f}" for l, s in zip(labels, scores)))

S = np.array(rows)
print("\n--- SCORE MATRIX HEALTH ---")
print("shape          :", S.shape)
print("any NaN        :", np.isnan(S).any(),  "| count:", int(np.isnan(S).sum()))
print("any -inf       :", np.isneginf(S).any(), "| count:", int(np.isneginf(S).sum()))
print("rows all-equal :", int(np.sum(np.ptp(S, axis=1) < 1e-6)), "of", S.shape[0])
print("per-class mean :", dict(zip(labels, np.nanmean(np.where(np.isfinite(S), S, np.nan), axis=0))))
```

Read the output like this:

| What you see | Diagnosis | Fix |
|---|---|---|
| NaN or -inf everywhere | Training diverged / zero-probability paths | Add covariance floor + smoothing (below) |
| Finite scores, but `before` column always highest by a wide margin | That model overfit to a near-singular covariance → inflated density | `covariance_type="diag"`, raise `min_covar` |
| Scores track sequence length, not class | You're comparing raw log-likelihoods across different `T` | Divide by `len(seq)` before argmax |
| Rows all-equal | Models are identical — training data leaked or all models fit the same pool | Check your per-class train split |

## The structural problem underneath

Even with the bug fixed, **17 test samples across 5 classes (~3-4 each) cannot validate anything.** A single sample flipping moves accuracy by 6 points. And a Gaussian HMM with `N` states over `D`-dim features has roughly `N² + N·D + N·D` free parameters — with `covariance_type="full"` it's `N·D²/2`. At `N=5, D=126` (MediaPipe two-hand landmarks) that's ~40,000 parameters per class, fit on maybe 10 videos. Baum-Welch will happily converge to a garbage local optimum.

Concrete constraints for your scale:

```python
from hmmlearn import hmm
from sklearn.decomposition import PCA

# 1. Reduce dimensionality HARD. Fit PCA on TRAIN ONLY.
pca = PCA(n_components=12, random_state=42).fit(X_train)   # 12, not 50
Xtr, Xte = pca.transform(X_train), pca.transform(X_test)

# 2. Small, left-to-right topology. Signs are monotonic in time.
def make_model(n_states=4):
    m = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",   # NOT "full" — singular covariances cause the -inf
        min_covar=1e-2,           # floor: prevents variance -> 0 -> likelihood -> inf
        n_iter=200,
        tol=1e-3,
        init_params="mc",         # we set s and t ourselves
        params="mct",
        random_state=42,
    )
    # left-to-right: start in state 0, only self-loop or advance
    m.startprob_ = np.eye(n_states)[0]
    t = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        t[i, i], t[i, i + 1] = 0.6, 0.4
    t[-1, -1] = 1.0
    m.transmat_ = t
    return m
```

And normalize at prediction time:

```python
def predict(seq, models, labels):
    T = max(len(seq), 1)
    scores = np.full(len(labels), -np.inf)
    for i, lab in enumerate(labels):
        try:
            s = models[lab].score(seq) / T      # length-normalized
            if np.isfinite(s):
                scores[i] = s
        except Exception:
            pass
    if not np.isfinite(scores).any():
        return None            # explicit abstain — NEVER silently argmax into class 0
    return labels[int(np.argmax(scores))]
```

That `return None` matters. An abstention you can count is infinitely more debuggable than a silent collapse onto the alphabetically-first label.

## Evaluation fix

Stop using a single 17-sample holdout. Use stratified k-fold over all your data and report mean ± std:

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Also always pass `zero_division=0` to `classification_report` so the 0.00 precisions are explicit rather than warnings you might scroll past.

---

Two things I need to know to go further:

1. **What are your features?** Raw frame pixels, MediaPipe hand/pose landmarks, or CNN embeddings? And what's `D`?
2. **How many training videos per class?** If it's under ~15, the honest answer is that no HMM configuration will work and you need DTW + k-NN instead — it has effectively zero parameters to fit and is the right tool at this data scale.

Tell me those and I'll write the full corrected pipeline.


our model is in a notebook found in /Users/lyrics/Desktop/client/evance ai project/dataset/ipynb_checkpoints the notebook itself is called 02_hmm_wlasl_baseline.ipynb  now can you help interprate on why we getting weak performance, is it model issue or development issue or data iissue and how can we solve to improve


Install is still compiling in the background — but I don't need it. Your logs already contain the smoking gun. Writing the diagnosis and the corrected pipeline now.I read the notebook and its saved outputs. Short answer: **all three, in a specific causal chain.** A development bug is what produced 0.2353 exactly; a data-scale problem is what triggered the bug; a feature-design problem is why patching the bug alone won't get you far.

---

## 1. Development issue — the direct cause (proof is in your own logs)

Cell 20's log shows this, and it's the whole story:

```
INFO | Training HMM for class: before, sequences: 11
INFO | Training HMM for class: computer, sequences: 10
WARNING | Some rows of transmat_ have zero sum because no transition
          from the state was ever observed.      x14
INFO | Training HMM for class: cool, sequences: 10
WARNING | Some rows of transmat_ have zero sum ...
```

`before` is the **only** model that trained cleanly. The other four have zero-sum transition rows, which makes `model.score()` return `-inf` for essentially any test sequence.

Now look at cell 22:

```python
scores[class_id] = model.score(sequence)   # -inf for 4 of 5 classes
return max(scores, key=scores.get)
```

When every value is `-inf`, Python's `max` doesn't error — it returns the **first key in insertion order**. Insertion order is `class_models` iteration order, which is `enumerate(label_encoder.classes_)`, i.e. class_id `0`. LabelEncoder sorts alphabetically, so class 0 is **`before`**.

That is precisely your result: `before` recall 1.00, everything else 0.00, accuracy 4/17 = 0.2353 = the majority-class baseline exactly.

Second bug reinforcing it: your abstain guard never fires.

```python
if not scores:      # scores is NEVER empty — it always has 5 keys
    return None
```

Hence `Valid predictions: 17 out of 17`. A `-inf` was silently laundered into a confident prediction. The guard needed was `np.isfinite(...).any()`, not `if not scores`.

---

## 2. Data issue — the feature vector is 86% noise

Cell 12 builds each frame as:

| Block | Dims | Share |
|---|---|---|
| pose | 99 | 6.1% |
| left hand | 63 | 3.9% |
| right hand | 63 | 3.9% |
| **face** | **1404** | **86.2%** |

`before`, `computer`, `cool`, `cousin`, `drink` are distinguished by **handshape and hand trajectory**. `StandardScaler` weights all 1629 dims equally, so PCA's leading components are facial geometry and head pose. Your `Explained variance ratio: 0.9985` from 30 of 1629 components confirms it — variance is concentrated in a handful of global modes (head position, body scale, framing), not fine hand detail. The HMMs are modelling faces.

Three more feature defects:

**Zero-fill is an outlier, not a neutral value.** `flatten_landmarks` returns zeros when MediaPipe misses a hand — common in WLASL. MediaPipe x/y are image-normalised with mean ≈ 0.5, so after `StandardScaler` a missing hand becomes a large-magnitude spike. PCA ends up modelling detector dropout.

**No spatial normalisation.** Raw image coordinates mean signer position in frame and camera distance vary more across clips than the signs do.

**No velocity features.** A Gaussian HMM emission describes one static frame. Sign identity is substantially in the motion. Deltas are the standard fix (same reason speech HMMs use delta-cepstra).

Also worth checking: you requested 8 classes and got 5. `thin`, `go`, `help` had 15/16/14 labelled samples but **zero** files passing `Path(sample.filepath).exists()`. All-or-nothing per class points at a download failure, not random missingness. Worth auditing `download_wlasl.py` — that's ~45 videos you should have.

---

## 3. Model issue — 330 parameters fit to ~350 correlated frames

`GaussianHMM(n_components=5, covariance_type="diag")` at D=30, per class:

| Component | Count |
|---|---|
| means (5×30) | 150 |
| variances (5×30) | 150 |
| transmat (5×5, ergodic) | 25 |
| startprob | 5 |
| **total** | **330** |

Training data per class: ~10 sequences × ~35 frames = **~350 frames**. And consecutive frames are near-duplicates, so effective sample size is maybe 30. That ratio is why Baum-Welch produced unvisited states and zero-sum transition rows. **The warnings were the model telling you it had no data.**

Ergodic topology is also wrong for signs, which are monotonic (preparation → stroke → retraction, never backwards). A left-to-right Bakis topology is both more accurate and 25 parameters cheaper.

---

## The fix

I wrote a complete corrected pipeline. **It rebuilds features from your existing `wlasl_hmm_sequences.joblib` cache — no MediaPipe re-run needed**, since the cached 1629-D vectors have a known layout and the face block can simply be sliced off.

What it changes:

| Area | Before | After |
|---|---|---|
| Features | 1629-D, 86% face | 309-D: upper-body pose + both hands + presence flags + Δ velocity |
| Coordinates | raw image-relative | centred on mid-shoulder, scaled by shoulder width |
| Missing hands | zero-fill → outlier | left at origin + explicit binary flag |
| PCA | 30 | 16 |
| Topology | ergodic, 5 states | left-to-right, 3 states, transitions **fixed** (`params="mc"`) |
| Params/class | 330 | 96 |
| Prediction | `max()` → silent class-0 | finite check, length-normalised, explicit `-1` abstain |
| Evaluation | one 17-sample holdout | 5-fold stratified CV + per-fold SD + DTW 1-NN reference |

Fixing the transitions (ADR-005 in the file) makes zero-sum rows *structurally impossible* rather than something you repair after the fact — the right trade at 10 sequences per class.

```bash
cd "/Users/lyrics/Desktop/client/evance ai project/dataset/ipynb_checkpoints"
python3 hmm_wlasl_fixed.py                 # diagnosis + corrected CV + saves package
python3 hmm_wlasl_fixed.py --diagnose-only # just print the -inf score matrix
python3 hmm_wlasl_fixed.py --states 4 --pca 20
```

It runs `diagnose_original()` first, which reproduces the old config and prints the raw score matrix — you'll see the `-inf` columns directly, which is the evidence you want in the write-up.

I verified syntax and the dimension arithmetic (153 / 156 / 309 all check out), but couldn't execute it end-to-end — my sandbox is still compiling scipy. Run it in the environment where the notebook already works.

---

## Set expectations honestly

With 13 samples per class you should target **0.45–0.65**, not 0.90. To go meaningfully higher, the data is the bottleneck, in this order:

1. **Fix the download** — recover `thin`/`go`/`help` and get back to 8 classes
2. **Raise samples per class** — 30+ per class is where HMMs on small-vocab sign start behaving; WLASL's tail is thin, so consider augmentation (mirroring, temporal jitter, small rotations)
3. **Signer-independent splits** — random stratified splits leak signer identity and inflate scores

If the DTW 1-NN baseline in the script beats the fixed HMM, that's a real finding worth reporting: at this data scale a zero-parameter method wins, and it's the honest baseline your LSTM/ST-GCN/Transformer notebooks should be compared against.

Want me to apply these fixes back into the `.ipynb` as replacement cells, or audit `03_lstm_wlasl_landmark_baseline.ipynb` for the same feature-design problems? The face-dominance issue almost certainly affects those too.


explain in simple terms what you mean and also what. do this mean -inf importance and meaning Ergodic topology i. A left-to-right Bakis topology is both more accurate and 25 parameters cheaper.

## The big picture, in one analogy

Think of each of your 5 HMMs as a **judge**. You show a video to all 5 judges and ask each "how likely is it that *you* produced this video?" Each returns a score. You pick the judge with the highest score, and their name becomes your prediction.

What happened in your run: **4 of the 5 judges were broken and returned "impossible" for every video.** Only the `before` judge answered at all. So `before` won 17 times out of 17 — not because it was right, but because it was the only one talking.

---

## What `-inf` means

Probabilities are tiny numbers, so HMMs work in **logarithms** instead. A probability of 0.001 becomes about −7. A probability of 0.000001 becomes about −14. Lower = less likely.

Now: what's the log of **zero**? Zero probability means *flat-out impossible*. The log of 0 is negative infinity — `-inf`. That's not a weird number, it's the model saying:

> "This video could not have come from me. Not unlikely. **Impossible.**"

Your 4 broken models said that about every single test video.

**Why it's dangerous:** `-inf` isn't an error. It doesn't crash. It's a perfectly valid float that quietly poisons the comparison downstream. Your code did:

```python
return max(scores, key=scores.get)
```

When all the scores are equally `-inf`, there's no winner — nothing is bigger than anything else. Python doesn't complain; it just hands back **whichever one it saw first**. That was class 0, and class 0 is `before` because scikit-learn sorts labels alphabetically.

So the "model" you shipped was really: *ignore the video, output `before`.* One missing `isfinite()` check turned a broken model into a confident-looking one.

**Why the models returned `-inf`:** each model has rules like "from stage 2 you can move to stage 4." If during training the model *never once observed* a move out of stage 3, it records that probability as 0. Later, if a test video needs to pass through stage 3 to be explained — probability 0 → log 0 → `-inf`. That's exactly what your log's `Some rows of transmat_ have zero sum` warnings were reporting, 14 times per class.

---

## Ergodic vs. left-to-right (Bakis)

An HMM breaks a sign into hidden **stages**. For the sign `drink`, roughly: hand rises → forms a C-shape → tilts to mouth → lowers.

The **transition matrix** is the rulebook for which stage can follow which.

### Ergodic — what you used (the default)

"Any stage can follow any stage." With 5 stages that's a full 5×5 grid of numbers to estimate:

```
        to→   S0    S1    S2    S3    S4
  S0          ?     ?     ?     ?     ?
  S1          ?     ?     ?     ?     ?
  S2          ?     ?     ?     ?     ?
  S3          ?     ?     ?     ?     ?
  S4          ?     ?     ?     ?     ?
```

Every `?` must be learned from data. But this also allows nonsense — the model is permitted to believe a sign goes *lowering hand → hand rising → tilting → C-shape*, backwards and scrambled. You're spending precious data learning that transitions which never happen have probability zero.

### Left-to-right / Bakis — what I switched you to

Signs run **forward in time and don't reverse.** You never return to the wind-up after the follow-through. So allow only two moves: **stay in this stage**, or **advance to the next one.**

```
        to→   S0    S1    S2    S3    S4
  S0         stay  next    0     0     0
  S1           0   stay  next    0     0
  S2           0     0   stay  next    0
  S3           0     0     0   stay  next
  S4           0     0     0     0   stay
```

("Bakis" is just the researcher's name attached to this shape. Common in speech recognition for the same reason — words also run forward.)

Two wins:

1. **More accurate** — it bakes in a fact you already know for free, instead of making the model discover it from 10 videos.
2. **Far cheaper** — most cells are hard zeros, so there's almost nothing left to estimate.

---

## The parameter count — and a correction

I said "25 parameters cheaper" in my last message. That was loose; let me give you the real number, because rows of a probability matrix must sum to 1, so not every cell is independently free.

| | Free numbers to learn |
|---|---|
| Ergodic 5-state transitions | 5 rows × 4 = **20** |
| Ergodic starting probabilities | **4** |
| **Total** | **24** |
| Bakis, transitions held fixed | **0** |

So the honest saving is **24 free parameters per class**, not 25. And in the script I went further — I don't just constrain the shape, I **freeze the transition numbers entirely** (`params="mc"`, meaning *train the means and covariances only*).

That sounds drastic, but consider the budget. Your old setup had **330** free parameters per class fit to roughly **350** frames — and those frames are near-duplicates of their neighbours, so the real independent sample count is more like 30. You were asking for ten times more information than the data contains. That's why Baum-Welch left stages unvisited and produced the zero rows.

The new setup: **96** parameters against the same data, with the transition structure supplied as known physics rather than guessed. Combined with cutting the face landmarks (86% of your input was modelling faces, not hands), that's the difference between a model that can't train and one that can.

The trade-off, stated plainly: fixed transitions mean the model can't learn that one sign is slower than another. With 10 examples per class that's a bargain. Once you get to 30+ per class, flip `TRAIN_TRANSITIONS = True` in the script — the sanitiser I wrote will repair any zero rows instead of letting them become `-inf`.


which areas were this that you touched in the fiie and explain

First, an important clarification: **I did not modify your notebook.** `02_hmm_wlasl_baseline.ipynb` is untouched, exactly as it was. I wrote a separate new file, `hmm_wlasl_fixed.py` (842 lines), that replaces the *back half* of the notebook's pipeline while reusing the expensive front half.

Here's the mapping.

## What I reused vs. replaced

| Notebook cell | What it does | Status |
|---|---|---|
| 2–10 | Load FiftyOne, pick classes | **Untouched** — reused as-is |
| 12–14 | Run MediaPipe, save `.joblib` cache | **Untouched** — I read the cache it produced |
| 16 | Train/test split | **Replaced** → Section 8 |
| 18 | Scaler + PCA(30) | **Replaced** → moved inside each CV fold, PCA 16 |
| 20 | Train the HMMs | **Replaced** → Section 4 |
| 22 | `predict_hmm` | **Replaced** → Section 5 ← *the actual bug* |
| 24–26 | Metrics + confusion matrix | **Replaced** → Section 8 |
| 28 | Save package | **Replaced** → Section 9 |

The big practical win: **MediaPipe never re-runs.** That extraction took you ~3 minutes across 67 videos. The cached vectors are 1629 numbers per frame in a known order — pose first, then left hand, right hand, face last. So I just slice off the face portion instead of re-processing video.

---

## The eight areas, in plain terms

### Section 2 — Load and check the cache (lines 189–235)

Before doing anything, verify the cache is what we think it is: every sequence is 2-dimensional, every frame is exactly 1629 numbers, sequence count matches label count, no sequence shorter than 2 frames.

**Why:** your original code assumed all of this silently. If the cache were ever regenerated with different settings, you'd get garbage results rather than an error. Checking costs milliseconds.

### Section 3 — Feature engineering (lines 237–323) ← **biggest change**

Turns each 1629-number frame into 309 numbers. Four things happen:

1. **Throw away the face.** 1404 of the 1629 numbers were face landmarks — 86% of your input. `before` vs `drink` is decided by the hands, not the eyebrows.

2. **Re-centre on the body.** MediaPipe gives you position *within the video frame*. So the same sign filmed from 2m away vs 1m away looks completely different to the model. I shift the origin to the midpoint between the shoulders and divide by shoulder width. Now "hand at chin height" means the same thing in every clip, regardless of camera distance.

3. **Handle missing hands properly.** When MediaPipe can't find a hand, your code filled in 63 zeros. After `StandardScaler`, those zeros became extreme outlier values — so PCA was partly modelling *"when does the detector fail"* rather than *"what is the signer doing."* Now a missing hand sits at the body origin (a neutral spot) and gets a separate on/off flag saying "this was missing." The model can learn from the absence instead of being confused by it.

4. **Add motion.** The original features described each frame as a frozen photo. I append the frame-to-frame *change* in every coordinate — how fast and which direction each landmark is moving. Sign identity is largely in the motion.

### Section 4 — Building the HMMs (lines 325–418)

Where the left-to-right structure gets applied. Three states instead of five, transitions locked to "stay here or move forward one," and the transition numbers frozen so training only fits the appearance of each stage.

Also adds a **variance floor** (`min_covar`). Without one, a model can decide a stage has *zero* variation, which makes its score blow up to infinity and lets one class hijack every prediction. The floor makes that impossible.

`sanitise_hmm` is a safety net for later: if you turn transition training back on once you have more data, it repairs any zero rows instead of letting them become `-inf`.

### Section 5 — Prediction (lines 420–465) ← **the actual bug fix**

Three changes to the ~10 lines that broke everything:

- Check `isfinite` before treating a score as real
- Return `-1` ("I don't know") when every class scored impossible, instead of silently defaulting to class 0
- Divide each score by sequence length, so long and short clips are comparable

That last one doesn't change which class wins for a single video, but it makes the numbers readable when you print them — which matters for debugging.

### Section 6 — DTW reference baseline (lines 467–496)

A completely different method as a sanity check: for a test video, find the most similar training video (allowing for speed differences) and copy its label. It has **zero parameters to fit**, which is exactly what you want when you only have 10 examples per class.

**Why it's there:** it tells you whether your HMM is genuinely learning. If DTW beats it, that's a real, publishable finding — not a failure.

### Section 7 — Diagnosis (lines 498–575)

Deliberately rebuilds your *original* broken configuration and prints the raw score table so you can see the `-inf` columns with your own eyes, then counts how many test videos had no valid score at all.

**Why:** so the diagnosis is evidence you can show someone, not just my claim.

### Section 8 — Evaluation (lines 577–688)

Replaces the single 17-video test set with 5-fold cross-validation: split the 67 videos five ways, train on four parts and test on the fifth, rotate. Every video gets predicted exactly once by a model that never saw it.

**Why:** with 17 test videos, one sample flipping moves your accuracy by 5.9 points. The number carries no information. Cross-validation gives you all 67 predictions *plus* a spread across folds, so you know how stable the result is.

The `report` function also prints **"distinct classes predicted: N of 5"** and shouts `** COLLAPSED **` if that's 1. That single line would have caught your bug immediately.

Critically: the scaler and PCA are re-fit **inside each fold**, on training data only. Fitting them once on everything would leak test information and inflate your score.

### Section 9 — Saving (lines 690–724)

Same idea as your cell 28, but the metadata now records per-fold accuracies, the standard deviation, the majority-class baseline, the DTW comparison, and the old 0.2353 — so the file itself documents the improvement.

---

## Two things I added that weren't in the original at all

**Named error codes** (Section 1). Every failure raises something like `[HMM-1002] Sequence 4 has 1629 features, expected...` — greppable in logs, unlike a bare `KeyError`.

**A detection-rate readout.** On startup it prints what fraction of frames MediaPipe actually found each hand in:

```
Landmark detection rate: pose=98.2%  left-hand=41.3%  right-hand=67.8%
```

Watch this number. If a hand is only detected in 40% of frames, no model architecture will save you — that's a video-quality problem you'd fix upstream, and you'd never have known it was there.

---

Want me to also fold these changes back into the `.ipynb` as replacement cells so you can run it in Jupyter, or should I check whether `03_lstm_wlasl_landmark_baseline.ipynb` has the same face-dominance and zero-fill problems?


write new  complete notebook called 022_hmm_wlasl_baselin.ipynb with the correct recommendations

Now building the notebook. I'll generate it programmatically so the JSON is guaranteed valid.Now verifying — JSON schema, per-cell syntax compile, and cross-cell name resolution.Two flags — let me check whether they're real.The rerun dropped its accumulator — re-running that check properly.**39 cells (19 code, 20 markdown)**, sitting next to your original. `02_hmm_wlasl_baseline.ipynb` is untouched.

## Sections

| § | Content | vs. v1 |
|---|---|---|
| 1–4 | Imports, config + error codes, dataset load, label helper | Same, plus named error codes |
| 5 | Class selection **+ availability audit** | New — prints a table exposing the ~45 missing videos |
| 6–7 | MediaPipe extractor, cache load | Unchanged — **reuses your cache, no re-extraction** |
| 8 | Feature engineering, ADR-001…004 | **1629-D → 309-D**, face dropped, body-centred, presence flags, velocity |
| 8b | Landmark detection-rate audit | New |
| 9 | HMM construction, ADR-005 | Bakis topology, 3 states, fixed transitions, variance floor |
| 10 | Prediction | **The bug fix**, with inline unit checks |
| 11 | `diagnose_v1()` | New — rebuilds the broken config and prints the `-inf` matrix |
| 12 | DTW + 1-NN | New reference baseline |
| 13 | 5-fold stratified CV, ADR-006 | Replaces the 17-sample holdout |
| 14–15 | Confusion matrices, save package | Metadata now records fold spread + baseline |
| 16 | `predict_video()` end-to-end | New, abstains rather than guessing |
| 17 | What to change next, in order | New |

Run top to bottom. Section 11 prints the evidence for your write-up; section 13 gives the honest number.

## Verification

| Check | Result |
|---|---|
| nbformat 4 schema | pass |
| Syntax, all 19 code cells | pass |
| Cross-cell name resolution | pass — every name defined before use |
| Dimension arithmetic | 153 / 156 / 309 confirmed |
| `predict_from_scores` on all-`-inf` | returns `-1` (abstains), **not** class 0 |
| Bakis matrix | rows sum to 1, strictly forward, zero-sum rows impossible |
| Translation invariance | max diff 4.4e-16 |
| Scale invariance | max diff 0.0 |
| Missing hand handling | block zeroed, flag flips to 0 |
| DTW self vs. cross distance | 3.7e-09 vs. 1.153 |

Two things I want to flag rather than bury:

**The first name-check pass flagged three names** (`_norm`, `tf`, `tf`). I traced each — all are nested function definitions my AST visitor didn't track. False positives; the corrected checker passes clean.

**My first DTW test failed on my own threshold, not the code.** `dtw(a,a)` returns 3.7e-09 rather than exact zero because the Gram expansion `|x-y|² = |x|² - 2xy + |y|²` loses precision to cancellation, and `sqrt` amplifies it. Harmless for 1-NN ranking against distances of order 1, but worth knowing it's there.

I could not execute the notebook end-to-end — my sandbox is still compiling scipy, and it has no FiftyOne/MediaPipe/WLASL anyway. Everything above is static and unit-level verification. Run it in your Docker environment where v1 already worked.

Want me to port sections 8 and 8b into `03_lstm_wlasl_landmark_baseline.ipynb`? Those notebooks consume the same 1629-D vectors, so they carry the same face-dominance and zero-fill problems — and fixing features there will likely help more than any architecture change.


which assumption does hmm have?

HMMs rest on **six assumptions**. Here they are with how each holds up against your WLASL data.

## 1. First-order Markov property

> The next hidden state depends only on the current state — not on how you got there.
> $P(q_{t+1} \mid q_t, q_{t-1}, \dots, q_1) = P(q_{t+1} \mid q_t)$

**Your data:** reasonable. Signs really are stage-to-stage (preparation → stroke → retraction). The left-to-right topology encodes exactly this.

## 2. Conditional (output) independence — **the one that bites you**

> Given the current hidden state, the current observation is independent of every other observation and state.
> $P(o_t \mid q_t, \text{everything else}) = P(o_t \mid q_t)$

**Your data: badly violated.** You sample every 2nd frame at ~25fps, so consecutive observations are ~80ms apart — near-duplicate images. Frame *t* is enormously informative about frame *t+1*, but the model assumes it tells you nothing once you know the state.

**Consequence:** the model double-counts evidence. Your ~350 training frames per class behave statistically like maybe 20–30 independent observations. This is *why* the v1 models had unvisited states and zero-sum transition rows — the data was thinner than the frame count suggested. Adding velocity features helps a little (they make each observation less redundant), but the assumption stays violated. It's the main reason you can't just scale up the state count.

## 3. Stationarity / homogeneity

> Transition and emission probabilities are constant over time and across sequences.

**Your data: violated across signers.** WLASL clips come from different people, cameras, and lighting. The same hidden state emits different observations depending on who's signing. Your shoulder-scaled, body-centred coordinates remove the worst of it (framing, distance), but signer-specific style remains. This is also why signer-independent splits matter — random splits let the model exploit a violation it shouldn't have access to.

## 4. A finite, fixed number of discrete hidden states

> There are exactly *N* states, chosen in advance, and they're categorical.

**Your data:** an approximation you impose, not a fact. A sign's motion is continuous; carving it into 3 stages is a modelling choice. `N_HMM_STATES` is a hyperparameter with no principled value — you pick it by validation, which at 67 sequences is itself noisy.

## 5. A specific emission family

> Each state emits observations from an assumed distribution. `GaussianHMM` assumes a single multivariate Gaussian per state; `covariance_type="diag"` further assumes the feature dimensions are **conditionally independent given the state**.

**Your data:** partly justified. PCA makes components globally uncorrelated, which is not the same as conditionally uncorrelated within a state — but it's much closer than raw landmarks would be. Diagonal is the right call anyway: full covariance at D=16 needs 136 numbers per state and would go singular on your data.

The unimodality assumption is shakier — if two signers perform a stage differently, that state's true distribution is bimodal and a single Gaussian averages them into a blob sitting between both.

## 6. Geometric state durations — **and a bug this exposes in my file**

This one is *implied*, not stated. A self-loop probability $p$ means the time spent in a state follows a geometric distribution with mean $\frac{1}{1-p}$. Real sign phases have roughly bell-shaped durations, not geometric (geometric says the most likely duration is always 1 frame).

Checking my own numbers against your data:

| | |
|---|---|
| `self_loop` I set | 0.70 |
| Implied mean duration | $1/(1-0.7) = 3.3$ frames |
| States 0 and 1 combined | ~7 frames |
| Your median sequence | ~35 frames |
| Frames dumped into the absorbing state 2 | **~28** |

That's lopsided — the model rushes through the first two stages in three frames each and explains the remaining 80% of the sign with one state. Since I *fixed* the transitions rather than learning them, this value matters much more than it would have in v1.

The principled setting is $p = 1 - \frac{N_{\text{states}}}{\bar{T}}$, which for 3 states and ~35 frames gives:

$$p = 1 - \frac{3}{35} \approx 0.91$$

Better still, compute it from your actual data rather than hard-coding it:

```python
def left_to_right_transmat(n_states, mean_length=None, self_loop=0.7):
    if mean_length is not None and mean_length > n_states:
        self_loop = 1.0 - (n_states / float(mean_length))
    ...
```

then pass `mean_length=np.median([len(s) for s in train_seqs])`. Want me to patch that into the notebook?

---

## Summary

| Assumption | Holds for your data? | Impact |
|---|---|---|
| First-order Markov | Reasonable | Low |
| **Conditional independence** | **Badly violated** | **High — the core limitation** |
| Stationarity | Violated across signers | Medium |
| Fixed discrete states | Imposed, not true | Medium — tuning noise |
| Gaussian, diagonal emissions | Approximate | Low–medium |
| Geometric durations | False | Medium — fixable via `self_loop` |

The honest framing for your write-up: **assumption 2 is the reason HMMs plateau on video.** It's precisely what LSTMs, ST-GCNs, and Transformers drop — they let observations depend on each other directly instead of routing everything through a single discrete state. That's the argument connecting notebook 022 to notebooks 03/04/05, and it's stronger stated as a modelling-assumption failure than as "the accuracy was low."

If you want to keep the HMM framing but relax the worst assumption, a **hidden semi-Markov model** (HSMM) fixes the duration assumption by modelling state durations explicitly. It won't fix conditional independence, though — nothing in the HMM family does.

how did  come up with 0.7 in     
self_loop I set    0.70


Honest answer: **I didn't derive it. I inherited it as a convention and never checked it against your data.**

0.7 is a common starter value for left-to-right HMMs in speech/gesture tutorials — "usually stay, sometimes advance" — and I reached for it while writing `left_to_right_transmat`. That's exactly the kind of unjustified magic number I told you I'd refuse to write. It's the one number in the file with no ADR behind it, and since I also *froze* the transition matrix, it silently became one of the most consequential settings in the notebook.

Let me show you what it actually does.## What 0.7 actually costs you

With `self_loop = 0.7`, state 2 swallows **81% of every sequence** and the model uses an effective **1.86 of its 3 states**. I removed parameters to fit your data budget, then quietly wasted a third of what was left.

The emissions still train, so it wouldn't produce garbage — state 2's Gaussian just becomes a broad blob covering most of the sign. But that's precisely the resolution I was claiming to buy by going left-to-right in the first place.

## Where the right number comes from

You want each state to cover roughly an equal share of the sign. A self-loop probability $p$ gives geometric state durations with mean $\frac{1}{1-p}$. Set that equal to frames-per-state:

$$\frac{1}{1-p} = \frac{T}{N} \quad\Longrightarrow\quad p = 1 - \frac{N}{T}$$

For $N = 3$, $T = 35$: $p \approx 0.914$. The table confirms it — 31.8 / 27.1 / 41.1%, and 2.96 effective states. Not perfectly flat, because the final state is absorbing and soaks up the remainder, but close.

Note the dependency: **$p$ is not a free choice, it's a function of your sequence length.** Hard-coding it means the notebook silently breaks if you change `MAX_FRAMES` or `FRAME_STRIDE`. Right now `MAX_FRAMES=60, FRAME_STRIDE=2` caps you at 30 frames; halve the stride and 0.914 becomes wrong too.

So the fix isn't "0.7 → 0.914" — it's deriving it from the data each run:

```python
def left_to_right_transmat(n_states, mean_length=None, self_loop=None):
    """
    Bakis topology. If mean_length is given, self_loop is derived so each state
    covers ~mean_length/n_states frames:  p = 1 - n_states/mean_length.

    ADR-007: a hard-coded self_loop is wrong whenever MAX_FRAMES or FRAME_STRIDE
    changes, and because transitions are FROZEN (ADR-005) the model cannot correct
    a bad value by learning. At p=0.7, n=3, T=35 the final state absorbs 81% of
    every sequence and only 1.86 of 3 states are effectively used.
    """
    if n_states < 1:
        raise ValueError("n_states must be >= 1")

    if self_loop is None:
        if mean_length is None or mean_length <= n_states:
            self_loop = 0.9   # conservative fallback for very short sequences
        else:
            self_loop = 1.0 - (n_states / float(mean_length))
    if not 0.0 < self_loop < 1.0:
        raise ValueError(f"self_loop must be in (0,1), got {self_loop}")

    t = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states - 1):
        t[i, i]     = self_loop
        t[i, i + 1] = 1.0 - self_loop
    t[n_states - 1, n_states - 1] = 1.0
    return t
```

`build_hmm` then takes `mean_length` and passes it through, and `fit_class_models` computes it per class from its own training sequences — since `drink` and `computer` may well differ in length.

## The general lesson

Every other constant in that file traces to something: 16 PCA components to a parameter-vs-data-count argument, 3 states to the same, `min_covar` to preventing singular densities, dropping face to 86.2% of the input vector. `self_loop=0.7` traced to *"that's the number people use."*

It's also the failure mode I'd flagged in v1 — freezing a parameter is only safe if you've checked its value. Worth grepping the rest of your notebooks for the same pattern: constants that arrived as convention and were never validated against your actual sequence lengths.

Want me to apply this to `022_hmm_wlasl_baselin.ipynb`? It touches three cells — `left_to_right_transmat`, `build_hmm`, and `fit_class_models` — and I'd add the occupancy table above as a printed sanity check so the chosen value is visible every run rather than buried in a default argument.


btw how did you handle each assumptions and how did you sort it, alsso what is the importance of each assumption

Two honest framings, because they differ.

## How I actually worked (failure-driven, not assumption-driven)

I did **not** start from the assumption list. I started from the broken output and worked backwards:

1. Why is accuracy exactly the majority baseline? → found the `-inf` argmax bug
2. Why did 4 of 5 models train degenerately? → parameter count vs. data
3. Why is there so little signal? → 86% of the input was face
4. Can I even trust the number? → 17-sample holdout, no

The assumption mapping below is **retrospective**. I'm laying it out that way because it's the more useful frame for your write-up, but I'd be overselling if I claimed I derived the fixes from first principles.

## Sorting criterion

I ranked by **severity of violation × leverage available inside the HMM family**. An assumption that's badly violated but unfixable ranks below one that's moderately violated and cheap to fix — because the first only gets you a caveat paragraph, the second gets you accuracy.

| # | Assumption | Violated? | Leverage | Priority | Status |
|---|---|---|---|---|---|
| 5b | Non-degenerate emissions | Fatally (v1) | High | **1** | Fixed |
| 3a | Stationarity — *camera/position* | Severely | High | **2** | Fixed |
| 2 | Conditional independence | Severely | **Low** | **3** | Mitigated only |
| 4 | Fixed state count | Imposed | Medium | **4** | Fixed |
| 5a | Diagonal Gaussian emissions | Mildly | Medium | **5** | Fixed |
| 6 | Geometric durations | Yes | Medium | **6** | Botched → patch pending |
| 1 | First-order Markov | Barely | — | **7** | Exploited |
| 3b | Stationarity — *signer style* | Severely | Low here | **8** | Flagged only |

Note #2 sits third despite being the worst violation. That's the leverage term doing its job — nothing inside the HMM family fixes it.

---

## Assumption by assumption

### 1. First-order Markov — *exploited, not just satisfied*

**Why it matters:** it's the model's definition. Break it and you need higher-order or context-dependent HMMs.

**What I did:** rather than merely tolerating it, I used it as a *constraint*. The Bakis matrix hard-codes "signs run forward" as zeros:

```python
t[i, i]     = self_loop      # stay
t[i, i + 1] = 1.0 - self_loop  # advance
# everything else stays 0 -- backward transitions structurally impossible
```

v1's ergodic matrix spent 24 free parameters *discovering* that backward transitions don't happen. Encoding it as structure is free.

**Verified:** `np.allclose(np.tril(T, -1), 0)` passed.

---

### 2. Conditional independence — *the one I could not fix*

**Why it matters:** this is what caps HMM performance on video, full stop. The model assumes frame *t* tells you nothing about frame *t+1* once you know the state. At 80ms spacing that's flatly false, so evidence gets double-counted and your 350 frames per class behave like ~20–30 independent observations.

**What I did — three mitigations, no fix:**

*Absorb dependence into the observation.* Velocity features make $o_t$ a function of frames $t$ and $t-1$, so first-order temporal dependence moves *inside* the observation instead of being left for the state to carry:

```python
delta[1:] = np.diff(coords, axis=0)
```

This is the same trick speech HMMs use with delta-cepstra, for exactly this reason.

*Size the model to effective sample count, not nominal.* Every capacity decision — 3 states not 5, 16 PCA components not 30, frozen transitions — comes from assuming the real information content is ~10× below the frame count. 330 → 96 parameters.

*Stop trusting frame counts.* Cross-validation at the **sequence** level, never the frame level. Splitting on frames would leak catastrophically here.

**What I did not do:** fix it. You can't, within HMMs. This is the honest headline for your comparison against notebooks 03/04/05 — LSTMs and Transformers exist precisely because they let observations depend on each other directly instead of routing everything through one discrete state.

---

### 3. Stationarity — *split into two, handled asymmetrically*

**Why it matters:** if the same hidden state emits different observations depending on who's signing or where they stand, the emission Gaussians become mixtures of incompatible things and the state means drift toward meaningless averages.

**3a — camera and position: fixed.** Body-centring plus shoulder scaling (ADR-003):

```python
out[:, 0] = (block[:, 0] - origin_xy[0]) / scale
out[:, 1] = (block[:, 1] - origin_xy[1]) / scale
```

**Verified:** translation invariance max diff 4.4e-16, scale invariance exactly 0.

Also fixed a second-order case — when MediaPipe loses the pose, the reference frame carries forward instead of jumping:

```python
else:
    origin_xy, scale = fallback   # reuse last valid body frame
```

Without that, a dropped pose detection injects a discontinuity into every landmark simultaneously.

**3b — signer style: flagged, not fixed.** Requires signer-independent splits and WLASL signer metadata you haven't wired up. It's in section 17. Ignoring it means your CV number is optimistic.

---

### 4. Fixed discrete state count — *reduced and made honest*

**Why it matters:** $N$ has no correct value; you pick it. Every state costs $2D$ parameters, so overestimating burns your data budget, and at 67 sequences your selection procedure is itself noisy.

**What I did:** 5 → 3, derived from the parameter budget rather than tuned (you don't have the data to tune it). Then made the trade-off explicit rather than hidden — section 17 says raise it only past ~20 samples/class, and section 13 prints per-fold SD so you can see whether a change exceeds the ±0.06 noise floor.

---

### 5. Emission family — *split into two, one of them was fatal*

**Why 5b matters most:** without a variance floor, a state can collapse to zero variance, its density goes to $+\infty$, and one class hijacks every prediction. Combined with zero-sum transition rows producing $-\infty$ elsewhere, this is what made v1's scores meaningless.

```python
min_covar=MIN_COVAR   # 1e-3 -- v1 had no floor at all
```

Plus `sanitise_hmm` clipping post-fit, and — the actual load-bearing fix — `predict_from_scores` refusing to argmax over non-finite scores.

**Why 5a matters less:** `covariance_type="diag"` assumes features are conditionally independent given the state. PCA makes them *globally* uncorrelated, which isn't the same as conditionally uncorrelated within a state, but it's close enough. And the alternative is worse: full covariance at D=16 needs 136 numbers per state versus 16, and would go singular.

**Not handled:** unimodality. If two signers perform a stage differently, that state's true distribution is bimodal and a single Gaussian sits between both peaks, matching neither. The fix is `GMMHMM` — which needs more data, so it's premature.

---

### 6. Geometric durations — *the one I got wrong*

**Why it matters:** more than usual in my design, because I *froze* the transitions. A learned transition matrix partially self-corrects a bad initialization; a frozen one cannot.

**What I did:** picked `self_loop = 0.7` by convention and never checked it. Result: 81% of every sequence lands in the absorbing state, 1.86 of 3 states effectively used. I removed parameters to fit your data budget, then wasted a third of what remained.

**Correct approach:** derive from data, $p = 1 - N/\bar{T}$, recomputed per class.

**Real fix:** an HSMM models durations explicitly. Out of scope at 67 sequences.

---

## Two implicit assumptions I should also name

**Equal class priors.** `argmax` over per-class likelihoods is Bayes-optimal *only* under a uniform prior. Yours are 12–15 per class — near enough that it's benign, and I'd rather leave it uniform than fit priors on 67 samples. But if you scale to imbalanced classes, this becomes a real bug.

**Sequences i.i.d. given class.** `fit_class_models` concatenates all sequences of a class and passes `lengths`. That treats them as independent draws — which fails if the same signer contributes several clips of one gloss. Same root as 3b, same fix: signer-aware handling.

---

## The one-line version for your paper

> Of the six HMM assumptions, four were addressable through feature design and capacity control; the geometric-duration assumption is addressable only by moving to an HSMM; and conditional independence — the assumption most violated by video data — is not addressable within the HMM family at all, which is the principled motivation for the recurrent and attention-based models evaluated in notebooks 03–05.

Want me to apply the `self_loop` patch now, and add an "assumption audit" cell that prints the state-occupancy table plus detection rates on every run — so the assumptions get checked rather than asserted?


this that you call state which is N and 3 explain in simple human terms what it means

## A "state" is a phase of the sign

Think about how you'd describe the sign **drink** to someone:

1. Hand comes up toward the mouth
2. Hand forms a C-shape and holds near the lips
3. Hand tilts, then drops back down

Those three moments are what the model calls **states**. `N = 3` means: *"assume every sign is made of 3 phases, and figure out what they are."*

That's it. `N` is you telling the model **how many chapters to divide the sign into.**

---

## Why they're called "hidden"

You never labeled any of this. Your data is just: *this video → the word `drink`*. Nobody went through frame by frame writing "frames 1–8 are the hand rising, frames 9–20 are the C-shape at the mouth."

So the phases are **hidden** — the model has to discover them on its own by noticing that videos of `drink` tend to pass through three recognizably different-looking stretches. Baum-Welch is the algorithm doing that discovering.

Contrast: the *observations* are visible (you have the landmark numbers for every frame). The *phase each frame belongs to* is hidden.

---

## What a state actually contains

Each state is really just two things:

- **A typical pose** — "in phase 2, the hand is usually about here, in about this shape"
- **How much wiggle is allowed** — "but it can vary this much and still count"

In the code those are `means_` and `covars_`. That's the whole thing. Three states = three typical poses plus their tolerances.

This is also why states cost so much: each one stores 16 numbers for the typical pose plus 16 for the tolerance. Three states = 96 numbers to learn per sign. Five states would be 160 — and you only have ~10 videos per sign to learn them from.

---

## The confusion worth heading off

**States are not your 5 words.** This trips people up constantly.

You have **5 separate models**, one per gloss. Each has its own private 3 states:

```
model "before"     -> its own phase 1, phase 2, phase 3
model "computer"   -> its own phase 1, phase 2, phase 3
model "cool"       -> its own phase 1, phase 2, phase 3
model "cousin"     -> its own phase 1, phase 2, phase 3
model "drink"      -> its own phase 1, phase 2, phase 3
```

15 states in total, in 5 sealed groups. `drink`'s phase 2 and `cool`'s phase 2 have nothing to do with each other — they're not comparable, they're not shared, they never interact.

Classification works by asking all 5 models *"could you have produced this video?"* and taking whoever answers loudest. The states are the internal machinery each model uses to answer.

---

## What a trained model looks like on a real clip

Run Viterbi and it labels each frame with a phase:

```
frame:  1  2  3  4  5  6  7  8  9 10 11 12 ... 33 34 35
state:  0  0  0  0  0  1  1  1  1  1  1  2 ...  2  2  2
        └── hand rising ──┘└── C-shape held ──┘└─ lower ─┘
```

Two rules govern that row of numbers:

- **Stay or advance, never go back.** That's the left-to-right constraint — you can go `0 0 0 1 1 2`, never `0 1 0`. Signs run forward.
- **How long you linger in each phase** is set by `self_loop`.

---

## And here's where 0.7 shows up

That second rule is exactly the setting I got wrong. With `self_loop = 0.7`, a typical decoded sequence looks like:

```
state:  0  0  0  1  1  1  2  2  2  2  2  2  2  2  2  2 ... 2  2  2
        └─ 3 ─┘  └─ 3 ─┘  └────────────── 28 frames ──────────────┘
```

Phase 3 eats 81% of the sign. You bought three chapters and got one long one with two brief prologues. Fixing `self_loop` to ~0.914 spreads it out properly:

```
state:  0  0  0  0  0  0  0  0  0  0  0  1  1  1 ... 2  2  2  2
        └────── ~11 frames ──────┘  └─ ~9 ─┘  └──── ~15 ────┘
```

---

## So why 3 and not 10?

Purely a budget question. More phases = finer description of the sign = more numbers to learn:

| States | Numbers to learn per sign | Videos you have per sign |
|---|---|---|
| 3 | 96 | ~10 |
| 5 | 160 | ~10 |
| 10 | 320 | ~10 |

v1 used 5 states on 30-dimensional features — 330 numbers from ~10 videos. The model ran out of data, left phases completely unvisited, and that's what produced the `Some rows of transmat_ have zero sum` warnings. Those warnings were literally the model saying *"I was given phases I never had enough examples to see."*

Three is roughly the most your current data supports. Get to 30 videos per sign and 4–5 becomes reasonable — which is why section 17 ties raising `N_HMM_STATES` to a sample-count threshold rather than just suggesting you try it.


you diid split dataa for training and testiing how did you do it and how much was trainng and tstng also explain 2026-08-16 12:49:46,024 | INFO    | fold 1/5 | train=53 test=14 | models=5 | evr=0.888 | acc=0.2857 | abstain=0
2026-08-16 12:49:48,147 | INFO    | fold 2/5 | train=53 test=14 | models=5 | evr=0.891 | acc=0.5000 | abstain=0
2026-08-16 12:49:49,514 | INFO    | fold 3/5 | train=54 test=13 | models=5 | evr=0.891 | acc=0.5385 | abstain=0
2026-08-16 12:49:50,913 | WARNING | Model is not converging.  Current: -10442.179654981406 is not greater than -10442.179654599595. Delta is -3.818113327724859e-07
2026-08-16 12:49:51,099 | INFO    | fold 4/5 | train=54 test=13 | models=5 | evr=0.889 | acc=0.6923 | abstain=0
2026-08-16 12:49:52,519 | INFO    | fold 5/5 | train=54 test=13 | models=5 | evr=0.884 | acc=0.5385 | abstain=0

######################################################################
# Out-of-fold results | 67 sequences | 5-fold stratified CV
# Majority-class baseline = 

Good — it ran clean. Let me verify the arithmetic before interpreting.**0.2353 → 0.5075.** The collapse is gone: 5 of 5 classes predicted, 0 abstentions, every class above the 0.2239 baseline.

## How the splitting works

**v1** did one split and stopped:

```python
train_test_split(test_size=0.25, stratify=y_encoded)   # 50 train / 17 test
```

One model, evaluated once, on 17 videos. 50 of your 67 videos were never tested on at all.

**v2** rotates the test set through all the data:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

| Fold | Train | Test |
|---|---|---|
| 1 | 53 | 14 |
| 2 | 53 | 14 |
| 3 | 54 | 13 |
| 4 | 54 | 13 |
| 5 | 54 | 13 |
| | | **67 total** |

67 doesn't divide by 5 (`67 = 5×13 + 2`), so two folds get the spare videos. Every fold is **80% train / 20% test**, and every video is tested exactly once by a model that never saw it. That's 67 out-of-fold predictions instead of 17, and 25 HMMs trained instead of 5 (plus 5 more for the final saved model).

"Stratified" means each fold keeps the class mix intact — you never get a fold with no `drink` in it.

---

## Reading the log line

```
fold 1/5 | train=53 test=14 | models=5 | evr=0.888 | acc=0.2857 | abstain=0
```

**`models=5`** — all five classes produced a trainable HMM. In v1 this was nominally 5 but four were broken. Five healthy models is the first thing to check.

**`evr=0.888`** — the 16 PCA components retain 88.8% of the variance in your 309-D features.

This number is more interesting than it looks. **v1 reported `evr=0.9985` from 30 of 1629 dimensions** — and that was a warning sign, not a good result. Capturing 99.85% of the variance from 1.8% of the dimensions means the variance was concentrated in a handful of global modes (head position, body scale, framing) rather than distributed across real signal. 0.888 from 16 of 309 is *healthier*: the features are information-dense, so the components have to work harder.

**`abstain=0`** — no video had all five classes score `-inf`. The safety net I added never had to fire. That's the direct confirmation the `-inf` problem is gone, not merely hidden.

---

## The convergence warning is benign

```
Model is not converging. Current: -10442.179654981406
is not greater than -10442.179654599595. Delta is -3.818e-07
```

EM should increase log-likelihood every iteration. Here it went *down* — by 0.00000038, on a value of 10,442. That's a relative change of 3.7e-11, and it's **262× smaller than your `tol=1e-4`**.

This is floating-point round-off, not divergence. EM is summing over 54 sequences × ~35 frames × 16 dimensions, so accumulation error at 1e-7 is entirely normal. hmmlearn's message is poorly worded — it means *"converged to numerical precision"*, not *"failed."*

The tell: **fold 4 scored your highest accuracy, 0.6923.** Ignore it.

---

## What the results actually say

| | Accuracy |
|---|---|
| Random guess (5 classes) | 0.2000 |
| Majority baseline | 0.2239 |
| v1 | 0.2353 |
| **HMM v2** | **0.5075** (34/67) |
| **DTW 1-NN** | **0.6119** (41/67) |

Quote **0.5075**, not the 0.5110 fold-mean — they differ only because folds are unequally sized, and the pooled figure is the honest one.

**Fold 1 looks alarming but isn't.** 0.2857 against ~0.57 for the others is a gap of just **3.9 videos**. With 14-video folds, one video is worth 7 accuracy points. Your ±0.13 SD is almost entirely this one fold. Don't over-read it — but do report the spread, because a mean of 0.51 that ranges 0.29–0.69 is a genuinely different claim from a tight 0.51.

---

## DTW beats the HMM — and that's a real finding

The gap is 0.6119 vs 0.5075: **7 videos out of 67**.

Testing it honestly: unpaired z = 1.22, which is **not significant** at p < 0.05. A paired McNemar test would be more sensitive and might reach significance, but it needs the per-sample predictions — I can add that cell if you want a defensible claim.

What *is* solid is the consistency:

| gloss | HMM F1 | DTW F1 |
|---|---|---|
| before | 0.64 | 0.73 |
| computer | 0.48 | 0.56 |
| cool | **0.33** | **0.40** |
| cousin | 0.47 | 0.62 |
| drink | 0.61 | 0.72 |

DTW wins on **all five** classes, and the two methods rank class difficulty almost identically (r = 0.97). Two structurally unrelated algorithms agreeing that closely means **the difficulty lives in the data, not the model.**

Why DTW wins is the textbook bias-variance story. DTW fits **zero parameters** — it just measures similarity. The HMM must estimate **96 numbers per class from ~11 videos**. At this sample size, the flexible model loses to the rigid one.

This is worth stating plainly in your write-up rather than buried: *at 13 sequences per class, a non-parametric method outperforms a trained temporal model.* It also makes a testable prediction — your LSTM, ST-GCN and Transformer notebooks have *more* parameters than the HMM, so unless you grow the dataset, expect them to land at or below 0.61 too. If they report much higher, suspect leakage.

`cool` is your problem class in both methods (F1 0.33 / 0.40). Worth pulling those 13 videos and checking their hand-detection rates specifically — that's a data issue, not a modelling one.

---

## One caveat on your number

**You ran this with `self_loop = 0.7`** — the value I couldn't justify. So 0.5075 was achieved while ~81% of every sequence collapsed into a single absorbing state, using an effective 1.86 of 3 states.

Fixing it to a data-derived `1 - N/T` is the cheapest remaining experiment, and it plausibly closes part of the DTW gap. Want me to patch it and add a McNemar cell so you can state the HMM-vs-DTW comparison with a p-value? Both are small edits to the existing notebook.

is DTW different model from HMM or same but different sample size

**Completely different models — and deliberately identical sample size.** That's what makes the comparison meaningful.

## Same data, verified

Both methods get the exact same inputs inside each fold. From `run_cv`:

```python
tr_seqs = [tf(feature_seqs[i]) for i in tr_idx]   # same PCA features
te_seqs = [tf(feature_seqs[i]) for i in te_idx]

models = fit_class_models(tr_seqs, y[tr_idx], class_ids, ...)   # HMM
S = score_matrix(te_seqs, models, class_ids)
oof_hmm[te_idx] = predict_from_scores(S, class_ids)

oof_dtw[te_idx] = dtw_1nn_predict(tr_seqs, y[tr_idx], te_seqs)  # DTW -- same tr_seqs
```

Same 53–54 training sequences, same folds, same scaler, same PCA, same 16 dimensions. **Only the algorithm differs.** If I'd given DTW more data the comparison would be worthless.

## The analogy

**HMM** is studying by writing summary notes, then throwing the textbook away. It compresses ~11 videos per class down to 96 numbers — three typical poses plus tolerances — and answers new questions from those notes alone. The original videos are gone.

**DTW + 1-NN** keeps every past paper. When a new video arrives it flips through all 54 stored training sequences, finds the closest match, and copies that label. Nothing is ever summarised.

## Side by side

| | HMM | DTW + 1-NN |
|---|---|---|
| Type | parametric, generative, probabilistic | non-parametric, instance-based |
| Parameters fitted | 96 per class | **zero** |
| Training | Baum-Welch (EM), iterative | none — it just stores the data |
| Stored at test time | 5 small models | all 54 sequences |
| Cost per prediction | 5 cheap `score()` calls | 54 DTW alignments (much slower) |
| Output | log-likelihood (a probability) | a distance (not a probability) |
| Adding a 6th class | retrain | drop in the examples, done |
| Assumptions | the six we discussed | essentially one: same sign ⇒ alignable in time |

That last row is the crux. **DTW makes almost no assumptions**, so it has almost nothing to get wrong. It never assumes conditional independence, never assumes Gaussian emissions, never assumes geometric durations. Your entire assumption audit simply doesn't apply to it.

## They are cousins, though

There's a real connection worth knowing: **DTW is roughly Viterbi decoding on a degenerate HMM** — one where every frame of a stored template is its own state, transitions are strictly left-to-right, and "emission probability" is replaced by raw distance.

So think of it as an HMM with *maximum* states and *zero* estimation. The HMM compresses 35 frames into 3 states; DTW keeps all 35 as-is.

## Why the zero-parameter method won

Classic bias-variance. The HMM must estimate 96 numbers from ~11 videos — roughly 9 videos per 96 parameters. Every estimate carries error, and those errors compound. DTW estimates nothing, so it has no estimation error to accumulate.

At small *n*, the rigid method wins. That's expected, not embarrassing.

## When each would take over

**DTW's ceiling:** it never generalises. One mislabelled or badly-tracked training video becomes a permanent trap that captures every similar test video. Prediction cost grows linearly with your dataset — at 2000 WLASL glosses × 30 videos it would be unusable in real time.

**HMM's advantage arrives with data.** It compresses, so it stays fast regardless of dataset size, and it *smooths over* noisy examples rather than memorising them. It also gives you calibrated probabilities, which matters if you ever want to say "I'm not confident" or chain signs into sentences.

**Crossover is roughly 30–50 sequences per class.** You're at 13.

## What to write down

> HMM and DTW are structurally distinct approaches evaluated on identical folds and identical features. The non-parametric method outperforms the parametric one (0.6119 vs 0.5075), and the two agree closely on per-class difficulty (r = 0.97), indicating the limiting factor is dataset size rather than model choice.

That framing does real work for you: it means adding *more parameters* — LSTM, ST-GCN, Transformer — is pushing in the wrong direction until the data grows. Fixing your video download is worth more than any architecture change on that list.



