"""Per-model deep-dive PDFs."""
import json
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak
from pdfkit_ksl import (NumberedDoc, titleblock, H1, H2, P, sp, rule, bullets,
                        figure, table, callout, code, caption, styles,
                        BLUE, ORANGE, AQUA, RED, GREEN, VIOLET)

R = json.load(open("results.json")); L = json.load(open("lang_results.json"))
M = R["models"]; base = R["meta"]["majority_baseline"]; classes = R["meta"]["classes"]
def A(m): return f"{M[m]['accuracy']:.3f}"
def F(m): return f"{M[m]['weighted_f1']:.3f}"
def W(m): return f"{M[m]['wer']:.3f}"
def SD(m): return f"±{M[m]['fold_acc_std']:.3f}"
def folds(m): return ", ".join(f"{x:.2f}" for x in M[m]["fold_accuracies"])

def result_table(key):
    d = M[key]
    rows = [["Metric","Value (5-fold CV)"],
            ["Accuracy", A(key)], ["Weighted precision", f"{d['weighted_precision']:.3f}"],
            ["Weighted recall", f"{d['weighted_recall']:.3f}"], ["Weighted F1", F(key)],
            ["Word Error Rate (WER)", W(key)],
            ["Fold accuracies", folds(key)],
            ["Cross-val SD", SD(key)],
            ["Distinct classes predicted", f"{d['distinct_classes_predicted']} of 5"],
            ["Majority baseline", f"{base:.3f}"]]
    return table(rows, [55*mm, 108*mm], align="LEFT")

def build(fname, title, subtitle, meta, accent, body):
    story = titleblock(title, subtitle, meta, accent)
    for el in body:                       # flatten any nested lists (e.g. bullets())
        if isinstance(el, list):
            story += el
        else:
            story.append(el)
    doc = NumberedDoc(f"/home/claude/ksl/pdfs/{fname}",
                      footer=f"KSL Model Analysis · {title.split('—')[0].strip()}")
    doc.build(story); print("built", fname)

# ==========================================================================
# 1. RANDOM FOREST
# ==========================================================================
build("01_RandomForest_Analysis.pdf",
      "Random Forest — Non-Temporal Statistical Baseline",
      "Per-model deep dive · the control condition for every temporal model",
      [f"Result: {A('Random Forest')} accuracy · {F('Random Forest')} weighted F1 · WER {W('Random Forest')} · fold SD {SD('Random Forest')}",
       "Notebook: RandomForest.ipynb (notebook 01)"], GREEN, [
    H1("1&nbsp;&nbsp;Theory"),
    P("A Random Forest is an ensemble of decision trees, each trained on a bootstrap sample of the data and a "
      "random subset of features, voting on the final class. It is a strong, low-variance classifier for "
      "tabular data of modest size. Crucially for this study it is <b>non-temporal</b>: it sees one fixed-length "
      "feature vector per video and has no notion of frame order. That makes it the deliberate <b>control "
      "condition</b> — the score a temporal model must beat to justify modelling time at all."),
    H1("2&nbsp;&nbsp;Method of execution"),
    P("Each video is a sequence of 1,629-D MediaPipe landmark frames. The notebook collapses that sequence into "
      "a single vector by taking four statistics across the time axis — mean, standard deviation, minimum and "
      "maximum of every one of the 1,629 dimensions — giving a <b>6,516-D</b> summary. This is standardized and "
      "fed to a class-balanced forest of 300 trees."),
    code("stats = [frames.mean(0), frames.std(0), frames.min(0), frames.max(0)]  # 4 × 1629\n"
         "vector = concatenate(stats)                                            # 6516-D\n"
         "RandomForestClassifier(n_estimators=300, class_weight='balanced')"),
    P("The key modelling decision (and deliberate weakness) is the pooling step: <b>mean/std/min/max destroy the "
      "order of frames.</b> A sign performed forwards and the same poses shuffled in time produce an identical "
      "feature vector. That is exactly the information a sign carries, thrown away on purpose to establish the "
      "baseline."),
    H1("3&nbsp;&nbsp;Results"),
    result_table("Random Forest"), sp(4),
    figure("m_rf.png", "Random Forest confusion matrix and per-class F1 under 5-fold CV.", 158*mm),
    H1("4&nbsp;&nbsp;Why these results"),
    P(f"At {A('Random Forest')} the forest sits mid-table — above the HMM, BiLSTM and ST-GCN, below the "
      "Transformer and DTW. It clears the baseline comfortably because the <i>average</i> and <i>spread</i> of "
      "hand and body position already separate these five words to a useful degree, and forests are robust when "
      "data is scarce. Its low fold spread confirms it is a stable estimator. But it has a hard ceiling: any two "
      "signs that share pose statistics and differ only in movement order are indistinguishable to it — no "
      "amount of extra trees fixes that."),
    callout("What makes it stronger / weaker",
      "<b>Stronger:</b> robust on tiny data, fast, no temporal assumptions to violate, naturally handles the "
      "high-dimensional vector. <b>Weaker:</b> temporally blind by construction — the ceiling is set by how much "
      "signs happen to differ in static statistics. It is a yardstick, not a candidate for the final system.", GREEN),
    H1("5&nbsp;&nbsp;Role in the hybrid"),
    P("Random Forest is <b>not</b> a component of the recommended hybrid — its value is diagnostic. Because it "
      "captures only static, order-free information, the gap between it and a temporal model measures how much a "
      "given architecture actually extracts from motion. That the trained two-stream hybrid clears it by ~15 "
      "points is direct evidence the hybrid is using temporal structure, not just average pose.")])

# ==========================================================================
# 2. HMM  (richest — includes assumptions)
# ==========================================================================
build("02_HMM_Analysis.pdf",
      "Hidden Markov Model — Probabilistic Temporal Baseline",
      "Per-model deep dive · the −∞ collapse, the corrected pipeline, and the six HMM assumptions",
      [f"Result (corrected): {A('HMM (corrected)')} accuracy · {F('HMM (corrected)')} weighted F1 · WER {W('HMM (corrected)')} · fold SD {SD('HMM (corrected)')}",
       "Notebooks: 02_hmm_wlasl_baseline.ipynb (original) → 022_hmm_wlasl_baselin.ipynb (corrected)"], VIOLET, [
    H1("1&nbsp;&nbsp;Theory"),
    P("An HMM models a sign as a walk through a small number of <b>hidden stages</b> (for <font face='Courier'>"
      "drink</font>: hand rises → C-shape at mouth → lowers). Each stage emits a typical pose with some "
      "tolerance (a Gaussian), and a transition matrix governs which stage may follow which. One HMM is trained "
      "per class; a new video is scored by every class model and assigned to whichever class was most likely to "
      "have produced it (argmax of log-likelihood). It is generative, interpretable, and — once corrected — very "
      "parameter-light."),
    callout("The original model collapsed — and it was a bug, not just weak performance.",
      "As first run, the HMM scored 0.235 and predicted <font face='Courier'>before</font> for all 17 test "
      "videos. Cause: four of the five class-models trained with zero-sum transition rows, so their "
      "<font face='Courier'>score()</font> returned −∞ for every test video. The prediction code did "
      "<font face='Courier'>max(scores)</font> over values that were all −∞; Python returns the <i>first</i> key "
      "on a tie, and <font face='Courier'>before</font> is alphabetically first. A single missing "
      "<font face='Courier'>isfinite()</font> check turned a broken model into a confident-looking one that "
      "output 0.235 = the majority-class baseline exactly.", RED),
    H1("2&nbsp;&nbsp;Method of execution — the corrected pipeline"),
    P("The corrected model (notebook 022) keeps the expensive MediaPipe cache and rebuilds the modelling half:"),
    bullets([
      "<b>Drop the face.</b> 1,404 of the 1,629 numbers were face landmarks (86%); signs here are decided by the "
      "hands, so the face is removed and only upper-body pose + both hands are kept.",
      "<b>Body-centre &amp; scale.</b> Coordinates are re-centred on the mid-shoulder and divided by shoulder "
      "width, so camera distance and signer position no longer change the features.",
      "<b>Presence flags + velocity.</b> A binary flag marks a missing hand (instead of a misleading zero-fill), "
      "and frame-to-frame deltas add motion the static frame lacks.",
      "<b>16 PCA components</b>, then a <b>3-state left-to-right (Bakis)</b> Gaussian HMM with a variance floor "
      "(<font face='Courier'>min_covar</font>) and <b>length-normalised, finite-checked scoring</b> that "
      "abstains rather than silently defaulting to class 0."]),
    P("A left-to-right topology means a sign can only stay in a stage or advance — never run backwards — which "
      "encodes a fact we already know (signs run forward) for free and makes the zero-sum collapse "
      "structurally impossible."),
    H1("3&nbsp;&nbsp;Results"),
    result_table("HMM (corrected)"), sp(4),
    figure("m_hmm.png", "Corrected HMM confusion matrix and per-class F1. The diagonal is present but weak — the "
           "model is the least accurate of the proper set.", 158*mm),
    callout("On the naive vs corrected numbers — an honest note.",
      f"Under cross-validation the <i>corrected</i> HMM scores {A('HMM (corrected)')} and always predicts all "
      f"five classes (no collapse, guaranteed). The naive configuration, re-run across CV folds, does not "
      f"collapse on every split, so its pooled number can look higher — but that is exactly the instability of a "
      f"single hold-out. The correction's value is <b>reliability</b>: no −∞, no class-0 default, lower variance, "
      f"and interpretable topology. Raw accuracy at ~13 examples per class is capped by theory, below.", VIOLET),
    PageBreak(),
    H1("4&nbsp;&nbsp;Why these results — the six HMM assumptions against video"),
    P("The HMM is the weakest proper model here for a principled reason: video violates its core assumptions. "
      "All six, and how each holds up:"),
    table([["#","Assumption","Holds for video?","Impact"],
      ["1","First-order Markov (next stage depends only on current)","Reasonable","Low — encoded by Bakis topology"],
      ["2","Conditional independence (frame independent given stage)","Badly violated","HIGH — the core limitation"],
      ["3","Stationarity (same emissions across signers/time)","Violated across signers","Medium"],
      ["4","Fixed discrete number of states","Imposed, not true","Medium — tuning noise"],
      ["5","Gaussian, diagonal emissions","Approximate","Low–medium"],
      ["6","Geometric state durations","False","Medium — fixable via self-loop / HSMM"]],
      [8*mm,66*mm,42*mm,44*mm], align="LEFT", font=7.9),
    caption("Table. The six assumptions. #2 is the one that caps HMMs on video."),
    P("<b>Assumption 2 is the killer.</b> The HMM assumes that, once you know the hidden stage, the current frame "
      "tells you nothing about the next. But frames sampled ~80 ms apart are near-duplicates, so each carries "
      "little new information. The practical effect: ~13 videos × ~35 frames <i>looks</i> like a lot of data but "
      "behaves like maybe 20–30 independent observations. Baum-Welch is starved, states go unvisited, and "
      "accuracy plateaus. Adding velocity helps a little; nothing inside the HMM family fixes it. <b>This is the "
      "principled reason the field moved to LSTMs, ST-GCNs and Transformers</b> — they let frames depend on each "
      "other directly instead of routing everything through one discrete stage."),
    H1("5&nbsp;&nbsp;What makes it stronger / weaker"),
    callout("Summary",
      "<b>Stronger:</b> interpretable (you can read the stages off a Viterbi decode), gives calibrated "
      "probabilities useful for chaining signs into sentences, and — once corrected — needs only ~96 parameters "
      "per class. <b>Weaker:</b> the conditional-independence and geometric-duration assumptions cap it on video; "
      "at this data scale it is beaten even by a zero-parameter memoriser (DTW).", VIOLET),
    H1("6&nbsp;&nbsp;Role in the hybrid"),
    P("The HMM is not part of the recommended recognition core, but it earns two roles. First, <b>as evidence</b>: "
      "its assumption audit is the theoretical argument for why the hybrid's attention and graph streams are "
      "needed — they exist precisely to drop assumption 2. Second, <b>as a future probabilistic layer</b>: HMMs "
      "shine at sequencing and giving calibrated confidence, so once recognition feeds continuous signing, an "
      "HMM/HSMM layer over recognized glosses is a natural way to model sentence structure — a role that plays to "
      "its strengths rather than its weakness.")])

# ==========================================================================
# 3. BiLSTM
# ==========================================================================
build("03_BiLSTM_Analysis.pdf",
      "Bidirectional LSTM — Deep Recurrent Model",
      "Per-model deep dive · learned temporal memory over the landmark sequence",
      [f"Result: {A('BiLSTM')} accuracy · {F('BiLSTM')} weighted F1 · WER {W('BiLSTM')} · fold SD {SD('BiLSTM')}",
       "Notebook: 03_lstm_wlasl_landmark_baseline.ipynb"], ORANGE, [
    H1("1&nbsp;&nbsp;Theory"),
    P("A Long Short-Term Memory network processes a sequence frame by frame, maintaining a gated memory cell that "
      "decides what to keep, forget and output. A <b>bidirectional</b> LSTM runs two of these — one forward, one "
      "backward — so the representation of any frame is informed by both its past and its future. This is the "
      "first model in the study to genuinely drop the HMM's frame-independence assumption: information flows "
      "directly from frame to frame through the memory, not through a single discrete stage."),
    H1("2&nbsp;&nbsp;Method of execution"),
    P("The 1,629-D landmark frames are passed through a linear input projection, then a 2-layer bidirectional "
      "LSTM (hidden size 128, dropout 0.3). The final forward and backward hidden states are concatenated and "
      "fed to a small MLP classifier. Padding frames are masked via packed sequences so the recurrence only sees "
      "real frames; training uses Adam with weight decay."),
    code("BiLSTM(input→proj→2×biLSTM(128)→concat(final states)→MLP→5 classes)\n"
         "Adam(lr=1e-3, weight_decay=1e-4), dropout 0.3, masked/packed sequences"),
    H1("3&nbsp;&nbsp;Results"),
    result_table("BiLSTM"), sp(4),
    figure("m_lstm.png", "BiLSTM confusion matrix and per-class F1 under 5-fold CV.", 158*mm),
    H1("4&nbsp;&nbsp;Why these results"),
    P(f"At {A('BiLSTM')} the BiLSTM clears the baseline but posts <b>the highest fold-to-fold variance of any "
      f"model</b> (SD {SD('BiLSTM')}; folds: {folds('BiLSTM')}). Recurrent networks have many parameters and "
      "need substantial data to constrain them; with ~54 training videos the model lands in very different "
      "places depending on the split — one fold reaches 0.69, another 0.31. Its central tendency is modest not "
      "because recurrence is wrong for sign, but because there is far too little data to train it well."),
    callout("What makes it stronger / weaker",
      "<b>Stronger:</b> a proper temporal model that captures order and long-range dependence; the architecture "
      "that scales best as data grows. <b>Weaker:</b> data-hungry and high-variance at this scale; a long "
      "recurrence over a high-dimensional input is hard to pin down on 13 examples per class.", ORANGE),
    H1("5&nbsp;&nbsp;Role in the hybrid"),
    P("The BiLSTM is included in the soft-vote ensemble but is <b>not</b> part of the recommended two-stream "
      "hybrid, which pairs the graph (ST-GCN) and attention (Transformer) streams instead. The reason is "
      "variance: the two-stream design deliberately combines the <i>lowest-variance</i> deep model (ST-GCN) with "
      "the <i>most expressive</i> one (Transformer). The BiLSTM's high variance would add noise rather than a "
      "complementary signal. It becomes a strong candidate again once the dataset is large enough to tame it.")])

# ==========================================================================
# 4. ST-GCN
# ==========================================================================
build("04_STGCN_Analysis.pdf",
      "ST-GCN — Spatial-Temporal Graph Convolutional Network",
      "Per-model deep dive · the anatomical prior, and the low-variance workhorse",
      [f"Result: {A('ST-GCN')} accuracy · {F('ST-GCN')} weighted F1 · WER {W('ST-GCN')} · fold SD {SD('ST-GCN')}",
       "Notebook: 04_stgcn_wlasl_landmark_baseline.ipynb"], AQUA, [
    H1("1&nbsp;&nbsp;Theory"),
    P("An ST-GCN treats each frame not as a flat vector but as a <b>skeleton graph</b>: joints are nodes, and "
      "the body's physical connections (wrist–elbow, finger joints) are edges. It convolves over that graph in "
      "space (a joint's neighbours inform it) and over time (a temporal convolution across frames). The "
      "inductive bias is anatomical: connected joints move together, and this structure is given to the model "
      "rather than learned from scratch — a powerful prior when data is scarce."),
    H1("2&nbsp;&nbsp;Method of execution"),
    P("Each frame is a 75-node graph (33 pose + 21 + 21 hand joints; face dropped) with 3 coordinates per node. "
      "A hand-built adjacency matrix encodes the MediaPipe pose and hand edges, adds self-loops, and is "
      "symmetrically normalized. Stacked ST-GCN blocks each perform a spatial graph convolution "
      "(<font face='Courier'>einsum</font> over the adjacency) followed by a temporal convolution, with "
      "BatchNorm, ReLU, dropout and a residual path; a global average pool feeds a linear classifier. The graph "
      "convolution is written directly — no external graph-learning library."),
    code("A = normalize(pose_edges + hand_edges + self_loops)     # 75×75\n"
         "block: x → einsum('nctv,vw->nctw', x, A) → 1×1 conv → temporal conv(k) → BN/ReLU/res\n"
         "3 blocks (3→32→64→64) → global pool → linear → 5 classes"),
    H1("3&nbsp;&nbsp;Results"),
    result_table("ST-GCN"), sp(4),
    figure("m_stgcn.png", "ST-GCN confusion matrix and per-class F1 under 5-fold CV.", 158*mm),
    H1("4&nbsp;&nbsp;Why these results"),
    P(f"ST-GCN reaches {A('ST-GCN')} with the <b>lowest fold variance of any deep model</b> (SD {SD('ST-GCN')}). "
      "That stability is the story: the skeleton graph is a correct, strong prior, so the network does not have "
      "to discover body structure from 54 videos — it is told. That keeps learning well-behaved where the LSTM "
      "and Transformer swing wildly across folds. Its accuracy ceiling is held down here by two things: the face "
      "(and its mouthing cues) is dropped from the graph, and the sample is tiny. It is the most <i>reliable</i> "
      "deep model even though it is not the most accurate single one."),
    callout("What makes it stronger / weaker",
      "<b>Stronger:</b> anatomically grounded, parameter-efficient, and the most stable deep model at this data "
      "scale — exactly the property you want in a component you will fuse. <b>Weaker:</b> the fixed graph can't "
      "learn long-range frame-to-frame relations the way attention can, and dropping the face loses "
      "facial-expression signal that matters for some signs.", AQUA),
    H1("5&nbsp;&nbsp;Role in the hybrid — a core component"),
    P("ST-GCN is <b>one of the two streams in the recommended hybrid.</b> It contributes the stable, low-variance "
      "anatomical representation, while the Transformer stream contributes temporal expressiveness. Fusing their "
      "features gives the hybrid ST-GCN's reliability together with the Transformer's ceiling — which is why the "
      "two-stream model tops the study. In short: ST-GCN is the hybrid's <b>backbone of stability</b>.")])

# ==========================================================================
# 5. TRANSFORMER
# ==========================================================================
build("05_Transformer_Analysis.pdf",
      "Transformer Encoder — Self-Attention Model",
      "Per-model deep dive · the best single architecture and the hybrid's namesake",
      [f"Result: {A('Transformer')} accuracy · {F('Transformer')} weighted F1 · WER {W('Transformer')} · fold SD {SD('Transformer')}",
       "Notebook: 05_transformer_wlasl_landmark_baseline.ipynb"], BLUE, [
    H1("1&nbsp;&nbsp;Theory"),
    P("A Transformer encoder uses <b>self-attention</b>: every frame computes a weighted combination of every "
      "other frame, with the weights learned from the data. Unlike the LSTM (which passes information "
      "step-by-step) or the ST-GCN (which uses a fixed graph), attention lets any frame relate directly to any "
      "other, at any distance, with no recurrence and no hand-specified structure. It is the most flexible "
      "temporal model — and the architecture the thesis's 'Hybrid Multimodal Transformer' is built around."),
    H1("2&nbsp;&nbsp;Method of execution"),
    P("The 1,629-D frames are linearly projected to a 128-D model space and given <b>sinusoidal positional "
      "encodings</b> (so the otherwise order-agnostic attention knows frame order). A 2-layer, 4-head "
      "Transformer encoder (feed-forward 256, GELU, dropout 0.3) processes the sequence with padding masked out; "
      "the valid frames are mean-pooled, LayerNorm-ed, and classified by a linear head."),
    code("frames(1629) → Linear(128) → +positional encoding → TransformerEncoder(2 layers, 4 heads)\n"
         "→ masked mean-pool → LayerNorm → Linear → 5 classes"),
    H1("3&nbsp;&nbsp;Results"),
    result_table("Transformer"), sp(4),
    figure("m_transformer.png", "Transformer confusion matrix and per-class F1 under 5-fold CV.", 158*mm),
    H1("4&nbsp;&nbsp;Why these results"),
    P(f"The Transformer is the <b>best individual architecture</b> at {A('Transformer')}. Attention captures the "
      "whole-sign temporal structure — the relationship between the wind-up and the stroke, however far apart — "
      "that averaging (RF) and stage-independence (HMM) cannot. Its fold spread is high (SD "
      f"{SD('Transformer')}; folds {folds('Transformer')}), because attention is data-hungry and 54 videos "
      "under-determine it, but its central tendency is the strongest of the single models. As data grows it is "
      "the architecture with the most headroom."),
    callout("What makes it stronger / weaker",
      "<b>Stronger:</b> the most expressive temporal model, no fixed structural assumptions, the natural "
      "backbone for a full KSL system. <b>Weaker:</b> the highest appetite for data of any model here — its "
      "variance will only come down with more examples.", BLUE),
    H1("5&nbsp;&nbsp;Role in the hybrid — a core component"),
    P("The Transformer is <b>the second stream of the recommended hybrid.</b> It supplies temporal expressiveness "
      "— the ability to weigh any frame against any other — which it pairs with ST-GCN's anatomical stability. "
      "The two are complementary: attention is powerful but high-variance, the graph is stable but structurally "
      "constrained, and jointly training their fused features lets the model take the best of both. This is the "
      "empirical and architectural heart of the 'Hybrid Multimodal Transformer'.")])

# ==========================================================================
# 6. DTW
# ==========================================================================
build("06_DTW_Analysis.pdf",
      "DTW + 1-NN — Non-Parametric Reference",
      "Per-model deep dive · the zero-parameter memoriser that beats every trained single model",
      [f"Result: {A('DTW 1-NN')} accuracy · {F('DTW 1-NN')} weighted F1 · WER {W('DTW 1-NN')} · fold SD {SD('DTW 1-NN')}",
       "Method: Dynamic Time Warping distance + 1-nearest-neighbour (added as the honest baseline)"], GREEN, [
    H1("1&nbsp;&nbsp;Theory"),
    P("Dynamic Time Warping measures how similar two sequences are while allowing them to be stretched or "
      "compressed in time — so the same sign performed slowly or quickly still matches. Paired with "
      "<b>1-nearest-neighbour</b>, classification is simply: find the most similar training video and copy its "
      "label. It fits <b>zero parameters</b> — it stores the training data and compares. It makes essentially "
      "one assumption (two performances of the same sign can be aligned in time), so there is almost nothing for "
      "it to get wrong."),
    P("There is a neat connection to the HMM: DTW is roughly Viterbi decoding on a degenerate HMM where every "
      "frame of a stored template is its own state. Where the HMM compresses 35 frames into 3 states, DTW keeps "
      "all 35 — maximum states, zero estimation."),
    H1("2&nbsp;&nbsp;Method of execution"),
    P("Each video is reduced to the same 16-D PCA landmark features used by the corrected HMM. A test video's "
      "DTW distance to every training video is computed via the standard dynamic-programming recurrence, and the "
      "nearest training video's label is assigned. It is evaluated on the identical folds as every other model, "
      "so the comparison is exact."),
    code("D[i,j] = dist(a_i, b_j) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])   # DTW recurrence\n"
         "predict(test) = label of argmin over training videos of DTW(test, train)"),
    H1("3&nbsp;&nbsp;Results"),
    result_table("DTW 1-NN"), sp(4),
    figure("m_dtw.png", "DTW + 1-NN confusion matrix and per-class F1 under 5-fold CV.", 158*mm),
    H1("4&nbsp;&nbsp;Why these results"),
    P(f"DTW is the <b>strongest single method</b> at {A('DTW 1-NN')}, beating every trained architecture. This is "
      "the textbook bias–variance trade-off: with ~13 examples per class, every parameter a model estimates "
      "carries error, and those errors compound. DTW estimates <i>nothing</i>, so it has no estimation error to "
      "accumulate — the rigid method wins at small sample size. That two structurally unrelated methods (DTW and "
      "the HMM) agree closely on which classes are hard is itself evidence that the difficulty lives in the "
      "data, not the model."),
    callout("What makes it stronger / weaker",
      "<b>Stronger:</b> unbeatable at tiny scale, no assumptions to violate, trivial to extend with new classes. "
      "<b>Weaker:</b> it never generalizes or compresses — prediction cost grows with the dataset (every test "
      "video compared to all training videos) and it gives distances, not calibrated probabilities. Unusable in "
      "real time at 2,000-gloss scale.", GREEN),
    H1("5&nbsp;&nbsp;Role in the hybrid"),
    P("DTW is the <b>honest baseline and low-data fallback</b>, not a component of the trained hybrid. Its "
      "importance is as the bar to beat: any trained model — including the hybrid — must exceed DTW to justify "
      "its parameters. The two-stream hybrid is the <b>only</b> model that clears it, which is what makes the "
      "hybrid's result meaningful. Keep DTW in the study as the reference against which real progress is "
      "measured, and expect trained models to overtake it decisively only once the dataset reaches ~30–50 "
      "videos per class.")])

# ==========================================================================
# 7. HYBRID
# ==========================================================================
build("07_Hybrid_Model_Analysis.pdf",
      "Hybrid Multimodal Transformer — The Recommended Recognition Core",
      "Per-model deep dive · design, training, and why it beats every component",
      [f"Result: {A('Hybrid (Two-Stream)')} accuracy · {F('Hybrid (Two-Stream)')} weighted F1 · WER {W('Hybrid (Two-Stream)')} · fold SD {SD('Hybrid (Two-Stream)')}",
       "Two-stream feature-level fusion: ST-GCN (graph) + Transformer (attention), trained end-to-end"], ORANGE, [
    H1("1&nbsp;&nbsp;Theory and design"),
    P("The hybrid combines two complementary views of a sign into one trained model. The <b>graph stream</b> "
      "(ST-GCN) encodes anatomical structure — how connected joints move together — and is the most stable deep "
      "model in the study. The <b>attention stream</b> (Transformer) encodes long-range temporal relationships — "
      "how any moment of the sign relates to any other — and is the most expressive single model. Each stream "
      "produces an embedding; the two embeddings are <b>concatenated (feature-level fusion)</b> and a fusion "
      "head classifies. Because both streams are trained together, the head learns how structure and motion "
      "<i>interact</i>, not just their separate votes."),
    code("graph  : skeleton(75 joints) → ST-GCN blocks → pooled embedding (64)\n"
         "landmark: hands+pose(225) → Transformer encoder → pooled embedding (96)\n"
         "fuse    : concat[64+96] → MLP → 5 classes   (trained end-to-end)"),
    callout("Two fusion strategies were tested — and the choice matters.",
      f"<b>Feature-level (two-stream):</b> fuse the streams' internal embeddings and train jointly → "
      f"{A('Hybrid (Two-Stream)')}. <b>Decision-level (soft-vote):</b> train BiLSTM/ST-GCN/Transformer "
      f"separately and average their output probabilities → {A('Hybrid (Soft-Vote)')}. Feature fusion wins "
      f"decisively because it can learn cross-stream interactions; soft-voting only averages three data-starved "
      f"models and mostly averages their noise.", ORANGE),
    H1("2&nbsp;&nbsp;Results"),
    result_table("Hybrid (Two-Stream)"), sp(4),
    figure("m_hybrid.png", "Two-stream hybrid confusion matrix and per-class F1 — the strongest diagonal in the "
           "study and the only model with F1 ≥ 0.54 on every class.", 158*mm),
    H1("3&nbsp;&nbsp;Why it wins"),
    P(f"The hybrid reaches {A('Hybrid (Two-Stream)')} / F1 {F('Hybrid (Two-Stream)')} — above the best single "
      f"deep model (Transformer, {A('Transformer')}) and the non-parametric baseline (DTW, {A('DTW 1-NN')}). It "
      "inherits ST-GCN's low variance <i>and</i> the Transformer's expressiveness: the graph stream keeps the "
      "model anchored when data is scarce, while the attention stream lifts the ceiling. It is also the most "
      "<b>balanced</b> model — the only one scoring F1 ≥ 0.54 on all five classes, where every single model has "
      "at least one class it fails badly."),
    H1("4&nbsp;&nbsp;Honest limitations"),
    P(f"The hybrid's fold spread is the highest in the study (SD {SD('Hybrid (Two-Stream)')}; folds "
      f"{folds('Hybrid (Two-Stream)')}) — its lead is driven partly by one very strong fold. The direction is "
      "clear and consistent (it tops the pooled ranking), but at ~13 examples per class the <i>margin</i> is not "
      "yet statistically firm. A paired McNemar test against DTW and, above all, more data are needed before the "
      "advantage is stated as a headline number. It also currently drops the face; adding a facial-expression "
      "stream is a natural third modality."),
    H1("5&nbsp;&nbsp;Recommendation"),
    callout("This is the model to carry forward.",
      "Build the recognition core of the KSL system as this <b>two-stream ST-GCN + Transformer hybrid</b>. It is "
      "the only model that empirically beats the DTW baseline, it is the most balanced across classes, and its "
      "architecture is exactly the 'Hybrid Multimodal Transformer' the thesis proposes — now with measured "
      "evidence behind it. Immediate next steps: signer-independent evaluation, a paired significance test, more "
      "data per class, and a third (facial) stream.", ORANGE)])

# ==========================================================================
# 8. LANGUAGE LAYER
# ==========================================================================
def lrow(d): return [f"{d['BLEU']:.1f}", f"{d['ROUGE1']:.1f}", f"{d['ROUGE2']:.1f}",
                     f"{d['ROUGEL']:.1f}", f"{d['METEOR']:.1f}", f"{d['WER']:.1f}", f"{d['exact_match']:.0f}%"]
lang_rows = [["Condition / translator","BLEU","R-1","R-2","R-L","METEOR","WER↓","Exact"],
    ["Exact rule-based — clean"]+lrow(L["clean"][0]),
    ["Fuzzy overlap — clean"]+lrow(L["clean"][1]),
    ["Exact rule-based — noisy"]+lrow(L["noisy"][0]),
    ["Fuzzy overlap — noisy"]+lrow(L["noisy"][1]),
    ["Fuzzy nearest-NN — generalization"]+lrow(L["generalization"][0])]
build("08_Language_Translation_Layer_Analysis.pdf",
      "Gloss → Healthcare Text — The Language Layer",
      "Per-model deep dive · where BLEU, ROUGE, METEOR and WER actually apply",
      ["Task: turn a recognized gloss (e.g. 'pain where') into a natural sentence ('Where is the pain?')",
       "Models: exact rule-based dictionary · fuzzy token-overlap matcher · mT5 (planned)"], BLUE, [
    H1("1&nbsp;&nbsp;Why this layer needs different metrics"),
    P("Recognition outputs a <i>label</i>; this layer outputs a <i>sentence</i>. That is why the translation "
      "metrics your review asked for — BLEU, ROUGE, METEOR — live here and not on the recognition models. Each "
      "compares generated text to a reference: BLEU on n-gram precision, ROUGE on n-gram/subsequence recall, "
      "METEOR on unigram matches with a word-order penalty, and WER on word-level edits. Reporting them on a "
      "classifier's single-label output would be a category error."),
    H1("2&nbsp;&nbsp;Method of execution"),
    P("Two baseline translators are evaluated on the 20-phrase healthcare set. The <b>exact rule-based</b> "
      "translator looks up the normalized gloss in a dictionary. The <b>fuzzy token-overlap</b> translator scores "
      "each stored gloss by Jaccard token overlap and returns the best match above a threshold — so it tolerates "
      "reordered or partially-dropped words. Both are tested three ways: on clean gloss input, on simulated noisy "
      "recognizer output (words reordered / dropped), and on a leave-one-out generalization test where the exact "
      "phrase is removed and the translator must retrieve the nearest <i>different</i> one."),
    H1("3&nbsp;&nbsp;Results"),
    table(lang_rows, [52*mm,14*mm,12*mm,12*mm,12*mm,17*mm,14*mm,15*mm], align="CENTER",
          highlight_rows={5:[0,1,2,3,4,5,6,7]}, font=8.0), sp(2),
    caption("All scores 0–100. R-1/R-2/R-L = ROUGE-1/2/L. Highlighted row = the generalization test."),
    figure("5_language.png", "Translation quality across the three conditions.", 155*mm),
    H1("4&nbsp;&nbsp;Why these results"),
    P("On known phrases both translators are perfect (BLEU 100, WER 0). The moment input is perturbed — which is "
      "what an imperfect recognizer produces — the exact matcher collapses (BLEU 0, WER 93, 10% coverage) while "
      "the fuzzy matcher recovers every phrase (BLEU 100, WER 0). <b>This is the empirical case for "
      "order-insensitive matching downstream of recognition.</b> The generalization row is the honest one: "
      "asked for a phrase it has never seen, the lookup translator retrieves a near-neighbour, scoring BLEU 9.8 "
      "/ ROUGE-L 39.5 / METEOR 37.3 — the metrics correctly registering a partial match. A dictionary cannot "
      "compose novel wordings; that is the precise gap a trainable model fills."),
    callout("What makes it stronger / weaker, and the next step",
      "<b>Stronger:</b> the fuzzy matcher is simple, fast, and robust to recognizer noise — the right choice for "
      "an MVP on a fixed phrase set. <b>Weaker:</b> both translators are closed-set lookups; they cannot "
      "generalize to unseen phrasings, so BLEU/ROUGE/METEOR on novel input are low. The planned fix — already "
      "scaffolded as T5-format training data — is to fine-tune <b>mT5-small</b> for English/Kiswahili/KSL "
      "gloss→text, which can generate rather than retrieve. These BLEU/ROUGE/METEOR numbers become the baseline "
      "that mT5 must beat.", BLUE),
    H1("5&nbsp;&nbsp;Role in the hybrid pipeline"),
    P("This layer sits between recognition and the doctor: recognized gloss → healthcare sentence. In the "
      "recommended pipeline it runs the fuzzy matcher today (robust to the recognizer's mistakes) and a "
      "fine-tuned mT5 next. Note the two stages are evaluated separately by necessity — recognition is "
      "benchmarked on WLASL words, the language layer on healthcare phrases — so the honest end-to-end estimate "
      "is the product of the two stages' quality, not a single trained number. Closing that gap requires a KSL "
      "healthcare video corpus, which is the project's central data need.")])

print("all per-model PDFs built")
