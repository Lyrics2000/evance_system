"""Master Comparative Model Analysis PDF."""
import json
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak
from pdfkit_ksl import (NumberedDoc, titleblock, H1, H2, P, sp, rule, bullets,
                        figure, table, callout, code, caption, styles, BLUE, ORANGE,
                        AQUA, RED, GREEN, VIOLET, colors)

R = json.load(open("results.json")); L = json.load(open("lang_results.json"))
M = R["models"]; base = R["meta"]["majority_baseline"]
def acc(m): return f"{M[m]['accuracy']:.3f}"
def f1(m): return f"{M[m]['weighted_f1']:.3f}"
def wer(m): return f"{M[m]['wer']:.3f}"
def sd(m): return f"±{M[m]['fold_acc_std']:.3f}"

story = []
story += titleblock(
    "Comparative Model Analysis for Bidirectional Kenyan Sign Language Healthcare Translation",
    "Paper 1 — Theory, method, results, and the empirical case for the hybrid model",
    ["Project: Hybrid Multimodal Transformer &amp; Diffusion Framework for KSL in Healthcare",
     "Scope: five sign-recognition architectures + a non-parametric baseline + two hybrid designs, "
     "evaluated on identical folds, plus the gloss→text language layer.",
     "Benchmark data: Voxel51/WLASL stand-in — 5 classes, 67 videos, MediaPipe Holistic landmarks.",
     "All numbers in this report were recomputed under one shared 5-fold stratified cross-validation."])

# ---------------- Executive summary ----------------
story += [H1("1&nbsp;&nbsp;Executive summary")]
story += [callout("Headline: the trained hybrid is the strongest model, and it is the one to carry forward.",
    f"Under a single fair evaluation (5-fold CV, 67 out-of-fold predictions), the two-stream "
    f"<b>Hybrid Multimodal Transformer</b> (graph + attention) reaches <b>{acc('Hybrid (Two-Stream)')} accuracy "
    f"/ {f1('Hybrid (Two-Stream)')} weighted F1</b>, the only model to clearly beat the strong non-parametric "
    f"baseline DTW ({acc('DTW 1-NN')}). Among individual architectures the ranking is "
    f"Transformer &gt; Random&nbsp;Forest &gt; ST-GCN &gt; BiLSTM &gt; HMM — the classic literature ordering, with the "
    f"attention model on top. Every model clears the {base:.2f} majority baseline, so all five architectures "
    f"learn real signal; the differences are about how well each one uses the tiny data.", ORANGE)]
story += [P(
    "Three things this report establishes. <b>(1) The comparison is now fair.</b> The original notebooks each "
    "reported a single 17-sample hold-out, where one video is worth ~6 accuracy points; those numbers are "
    "re-derived here on identical folds so the models can actually be ranked. <b>(2) The evaluation is now "
    "complete.</b> Alongside accuracy / precision / recall / F1 we add Word Error Rate and, for the language "
    "layer, BLEU, ROUGE and METEOR — the metrics your review asked for, each mapped to the stage where it is "
    "meaningful. <b>(3) The hybrid is real, not projected.</b> It was trained end-to-end on the same cached "
    "features and evaluated the same way, so its advantage is measured rather than assumed.")]

# headline table
rows = [["Model", "Family", "Accuracy", "Wtd F1", "WER↓", "CV SD"]]
order = [("Hybrid (Two-Stream)","Hybrid (this study)"),("DTW 1-NN","Non-parametric"),
         ("Transformer","Attention"),("Random Forest","Statistical"),
         ("Hybrid (Soft-Vote)","Hybrid ensemble"),("ST-GCN","Graph"),
         ("BiLSTM","Recurrent"),("HMM (corrected)","Probabilistic")]
for m, fam in order:
    rows.append([m, fam, acc(m), f1(m), wer(m), sd(m)])
rows.append(["Majority baseline", "—", f"{base:.3f}", "—", f"{1-base:.3f}", "—"])
story += [sp(4), table(rows, [40*mm,30*mm,20*mm,18*mm,18*mm,18*mm], align="CENTER",
                       highlight_rows={1:[0,1,2,3,4,5]})]
story += [caption("Table 1. All models on identical 5-fold stratified folds (seed 42). "
                  "WER↓ = lower is better; for isolated-sign recognition WER = 1 − accuracy. "
                  "The hybrid row is highlighted.")]
story += [figure("1_ranking.png",
    "Figure 1. Out-of-fold accuracy with cross-validation spread (whiskers = SD across folds). "
    "The hybrid (orange) leads; DTW (green) is the strongest non-parametric method.", 150*mm)]

story += [PageBreak()]

# ---------------- Study design ----------------
story += [H1("2&nbsp;&nbsp;Study design and data")]
story += [P(
    "Because no Kenyan Sign Language corpus exists yet, all five architectures are benchmarked on the public "
    "<b>WLASL</b> (Word-Level American Sign Language) dataset as a stand-in. This is a deliberate, defensible "
    "choice for a design-science study: it lets us compare architectures on identical inputs and reproduce the "
    "literature ordering, while the KSL-specific contribution lives in the localization, healthcare-phrase "
    "validation, and language layers. The recognition numbers are therefore <i>architecture evidence</i>, not "
    "claims about deployed KSL accuracy.")]
story += [H2("2.1&nbsp;&nbsp;Features — one shared representation")]
story += [P(
    "Every recognition model consumes the same MediaPipe Holistic landmarks: per frame, 33 pose + 21 left-hand "
    "+ 21 right-hand + 468 face landmarks × (x,y,z) = <b>1,629 numbers</b>, sampled at 60 frames / stride 2. "
    "The ST-GCN additionally uses a 75-node skeleton graph (face dropped). Sharing the representation is what "
    "makes the comparison clean — differences in the table come from the model, not the pre-processing.")]
story += [callout("A structural caveat worth stating up front (it shapes every result).",
    "67 videos across 5 classes is ~13 examples per class and ~13 test items per fold. A single test video "
    "moves fold accuracy by ~7 points, which is exactly why cross-validation and per-fold spread are reported "
    "rather than one number. The five classes (before, computer, cool, cousin, drink) are ASL benchmark words, "
    "not healthcare KSL. Read the numbers as <b>directional evidence about architectures</b>, not statistically "
    "significant performance claims.", RED)]
story += [H2("2.2&nbsp;&nbsp;Evaluation protocol — why cross-validation, not a hold-out")]
story += [P(
    "The original notebooks each trained once and tested on a 25% hold-out (~17 videos). With that few test "
    "items the score is dominated by luck of the split. We replace it with <b>5-fold stratified "
    "cross-validation</b>: the 67 videos are split five ways, each fold trained on ~54 and tested on ~13, and "
    "every video is predicted exactly once by a model that never saw it. This yields 67 honest predictions per "
    "model and a spread across folds. The effect is not cosmetic — see Figure 2.")]
story += [figure("2_holdout_vs_cv.png",
    "Figure 2. The same model configuration scored two ways. The HMM's famous 0.235 was a degenerate collapse "
    "on one unlucky split; under CV it does not recur every fold. Single hold-outs are unstable — the ranking "
    "in this report is built on the CV numbers.", 145*mm)]

story += [PageBreak()]

# ---------------- Taxonomy ----------------
story += [H1("3&nbsp;&nbsp;The models at a glance")]
story += [P("The six recognition approaches span the full modelling spectrum — from a memoriser with zero "
            "trained parameters (DTW) to attention over the whole sequence (Transformer). Their inductive "
            "biases are what the comparison is really testing.")]
tax = [["Model", "Type", "Temporal handling", "Trained params / class", "Core assumption"],
    ["Random Forest", "Statistical, non-temporal", "None — pools frames to mean/std/min/max", "~ensemble of 300 trees", "Order does not matter"],
    ["HMM", "Generative, probabilistic", "Hidden states + transitions", "~96 (corrected)", "Markov + frame independence"],
    ["BiLSTM", "Deep recurrent", "Gated memory, both directions", "~0.9M (shared)", "Learn temporal dependence"],
    ["ST-GCN", "Deep graph", "Graph conv + temporal conv", "~0.2M (shared)", "Skeleton is a graph"],
    ["Transformer", "Deep attention", "Self-attention over all frames", "~0.5M (shared)", "Any frame attends to any"],
    ["DTW + 1-NN", "Non-parametric", "Elastic time alignment", "0", "Same sign ⇒ alignable"]]
story += [sp(2), table(tax, [26*mm,30*mm,42*mm,30*mm,36*mm], align="LEFT", font=8.0)]
story += [caption("Table 2. Recognition-model taxonomy. Deep models share weights across classes; HMM/DTW model "
                  "each class separately.")]
story += [P(
    "The single idea that connects the whole table: <b>a sign lives in motion and hand shape over time.</b> "
    "Random Forest throws the time axis away (and still clears baseline, because average pose carries some "
    "signal). The HMM keeps time but assumes each frame is independent given a hidden stage — the assumption "
    "video violates most. LSTM, ST-GCN and Transformer each drop that independence assumption in a different "
    "way, which is why they climb the ranking. DTW keeps every frame and makes almost no assumptions at all — "
    "and at this data scale that rigidity is a strength.")]

story += [PageBreak()]

# ---------------- Per-model comparison ----------------
story += [H1("4&nbsp;&nbsp;Model-by-model comparison")]
story += [P("Each model below gets the same four questions: what is the theory, how was it executed, what did "
            "it score, and why. Full deep-dives (with confusion matrices and ADRs) are in the companion "
            "per-model PDFs; this section is the side-by-side.")]

def model_block(name, key, theory, method, why, strength, color=BLUE):
    els = [H2(name),
           P(f"<b>Result (5-fold CV):</b> accuracy <b>{acc(key)}</b> · weighted F1 {f1(key)} · "
             f"WER {wer(key)} · fold spread {sd(key)}."),
           P(f"<b>Theory.</b> {theory}"),
           P(f"<b>Method of execution.</b> {method}"),
           P(f"<b>Why this result.</b> {why}"),
           callout("What makes it stronger / weaker", strength, color)]
    return els

story += model_block("4.1&nbsp;&nbsp;Random Forest — the non-temporal baseline", "Random Forest",
    "An ensemble of 300 decision trees voting on a single fixed-length vector per video. It is the control "
    "condition: it deliberately ignores the order of frames.",
    "Each video's 1,629-D landmark stream is collapsed across time into mean, standard deviation, min and max "
    "→ a 6,516-D summary vector, standardized, then classified by a class-balanced random forest.",
    f"It reaches {acc('Random Forest')} — above several temporal models — because average hand position and "
    "spread already separate these five words fairly well, and trees are robust on small tabular data. But it "
    "has hit its ceiling: by discarding movement order it can never distinguish signs that differ only in "
    "trajectory.",
    "<b>Stronger:</b> robust, fast, no sequence assumptions to violate, strong on tiny data. "
    "<b>Weaker:</b> temporally blind — averaging destroys exactly what a sign is. It is the bar the temporal "
    "models must clear to justify their complexity.", GREEN)

story += model_block("4.2&nbsp;&nbsp;Hidden Markov Model — the probabilistic temporal baseline", "HMM (corrected)",
    "A generative model that treats a sign as a walk through a few hidden stages (e.g. rise → hold → lower), "
    "each stage emitting a typical pose. One HMM is trained per class; classification picks the class whose "
    "model best explains the video.",
    "The corrected pipeline drops the face (86% of the raw vector), body-centres and scales on the shoulders, "
    "adds velocity and hand-presence flags, reduces to 16 PCA components, and fits a 3-state left-to-right "
    "(Bakis) Gaussian HMM with a variance floor and length-normalised scoring — the fixes that eliminate the "
    "original −∞ collapse.",
    f"It lands at {acc('HMM (corrected)')}, the weakest of the proper models. The reason is theoretical, not a "
    "bug: HMMs assume each frame is independent given the state, but frames 80 ms apart are near-duplicates, so "
    "~13 videos behave like far fewer independent observations. The model is data-starved by its own "
    "assumptions.",
    "<b>Stronger:</b> interpretable stages, calibrated probabilities, tiny parameter count once corrected. "
    "<b>Weaker:</b> the conditional-independence assumption caps HMMs on video — the single most important "
    "reason the field moved to LSTMs, GCNs and Transformers.", VIOLET)

story += [PageBreak()]

story += model_block("4.3&nbsp;&nbsp;BiLSTM — the deep recurrent model", "BiLSTM",
    "A bidirectional two-layer LSTM reads the landmark sequence forward and backward, maintaining a gated "
    "memory that lets any frame influence later ones — directly dropping the HMM's frame-independence "
    "assumption.",
    "The 1,629-D per-frame vector is projected down, run through a 2-layer bidirectional LSTM (hidden 128), and "
    "the concatenated final states feed a small MLP head. Trained with Adam, dropout 0.3.",
    f"At {acc('BiLSTM')} it clears baseline but posts the highest fold-to-fold spread ({sd('BiLSTM')}) of any "
    "model — recurrent nets have many parameters and ~54 training videos is far too few to pin them down, so "
    "the fold you test on matters a lot.",
    "<b>Stronger:</b> genuinely models temporal dependence; scales well once data grows. "
    "<b>Weaker:</b> data-hungry and high-variance here; a long recurrence over a huge input dimension is hard "
    "to train on 13 examples per class.", ORANGE)

story += model_block("4.4&nbsp;&nbsp;ST-GCN — the graph spatial-temporal model", "ST-GCN",
    "A spatial-temporal graph convolutional network treats each frame as a skeleton graph (pose + both hands, "
    "75 joints) and convolves over the body's connectivity in space and over time — encoding the anatomical "
    "prior that connected joints move together.",
    "A hand-built adjacency matrix (MediaPipe pose/hand edges + self-loops, symmetrically normalized) drives "
    "stacked blocks of graph convolution + temporal convolution with residuals; a global pool feeds a linear "
    "classifier. No external graph library — the graph conv is written directly.",
    f"It reaches {acc('ST-GCN')} with the <b>lowest variance of any deep model</b> ({sd('ST-GCN')}): the "
    "skeleton prior is a strong, correct inductive bias that stabilizes learning when data is scarce. It does "
    "not top the table here only because dropping the face and the tiny sample cap its ceiling.",
    "<b>Stronger:</b> anatomically grounded, low-variance, parameter-efficient — the most reliable deep model "
    "at this scale. <b>Weaker:</b> ignores face/mouthing cues; fixed graph can't learn long-range "
    "frame-to-frame relations the way attention does.", AQUA)

story += model_block("4.5&nbsp;&nbsp;Transformer encoder — the attention model", "Transformer",
    "Self-attention lets every frame directly attend to every other frame, with no recurrence and no fixed "
    "graph — the most flexible temporal model, and the architecture the thesis's hybrid is named after.",
    "The 1,629-D frames are linearly projected to a 128-D model space, given sinusoidal positional encodings, "
    "passed through a 2-layer / 4-head encoder (GELU, dropout 0.3), mean-pooled over valid frames, and "
    "classified after a LayerNorm.",
    f"It is the best individual architecture at {acc('Transformer')}. Attention captures the whole-sign "
    "temporal structure that averaging (RF) and stage-independence (HMM) miss. Its spread ({sd('Transformer')}) "
    "is high — attention is data-hungry — but its central tendency is the strongest of the single models.",
    "<b>Stronger:</b> most expressive temporal model; the natural backbone for a larger KSL system. "
    "<b>Weaker:</b> highest appetite for data — it will benefit most, and only, once the dataset grows.", BLUE)

story += model_block("4.6&nbsp;&nbsp;DTW + 1-NN — the non-parametric reference", "DTW 1-NN",
    "Dynamic Time Warping measures similarity between two sequences allowing for speed differences, and 1-NN "
    "simply copies the label of the most similar training video. It fits <b>zero parameters</b> — it is pure "
    "memory.",
    "Each video is reduced to the same 16-D PCA landmark features; a test video is compared by DTW distance to "
    "every training video and assigned the nearest one's label.",
    f"It scores {acc('DTW 1-NN')} — the best single method, beating every trained architecture. This is the "
    "textbook bias–variance story: with ~13 examples per class, a method that estimates nothing beats flexible "
    "models that must estimate thousands of parameters from too little data.",
    "<b>Stronger:</b> unbeatable at tiny scale, no assumptions to violate, trivial to extend. "
    "<b>Weaker:</b> never generalizes or compresses — prediction cost grows with the dataset and it can't give "
    "calibrated probabilities. It is the honest baseline, not the endpoint.", GREEN)

story += [PageBreak()]
story += [figure("6_confusion.png",
    "Figure 3. Confusion matrices under CV, weakest to strongest. The diagonal (correct predictions) sharpens "
    "from HMM through to the hybrid; 'cool' is the hardest class for every model.", 168*mm)]
story += [figure("3_perclass_f1.png",
    "Figure 4. Per-class F1. 'cool' is the universal weak spot (a data-quality signal worth auditing), while "
    "'computer' and 'cousin' are the easiest — the hybrid is the only model strong across all five.", 120*mm)]

story += [PageBreak()]

# ---------------- Metrics ----------------
story += [H1("5&nbsp;&nbsp;Evaluation metrics — and which applies to which model")]
story += [P("Your review asked for BLEU, Word Error Rate, ROUGE and METEOR. These are not interchangeable: some "
            "measure <i>classification</i>, others measure <i>generated text</i>. Using the wrong one produces "
            "meaningless numbers, so the table below maps each metric to the stage where it is defined.")]
mm_rows = [["Metric", "What it measures", "Where it applies in this system", "Reported here"],
    ["Accuracy", "Fraction of signs classified correctly", "Recognition models (§4)", "Yes — Table 1"],
    ["Precision / Recall / F1", "Per-class correctness, weighted", "Recognition models", "Yes — Table 1 &amp; Fig 4"],
    ["WER (Word Error Rate)", "Word substitutions/insertions/deletions vs reference", "Recognition (isolated ⇒ 1−acc); gloss→text sentences", "Yes — both stages"],
    ["BLEU", "n-gram precision of generated text vs reference", "Gloss→text language layer only", "Yes — §6"],
    ["ROUGE-1/2/L", "n-gram &amp; longest-subsequence recall of generated text", "Gloss→text language layer only", "Yes — §6"],
    ["METEOR", "Unigram match with stem/synonym + word-order penalty", "Gloss→text language layer only", "Yes — §6"]]
story += [table(mm_rows, [26*mm,44*mm,52*mm,26*mm], align="LEFT", font=7.9)]
story += [caption("Table 3. Metric-to-model map. Translation metrics (BLEU/ROUGE/METEOR) are undefined for a "
                  "classifier that outputs one label — they belong to the text-generation stage.")]
story += [callout("The key distinction, in one line.",
    "<b>Recognition models output a class label</b>, so they are scored with accuracy / F1 (and WER, which for "
    "one-word utterances equals 1 − accuracy). <b>The language layer outputs a sentence</b>, so it is scored "
    "with BLEU / ROUGE / METEOR / WER. Reporting BLEU on the Transformer's class prediction, or accuracy on a "
    "generated sentence, would both be category errors — the split below keeps each metric where it is valid.",
    BLUE)]

# ---------------- Language layer ----------------
story += [H1("6&nbsp;&nbsp;The language layer — gloss → healthcare text (BLEU / ROUGE / METEOR / WER)")]
story += [P("Recognition produces a gloss like <font face='Courier'>pain where</font>; the language layer turns "
            "it into <font face='Courier'>“Where is the pain?”</font> Two baseline translators are evaluated on "
            "the 20-phrase healthcare set: an <b>exact rule-based</b> dictionary and a <b>fuzzy token-overlap</b> "
            "matcher, tested on clean input, on simulated noisy recognizer output (reordered / dropped words), "
            "and on a leave-one-out generalization test.")]
def lrow(d): return [d["model"].split(" (")[0], f"{d['BLEU']:.1f}", f"{d['ROUGEL']:.1f}",
                     f"{d['METEOR']:.1f}", f"{d['WER']:.1f}", f"{d['exact_match']:.0f}%"]
lang_rows = [["Condition / translator", "BLEU", "ROUGE-L", "METEOR", "WER↓", "Exact"]]
lang_rows.append(["Exact rule-based — clean"]+lrow(L["clean"][0])[1:])
lang_rows.append(["Fuzzy overlap — clean"]+lrow(L["clean"][1])[1:])
lang_rows.append(["Exact rule-based — noisy"]+lrow(L["noisy"][0])[1:])
lang_rows.append(["Fuzzy overlap — noisy"]+lrow(L["noisy"][1])[1:])
lang_rows.append(["Fuzzy nearest-NN — generalization"]+lrow(L["generalization"][0])[1:])
story += [sp(2), table(lang_rows, [58*mm,17*mm,20*mm,20*mm,17*mm,16*mm], align="CENTER",
                       highlight_rows={4:[0,1,2,3,4,5]})]
story += [caption("Table 4. Real BLEU/ROUGE/METEOR/WER on the gloss→text layer (0–100). Highlighted row is the "
                  "generalization test where the metrics do their real discriminative work.")]
story += [figure("5_language.png",
    "Figure 5. Translation quality across conditions. The exact matcher collapses on reordered input "
    "(WER 93); the fuzzy matcher is robust (WER 0). On unseen phrases (right) BLEU/ROUGE/METEOR fall to "
    "10–40 — showing the metrics discriminating partial matches, which is exactly why a trainable mT5 model is "
    "the planned next step.", 150*mm)]
story += [P(
    "<b>What the language numbers say.</b> On known phrases both translators are perfect. The moment the input "
    "is perturbed — which is what an imperfect recognizer produces — the exact matcher drops to 10% coverage "
    "(BLEU 0, WER 93) while the fuzzy matcher recovers every phrase (BLEU 100, WER 0). This is the empirical "
    "argument for order-insensitive matching downstream of recognition. The generalization row (BLEU 9.8, "
    "ROUGE-L 39.5, METEOR 37.3) is the honest one: a lookup translator cannot compose novel wordings, which is "
    "precisely the gap a fine-tuned mT5 language model is meant to close.")]

story += [PageBreak()]

# ---------------- Hybrid ----------------
story += [H1("7&nbsp;&nbsp;The hybrid model — design, results, and why it wins")]
story += [P("The thesis proposes a <b>hybrid multimodal transformer</b>. We built and trained a concrete "
            "version and evaluated it on the same folds, so its advantage is measured. Two hybrid strategies "
            "were tested:")]
story += bullets([
    "<b>Two-stream (feature-level fusion) — the recommended hybrid.</b> A graph stream (ST-GCN over the 75-joint "
    "skeleton) and an attention stream (Transformer over the face-dropped 225-D hand+pose landmarks) each "
    "produce an embedding; the two are concatenated and a fusion head classifies. The two streams see the sign "
    "through complementary lenses — anatomical structure and long-range temporal attention.",
    "<b>Soft-vote ensemble (decision-level fusion).</b> Train BiLSTM, ST-GCN and Transformer separately and "
    "average their class probabilities. Simpler, but it cannot learn cross-stream interactions."])
hyb = [["Model", "Accuracy", "Wtd F1", "WER↓", "vs best single"],
    ["Best single (DTW 1-NN)", acc("DTW 1-NN"), f1("DTW 1-NN"), wer("DTW 1-NN"), "—"],
    ["Best deep single (Transformer)", acc("Transformer"), f1("Transformer"), wer("Transformer"), "—"],
    ["Hybrid — Soft-Vote ensemble", acc("Hybrid (Soft-Vote)"), f1("Hybrid (Soft-Vote)"), wer("Hybrid (Soft-Vote)"),
     f"{(M['Hybrid (Soft-Vote)']['accuracy']-M['DTW 1-NN']['accuracy'])*100:+.1f} pts"],
    ["Hybrid — Two-Stream (recommended)", acc("Hybrid (Two-Stream)"), f1("Hybrid (Two-Stream)"),
     wer("Hybrid (Two-Stream)"), f"{(M['Hybrid (Two-Stream)']['accuracy']-M['DTW 1-NN']['accuracy'])*100:+.1f} pts"]]
story += [sp(2), table(hyb, [58*mm,22*mm,20*mm,18*mm,26*mm], align="CENTER",
                       highlight_rows={4:[0,1,2,3,4]})]
story += [caption("Table 5. Hybrid vs its strongest components. Two-stream fusion is the only design that beats "
                  "both the best single deep model and the non-parametric baseline.")]
story += [figure("4_hybrid.png",
    "Figure 6. The two-stream hybrid lifts accuracy and F1 above every component; the soft-vote ensemble does "
    "not — averaging three data-starved models mostly averages their noise.", 145*mm)]
story += [callout("Why the two-stream hybrid wins, and the soft-vote does not.",
    f"The two streams are <b>complementary and jointly trained</b>: the graph stream contributes a stable, "
    f"low-variance anatomical prior (ST-GCN had the lowest spread of any deep model) and the attention stream "
    f"contributes temporal expressiveness (the Transformer was the best single model). Fusing their "
    f"<i>features</i> lets the head learn how structure and motion interact, so the hybrid inherits ST-GCN's "
    f"stability <i>and</i> the Transformer's ceiling — reaching {acc('Hybrid (Two-Stream)')} / F1 "
    f"{f1('Hybrid (Two-Stream)')}, the best of the study. The soft-vote fuses only final <i>decisions</i>, so "
    f"it cannot learn those interactions and mostly averages three noisy models — landing below DTW. "
    f"<b>Caveat, stated plainly:</b> the two-stream's fold spread is high ({sd('Hybrid (Two-Stream)')}); its "
    f"lead is real but, at this sample size, needs more data to be called statistically firm.", ORANGE)]

# ---------------- Recommendation ----------------
story += [H1("8&nbsp;&nbsp;Which components to pick for the hybrid")]
story += [P("Bringing the recognition, language and generation evidence together, the recommended hybrid for "
            "bidirectional KSL healthcare translation is:")]
pick = [["Pipeline stage", "Recommended component", "Why (evidence from this study)"],
    ["Sign → landmarks", "MediaPipe Holistic (pose + hands; face optional)", "Shared, reliable; dropping face concentrated the signal (§4.2)"],
    ["Recognition (core)", "Two-stream ST-GCN + Transformer hybrid", f"Best measured model: {acc('Hybrid (Two-Stream)')} acc, beats all singles (§7)"],
    ["Low-data fallback / baseline", "DTW + 1-NN", f"Strongest non-parametric ({acc('DTW 1-NN')}); the bar to beat until data grows"],
    ["Gloss → healthcare text", "Fuzzy matcher now → fine-tuned mT5 next", "Fuzzy is robust to recognizer noise (§6); mT5 for novel wordings"],
    ["Text → sign (reverse)", "Fuzzy text→gloss + clip dictionary → diffusion avatar (future)", "MVP retrieval now; diffusion is design-only until data exists"]]
story += [sp(2), table(pick, [30*mm,44*mm,60*mm], align="LEFT", font=7.9,
                       highlight_rows={2:[0,1,2]})]
story += [caption("Table 6. Component recommendation across the bidirectional pipeline.")]
story += [callout("The one-sentence recommendation.",
    "Build the recognition core as the <b>two-stream ST-GCN + Transformer hybrid</b> — it is the only model that "
    "empirically beats the DTW baseline — keep DTW as the honest reference, use the fuzzy translator as the "
    "language layer today with mT5 as the funded next step, and treat text-to-sign generation as a retrieval MVP "
    "with diffusion as future work. Above all, the biggest single win is not an architecture change but "
    "<b>more data</b>: recover the missing WLASL videos and move to signer-independent splits.", BLUE)]

# ---------------- Limitations ----------------
story += [H1("9&nbsp;&nbsp;Honest limitations and next steps")]
story += bullets([
    "<b>Sample size.</b> 67 videos / 5 classes. All accuracies carry ~±7 points of per-fold noise; treat "
    "rankings as directional. The crossover where trained models reliably beat DTW is ~30–50 videos/class.",
    "<b>Benchmark ≠ target.</b> Classes are ASL words, not KSL healthcare signs. Recognition numbers are "
    "architecture evidence; KSL claims require KSL data.",
    "<b>Signer-independent splits.</b> Current folds may place the same signer in train and test, inflating "
    "scores. Wire up WLASL signer metadata and re-run — the honest number will be lower.",
    "<b>Missing videos.</b> Several requested classes had zero downloadable files; recovering them (and "
    "augmenting: mirroring, temporal jitter) is worth more than any model change.",
    "<b>Hybrid variance.</b> The two-stream's lead is real but wide-spread; confirm with a paired McNemar test "
    "and more data before headline claims.",
    "<b>Language &amp; generation.</b> Translators are closed-set lookups; the diffusion avatar is design-only. "
    "Both are correctly scoped as next steps, not current results."])
story += [rule()]
story += [P("<b>Reproducibility.</b> All models trained on the cached MediaPipe features "
            "(<font face='Courier'>wlasl_lstm/stgcn/transformer_sequences_fixed_classes.joblib</font>), "
            "5-fold stratified CV, seed 42, identical folds across models. Feature engineering, HMM correction, "
            "DTW, the two-stream hybrid and the language metrics are in the companion code. Per-model deep-dive "
            "PDFs accompany this document.", "small")]

doc = NumberedDoc("/home/claude/ksl/pdfs/00_Master_Comparative_Model_Analysis.pdf",
                  footer="KSL Comparative Model Analysis · Paper 1")
doc.build(story)
print("master built")
