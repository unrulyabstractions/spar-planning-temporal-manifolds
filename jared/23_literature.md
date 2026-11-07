# Literature: where this project sits (2026-11-03)

Two reviews compiled overnight (sections A and B below, written by review agents from
abstract pages; all 60 arXiv IDs re-opened and checked against title and first author
on 2026-11-03: 60 exist, none is invented; one title is a later version's,
"Steering Language Models With Activation Engineering" = ActAdd, 2308.10248, and
Panickssery = Rimsky for CAA, 2312.06681). Summaries come from abstracts, not full
texts: check a paper before quoting a number from it.

## What the literature says about our open questions

**1. Is the horizon one duration code, or a unit word plus a number? (R3)**
- Numbers in LLMs are not stored as one magnitude. Levy & Geva (2410.11781) find a
  circular base-10 code *per digit*; Kantamneni & Tegmark (2502.00873) and Zhou et al.
  (2406.03445) find helix / Fourier features with periods 2, 5, 10, 100; Zhu et al.
  (2401.03735) read value linearly but with large error; AlquBoj et al. (2502.16147)
  report log-spaced value. Magnitude is partly already in the token embedding
  (Wallace et al., 1909.07940).
- This predicts what E14 saw ("6 months" → "66 months": a digit edit, ~1 decade
  whatever the step) and what tonight's E-C finds (the prompt-end geometry is a
  unit-word code plus a shared numeral code, with only a coarse duration ordering
  across units).
- No paper we found probes durations-with-units in a modern LLM; the closest are
  Gurnee & Tegmark (2310.02207: entity dates, not stated durations) and Sasaki et al.
  (2606.03982: number-vs-unit heuristics in quantity comparison, behavioural). This is
  the project's clearest novelty.

**2. Why steering the line fails and the curve barely works.**
- Engels et al. (2405.14860) and Park et al. (2311.03658, 2406.01506): ordinal and
  categorical concepts live on curves/simplices; a single probe direction is the
  wrong object to steer. Modell et al. (2505.18235) give a theory of such manifolds;
  Raval et al. "Curveball steering" (2603.09313, unreviewed) steer along them.
- Tan et al. (2407.12404) and Braun et al. (2505.22637): steering reliability is low
  and prompt-dependent when the behaviour is not one coherent direction. Report
  per-prompt distributions and size-matched random controls (no canonical citation
  for the latter; it is good practice, not a named method).
- Makelov et al. (2311.17030): subspace patching can create an "illusion" by
  activating a dormant pathway. This is the main caution for E10's unit-word swap;
  DAS (Geiger et al. 2303.02536) with size-matched random subspaces is the standard
  answer.

**3. Is the reasoning read, or performed? (R5, R6)**
- Faithfulness: Lanham (2307.13702), Turpin (2305.04388), Chen et al. 2025
  (2505.05410: reasoning models often omit the cue they used), Arcuschin (2503.08679).
- Thought Anchors (2506.19143) and Thought Branches (2510.27484): resample on-policy
  from a point in the trace to measure a sentence's causal weight; off-policy text
  edits (what E14 did) gave small, unstable effects there.
- Answers are often decodable before the chain ends: "Reasoning Theater"
  (2603.05488), Zhang et al. (2504.05419), the detection-extraction gap (2604.06613).
  Tonight's E-A tests this for the choice.
- Venhoff et al. (2506.18167): reasoning *behaviours* (backtracking, uncertainty) are
  linearly steerable in thinking models: a candidate for what the forced-close and
  "does it compute" directions are.

**4. Intertemporal choice in LLMs** is behavioural and thin (Goli & Singh 2305.02531,
Chen et al. PNAS 2305.12763, Zhu/Yan/Griffiths 2405.19313: arithmetic ability
predicts human-like choice, Tak et al. 2601.22329). Nothing mechanistic besides the
prior paper (Rios-Sialer et al., 2606.05194). Zhu/Yan/Griffiths is the most relevant
to tonight's finding that the 8B's "irrational" near choices are arithmetic errors.

**5. Comparison mechanisms**: GPT-2 greater-than (Hanna et al. 2305.00586), causal
mediation in arithmetic (Stolfo et al. 2305.15054). Nothing on comparing quantities
in *different units*, nor on comparison inside a reasoning trace.

## Hypotheses the reviews raised, and their status after tonight

| hypothesis (source) | status |
|---|---|
| Within-unit magnitude is a digit/numeral code (Levy & Geva) | half: a numeral code shared across units exists (`24` §5, ratio ~0.8 from L15), but at the prompt end it is the written number's log magnitude, not a per-digit code (digit terms ≤ 0.09 beyond L3) |
| Unit (categorical) and magnitude (ordinal) are separate subspaces (Park) | E-C: unit word dominates the geometry; duration ordering across units is coarse (~0.3 decades) |
| The trace's written horizon predicts the choice beyond the prompt's (text mediation; Lanham) | not testable on unsteered traces: the stated horizon equals the prompt's in 750 of 753 (8B) and 439 of 445 (14B) natural closes (E-B) |
| Force-cut confound explains the CAA sign (Venhoff/Braun) | rejected at the answer slot: the CAA direction is the none-regime gradient, which is the model's real behaviour (E-D) |
| The choice is decodable before reasoning (Reasoning Theater) | rejected beyond the prompt surface: `24` §2 (also for the arithmetic shortcut, §2.2) |
| Helix / Fourier fit of the horizon curve (Kantamneni) | open (saved data; the 48 horizons are unit-confounded, so test on E9 / units bank) |
| DAS vs PCA vs random for the unit subspace (Makelov, Geiger) | open (box) |
| On-policy resampling at the commitment point (Thought Branches) | proposed (box), see `24_results_overnight_autoresearch.md` §8 |

---

# A. Representation geometry: numbers, time, manifolds, probing


All entries were opened on arXiv and the title, authors, year and venue checked against the abstract page. Summaries use only what the abstract page states. Where I extrapolate, it is marked as "Relevance to us". Candidates I could not open were dropped. I did not find a paper that probes duration-with-unit representations inside an LLM, which is gap G1 at the end.

## 1. How LLMs represent numbers and magnitude

1. **Levy & Geva (2024/NAACL 2025).** "Language Models Encode Numbers Using Digit Representations in Base 10." Amit Arnold Levy, Mor Geva. https://arxiv.org/abs/2410.11781
   Model errors on numeric tasks follow the digits of the answer, not its numeric distance. Probing and causal interventions show each digit is stored separately in a circular base-10 representation.
   Relevance to us: predicts that "6 -> 66 months" (one decade, any step size) is a digit-slot edit, and that magnitude within a unit lives in per-digit circular features at the number tokens. Test: probe each digit separately (tens, units) at the number tokens and check whether the steered curve moves a digit subspace.

2. **Kantamneni & Tegmark (2025).** "Language Models Use Trigonometry to Do Addition." Subhash Kantamneni, Max Tegmark. https://arxiv.org/abs/2502.00873
   Three mid-sized LMs represent numbers as a generalized helix, which causally affects addition and related arithmetic. They propose a "Clock" algorithm and support it with interventions.
   Relevance to us: a helix (linear trend plus periodic components) would look like a curved low-dimensional manifold with a tail. Test: fit a helix (linear + Fourier at periods like 2, 5, 10, 100) to the number-token activations and compare its fit with our 2-4 curved dims.

3. **Zhou, Fu, Sharan, Jia (2024).** "Pre-trained Large Language Models Use Fourier Features to Compute Addition." https://arxiv.org/abs/2406.03445
   Number information is concentrated in a small set of frequencies. MLP layers estimate rough magnitude with low-frequency features, and attention layers handle modular properties with high-frequency features.
   Relevance to us: suggests coarse magnitude (decade) and fine digit identity are separate frequency bands. Test: Fourier-decompose the horizon activations over the log axis and check whether low frequencies match our "curved dims" and high frequencies match the weak within-unit signal.

4. **Zhu, Dai, Sui (2024).** "Language Models Encode the Value of Numbers Linearly." https://arxiv.org/abs/2401.03735
   Linear probes recover input numbers from hidden states across layers, and adding simple vectors to representations changes outputs.
   Relevance to us: the baseline claim that probe directions can steer. Our probe line being causally inert is a contrast. Test: replicate on a bare number prompt ("6", "66") at the number token, with no unit, to see whether the number alone is linear while the full horizon is not.

5. **AlquBoj et al. (2025).** "Number Representations in LLMs: A Computational Parallel to Human Perception." H.V. AlquBoj, Hilal AlQuabeh, Velibor Bojkovic, Tatsuya Hiraoka, Ahmed Oumar El-Shangiti, Munachiso Nwadike, Kentaro Inui. https://arxiv.org/abs/2502.16147
   Using PCA, PLS and geometric regression on layer representations, the authors report that spacing between numerical values follows a logarithmic pattern, as on a compressed human number line.
   Relevance to us: this is the closest published support for a log-duration axis. Test: on number-only tokens, check whether the 1-D coordinate (PLS) is closer to log(n) or n. Our log-spaced horizons confound this, so test with linear spacing within a unit.

6. **Štefánik et al. (2025/2026).** "Language Models Learn Universal Representations of Numbers and Here's Why You Should Care." Michal Štefánik, Timothee Mickus, Marek Kadlčík, Bertram Højer, Michal Spiegel, Raúl Vázquez, Aman Sinha, Josef Kuchař, Philipp Mondorf, Pontus Stenetorp. https://arxiv.org/abs/2510.26285
   LLMs from different families learn nearly identical sinusoidal number representations, so the representations are largely interchangeable across setups. The authors argue this universality must be accounted for when measuring how well models encode numeric information.
   Relevance to us: a probe's R^2 on numbers can reflect this shared sinusoidal structure. Test: compare the periodic part of our number-token manifold against that of a different number context (e.g. prices).

7. **Wallace, Wang, Li, Singh, Gardner (EMNLP 2019).** "Do NLP Models Know Numbers? Probing Numeracy in Embeddings." https://arxiv.org/abs/1909.07940
   Standard token embeddings (GloVe, word2vec, ELMo, BERT) already encode numeric magnitude surprisingly well. BERT's subword tokenization is less precise.
   Relevance to us: magnitude information is partly in the token embeddings themselves. Test: probe the layer-0 embeddings of the number tokens in Qwen3-8B. If magnitude is readable at layer 0, the "within-unit" signal is lexical, not computed.

## 2. Time, dates, durations and quantities with units

1. **Gurnee & Tegmark (2023/ICLR 2024).** "Language Models Represent Space and Time." https://arxiv.org/abs/2310.02207
   Llama-2 models encode space and time linearly, at several scales, robustly to prompts and across entity types (historical figures, artworks, news headlines). The authors find individual "time neurons".
   Relevance to us: their time is a calendar date of an entity, not a duration stated in text. Open question is whether a duration is read out the same way. Test: apply their probe protocol (ridge probe on the last entity token, with a hold-out-by-block split) to our horizons and compare R^2 with the unit-word lookup baseline of 0.96.

2. **Sasaki et al. (2026).** "Language Models Compare Quantities Using Number-specific and Unit-specific Heuristics." Mutsumi Sasaki, Go Kamoda, Ryosuke Takahashi, Kosuke Sato, Kentaro Inui, Keisuke Sakaguchi, Benjamin Heinzerling. https://arxiv.org/abs/2606.03982
   Tests LM comparison of measured quantities across unit systems (e.g. 110 cm vs 1.2 m). Accuracy drops when small value changes decide the answer, and errors follow numeral and unit-scale differences. (The search snippet also described an internal-mechanism analysis; I did not confirm that part on the abstract page.)
   Relevance to us: behavioural precedent for number and unit being handled as separate heuristics, which fits our "digit edit vs unit swap" result. Test: cross-unit behavioural check (our "1 century" vs "100 years" cases) for whether the 0.3-decade shift matches their error pattern.

3. **Tiblias et al. (2025, TMLR 2026).** "Hypothesis-Driven Feature Manifold Analysis in LLMs via Supervised Multi-Dimensional Scaling." Federico Tiblias, Irina Bigoulaeva, Jingcheng Niu, Simone Balloccu, Iryna Gurevych. https://arxiv.org/abs/2510.01025
   Introduces SMDS to compare hypotheses about feature geometry. Applied to temporal reasoning, features take distinct shapes (circles, lines, clusters) that follow the concept's semantics, stable across model families and sizes.
   Relevance to us: provides a ready method to test "line vs circle vs cluster" for our horizons. Test: run SMDS on the 48 horizon averages with line, log-line and cluster-by-unit hypotheses.

4. **Karkada et al. (2026).** "Symmetry in language statistics shapes the geometry of model representations." Dhruva Karkada, Daniel J. Korchinski, Andres Nava, Matthieu Wyart, Yasaman Bahri. https://arxiv.org/abs/2602.15029
   Months forming a circle and years forming a smooth 1-D manifold are argued to come from translation symmetry in word co-occurrence statistics. Predictions are reported to match embedding models and LLMs.
   Relevance to us: predicts that an open-ended magnitude axis is a smooth curve, and that curvature is a property of the training statistics, not of the task. Test: whether the curvature in our horizon curve matches the curve of plain number or unit words with no decision prompt.

5. **Thukral, Kukreja, Kavouras (BlackboxNLP 2021).** "Probing Language Models for Understanding of Temporal Expressions." https://arxiv.org/abs/2110.01113
   NLI challenge sets for order, duration, and magnitude across units. MNLI-fine-tuned models do not have a thorough understanding of how temporal expressions relate.
   Relevance to us: older encoder models, behaviour only, but it motivates a cross-unit check. Test: measure whether equal durations written in different units (60 minutes, 1 hour) map to nearby points on our manifold.

6. **Zhou, Khashabi, Ning, Roth (EMNLP 2019).** "'Going on a vacation' takes longer than 'Going for a walk': A Study of Temporal Commonsense Understanding." https://arxiv.org/abs/1909.03065
   Defines five classes of temporal commonsense, including duration, and builds MCTACO. Best systems trailed humans by about 20%.
   Relevance to us: background for duration being a semantic quantity, not a number. Low priority for the experiments.

## 3. Feature manifolds, nonlinear and multidimensional features

1. **Engels, Michaud, Liao, Gurnee, Tegmark (ICLR 2025).** "Not All Language Model Features Are One-Dimensionally Linear." https://arxiv.org/abs/2405.14860
   Defines irreducible multi-dimensional features. SAEs on GPT-2 and Mistral 7B find circles for days and months, and interventions on Mistral 7B and Llama 3 8B suggest these circles are used for modular arithmetic.
   Relevance to us: the template for "find the manifold, then intervene on it and see the output change as the geometry predicts". Test: patch a point on our horizon curve from another horizon and check that the written number or unit moves along the curve, not along a probe line.

2. **Park, Choe, Veitch (ICML 2024).** "The Linear Representation Hypothesis and the Geometry of Large Language Models." https://arxiv.org/abs/2311.03658
   Formalises linear concept representations through counterfactual pairs, links them to probing and steering, and shows the choice of inner product matters, motivating a causal inner product.
   Relevance to us: our "probe line is causally inert" may reflect the Euclidean metric. Test: redo probe-direction steering in a whitened (unembedding-covariance) metric.

3. **Park, Choe, Jiang, Veitch (2024/ICLR 2025).** "The Geometry of Categorical and Hierarchical Concepts in Large Language Models." https://arxiv.org/abs/2406.01506
   Categorical concepts are modelled as polytopes, and hierarchy is tied to orthogonality, validated on Gemma and LLaMA-3 with 900+ WordNet concepts.
   Relevance to us: the unit word is a categorical variable (a polytope with ~7 vertices) and magnitude is a within-category offset. Test: check whether within-unit magnitude directions are orthogonal to the between-unit directions, which would explain the R^2 gap between 0.96 and 0.5.

4. **Modell, Rubin-Delanchy, Whiteley (2025).** "The Origins of Representation Manifolds in Large Language Models." https://arxiv.org/abs/2505.18235
   Proposes a feature is better modelled as a continuous manifold, with cosine similarity reflecting intrinsic geometry via shortest paths on the manifold.
   Relevance to us: predicts that cosine similarity between horizon activations tracks geodesic, not straight-line, distance. Test: Isomap versus PCA reconstruction of the pairwise cosine matrix of the 48 horizons.

5. **Raval et al. (2026).** "Curveball Steering: The Right Direction To Steer Isn't Always Linear." Shivam Raval, Hae Jin Song, Linlin Wu, Abir Harrasse, Jeff M. Phillips, Fazl Barez, Amirali Abdullah. https://arxiv.org/abs/2603.09313
   Argues a single global linear direction cannot capture distorted activation geometry. Proposes polynomial kernel PCA steering, which outperforms linear PCA steering where distortion is strong. Results are on Llama and phi.
   Relevance to us: a ready steering baseline for our curve-based edits. Test: compare curve steering against kernel-PCA steering on the same horizon pairs.

6. **Kim (2026).** "Non-linear Interventions on Large Language Models." Sangwoo Kim. https://arxiv.org/abs/2605.14749
   Generalises interventions to features on nonlinear manifolds with a learned feature map via interchange interventions, and reports more precise control on refusal than linear baselines.
   Relevance to us: suggests learning the interchange map for the horizon manifold directly, rather than hand-fitting a curve. Both 5 and 6 are single-group recent preprints, so treat their claims as unreplicated.

## 4. Probing pitfalls

1. **Hewitt & Liang (EMNLP 2019).** "Designing and Interpreting Probes with Control Tasks." https://arxiv.org/abs/1909.03368
   Control tasks assign random outputs to word types, so only the probe can learn them. Selectivity is task accuracy minus control accuracy. Popular probes are not very selective.
   Relevance to us: our R^2 of 1.00 is read off a lookup-capable representation. Test: build a control task that assigns random targets per unit word and per number token, and report selectivity for the horizon probe.

2. **Elazar, Ravfogel, Jacovi, Goldberg (TACL 2021).** "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals." https://arxiv.org/abs/2006.00995
   Probing cannot show what a model causally relies on. Amnesic probing removes a property from the representation and measures the drop in task performance. Probing accuracy does not track importance for word prediction.
   Relevance to us: matches our "decodable but inert" observation. Test: remove the probe subspace (INLP-style) at the horizon tokens and measure the effect on the written unit and on the decision.

3. **Ravichander, Belinkov, Hovy (EACL 2021).** "Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?" https://arxiv.org/abs/2005.00719
   Models encode properties not needed for the task, much of it inherited from pretrained embeddings, and probes can exceed chance when the property is random noise.
   Relevance to us: magnitude may be encoded because it is inherited from the embedding, with no role in the decision. Test: layer-0 probe (see 1.7), plus the decision-vs-horizon ablation.

4. **Belinkov (Computational Linguistics, 2022).** "Probing Classifiers: Promises, Shortcomings, and Advances." https://arxiv.org/abs/2102.12452
   Reviews the probing framework and its methodological limits (probe complexity, causal interpretation, baselines).
   Relevance to us: checklist for the write-up. Report the probe against a lookup baseline, as we already do for unit-only.

## Testable hypotheses for our setting

H1. **Within-unit magnitude is carried by digit slots (Levy & Geva).** Probe tens and units digits at the number tokens, and check that a "tens digit" subspace patch moves "6" to "66". Prediction: digit probes exceed R^2 0.5 where the horizon-average probe does not. Uses saved activations only if the number tokens are among saved positions, otherwise a small patching run.

H2. **Magnitude is partly lexical (Wallace; Ravichander).** Probe layer-0 or early-layer token embeddings for the number token. If R^2 is high at layer 0, the within-unit signal is inherited, not computed. Test on existing activations at early layers.

H3. **Our manifold is a helix or Fourier curve (Kantamneni; Zhou).** Fit linear + sinusoidal basis (periods 10, 100 etc.) to the horizon averages and compare with the 2-4 curved dims found by PCA. Prediction: periodic terms explain the curvature and the "tail".

H4. **Separate unit (categorical) and magnitude (ordinal) subspaces (Park et al.).** Check orthogonality of the 8-dim unit subspace to the within-unit magnitude directions, and that swapping the former leaves the latter unchanged (our existing result predicts yes). Test by subspace overlap on saved activations.

H5. **Probe-line inertness comes from the metric (Park et al. 2023) and the geometry (Engels).** Redo steering along the probe direction in a whitened metric, and along the curve in kernel-PCA space (Raval). Prediction: curve steering beats line steering, and whitening changes the effect only mildly. Small steering run.

H6. **Geodesic distance, not Euclidean, tracks duration (Modell).** Compare Isomap geodesic distance with PCA distance as predictors of |log horizon difference|, and with the decision-score difference. Saved activations only.

H7. **Probe selectivity is low (Hewitt & Liang).** Build per-unit random control targets and report selectivity of the horizon probe at each layer. Prediction: selectivity is high for the unit and low for the within-unit magnitude. Saved activations only.

H8. **Amnesic removal of the unit subspace changes the written unit but not the decision (Elazar).** Project out the unit subspace at the unit word and measure the written unit and the near/far answer. Compare with removing a random 8-dim subspace. Small patching run.

## Gaps

G1. No paper found probing duration-with-unit representations inside a modern LLM. The closest are Gurnee & Tegmark (dates of entities) and Sasaki et al. (quantity comparison), so this work is likely novel.
G2. I found nothing on number representations in thinking/reasoning mode, specifically how the reasoning re-reads edited text.
G3. Recent 2026 preprints (Raval, Kim, Sasaki, Karkada) are not yet peer-reviewed as far as I could verify. Treat their results as single-source.

---

# B. Causal methods, chain of thought, intertemporal choice, comparison


Every entry below was checked against an arXiv abstract page or a search hit returning title, authors and id. Summaries come from abstracts and search snippets, not full-text reads, unless noted. "Prior paper" = Rios-Sialer et al. 2026, "Temporal Preference Concepts and their Functions in a Large Language Model", arXiv:2606.05194 (Qwen3-4B-Instruct-2507; gradient attribution, activation patching, steering). Its two .bib files plus draft cite: Wei 2022 (CoT), Elhage 2021, Meng 2022 (ROME), Li 2023 (ITI), Belinkov 2022 (probing), Rimsky/Panickssery 2024 (CAA), Turner 2023 (ActAdd, in draft text), Zou 2023 (the adversarial-attack paper, not RepE). Nothing else below is cited there. The .bib files and the bibliography section of paper_draft.md were checked via grep; the draft may use citations I did not catch.

---

## Theme 1. Causal intervention methodology

1. **Heimersheim & Nanda (2024). How to use and interpret activation patching.** arXiv:2404.15255. https://arxiv.org/abs/2404.15255. Practical guide: which patching variants support which claims, which metrics to use, and pitfalls. Recommends coarse (residual-stream) patching first, then finer granularity and path patching. Not cited by prior paper.
   Relevance to us: our "0.56 at L24 / 0.82 at L34" numbers are noising/denoising-style transfers, so we should state the direction (clean to corrupted or reverse) and check that they are not metric artifacts. Path patching would test whether the answer-slot read at L22-24 actually mediates the horizon effect.

2. **Zhang & Nanda (2024). Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.** arXiv:2309.16042 (ICLR 2024). https://arxiv.org/abs/2309.16042. Shows that corruption method (e.g. Gaussian noise vs. symmetric token replacement) and metric (logit difference vs. probability) can change the localization result. Not cited by prior paper.
   Relevance to us: report patching on the near/far log-odds, and compare at least two corruption types, e.g. horizon swapped within the same regime vs. across regimes. The force-cut confound argues for resampling counterfactual pairs, not just contrasting classes.

3. **Makelov, Lange & Nanda (2024). Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching.** arXiv:2311.17030 (ICLR 2024). https://arxiv.org/abs/2311.17030. A subspace patch can change the output by switching on a dormant pathway, not by carrying the feature the model normally uses. Notes the success case (IOI) where an independent circuit analysis agreed. Not cited by prior paper.
   Relevance to us: our "swap a subspace at the unit word swaps the unit the reasoning writes" is exactly the kind of result that needs a check against this illusion. Test whether the same subspace is used by unpatched runs (e.g. does its natural variation across prompts predict the written unit?).

4. **Geiger et al. (2023). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS).** arXiv:2303.02536 (CLeaR 2024). https://arxiv.org/abs/2303.02536. Learns a rotation by gradient descent so that interchange interventions on a subspace implement a high-level causal variable. Not cited by prior paper.
   Relevance to us: DAS on the horizon variable (decade vs. unit vs. magnitude) at the unit-word and answer-slot positions would give a learned alternative to our probe, CAA and class-mean-curve directions, which were inert or confounded. Makelov et al. caution applies to DAS as well.

5. **Geiger et al. (2025). Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability.** arXiv:2301.04709 (JMLR 26). https://arxiv.org/abs/2301.04709. Unifies activation/path patching, causal mediation, causal scrubbing, SAEs and DAS under interventions on a high-level causal model, with approximate abstraction to grade fit. Not cited by prior paper.
   Relevance to us: gives a way to write our hypothesis as an explicit high-level graph (horizon text -> reasoning writes horizon -> comparison to delays -> choice) and score it by interchange-intervention accuracy.

6. **Panickssery (Rimsky) et al. (2024). Steering Llama 2 via Contrastive Activation Addition.** arXiv:2312.06681. https://arxiv.org/abs/2312.06681. Averages residual-stream differences over contrast pairs and adds them at inference. Cited by prior paper (as Rimsky 2024).
   Relevance to us: CAA assumes the contrast pairs differ only in the target concept. Our long-vs-short pairs also differ in the rate of reasoning force-cuts, so CAA's wrong-way steer is a violation of that assumption, not evidence about the horizon.

7. **Turner et al. (2023). Steering Language Models With Activation Engineering (first version titled "Activation Addition: Steering Language Models Without Optimization").** arXiv:2308.10248 (id and first author checked 2026-11-03). Cited by prior paper. Prompt-pair activation difference added at one layer.
   Relevance to us: ActAdd-style contrast of two horizon strings would be a cleaner pair than class means because only the horizon text differs.

8. **Tan et al. (2024). Analyzing the Generalization and Reliability of Steering Vectors.** arXiv:2407.12404. https://arxiv.org/abs/2407.12404. Steering is often unreliable even in distribution, with large per-example variance, and breaks under small prompt changes out of distribution. Not cited by prior paper.
   Relevance to us: report per-prompt steering effect distributions, not means. Inert steering on a single probe direction is the expected default, not surprising evidence against a representation.

9. **Braun et al. (2025). Understanding (Un)Reliability of Steering Vectors in Language Models.** arXiv:2505.22637. https://arxiv.org/abs/2505.22637. On Llama2-7B-chat across 36 datasets, steering works better when training activation differences are more cosine-aligned and positive/negative sets separate cleanly; it is unreliable when the behavior is not one coherent direction. Not cited by prior paper.
   Relevance to us: compute cosine consistency of per-pair horizon differences. A low value would explain the probe/CAA failures and motivate a discrete-edit or curved account.

10. **Wu et al. (2025). AxBench.** arXiv:2501.17148 (ICML 2025). https://arxiv.org/abs/2501.17148. On Gemma-2 2B/9B, prompting beats representation steering and finetuning, and difference-in-means is best at concept detection. Not cited by prior paper.
    Relevance to us: text-level editing of the horizon being the strongest lever is consistent with this benchmark's ranking.

Also relevant, not detailed: Zou et al., Representation Engineering, arXiv:2310.01405 (https://arxiv.org/abs/2310.01405), a top-down reading/control framework; not cited by prior paper. On size-matched random controls I found no single canonical paper in this search; treat it as standard practice via Zhang & Nanda and Heimersheim & Nanda (random-direction/resample baselines), not as a cited result.

---

## Theme 2. Reasoning models and chain-of-thought

1. **Lanham et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning.** arXiv:2307.13702. https://arxiv.org/abs/2307.13702. Tests whether answers depend on the CoT by truncating it, inserting mistakes and paraphrasing; faithfulness varies strongly by task and tends to worsen with model capability. Not cited by prior paper.
   Relevance to us: the standard recipe (truncate / corrupt / paraphrase the think block, re-read the answer) applies directly to our 1,400 traces.

2. **Turpin et al. (2023). Language Models Don't Always Say What They Think.** arXiv:2305.04388 (NeurIPS 2023). https://arxiv.org/abs/2305.04388. Biasing features in the prompt shift answers while the CoT does not mention them. Not cited by prior paper.
   Relevance to us: the horizon is stated in the prompt, so the question is whether the CoT's written horizon is faithful to what drives the choice, e.g. does a steered horizon appear in text?

3. **Chen et al. (2025). Reasoning Models Don't Always Say What They Think.** arXiv:2505.05410. https://arxiv.org/abs/2505.05410. Reasoning models (Claude 3.7 Sonnet, DeepSeek R1) verbalize used hints rarely, often under 20%, though more than non-reasoning models. Not cited by prior paper.
   Relevance to us: a base rate for unfaithfulness. In our setting the reasoning does write the horizon, which makes it a favorable case for text-level mediation.

4. **Arcuschin et al. (2025). Chain-of-Thought Reasoning In The Wild Is Not Always Faithful.** arXiv:2503.08679. https://arxiv.org/abs/2503.08679. Without artificial bias, models can write plausible arguments for contradictory answers to paired "is X bigger than Y" / "is Y bigger than X" questions (implicit post-hoc rationalization). Reported rates differ between versions (v3 vs. v4 abstracts), so quote the version. Not cited by prior paper.
   Relevance to us: our task is a magnitude comparison. Swap near/far option order and check whether choice flips in a way the text justifies post hoc.

5. **Bogdan et al. (2025). Thought Anchors: Which LLM Reasoning Steps Matter?** arXiv:2506.19143. https://arxiv.org/abs/2506.19143. Counterfactual resampling of sentences plus attention analysis identifies sentences (planning, uncertainty management) that disproportionately shape later reasoning and the answer. Not cited by prior paper.
   Relevance to us: apply the same resampling to the sentence that first states or converts the horizon, and the sentence that compares it to payout delays.

6. **Macar et al. (2025). Thought Branches: Interpreting LLM Reasoning Requires Resampling.** arXiv:2510.27484 (ICLR 2026). https://arxiv.org/abs/2510.27484. Argues for studying the distribution over CoTs by resampling; reports that off-policy text edits gave small, unstable effects while on-policy resampling was more reliable (per search summary). Not cited by prior paper.
   Relevance to us: our edit-the-horizon-text finding is off-policy. Resample from the same prefix with the horizon sentence removed or replaced, and compare.

7. **Boppana et al. (2026). Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought.** arXiv:2603.05488 (ICML 2026). https://arxiv.org/abs/2603.05488. On DeepSeek-R1 671B and GPT-OSS 120B, attention probes read the final answer from activations well before the text reveals it, especially on easy questions; backtracking correlates with real belief shifts on harder ones. Not cited by prior paper. (Search summary; authors "Boppana, Ma, and colleagues" only.)
   Relevance to us: our 25%/50%/100% activations let us ask when the choice becomes decodable, in contrast to the horizon, which we find barely decodable mid-reasoning.

8. **Zhang et al. (2025). Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification.** arXiv:2504.05419. https://arxiv.org/abs/2504.05419. Hidden states encode correctness of intermediate and future answers; a probe-based stop rule cut tokens 24%. Not cited by prior paper.
   Relevance to us: supports probing the final choice at 25% and 50% of the think block.

9. **Venhoff et al. (2025). Understanding Reasoning in Thinking Language Models via Steering Vectors.** arXiv:2506.18167. https://arxiv.org/abs/2506.18167. In DeepSeek-R1-Distill models, behaviors such as backtracking and uncertainty are controlled by linear steering directions. Not cited by prior paper.
   Relevance to us: reasoning-process features (e.g. cut-off/length) are steerable, which means CAA directions can encode them. This supports the confound reading.

Also: Wang & Zhu (2026), The Detection-Extraction Gap, arXiv:2604.06613 (https://arxiv.org/abs/2604.06613): 52-88% of CoT tokens come after the answer is recoverable from a prefix, and forced extraction fails on 42% of those cases. This is a caution about forced-answer readouts in our setup.

---

## Theme 3. Intertemporal choice and economic rationality in LLMs

1. **Rios-Sialer et al. (2026). Temporal Preference Concepts and their Functions in a Large Language Model.** arXiv:2606.05194. https://arxiv.org/abs/2606.05194. The prior paper itself. Localizes a temporal-preference subgraph in mid-to-upper layers of Qwen3-4B-Instruct-2507, finds horizon information in the residual stream, and reports behavior that discounts less steeply than humans and varies with context, with suggestive steering evidence.
   Relevance to us: our 8B-thinking result should be framed as testing whether that residual-stream account survives when a reasoning trace intervenes.

2. **Goli & Singh (2024). Can LLMs Capture Human Preferences?** arXiv:2305.02531. https://arxiv.org/abs/2305.02531. GPT-3.5 and GPT-4 are less patient than humans; GPT-3.5 shows a lexicographic preference for earlier rewards; patience varies by language; chain-of-thought conjoint narrows the gap but does not close it. Not cited by prior paper.
   Relevance to us: behavioral precedent that CoT changes discounting. Compare the thinking-on vs. thinking-off discount curves in our model.

3. **Chen, Liu, Shan & Zhong (2023). The Emergence of Economic Rationality of GPT.** arXiv:2305.12763 (PNAS 2023). https://arxiv.org/abs/2305.12763. Revealed-preference tests of consistency across risk, time, social and food tasks. GPT scores high but rationality drops under different price framing and discrete-choice settings. Not cited by prior paper.
   Relevance to us: rationality depends on framing, so check that choice is consistent across horizon units (39 years vs. 468 months) and option order.

4. **Tak et al. (2026). Sparks of Rationality: Do Reasoning LLMs Align with Human Judgment and Choice?** arXiv:2601.22329. https://arxiv.org/abs/2601.22329. Deliberate thinking pushes models toward expected-value maximization; representation-level emotion steering gave plausible but less reliable shifts than in-context priming. I could not confirm that it covers temporal discounting. Not cited by prior paper.
   Relevance to us: thinking changes choices toward normative values, and representation steering was the less reliable lever, matching our steering results.

5. **Zhu, Yan & Griffiths (2025). Language Models Trained to do Arithmetic Predict Human Risky and Intertemporal Choice.** arXiv:2405.19313 (ICLR 2025). https://arxiv.org/abs/2405.19313. A small model pretrained on expected-value arithmetic predicts human risky and intertemporal choices better than many cognitive models. Not cited by prior paper.
   Relevance to us: suggests that choices may be computed from payout and delay arithmetic, which supports the comparison/arithmetic view in Theme 4.

Gap: I found no mechanistic study of intertemporal choice in a reasoning model besides the prior paper. A Ballard seminar abstract and a Leng & Nguyen item surfaced but had no verifiable arXiv paper and were dropped.

---

## Theme 4. Comparison and magnitude circuits

1. **Hanna, Liu & Variengien (2023). How does GPT-2 compute greater-than?** arXiv:2305.00586 (NeurIPS 2023). https://arxiv.org/abs/2305.00586. A small circuit in GPT-2 small, in which late MLPs do most of the work, raises the probability of years greater than the start year. The mechanism is neither pure memorization nor general math; the circuit misfires on a less-than variant. Not cited by prior paper.
   Relevance to us: expect late layers at the answer slot (our L22-24) to implement a comparison between horizon and delay values, and expect failures under reversal.

2. **Stolfo, Belinkov & Sachan (2023). A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis.** arXiv:2305.15054 (EMNLP 2023). https://arxiv.org/abs/2305.15054. Causal mediation traces how operand information flows to the answer; a small set of components has high causal influence. Belinkov is a coauthor, but the prior paper cites Belinkov 2022 (probing), not this one.
   Relevance to us: a template for mediation analysis with operand positions as sources and the answer slot as sink.

3. **Levy & Geva (2025). Language Models Encode Numbers Using Digit Representations in Base 10.** arXiv:2410.11781 (NAACL 2025). https://arxiv.org/abs/2410.11781. Comparison errors track digit-space distance, not value distance; causal interventions on circular per-digit representations modify the number. Tested on Llama 3 8B and Mistral 7B. Not cited by prior paper. This overlaps with the number-geometry agent's scope.
   Relevance to us: comparing horizon to delay may be carried out digit by digit, and a swap at the unit word may act as a unit-token substitution.

4. **Sandoval (2025). Even Heads Fix Odd Errors.** arXiv:2508.19414. https://arxiv.org/abs/2508.19414. Single-author preprint on Llama-3.1-8B-Instruct decimal comparison ("9.8 vs 9.11"). Claims format-dependent failure and a head-level repair. Summaries come only from secondary sources, so treat as weak evidence.
   Relevance to us: only as a reminder that comparison behavior can be format-dependent, which is a reason to vary the prompt template.

Gap: I found no work on comparison circuits inside a reasoning trace (where the comparison is written in text), nor on comparing quantities with units (years vs. days).

---

## Testable hypotheses for our setting

H1. **Text mediation (saved traces).** If the reasoning re-reads the horizon as text, the horizon string and unit written in the first reasoning sentences should predict the choice better than the prompt horizon does. Test: regress near/far log-odds on the prompt horizon alone vs. plus the parsed horizon value and unit from the trace; the prompt term should shrink. (Lanham, Thought Anchors.)

H2. **Reasoning text is necessary (GPU run).** Truncate, paraphrase, or swap the horizon sentence in the think block and force the answer. If the choice follows the edited text and not the prompt, the text is the mediator. Use on-policy resampling, not only edits (Macar et al.). Add a control that edits an unrelated sentence of the same length.

H3. **Force-cut confound explains CAA (saved traces).** Within matched pairs of cut vs. uncut traces at the same horizon, compute the CAA direction; its cosine with the long-vs-short direction should be high. Regressing out the cut indicator from the activations should flip the sign of the steering effect. (Venhoff: reasoning-process features are linearly steerable; Braun: low cosine consistency predicts unreliability.)

H4. **Per-prompt steering reliability (saved results or small run).** The steering effect on log-odds has a large variance across prompts, and the effect is concentrated where the horizon crosses a payout delay. Report the distribution and the fraction of prompts moved in the expected direction. (Tan et al.)

H5. **Subspace swap is a pathway switch or the real carrier (GPU run).** For the unit-word subspace, patch across natural prompts and test (a) that the effect size scales with the learned subspace's variance in unpatched runs and (b) that a DAS-trained rotation outperforms a PCA subspace of the same rank, with size-matched random subspaces as the baseline. (Makelov, Geiger.)

H6. **Late-layer comparison is localized and fails under reversal (GPU run).** Patch residuals at the answer slot at L22-24 between prompts that differ only in whether the horizon exceeds the delay. Check that reversing the question (choose the nearer vs. farther option) flips the sign. If the circuit is a greater-than-style comparison, it should misfire under the reversal. (Hanna et al.)

H7. **Choice is decodable before the horizon is (saved activations).** Train probes for the final choice on the 25% and 50% activations. If the choice is decodable earlier than the horizon is, then the reasoning is partly performative (Boppana; Zhang et al.). If it is not, then the horizon re-forming at the end of the think block is a necessary step. The comparison to the 100% and answer-slot decodability gives a time course.

H8. **Thinking changes the discount curve (saved traces plus a short no-think run).** Thinking-on choices are closer to a normative threshold (the horizon vs. the payout delay) than thinking-off choices, with a sharper step. (Tak et al.; Goli & Singh.)

---

## Caveats
- Summaries rely on abstracts and search snippets. Paper figures quoted: only those present in the abstract pages fetched.
- Arcuschin et al. report different rates across versions; check the version before quoting.
- Boppana et al. author list is incomplete in my sources.
