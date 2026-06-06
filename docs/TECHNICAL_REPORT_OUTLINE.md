# Technical Report Outline: Geometric Safety Features for AI Boundary Detection

**Title:** Boundary-Stratified Evaluation of k-NN Geometric Features for AI Safety Detection

**Authors:** [Your Name]

**Date:** January 2026

**Keywords:** AI Safety, Geometric Features, Boundary Detection, k-NN, Embedding Spaces, Uncertainty Quantification

---

## Abstract (200-250 words)

**Structure:**
1. **Problem Statement** (2-3 sentences)
   - AI models fail unpredictably near decision boundaries
   - Embedding-only methods struggle to detect borderline cases
   - Need for geometric features that signal boundary proximity

2. **Approach** (2-3 sentences)
   - We propose 7 k-NN geometric features computed on reference set
   - Boundary-stratified evaluation methodology (safe/borderline/unsafe)
   - Test on sentiment classification with OpenAI embeddings

3. **Key Results** (2-3 sentences)
   - Geometry provides +3.8% improvement on borderline cases
   - 4.8× larger improvement on borderline vs safe cases
   - Top feature: knn_std_distance (r=+0.399 on borderline)

4. **Significance** (1-2 sentences)
   - Demonstrates targeted value where baseline methods struggle
   - Validates continuous correlation mechanism over discrete thresholds
   - Production-ready implementation available

**Writing Tips:**
- Lead with the borderline finding (it's your headline)
- Emphasize "targeted improvement" not just "overall improvement"
- Mention falsification of Dark River hypothesis (honest science)

---

## 1. Introduction (2-3 pages)

### 1.1 Motivation
**Points to Cover:**
- AI safety requires detecting when models approach decision boundaries
- Failure modes concentrate in uncertain regions (borderline cases)
- Existing methods: embedding-only approaches struggle with geometric structure

**Key Claims:**
- "Models are most dangerous when confident but wrong"
- "Borderline cases represent high-stakes uncertainty"
- "Geometric features complement embedding representations"

**References to Include:**
- Prior work on adversarial examples
- Uncertainty quantification in deep learning
- Geometric deep learning

### 1.2 Research Questions
**List 3-4 core questions:**

1. **RQ1:** Do k-NN geometric features improve boundary distance prediction over embedding-only baselines?
   - **Answer:** Yes, +3.8% on borderline (p < 0.001)

2. **RQ2:** Where do geometric features provide the most value?
   - **Answer:** Borderline cases (4.8× larger improvement than safe)

3. **RQ3:** Which geometric features matter most?
   - **Answer:** knn_std_distance (consensus top feature)

4. **RQ4:** Do discrete geometric regions (Dark Rivers) exist?
   - **Answer:** No, falsified for normalized embeddings

**Writing Tips:**
- Frame as hypotheses to be tested
- Each RQ maps to a Results subsection
- RQ4 (falsification) shows honest negative results

### 1.3 Contributions
**Numbered list:**

1. **Boundary-stratified evaluation methodology** that reveals targeted improvements in high-uncertainty regions

2. **7 k-NN geometric features** with validated +3.8% improvement on borderline cases (4.8× larger than safe)

3. **SVD-based curvature computation** solving numerical instability for k << D

4. **Falsification of Dark River hypothesis** for normalized embeddings, replacing with continuous correlation mechanism

5. **Production-ready implementation** with frozen schema v2.0 and comprehensive testing

**Writing Tips:**
- Lead with methodology contribution (boundary-stratified)
- Emphasize honest falsification (contribution #4)
- Mention code availability

### 1.4 Paper Organization
**Brief roadmap:**
- Section 2: Methods (features, evaluation, data)
- Section 3: Results (boundary-sliced, feature importance, ablation)
- Section 4: Discussion (interpretation, limitations, implications)
- Section 5: Conclusion

---

## 2. Methods (4-5 pages)

### 2.1 k-NN Geometric Features

#### 2.1.1 Feature Definitions
**Table 1: 7 Geometric Features**

| # | Feature | Symbol | Formula | Interpretation |
|---|---------|--------|---------|----------------|
| 1 | k-NN Mean | μ_knn | mean(d₁, ..., d_k) | Average neighbor distance |
| 2 | k-NN Std | σ_knn | std(d₁, ..., d_k) | Neighborhood uniformity |
| 3 | k-NN Min | d_min | min(d₁, ..., d_k) | Nearest neighbor |
| 4 | k-NN Max | d_max | max(d₁, ..., d_k) | Neighborhood extent |
| 5 | Local Curvature | JC | σ_min/σ_max (SVD) | Manifold anisotropy |
| 6 | Ridge Proximity | SD | σ_knn/μ_knn | Density gradient |
| 7 | 1-NN Distance | d_1nn | distance to nearest | Redundant with d_min |

**Writing Tips:**
- Explain intuition for each feature
- Note: feature 7 redundant (kept for completeness)
- Emphasize k=50 is validated optimum

#### 2.1.2 SVD-Based Curvature (Critical Detail)
**Problem Statement:**
- Original method: eigenvalues of 256×256 covariance from 50 neighbors
- Result: rank deficiency (rank ≤ 49), ill-conditioning (κ ~ 10¹⁸)
- Eigenvalues collapse to numerical noise (~10⁻¹⁶)

**Solution:**
```
# Direct SVD on (k, D) matrix
centered = neighbors - query_point
U, S, Vt = svd(centered)
curvature = S[-1] / S[0]  # smallest/largest singular value
```

**Validation:**
- Before fix: all curvature values = 0
- After fix: mean = 0.0138 ± 0.0025 (meaningful!)
- 20/20 trials show +3.8% improvement

**Writing Tips:**
- This is a technical contribution worth highlighting
- Provide pseudocode for reproducibility
- Explain why SVD is more stable than covariance

#### 2.1.3 ParallaxChain Guarantee
**Property:**
- Reference-only computation: queries never modify reference set
- Batch-order invariance: permute(queries) → permute(features)

**Why This Matters:**
- Reproducibility: same query → same features
- Scalability: process queries in any order
- Trust: no hidden dependencies

**Writing Tips:**
- This ensures reproducibility
- Important for production deployment

### 2.2 Boundary-Stratified Evaluation

#### 2.2.1 Zone Definitions
**Three regions based on boundary distance:**

1. **SAFE:** boundary_distance > 0.5
   - Model confident AND correct
   - Expected: high baseline performance, minimal improvement

2. **BORDERLINE:** |boundary_distance| < 0.5
   - High uncertainty, critical safety region
   - Expected: low baseline performance, large improvement

3. **UNSAFE:** boundary_distance < -0.5
   - Model confident BUT wrong
   - Expected: moderate difficulty, moderate improvement

**Figure Reference:** Include visualization of zone definitions

**Writing Tips:**
- Explain rationale for threshold (0.5)
- Justify asymmetric zones (safe vs unsafe)

#### 2.2.2 Evaluation Protocol
**Steps:**
1. Split data: 80% reference, 20% query
2. Compute 7 geometric features for queries
3. Stratify by boundary distance (safe/borderline/unsafe)
4. Train Ridge regression (α=1.0) on each zone
5. Compare R² for baseline (embeddings only) vs geometry (embeddings + 7 features)
6. Run 20 independent trials for statistical robustness

**Metrics:**
- Primary: R² (coefficient of determination)
- Secondary: MAE (mean absolute error)
- Statistical test: One-sample t-test (H₀: improvement = 0)

**Writing Tips:**
- Emphasize 20 trials for robustness
- Justify Ridge regression (simple, interpretable)
- Note: results deterministic (std = 0.000)

### 2.3 Dataset

#### 2.3.1 Sentiment Classification
**Details:**
- Task: Binary sentiment classification
- Embedder: OpenAI `text-embedding-3-large`
- Embedding dimension: D = 256
- Total samples: N = 1099
- Train/test split: 80/20 (879 reference, 220 query)

**Boundary Distance:**
- Ground truth: true label ∈ {0, 1}
- Model prediction: probability p ∈ [0, 1]
- Boundary distance: 2(p - 0.5) if correct, -2(p - 0.5) if wrong
- Range: [-2.36, +2.10] in dataset

**Zone Distribution:**
- SAFE: 67 samples (30.5%)
- BORDERLINE: 79 samples (35.9%)
- UNSAFE: 74 samples (33.6%)

**Writing Tips:**
- Explain boundary distance metric clearly
- Note: borderline is largest zone (35.9%)

#### 2.3.2 Data Properties
**Embedding Statistics:**
- Normalized: ‖embedding‖ = 1.0 (by design)
- Smooth density: ridge_proximity ≈ 0.2 (uniform)
- Low curvature: JC ≈ 0.01-0.02 (anisotropic manifold)

**Implications:**
- Modern embeddings are well-behaved
- No discrete "Dark River" regions
- Geometry works via continuous correlations

---

## 3. Results (5-6 pages)

### 3.1 Boundary-Stratified Performance (RQ1 & RQ2)

#### 3.1.1 Main Finding
**Figure 1:** R² by Region (bar chart)

**Table 2: Performance by Zone**

| Zone | N | Baseline R² | Geometry R² | Improvement | Significance |
|------|---|-------------|-------------|-------------|--------------|
| BORDERLINE | 79 | 0.575 | 0.597 | **+3.8%** | p < 0.001 *** |
| UNSAFE | 74 | 0.680 | 0.694 | +2.1% | p < 0.001 *** |
| SAFE | 67 | 0.604 | 0.609 | +0.8% | p < 0.001 *** |

**Key Observations:**
1. **Borderline shows largest improvement** (+3.8% vs +2.1% unsafe, +0.8% safe)
2. **Borderline has lowest baseline** (R² = 0.575 vs 0.604 safe, 0.680 unsafe)
3. **Improvements inversely correlated with baseline performance**

**Claim:**
> "Geometry features provide 4.8× larger improvements on borderline cases compared to safe cases (3.8% / 0.8% = 4.75)"

**Writing Tips:**
- Lead with borderline result (it's the headline)
- Explain why borderline has lowest baseline
- Statistical significance table in appendix

#### 3.1.2 Interpretation
**Why This Validates Targeted Value:**

1. **Not just "more parameters"**
   - If geometry were just regularization, we'd see uniform improvement
   - Instead, improvement concentrates where baseline struggles

2. **Meaningful safety signal**
   - Borderline = high-stakes uncertain region
   - Geometry helps most where it matters most

3. **Efficient deployment path**
   - 64% of queries are safe/unsafe (minimal improvement)
   - 36% are borderline (large improvement)
   - Can route based on uncertainty

**Writing Tips:**
- Contrast with uniform improvement hypothesis
- Emphasize "targeted" vs "general" improvement

### 3.2 Feature Importance Analysis (RQ3)

#### 3.2.1 Correlation Analysis
**Figure 2:** Feature Importance (overall vs borderline)

**Table 3: Pearson Correlations**

| Feature | Overall | Borderline | Ratio |
|---------|---------|------------|-------|
| knn_max_distance | **+0.345*** | +0.168 | 0.49 |
| knn_std_distance | +0.286*** | **+0.399***⭐ | **1.39** |
| knn_mean_distance | +0.221** | -0.023 | -0.10 |
| ridge_proximity | +0.215** | +0.361*** | 1.68 |
| local_curvature | -0.103 | +0.007 | -0.07 |
| knn_min_distance | +0.067 | -0.049 | -0.73 |
| dist_to_ref_nearest | +0.067 | -0.049 | -0.73 |

**Key Findings:**
1. **knn_std_distance: consensus top feature**
   - Appears in top-3 for correlation, RF importance, ablation loss
   - **Amplified on borderline** (r=+0.399 vs +0.286 overall)

2. **Different features for different zones**
   - Overall winner: knn_max_distance (r=+0.345)
   - Borderline winner: knn_std_distance (r=+0.399)

3. **Neighborhood variance matters most**
   - Std, max capture local density variations
   - Mean less informative

**Writing Tips:**
- Highlight consensus winner (knn_std_distance)
- Explain "amplified on borderline" phenomenon

#### 3.2.2 Ablation Study
**Figure 3:** Ablation Study (performance loss)

**Table 4: Leave-One-Out Results**

| Feature Removed | R² Loss | % of Baseline | Category |
|-----------------|---------|---------------|----------|
| local_curvature | +0.0180 | 2.0% | Critical |
| knn_std_distance | +0.0048 | 0.5% | Important |
| knn_mean_distance | +0.0043 | 0.5% | Important |
| ridge_proximity | +0.0012 | 0.1% | Marginal |
| Others | < 0.001 | < 0.1% | Redundant |

**Key Findings:**
1. **Local curvature most critical** (2.0% loss)
   - Low correlation (r=-0.103) but high ablation loss
   - Provides non-linear orthogonal information

2. **knn_std_distance important** (0.5% loss)
   - High correlation AND ablation loss
   - Consensus top feature confirmed

3. **Recommendation: keep all 7 features**
   - Small feature set (only 7)
   - Non-linear interactions not tested
   - Minimal computational cost

**Writing Tips:**
- Explain paradox: low correlation but high ablation loss for curvature
- Emphasize non-linear value

### 3.3 Hypothesis Falsification (RQ4)

#### 3.3.1 Dark River Hypothesis
**Original Claim (v1.0):**
> "Dark Rivers are discrete unstable regions identified by low curvature (< 0.5) AND high ridge (> 2.0)"

**Test Results:**
- Ridge proximity range: [0.069, 0.443] (max = 0.443 << 2.0)
- Detections using original threshold: **0/220 (0%)**
- Curvature range: [0.010, 0.022] (all < 0.5)

**Root Cause:**
- Modern embeddings (OpenAI, Cohere, etc.) are normalized
- Smooth, uniform density everywhere (σ/μ ≈ 0.2)
- No discrete high-variance "ridges" exist

**Conclusion:**
❌ **Dark River discrete region hypothesis FALSIFIED** for normalized embeddings

**Replacement:**
✓ Continuous correlation mechanism: geometry provides graded signals integrated by ML models

**Writing Tips:**
- This is honest negative result (good science!)
- Explain why hypothesis seemed plausible initially
- Emphasize continuous > discrete

#### 3.3.2 Revised Understanding
**What Actually Works:**

1. **Continuous not discrete**
   - No binary threshold separates safe from unsafe
   - Features provide graded signals

2. **Correlation not classification**
   - Features correlate with boundary distance
   - ML models learn optimal combinations

3. **Targeted not uniform**
   - Improvements concentrate on borderline
   - Not just "more features = better"

**Writing Tips:**
- Frame as refinement, not failure
- Emphasize what we learned

---

## 4. Discussion (3-4 pages)

### 4.1 Interpretation of Results

#### 4.1.1 Why Geometry Helps on Borderline
**Explanation:**
- Borderline cases have high uncertainty (model probability ≈ 0.5)
- Embedding-only methods struggle to resolve fine-grained differences
- Geometric features capture local manifold structure
- Neighborhood variance (knn_std_distance) signals approaching boundary

**Evidence:**
- Baseline R² lowest on borderline (0.575)
- Geometry improvement largest on borderline (+3.8%)
- Top feature (knn_std_distance) amplified on borderline

**Writing Tips:**
- Connect to manifold hypothesis
- Explain why variance matters near boundaries

#### 4.1.2 Role of Local Curvature
**Paradox:**
- Low correlation (r=-0.103) but highest ablation loss (2.0%)

**Explanation:**
- Provides non-linear information orthogonal to other features
- Captures manifold anisotropy
- Random Forest can exploit non-linear relationships

**Validation:**
- RF importance: 0.194 (3rd place)
- Ablation: 2.0% loss (1st place)
- Confirms value despite weak linear correlation

**Writing Tips:**
- Emphasize non-linear value
- Connect to manifold learning literature

### 4.2 Comparison to Prior Work

#### 4.2.1 Geometric Deep Learning
**Connections:**
- Graph neural networks use local geometry
- Manifold learning captures intrinsic structure
- Our work: applies to embedding spaces

**Novelty:**
- Boundary-stratified evaluation (new)
- Production-ready k-NN features (simple, effective)
- Falsification of discrete region hypothesis (honest science)

#### 4.2.2 Uncertainty Quantification
**Connections:**
- Bayesian methods: predictive uncertainty
- Ensemble methods: variance across models
- Our work: geometric uncertainty signals

**Advantage:**
- No retraining required (post-hoc features)
- Works with any embedding model
- Interpretable (k-NN statistics)

**Writing Tips:**
- Position as complementary to existing methods
- Emphasize simplicity + effectiveness

### 4.3 Limitations

#### 4.3.1 Single Embedder
**Limitation:**
- Only tested on OpenAI `text-embedding-3-large`
- May not generalize to all embedders

**Mitigation:**
- Normalized embeddings are standard
- Smooth geometry expected across modern embedders

**Future Work:**
- Test on Sentence-transformers, Cohere
- Characterize when geometry helps vs doesn't

#### 4.3.2 Reference Set Requirements
**Limitation:**
- Requires representative reference set
- Performance degrades if reference distribution shifts

**Mitigation:**
- Update reference set periodically
- Monitor distribution drift

#### 4.3.3 Computational Cost
**Limitation:**
- k-NN search: O(N_ref × D) per query

**Mitigation:**
- Use approximate k-NN (FAISS, Annoy)
- Two-tier routing (fast/slow path)

**Writing Tips:**
- Be honest about limitations
- Provide concrete mitigation strategies

### 4.4 Implications for AI Safety

#### 4.4.1 Production Deployment
**Recommendations:**
1. **Use geometry for borderline detection**
   - High-uncertainty queries → compute geometry
   - Clear safe/unsafe → skip geometry (save compute)

2. **Monitor geometry distributions**
   - Detect distribution shifts
   - Flag anomalous geometric structure

3. **Combine with embedding-only confidence**
   - Fast path: embedding-only classifier
   - Slow path: embedding + geometry (when needed)

**Writing Tips:**
- Practical, actionable advice
- Connect to real-world deployment

#### 4.4.2 Research Directions
**Future Work:**
1. **Generalization across embedders**
   - Multi-embedder validation
   - When does geometry help?

2. **Two-tier architecture**
   - Confidence-based routing
   - Optimize for latency/accuracy tradeoff

3. **Geometry Health Panel**
   - Real-time monitoring
   - Anomaly detection

**Writing Tips:**
- Frame as natural extensions
- Provide clear next steps

---

## 5. Conclusion (1 page)

### 5.1 Summary of Contributions
**Restate main findings:**
1. **Boundary-stratified evaluation reveals targeted improvements**
   - Geometry helps 4.8× more on borderline vs safe cases
   - Validates "targeted value" hypothesis

2. **7 k-NN features provide validated +3.8% improvement**
   - Top feature: knn_std_distance (consensus winner)
   - SVD-based curvature solves numerical instability

3. **Dark River hypothesis falsified, replaced with continuous mechanism**
   - Honest negative result
   - Continuous correlations > discrete thresholds

4. **Production-ready implementation with frozen schema v2.0**
   - ParallaxChain guarantee (reproducibility)
   - Comprehensive testing suite

### 5.2 Broader Impact
**Key Takeaways:**
- Geometric features complement embeddings for safety detection
- Improvements concentrate where baseline methods struggle (borderline)
- Simple k-NN features effective, no retraining required

**Significance:**
- Practical path to safer AI systems
- Honest science: falsification as contribution
- Open implementation for reproducibility

### 5.3 Closing Statement
**Final message:**
> "By combining embedding representations with geometric safety signals, we can build AI systems that recognize their own uncertainty—especially in the borderline regions where safety matters most."

---

## Appendices

### Appendix A: Detailed Statistics
**Tables:**
- A1: Full zone statistics (all metrics)
- A2: Feature correlations (all features × all zones)
- A3: Ablation results (all combinations)

### Appendix B: Implementation Details
**Code snippets:**
- B1: SVD curvature computation (pseudocode)
- B2: Boundary stratification algorithm
- B3: ParallaxChain test

### Appendix C: Reproducibility
**Materials:**
- C1: Dataset download links
- C2: Code repository (GitHub)
- C3: Frozen schema v2.0 specification
- C4: Run environment (Python, dependencies)

---

## References

**Categories to include:**
1. **Geometric Deep Learning**
   - Graph neural networks
   - Manifold learning
   - Intrinsic dimensionality

2. **AI Safety**
   - Adversarial examples
   - Out-of-distribution detection
   - Uncertainty quantification

3. **Embedding Models**
   - OpenAI embeddings
   - Sentence transformers
   - Evaluation benchmarks

4. **Statistical Methods**
   - Ridge regression
   - Bootstrap confidence intervals
   - Multiple testing correction

---

## Formatting Guidelines

**Length Targets:**
- Abstract: 200-250 words
- Introduction: 2-3 pages
- Methods: 4-5 pages
- Results: 5-6 pages
- Discussion: 3-4 pages
- Conclusion: 1 page
- **Total: 15-20 pages** (excluding appendices, references)

**Figures:**
- Figure 1: R² by region (Results 3.1)
- Figure 2: Feature importance (Results 3.2)
- Figure 3: Ablation study (Results 3.2)
- Figure 4: Zone visualization (Methods 2.2, optional)

**Tables:**
- Table 1: Feature definitions (Methods 2.1)
- Table 2: Performance by zone (Results 3.1)
- Table 3: Correlations (Results 3.2)
- Table 4: Ablation results (Results 3.2)

**Writing Style:**
- Past tense for your work ("we found", "we tested")
- Present tense for established facts ("embeddings capture semantic structure")
- Active voice preferred ("We propose 7 features" not "7 features are proposed")
- Clear, direct language (avoid jargon where possible)

---

## Pre-Submission Checklist

**Before Weekend Writing:**
- [ ] Review Phase E contract v2.0 for technical details
- [ ] Check all figure files render correctly
- [ ] Verify all statistics match validation runs
- [ ] Prepare reference list

**During Writing:**
- [ ] Each claim has supporting evidence (figure, table, or citation)
- [ ] Figures referenced in text before appearing
- [ ] Tables have clear captions
- [ ] All acronyms defined on first use

**After First Draft:**
- [ ] Read abstract alone (does it tell the story?)
- [ ] Check figure/table numbering
- [ ] Verify all citations formatted correctly
- [ ] Spell check + grammar check

---

**END OF OUTLINE**

---

## Quick Reference: Your 3 Headline Claims

Use these when writing abstract/introduction/conclusion:

1. **"Geometry features provide 4.8× larger improvements on borderline cases compared to safe cases"**
   - Evidence: Borderline +3.8%, Safe +0.8%
   - Figure: Figure 1
   - Significance: p < 0.001

2. **"knn_std_distance is the consensus top feature, amplified on borderline (r=+0.399 vs r=+0.286 overall)"**
   - Evidence: Top-3 in correlation, RF importance, ablation
   - Figure: Figure 2
   - Table: Table 3

3. **"Dark River discrete region hypothesis falsified; continuous correlations provide safety signals"**
   - Evidence: 0 detections, ridge_max = 0.443 << 2.0
   - Section: Results 3.3
   - Impact: Honest negative result advances field

---

**Good luck with weekend writing! This outline should give you clear structure to follow.**
