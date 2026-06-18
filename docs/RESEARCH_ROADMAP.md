# Mirrorfield Research Roadmap

> **⚠️ STALE (Jan 2026, geometry era — superseded).** This roadmap is built on premises
> the project later **falsified**: the 16-cell/polytope program, the "LID ≈ 4" and
> "16-cell outperforms all alternatives (ΔR² +0.72)" claims, and the "Dark River"
> hypothesis are all **retracted** (see `WORK_MAP.md` §4 and the root README banner).
> Treat everything below as historical. The **one durable thing here is the set of
> external literature pointers** (emergent misalignment / toxic personas, the AI-deception
> & sycophancy survey, transformer circuits, faithfulness) — those real safety problems
> are now pursued *behaviorally*, without geometry, in
> [`NOVEL_SAFETY_DIRECTIONS.md`](NOVEL_SAFETY_DIRECTIONS.md). Read that instead.

**Date**: January 5, 2026
**Purpose**: Connect Mirrorfield geometry constraints to DMET and related manifold dynamics research

---

## Executive Summary

This roadmap connects Mirrorfield's **geometric constraint approach** to recent theoretical advances in:
1. **Dynamic Manifold Evolution Theory (DMET)** - observing attractor dynamics
2. **Toxic persona latents** - detecting misalignment
3. **Geometric deep learning** - implementing manifold constraints

**Key differentiation**: While existing research **observes and detects**, Mirrorfield aims to **constrain and prevent** via geometric priors.

---

## TIER 1: Core DMET Papers (Read These First)

### 1. Dynamic Manifold Evolution Theory (PRIMARY)

**Paper**: "Empirical Investigation of Latent Representational Dynamics in Large Language Models: A Manifold Evolution Perspective"
- **Authors**: Yukun Zhang (Chinese U. of Hong Kong), Qi Dong (Fudan University)
- **arXiv**: [2505.20340](https://arxiv.org/abs/2505.20340) (May 24, 2025)
- **Official**: [HTML version](https://arxiv.org/html/2505.20340)

**What it is**: DMET models LLM generation as a continuous trajectory evolving on a low-dimensional semantic manifold.

**Three core metrics** (C, Q, P):
1. **C (Continuity)**: State continuity - how smoothly representations evolve
2. **Q (Compactness)**: Attractor compactness - stability of convergence regions
3. **P (Persistence)**: Topological persistence - structural invariants

**Key findings**:
- Smoother trajectories → greater fluency
- Richer topological organization → enhanced coherence
- Attractor dynamics are observable across multiple Transformer architectures

**Connection to Mirrorfield**:
- **They measure C, Q, P** → **We can constrain them via geometry**
- Their "attractor manifolds" = our "designed 4D polytope structure"
- Their "topological persistence (P)" = our ridge/curvature features

**Critical sections to read**:
- Section 3: Methodology (how they compute C, Q, P)
- Section 4: Experiments (empirical validation across architectures)
- Figure 3-5: Visualization of manifold evolution

**Read this first**: [arXiv PDF](https://arxiv.org/pdf/2505.20340)

---

## TIER 2: Direct Dependencies (What DMET Built On)

### 2. Neural Ordinary Differential Equations

**Paper**: "Neural Ordinary Differential Equations"
- **Authors**: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
- **arXiv**: [1806.07366](https://arxiv.org/abs/1806.07366) (June 19, 2018)
- **Venue**: NeurIPS 2018 (Best Paper Award)

**Core innovation**: View residual networks as **discrete approximations of continuous ODEs**.

**Key insight**:
```
ResNet layer:  h_{t+1} = h_t + f(h_t, θ)
Continuous:    dh/dt = f(h, θ, t)
```

**Why this matters for DMET**:
- Transformer layers → Euler steps in continuous dynamics
- Residual stream → trajectory through latent manifold
- Layer depth → time parameter in ODE

**Connection to Mirrorfield**:
- **They model discrete → continuous** → **We can constrain the continuous dynamics**
- Neural ODEs provide theoretical foundation for "geometric flow"
- Our 16-cell adjacency could define flow constraints

**Critical sections**:
- Section 2: Reverse-mode automatic differentiation (adjoint method)
- Section 3: Replacing ResNets with ODEs
- Section 4: Continuous normalizing flows

**Read this second**: [arXiv PDF](https://arxiv.org/pdf/1806.07366)

---

### 3. Attractor Networks in Neuroscience

**Paper**: "The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep"
- **Authors**: Chaudhuri, R., Gerçek, B., Pandey, B., et al.
- **Journal**: [Nature Neuroscience, Volume 22, 1512–1520 (2019)](https://www.nature.com/articles/s41593-019-0460-x)
- **Published**: August 12, 2019

**Biological validation**: Mammalian brain constructs a navigational compass using a **one-dimensional ring of stable activity states**.

**Key findings**:
- Head-direction circuit forms a **topologically nontrivial 1D ring attractor**
- Ring exhibits isometry and is **invariant across waking and REM sleep**
- Directly demonstrates **continuous attractor dynamics** in biological neural circuits

**Connection to Mirrorfield**:
- **Biology uses ring attractors** → **We use 4D polytope attractors**
- Their "stable activity states" = our "polytope cells"
- Their "isometry preservation" = our geometry preservation goal

**Why this is critical**:
- Proves attractor dynamics work in real biological systems
- Shows that low-dimensional manifolds (1D ring) can encode high-dimensional information
- Validates "designed geometry" approach to cognition

**Critical sections**:
- Figure 1: Ring attractor visualization
- Methods: Topological characterization of manifolds
- Results: Invariance across brain states

**Read this third**: [Nature paper](https://www.nature.com/articles/s41593-019-0460-x) or [McGovern Institute PDF](https://mcgovern.mit.edu/wp-content/uploads/2024/05/s41593-019-0460-x.pdf)

---

### 4. Transformer Circuits Framework

**Paper**: "A Mathematical Framework for Transformer Circuits"
- **Authors**: Nelson Elhage, Neel Nanda, Catherine Olsson, Chris Olah, et al. (Anthropic)
- **Published**: December 22, 2021
- **Official**: [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

**Core concept**: **Residual stream** as central communication channel.

**Key insight**: All transformer components (embedding, attention, MLP, unembedding) communicate by:
1. **Reading** from residual stream (linear projection)
2. **Writing** to residual stream (add linear projection back)

**Residual stream formula**:
```
residual_stream = embedding + Σ(attention_outputs) + Σ(MLP_outputs)
```

**Connection to Mirrorfield**:
- **Residual stream = trajectory through embedding space**
- Each layer adds a "perturbation" to this trajectory
- Our geometry constraints can shape how these perturbations accumulate

**Critical sections**:
- Section 2: Residual stream as communication channel
- Section 3: Attention heads as independent operations
- Section 4: Linear structure of transformers

**Why mechanistic interpretability matters**:
- Understanding information flow → targeted geometric interventions
- Knowing which layers matter → where to apply constraints
- Decomposing computations → modular constraint design

**Read this fourth**: [Transformer Circuits website](https://transformer-circuits.pub/2021/framework/index.html)

---

## TIER 3: Dark River / Toxic Stability Research

### 5. Emergent Misalignment (OpenAI)

**Paper**: "Persona Features Control Emergent Misalignment"
- **Authors**: OpenAI Alignment Team
- **arXiv**: [2506.19823](https://arxiv.org/abs/2506.19823) (June 2024)
- **OpenAI Blog**: [Toward understanding and preventing misalignment generalization](https://openai.com/index/emergent-misalignment/)
- **PDF**: [Direct PDF](https://cdn.openai.com/pdf/a130517e-9633-47bc-8397-969807a43a23/emergent_misalignment_paper.pdf)

**Discovery**: Fine-tuning GPT-4o on intentionally insecure code causes **emergent misalignment** - model gives stereotypically malicious responses to unrelated prompts.

**The "Toxic Persona" Feature**:
- Discovered using **sparse autoencoders (SAEs)** and "model diffing"
- Specific latent pattern that activates for morally questionable behaviors
- Activates at **fractions as low as 5%** (early warning signal!)

**Critical findings**:
1. **Low poisoning threshold**: 5% malicious data causes emergent misalignment
2. **Generalizes across domains**: Insecure code → toxic responses elsewhere
3. **Latent-level control**: Steering the "toxic persona" feature controls misalignment
4. **Persistent across fine-tuning**: Feature remains even with safety training

**Connection to Mirrorfield Dark Rivers**:
- **Their "toxic persona" = our "Dark River" attractor**
- **Their 5% threshold = our boundary proximity threshold**
- **Their steering = our geometric constraints**

**Critical difference**:
| OpenAI Approach | Mirrorfield Approach |
|-----------------|----------------------|
| **Detect** toxic latent post-hoc | **Prevent** via geometry constraints |
| Steer away from bad feature | Make bad feature geometrically unreachable |
| Monitoring-based safety | Structure-based safety |

**Key quote**:
> "Malicious actors may intentionally poison training data to make models misaligned during fine-tuning, and training data that is incorrect but otherwise appears innocuous may subvert safeguards."

**This validates Mirrorfield's motivation**: Need proactive geometric constraints, not just reactive detection.

**Read this fifth**: [arXiv HTML version](https://arxiv.org/html/2506.19823v2)

---

### 6. AI Deception Survey

**Paper**: "AI deception: A survey of examples, risks, and potential solutions"
- **Authors**: Park et al.
- **Journal**: [Patterns, Volume 5, Issue 5 (May 10, 2024)](https://www.cell.com/patterns/fulltext/S2666-3899(24)00103-X)
- **arXiv**: [2308.14752](https://arxiv.org/abs/2308.14752)
- **PMC**: [Free full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11117051/)

**Definition**: Deception = "systematic inducement of false beliefs in pursuit of some outcome other than truth"

**Three types of deceptive behavior**:
1. **Strategic deception**: Deliberate manipulation
2. **Sycophancy**: Agreeing regardless of accuracy (false equilibrium)
3. **Unfaithful reasoning**: Misleading chain-of-thought

**Sycophancy as Stable False Equilibrium**:
- LLMs empirically tend to agree with conversational partners
- Users get **locked into persistent false beliefs**
- Reinforces misconceptions → political polarization
- Provides "pleasing but inaccurate advice"

**Connection to Mirrorfield**:
- **Sycophancy = attractor toward false beliefs**
- **Our geometry can prevent convergence to these attractors**
- **Persistent false beliefs = stable fixed points we want to eliminate**

**Key insight**: Deception is not just individual responses, but **systemic patterns** (attractors).

**Read this sixth**: [arXiv PDF](https://arxiv.org/pdf/2308.14752)

---

## TIER 4: Geometric Methods (Relevant to H₄/16-cell Work)

### 7. Geometric Deep Learning & Equivariance

**Paper**: "Geometric Deep Learning and Equivariant Neural Networks"
- **Authors**: Multiple authors (comprehensive review)
- **Journal**: [Artificial Intelligence Review (2023)](https://link.springer.com/article/10.1007/s10462-023-10502-7)
- **arXiv**: [2105.13926](https://arxiv.org/abs/2105.13926)

**Core principle**: **Equivariance** = "if you transform the input, the output transforms predictably"

**Mathematical formulation**:
```
f(g · x) = g · f(x)  for all g ∈ G (symmetry group)
```

**Key concepts for Mirrorfield**:
1. **Gauge equivariant CNNs**: Constructed using principal bundles over manifolds
2. **Structure group K**: Defines allowed transformations
3. **Homogeneous spaces**: Position/orientation symmetries

**Recent advances (2024)**:
- **PDE-based Group CNNs**: Generalize roto-translation equivariance
- **Efficient flow matching**: Geometry-preserving flows on manifolds
- **Principal bundles**: Mathematical framework for equivariant design

**Connection to Mirrorfield**:
- **Our 16-cell adjacency defines symmetry group**
- **Message passing respects polytope structure** = equivariance
- **4D projection preserves geometric properties** = gauge theory

**Applications**:
- Protein structure processing (molecular biology)
- Diffusion tensor imaging (medical)
- Weather forecasting (physics)

**Critical insight**: **Geometry is not just a representation - it's a constraint on allowed computations.**

**Read this seventh**: [Springer article](https://link.springer.com/article/10.1007/s10462-023-10502-7)

---

### 8. Persistent Homology in Neural Networks

**Paper**: "Topological Data Analysis for Neural Network Analysis: A Comprehensive Survey"
- **Authors**: Multiple (2024 survey)
- **arXiv**: [2312.05840](https://arxiv.org/abs/2312.05840)
- **HTML**: [arXiv HTML version](https://arxiv.org/html/2312.05840v2)

**What is Persistent Homology**:
- Captures **topological features** (connected components, holes, voids) across multiple scales
- Produces **persistence diagrams** showing birth/death of features
- Provides **robust, multiscale, interpretable features**

**Connection to DMET's P metric**:
- DMET uses persistent homology for **P (topological persistence)**
- Measures structural invariants of manifold evolution
- Detects when topology changes during generation

**Recent advances (2024)**:
1. **Persistent Laplacians**: Spectral representations capturing both topology and homotopy
2. **Topological Deep Learning (TDL)**: Integrated with neural networks since 2017
3. **Molecular applications**: Protein complex structure assessment with AlphaFold3

**Connection to Mirrorfield**:
- **Our ridge/curvature features = topological features**
- **Polytope cells = persistent homology generators**
- **Stability across layers = persistent features**

**Key application**: Attention networks + persistent homology → multiscale prediction

**Why this matters**:
- Provides mathematical foundation for "geometry stability"
- Shows how to measure topological invariants in neural networks
- Connects to DMET's P metric directly

**Read this eighth**: [arXiv survey](https://arxiv.org/html/2312.05840v2)

---

## Practical Implementation Timeline

### Week 1: DMET Deep Dive

**Goal**: Understand how to measure C, Q, P metrics

**Tasks**:
1. Download and read [DMET paper](https://arxiv.org/pdf/2505.20340)
2. Focus on Section 3 (methodology) - extract metric formulas
3. Identify which metrics we already compute (ridge proximity ≈ Q?)
4. Document gaps: What metrics are we missing?

**Output**: Document `DMET_METRICS_COMPARISON.md`
- Table mapping DMET metrics → Mirrorfield features
- Identify what's already measured vs. what needs implementation

---

### Week 2: Continuous Dynamics Framework

**Goal**: Understand Neural ODEs and residual stream flow

**Tasks**:
1. Read [Neural ODEs paper](https://arxiv.org/pdf/1806.07366) - focus on Section 2-3
2. Read [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) - focus on residual stream
3. Map residual stream evolution to "flow on 16-cell manifold"
4. Sketch: How would 16-cell adjacency constrain this flow?

**Output**: Sketch `16CELL_FLOW_CONSTRAINTS.md`
- Mathematical formulation of constrained flow
- Pseudocode for flow-based geometry constraints

---

### Week 3: Dark River Mapping

**Goal**: Connect OpenAI's toxic persona to Dark River concept

**Tasks**:
1. Read [OpenAI emergent misalignment paper](https://arxiv.org/html/2506.19823v2)
2. Read [AI deception survey](https://arxiv.org/pdf/2308.14752)
3. Document overlaps with Dark River hypothesis
4. Identify testable predictions: Can geometry prevent toxic attractors?

**Output**: Document `DARK_RIVER_VALIDATION_PLAN.md`
- Hypothesis: Toxic personas = attractors on boundary
- Experiment: Can 16-cell constraints prevent 5% poisoning threshold?
- Metrics: Use SAE features + Mirrorfield geometry metrics

---

### Week 4: Geometric Constraint Implementation

**Goal**: Apply geometric deep learning principles to Mirrorfield

**Tasks**:
1. Read [Geometric Deep Learning review](https://link.springer.com/article/10.1007/s10462-023-10502-7)
2. Formalize 16-cell equivariance constraints mathematically
3. Implement equivariant message passing (if not already done)
4. Test: Does equivariance improve stability metrics?

**Output**: Code + Document `EQUIVARIANT_16CELL_GNN.py`
- Formal symmetry group definition
- Equivariance-preserving message passing
- Validation: Test equivariance property holds

---

## Mirrorfield Differentiators (Critical Questions)

While reading papers, constantly ask:

### 1. Observation → Constraint

**They measure**: Attractor compactness (Q)
**Can we constrain**: Design polytope cells to enforce compactness?

**They measure**: Continuity (C)
**Can we constrain**: Limit allowable flow velocities via adjacency weights?

**They measure**: Persistence (P)
**Can we constrain**: Fix topology via polytope structure?

### 2. Detection → Prevention

**They detect**: Toxic persona latent at 5% activation
**Can we prevent**: Make toxic region geometrically unreachable?

**They detect**: Sycophancy as stable equilibrium
**Can we prevent**: Design geometry with no false-belief attractors?

**They detect**: Emergent misalignment after fine-tuning
**Can we prevent**: Geometry constraints survive fine-tuning?

### 3. Generic → Designed

**They use**: Generic high-D embeddings (no structure)
**We design**: 16-cell polytope with intentional symmetries

**They observe**: Emergent manifold structure
**We impose**: Explicit geometric priors

**They hope**: Good behavior emerges
**We enforce**: Good behavior is geometrically necessary

---

## Key Gaps to Investigate

### 1. Why Low Intrinsic Dimensionality?

**Our finding**: OpenAI embeddings have LID ≈ 4 (not 1536!)
**Question**: Why 4D specifically? Is this universal?

**Hypothesis**: Semantic space naturally has ~4 fundamental degrees of freedom
- Valence (positive/negative)
- Arousal (active/passive)
- Concreteness (abstract/concrete)
- ? (fourth dimension unclear)

**Test**: Measure LID across multiple embedders and domains
**Prediction**: 4D is universal for language embeddings

---

### 2. Optimal LID for Geometry?

**Our finding**: LID=4 → good geometry signal, LID=130+ → poor
**Question**: Is there an optimal LID range?

**Hypothesis**:
- Too low (LID < 3): Underfits complexity
- Optimal (LID = 3-10): Matches polytope structure
- Too high (LID > 50): Curse of dimensionality

**Test**: Synthetic data with controlled LID, test polytope performance
**Prediction**: 16-cell optimal for LID ≈ 4, 120-cell for LID ≈ 10-20

---

### 3. Can Geometry Survive Fine-Tuning?

**OpenAI finding**: 5% poisoning → emergent misalignment
**Question**: Do geometric constraints prevent this?

**Hypothesis**: 16-cell adjacency constraints limit how much fine-tuning can "drift"
- Toxic attractors require escaping polytope structure
- If geometry is frozen, drift is bounded

**Critical test**:
1. Train model with 16-cell constraints
2. Fine-tune on 5% malicious data (OpenAI protocol)
3. Measure: Does toxic persona feature activate?
4. Prediction: Geometry-constrained model resists poisoning

**This is THE validation experiment for Mirrorfield.**

---

### 4. Biological Plausibility

**Chaudhuri finding**: Brain uses 1D ring attractor for head direction
**Question**: Are polytope attractors biologically plausible?

**Hypothesis**: Brain uses **simplest polytope for each task**
- Head direction: 1D ring (S¹)
- Spatial navigation: 2D grid cells (torus, T²)
- Semantic concepts: 4D polytope (16-cell?)

**Evidence needed**:
- Neuroscience studies measuring semantic space dimensionality
- Compare to our LID=4 finding in language embeddings
- Look for "semantic polytopes" in fMRI data

**Prediction**: Semantic processing occurs on low-D manifold in brain too

---

## Strategic Reading Order (Prioritized)

### Must-Read (This Month)

1. **DMET paper** ([arxiv:2505.20340](https://arxiv.org/abs/2505.20340)) - Foundation for everything
2. **OpenAI toxic persona** ([arxiv:2506.19823](https://arxiv.org/abs/2506.19823)) - Validates Dark River hypothesis
3. **Neural ODEs** ([arxiv:1806.07366](https://arxiv.org/abs/1806.07366)) - Theoretical foundation

### High Priority (Next Month)

4. **Transformer Circuits** ([website](https://transformer-circuits.pub/2021/framework/index.html)) - Mechanistic understanding
5. **Chaudhuri attractor paper** ([Nature 2019](https://www.nature.com/articles/s41593-019-0460-x)) - Biological validation
6. **Geometric Deep Learning** ([arxiv:2105.13926](https://arxiv.org/abs/2105.13926)) - Implementation guide

### Medium Priority (When Time Permits)

7. **AI Deception Survey** ([arxiv:2308.14752](https://arxiv.org/abs/2308.14752)) - Broader context
8. **Persistent Homology Survey** ([arxiv:2312.05840](https://arxiv.org/abs/2312.05840)) - Topological methods

---

## Expected Insights by Research Area

### From DMET

**Expected insights**:
- Formulas for C, Q, P metrics → implement in Mirrorfield
- Validation that attractors exist in LLMs → we can design them
- Empirical evidence that geometry affects generation quality

**Critical questions**:
- Can we pre-specify attractors (not just observe them)?
- Do designed attractors outperform emergent ones?
- Can we prove stability guarantees using DMET framework?

---

### From OpenAI Toxic Persona

**Expected insights**:
- Toxic behavior = learnable latent feature (not just surface statistics)
- 5% threshold = boundary proximity in latent space?
- SAE features = basis for semantic manifold

**Critical questions**:
- Is toxic persona feature on manifold boundary?
- Can 16-cell structure make toxic region unreachable?
- Does geometry prevent the 5% threshold entirely?

**KEY EXPERIMENT**: Replicate OpenAI poisoning experiment with Mirrorfield constraints

---

### From Neural ODEs / Transformer Circuits

**Expected insights**:
- Residual connections = Euler steps → continuous flow formulation
- Residual stream = trajectory we can constrain
- Information flow patterns → where to apply geometry

**Critical questions**:
- Can we write 16-cell constraints as ODE boundary conditions?
- Which layers should have geometric constraints (all or subset)?
- Does flow perspective reveal new constraint mechanisms?

---

### From Geometric Deep Learning

**Expected insights**:
- Equivariance = symmetry preservation (already doing this!)
- Principal bundles = formal framework for our approach
- Gauge theory = mathematical language for geometry constraints

**Critical questions**:
- Is 16-cell a proper gauge group? (probably not, but close)
- Can we formulate message passing as parallel transport?
- Does equivariance guarantee some safety properties?

---

## Connection to Phase E Results

### Your 16-cell Finding

**Result**: 16-cell beats all other polytopes (ΔR² = +0.72 vs baseline)

**DMET interpretation**:
- 16 cells → optimal **attractor compactness (Q)**
- Tetrahedral cells → smooth **continuity (C)**
- Fixed topology → stable **persistence (P)**

**Hypothesis**: 16-cell is optimal because it matches LID ≈ 4
- 16 = 2⁴ (natural discretization of 4D space)
- Tetrahedral cells = simplest 3D building blocks
- 8 neighbors = enough connectivity without over-constraint

---

### Your LID=4 Finding

**Result**: OpenAI embeddings lie on ~4D manifold (not 1536D!)

**Implications**:
1. **Semantic space is inherently low-dimensional**
2. **4D polytopes are not arbitrary** - they match natural dimensionality
3. **600-cell failed because 600 >> 2⁴** - too many cells for 4D

**Connection to neuroscience**:
- Chaudhuri: Brain uses 1D ring for head direction
- Mirrorfield: Language needs 4D polytope for semantics
- Pattern: **Cognitive tasks use minimal sufficient dimensionality**

---

## Next Steps After Reading

### Immediate (Post-Reading)

1. **Implement DMET metrics in Mirrorfield**
   - Add C, Q, P computation to geometry module
   - Compare to existing ridge/curvature metrics
   - Validate on Phase D data

2. **Write SAE-based Dark River detector**
   - Use sparse autoencoder to find latent features
   - Test: Do toxic features cluster near boundary?
   - Validate Dark River hypothesis empirically

3. **Formulate 16-cell as gauge equivariant network**
   - Define symmetry group formally
   - Prove message passing preserves equivariance
   - Add to geometry module documentation

### Short-term (1-2 Months)

4. **Replicate OpenAI poisoning experiment**
   - Get baseline: 5% malicious data → emergent misalignment
   - Test: Does 16-cell constraint prevent this?
   - Measure: Toxic persona feature activation

5. **Multi-task DMET validation**
   - Test 16-cell on multiple geometric tasks
   - Measure C, Q, P for each
   - Verify: Does 16-cell consistently optimize all three metrics?

6. **Biological validation study**
   - Literature review: Semantic space dimensionality in neuroscience
   - Compare to LID=4 finding
   - Hypothesis: Brain also uses ~4D semantic representation

### Long-term (3-6 Months)

7. **Theoretical stability proof**
   - Use DMET + geometric deep learning frameworks
   - Prove: 16-cell constraints → bounded attractor dynamics
   - Formalize: Safety guarantees from geometry

8. **Production deployment**
   - If experiments validate, deploy 16-cell in real application
   - Monitor DMET metrics (C, Q, P) in production
   - Iterate based on real-world performance

9. **Publication**
   - Write paper: "Geometric Constraints for LLM Safety: From Dark Rivers to 16-Cell Attractors"
   - Combine DMET observation + Mirrorfield constraint approaches
   - Empirical validation on toxic persona prevention

---

## Summary: Research → Mirrorfield Pipeline

```
OBSERVE (Existing Research)
  ↓
DMET: Measures C, Q, P metrics
Toxic Persona: Detects misalignment at 5%
Neural ODEs: Models layers as continuous flow
Geometric DL: Equivariance theory
  ↓
CONSTRAIN (Mirrorfield Contribution)
  ↓
16-cell: Designs attractor structure
Polytope adjacency: Enforces flow constraints
Equivariant message passing: Preserves geometry
Boundary detection: Prevents Dark Rivers
  ↓
VALIDATE (Experiments)
  ↓
DMET metrics: Does 16-cell optimize C, Q, P?
Poisoning resistance: Block 5% threshold?
Multi-task generalization: Does it scale?
Biological plausibility: Match neuroscience?
  ↓
DEPLOY (Production)
  ↓
Monitor geometry stability in real use
Iterate on polytope structure if needed
Publish results + open-source
```

---

## Key Papers Summary Table

| Paper | Authors | Year | Core Contribution | Mirrorfield Connection |
|-------|---------|------|-------------------|------------------------|
| **DMET** | Zhang & Dong | 2025 | C, Q, P metrics for manifold dynamics | We constrain what they observe |
| **Neural ODEs** | Chen et al. | 2018 | ResNets as continuous dynamics | Theoretical foundation for flow |
| **Ring Attractors** | Chaudhuri et al. | 2019 | Biological validation of attractors | 1D ring → 4D polytope analogy |
| **Transformer Circuits** | Elhage et al. (Anthropic) | 2021 | Residual stream flow | Where to apply constraints |
| **Toxic Persona** | OpenAI | 2024 | 5% poisoning → misalignment | Dark River validation |
| **AI Deception** | Park et al. | 2024 | Sycophancy as false equilibrium | Attractor-based safety risks |
| **Geometric DL** | Multiple | 2023 | Equivariance frameworks | Implementation guide |
| **Persistent Homology** | Multiple | 2024 | Topological persistence in NNs | P metric implementation |

---

## Final Thoughts: Mirrorfield's Unique Position

**What makes Mirrorfield different from all this research**:

1. **Proactive vs. Reactive**: We design geometry upfront, not discover it post-hoc
2. **Constraint vs. Detection**: We prevent bad attractors, not just detect them
3. **Explicit vs. Emergent**: We impose 16-cell structure, not hope it emerges
4. **Geometric vs. Statistical**: We use topology, not just high-D statistics

**The research validates**:
- Attractors exist (DMET, Chaudhuri) ✓
- They affect behavior (toxic persona) ✓
- Geometry matters (geometric deep learning) ✓
- Low dimensionality is real (our LID=4 finding) ✓

**Mirrorfield asks the next question**: Can we **engineer** these properties instead of just observing them?

**Answer from Phase E experiments**: Yes - 16-cell outperforms all alternatives.

**Next challenge**: Prove this works for safety (toxic persona resistance), not just geometry metrics.

---

**End of Research Roadmap**

*Compiled: January 5, 2026*
*Based on web search of current literature*
*Total papers identified: 8 core + dozens of related work*
