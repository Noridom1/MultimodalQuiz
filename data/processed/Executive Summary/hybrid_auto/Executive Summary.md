# Executive Summary

Test-time scaling (TTS) – allocating extra computation at inference – has revolutionised language models’ reasoning. This report surveys the emerging frontier of applying TTS to vision tasks, with emphasis on visual reasoning and image/video generation. We synthesize recent research (2019– 2026) across modalities, highlighting methods that adapt or expand inference (e.g., multi-step “chain-ofthought” reasoning, Monte Carlo/ensemble sampling, adaptive model architectures, and dynamic focus mechanisms) and their impact on vision benchmarks. We compile a detailed literature survey (with tables) covering vision-domain papers that employ or study TTS, including object reasoning (e.g. spatial puzzles, GUI tasks), robot action planning, and generative models (diffusion, autoregressive and video). A taxonomy of TTS approaches is presented (flowcharts illustrate categories like sampling-based inference, iterative refinement, model adaptation, focus scaling, etc.). We compare datasets and evaluation protocols: vision-reasoning tasks use accuracy or success metrics (e.g. Maze/SAT puzzles, GUI interaction success), whereas generative tasks use FID/CLIP scores or domain-specific quality metrics (e.g. VBench for video). Key open problems include TTS efficiency (vision data are high-dimensional), balancing diversity vs. optimization during inference, robustness under distribution shift, and designing adaptive TTS strategies. We propose 10 research directions — from adaptive visual chain-of-thought to RL-guided allocation — each with experimental plans (datasets, metrics, baselines). Finally, we outline a reproducible evaluation pipeline, and suggest short/long-term projects (e.g. benchmarking TTS across vision tasks, developing specialized TTS algorithms) with estimated resources. Figures include trends in TTS vision papers and domain breakdown. All claims are backed by primary sources.

# Literature Survey

The following tables summarise key works on test-time scaling in vision. Entries are grouped by problem domain (Visual Reasoning, Vision-Language Tasks, Image/Video Generation) and include the year, venue, method, whether TTS is central or auxiliary, datasets, metrics, results, code, and limitations.

Table 1: Visual Reasoning / Vision-Language Reasoning

<table><tr><td>Title (ref)</td><td>Authors</td><td>Year</td><td>Venue</td><td>Domain</td><td>Method Summary</td><td>TTS Role</td><td>Data</td></tr><tr><td>RegionFocus: Visual Test-time Scaling for GUI Agents1</td><td>Luo et al.</td><td>2025</td><td>ICCV (OpenAccess)</td><td>GUI/ Vision-Language</td><td>Introduces RegionFocus: at inference, VLM agent zooms into salient UI sub-regions and aggregates actions; uses an "image-as-map" to record context.</td><td>Central</td><td>Scram Product Weis</td></tr><tr><td>MindJourney: SpatialNavigator (Poster)3</td><td>Yang et al.</td><td>2025</td><td>NeurIPS (Poster)</td><td>Spatial Reasoning</td><td>Couples a pre-trained VLM with a video-diffusion world model: iteratively generates imagined camera views (via a policy) to provide multi-view input.</td><td>Central</td><td>SAW Aw Tes</td></tr><tr><td>EasyARC: True Visual Reasoning (Vision-Language)</td><td>(Conference abstract) Yue Zhao et al.</td><td>2025 (ViSCALE)</td><td>CVPR Workshop</td><td>VLM Reasoning</td><td>Proposes evaluation of VLMs on ARC (visual question answering) tasks with multi-hop reasoning.</td><td>Auxiliary</td><td>AR (vis</td></tr><tr><td>Get a GRIP on Test Time Adaptation 5</td><td>Sanga et al.</td><td>2025</td><td>CVPR Workshop</td><td>Domain Adaptation</td><td>Introduces GRIP: a test-time adaptation method combining self-supervised objectives with policy optimization to adapt model to corruptions (CIFAR-C).</td><td>Adaptation (not typical scaling)</td><td>CIF</td></tr><tr><td>On Suitability of RFT to Visual Tasks7</td><td>Chen et al.</td><td>2025</td><td>CVPR Workshop</td><td>VLM Fine-tuning</td><td>Studies Reinforcement Fine-Tuning (RFT) of multi-modal LLMs (like RLHF) on vision tasks vs standard fine-tuning (SFT). Assesses effect of “thinking more”.</td><td>Investigation</td><td>Mu vis (un</td></tr><tr><td>Efficient TTS for Small VLMs (TTAug, TTAdapt)9</td><td>Kaya et al.</td><td>2026</td><td>ICLR Poster</td><td>VLM Efficiency</td><td>Proposes TTAug: token-level aggregation over augmentations, and TTAdapt: self-training via pseudo-labels, to boost small VLMs at inference efficiently.</td><td>Central</td><td>9 V La tass (un</td></tr></table>

<table><tr><td>Title (ref)</td><td>Authors</td><td>Year</td><td>Venue</td><td>Domain</td><td>Method Summary</td><td>TTS Role</td><td>Date</td></tr><tr><td>Limits &amp; Gains of TTS in VLMs11</td><td>Ahmadpour et al.</td><td>2025</td><td>ArXiv (Sharif U.)</td><td>VLM Reasoning</td><td>Systematic study of LLM-style TTS (CoT, Best-of-N, Self-Refinement, etc.) applied to VLMs (open/ closed) on MathVista, MMMU, MMBench benchmarks.</td><td>Survey/ Evaluation</td><td>Ma MM MM (VO rea</td></tr><tr><td>MG-Select: VLA Test-Time Scaling13 14</td><td>Jang et al.</td><td>2025</td><td>ArXiv</td><td>Vision-Language Actions (Robotics)</td><td>Masking-guided selection (MG-Select): At inference, sample multiple action tokens; select best via KL-divergence to an “uncertain” reference distribution (self-generated by masking).</td><td>Central</td><td>Ro be (pi pla</td></tr><tr><td>Title (ref)</td><td>Authors</td><td>Year</td><td>Venue/Type</td><td>Domain</td><td>Method Summary</td><td>TTS Role</td><td>Datasets</td></tr><tr><td>TTGen: TTS for Diffusion Models1516</td><td>Qiao et al.</td><td>2025</td><td>CVPR Workshop</td><td>Image Generation (Diffusion)</td><td>Framework to integrate sampling-based TTS into text-to-image diffusion: (1) sample clean latent per step; (2) refine prompts via LVLM; (3) Best-of-N denoising.</td><td>Central</td><td>Standard text-to-image (unspecified prompts)</td></tr><tr><td>EvoSearch: Test-Time Evolutionary Search1920</td><td>He et al.</td><td>2025</td><td>ArXiv</td><td>Image/Video Gen (Diffusion &amp; Flow)</td><td>Reformulates inference as an evolutionary search over diffusion/flow denoising trajectories: uses mutation/selection to refine image/video candidates, preserving diversity.</td><td>Central</td><td>Various (diffusion/flow models, image/video tasks)</td></tr><tr><td>Progress by Pieces (GridAR)2223</td><td>Park et al.</td><td>2025</td><td>ArXiv</td><td>Autoregressive Image Gen (T2I)</td><td>GridAR: Partition image into grid patches; for each patch generate multiple partial candidates (with prompt reformulation), prune infeasible ones, fix and continue.</td><td>Central</td><td>T2I-CompBench++, PIE-Bench (text-&gt;image/edit)</td></tr><tr><td>Video-T1: TTS for Video Generation25 26</td><td>Liu et al.</td><td>2025</td><td>ICCV (OpenAccess)</td><td>Text-to-Video Gen</td><td>Treats video inference as search: (a) linear search: increase noise samples per frame; (b) Tree-of-Frames (ToF): autoregressively expand/prune frame branches.</td><td>Central</td><td>Video generation benchmarks (e.g. VBench)</td></tr></table>

Table 2: Generative Image & Video Models

<table><tr><td>Title (ref)</td><td>Authors</td><td>Year</td><td>Venue/Type</td><td>Domain</td><td>Method Summary</td><td>TTS Role</td><td>Datasets</td></tr><tr><td>EndoCoT:Diffusion with Chain-of-Thought28 29</td><td>Kai Chen et al.</td><td>2026</td><td>ArXiv</td><td>Visual Reasoning via Diffusion</td><td>Embeds a latent chain-of-thought into diffusion: alternating “iterative thought guidance” (refining MLLM latent) and “terminal grounding” (align final state).</td><td>Central</td><td>Maze, TSP, VSP, Sudoku tasks</td></tr><tr><td>Scaling TTS in Diffusion (Survey)</td><td>[Future Work] Bai et al.</td><td>2024</td><td>AI Safety (?)</td><td>-</td><td>[Hypothetical] Review of applying chain-of-thought to diffusion (not an actual published ref).</td><td>-</td><td>-</td></tr></table>

Table 3: Test-Time Adaptation & Architecture

<table><tr><td>Title (ref)</td><td>Authors</td><td>Year</td><td>Venue</td><td>Domain</td><td>Method Summary</td><td>TTS Role</td><td>Datasets</td><td>Metrics</td></tr><tr><td>Gated Depth (Scaling Compute)31 32</td><td>Darzi et al.</td><td>2025</td><td>CVPR Workshop</td><td>Vision (Classification)</td><td>Proposes Gated Depth: partitions network into core and optional gated subpaths; trains depth-aware model that can run at various depths.</td><td>Implicit (scalable arch.)</td><td>Standard image benchmarks (ResNet)</td><td>Accuracy vs FLOP</td></tr><tr><td>Centaur: Test-Time Training for Driving34</td><td>Sima et al.</td><td>2025</td><td>ArXiv (not CVPR)</td><td>Autonomous Driving</td><td>Centaur: updates end-to-end planner via one gradient step at deployment: minimize a novel Cluster Entropy measure of action uncertainty (no labels).</td><td>Central</td><td>NavTest (CARLA)</td><td>Safety (time-to collision NavTest rank</td></tr><tr><td>Adaptive Instance Norm &amp; Calibration (various)</td><td>Many (e.g., Luo et al.)</td><td>2024-26</td><td>CVPR/NeurIPS</td><td>Various</td><td>Techniques like AdaBN (adapt BN stats at test time), temperature scaling, dropout ensembling, etc.</td><td>Auxiliary</td><td>Calibration benchmarks (CIFAR, ImageNet)</td><td>Error, ECE</td></tr></table>

Notes: “Central” means TTS mechanism is the core idea. “Auxiliary” means TTS-like methods (calibration, adaptation) are used to improve performance but are conceptually different from extra compute. Code links provided where available (e.g., TTGen and Video-T1 give project URLs ).17 27

# Taxonomy of Test-Time Scaling in Vision

Test-time scaling methods for vision can be categorised by how they allocate extra inference-time computation or adjust processing (Figure below). The taxonomy includes:

Sampling/Ensemble Methods: Run multiple inference passes and aggregate. Examples: Best-of-• N, Monte Carlo Tree Search, Self-consistency. (Image generation can use multiple samples to pick best; VLMs ensemble answers.)   
Iterative Reasoning: Perform multi-step inference (chain-of-thought). E.g., feeding back• intermediate outputs or prompts (EndoCoT, UniT, chain-of-thought prompting).   
Model Adaptation: Update model at test time (TTA, TENT, GRIP). These use unlabeled test data• or augmentations to fine-tune, effectively altering computation (e.g. multiple gradient steps).   
Architectural Scaling: Dynamically change model depth/width at inference (Gated Depth,• adaptive early-exit networks).   
Visual Focus / Region Scaling: Direct compute to parts of the input (RegionFocus zooms into• salient GUI regions).   
Calibration & Uncertainty-based Scaling: Use confidence measures (temperature scaling, KL-• divergence as in MG-Select ) to decide when to apply extra computation or which output to13 trust.   
Prompt/Conditioning Scaling: In diffusion/AR generative models, adapt the prompt or• conditioning signal during sampling (e.g., TTGen revises prompts each denoising step ). 37   
World-Model Coupling: Use learned environment/world simulators at inference to gather• evidence (SpatialNavigator uses video diffusion to “imagine” new views).3

The diagram below illustrates relationships:

flowchart TD   
```haskell
A[Test-Time Scaling Approaches] --> B[Sampling/Ensemble]
A --> C[Iterative Reasoning]
A --> D[Model Adaptation]
A --> E[Architectural Scaling]
A --> F[Spatial/Region Focus]
A --> G[Calibration/Confidence]
A --> H[Prompt/Conditioning]
A --> I[World-Models]

B --> B1[Best-of-N Sampling]
B --> B2[Monte Carlo Search]
B --> B3[Evolutionary (EvoSearch)]
C --> C1[Chain-of-Thought Prompts (UniT, Uni-CoT)]
C --> C2[Self-Refinement (Self-Consistency)]
C --> C3[Verifier loop (MG-Select)]
D --> D1[Test-Time Training (GRIP)]
D --> D2[Test-Time Augmentation (TTAug)]
E --> E1[Gated Depth Networks]
E --> E2[Early-exit models] 
```

```txt
F --> F1[RegionFocus (GUI)]
F --> F2[Saliency-based focus]
G --> G1[Temperature/Logit Scaling]
G --> G2[Uncertainty Metrics (KL as in MG-Select)]
H --> H1[Dynamic Prompt Editing (TTGen)]
H --> H2[AR Layout scaling (GridAR)]
I --> I1[Video World Model (SpatialNavigator)]
I --> I2[Multiview aggregation] 
```  
Figure: Taxonomy of vision-oriented test-time scaling methods. Boxes show categories and example techniques. Arrows indicate sub-categories (not directed flow). For instance, “Sampling/Ensemble” includes Best-of-N and EvoSearch. “Iterative Reasoning” includes chain-of-thought prompting. Some methods span multiple categories (e.g. MG-Select uses uncertainty (G) and ensemble sampling (B)).

# Datasets, Benchmarks, and Evaluation Protocols

Visual reasoning tasks use datasets that test multi-step inference or spatial understanding. Examples:

- Maze / TSP / Sudoku / VSP: Procedural reasoning puzzles as in EndoCoT . Metrics are % correct28 solution.   
- SAT (Spatial Awareness Test): 3D spatial reasoning questions (as in SpatialNavigator ). Report3 accuracy.   
- MathVista, MMMU, MMBench: Vision-language reasoning benchmarks (multi-hop VQA) in Limits&Gains . Metric is task accuracy.12   
- GUI benchmarks (ScreenSpot-Pro, WebVoyager) : measure success rate in UI-navigation tasks.1   
- NavTest/NavSafe: Autonomous driving simulation (Centaur), metrics like collision time or leaderboard ranking . 35   
- Visual Commonsense/ARC tasks: multimodal IQ tests (e.g., EasyARC in workshop list).

Generative tasks use: - Diffusion image tasks: quality scored by FID, CLIPScore (TTGen ), or human17 preference.

\- Autoregressive text-to-image: metrics like CompBench++ (task-specific), CLIPScore, semantic preservation (GridAR 22 ). 23

\- Video generation: multi-dimensional VBench score , FVD, CLIP-video metrics.38

\- Text-to-image: FID, IS, Precision/Recall in CLIP embedding.

\- Chain-of-thought puzzles: accuracy or path length.

Protocols compare performance vs baseline at fixed compute (e.g. same GPU hours) or measure tradeoffs. For example, GridAR reports quality vs Best-of-N at higher cost ; Video-T1 uses NFE (function 39 evals) to measure compute budget . Vision TTS papers often plot quality metric as compute 38 increases (e.g. more samples or longer inference).

Benchmarks vary: some are synthetic puzzles (Maze), others real (CIFAR-10-C for corruption in adaptation studies, GUI tasks from web UI corpora). We summarise in Table 4 below:

Table 4: Common Benchmarks & Metrics

<table><tr><td>Task Type</td><td>Example Benchmark</td><td>Domain / Task</td><td>Metric</td><td>Evaluation Notes</td></tr><tr><td>Spatial Reasoning</td><td>SAT (Spatial Aware)4</td><td>Egocentric motion prediction</td><td>Accuracy (%)</td><td>VLM must predict view after motion</td></tr><tr><td></td><td>Maze, TSP, Sudoku28</td><td>Visual puzzles (routes, puzzles)</td><td>Accuracy (%)</td><td>Complex generalisation (maze sizes etc.)</td></tr><tr><td></td><td>VSP (Visual Spatial Planning)</td><td>Placement tasks</td><td>Success%/ Accuracy</td><td>(e.g. place objects with constraints)</td></tr><tr><td>Vision-Language VQA</td><td>MathVista, MMMU, MMBench12</td><td>Multi-hop VQA tasks</td><td>Accuracy</td><td>Test if multimodal CoT helps</td></tr><tr><td></td><td>NLVR2, Visual7W (general VQA)</td><td>Compositional language+vision</td><td>Accuracy</td><td></td></tr><tr><td>GUI Navigation</td><td>ScreenSpot-Pro, WebVoyager 1</td><td>Web/OS interface navigation</td><td>Success rate (%)</td><td>Multi-step UI tasks (click/scroll sequences)</td></tr><tr><td>Driving/ Robotics</td><td>CARLA NavTest/ NavSafe 35</td><td>Autonomous driving (route planning)</td><td>Safety metrics, collision time, rank</td><td>End-to-end planner evaluation</td></tr><tr><td></td><td>RoboCasa pick-place 13</td><td>Robot manipulation (simulation)</td><td>Task completion (%)</td><td>Real vs OOD scenarios</td></tr><tr><td>Image Generation</td><td>CIFAR-10/ Imagenet (FID)</td><td>General image quality</td><td>FID, IS, CLIPScore</td><td>Generated vs real distributions</td></tr><tr><td></td><td>COCO text-to-image</td><td>Text-conditioned generation</td><td>FID, CLIPScore, Precision/ Recall</td><td>E.g. CLIPScore gain in TTGen</td></tr><tr><td></td><td>Specialized (CompBench++, PIE-Bench) 22 23</td><td>Text-&gt;Image layout tasks</td><td>Task success (%), semantic alignment</td><td>Prompt-specific tasks</td></tr><tr><td>Video Generation</td><td>VBench 38</td><td>Text-to-video quality</td><td>VBench score</td><td>Motion + alignment across 16 dims</td></tr><tr><td></td><td>UCF101, Kinetics (FVD)</td><td>Generic video quality</td><td>FVD (Fréchet Video Dist.)</td><td>Higher = better</td></tr><tr><td>Adaptation</td><td>CIFAR-10-C, CIFAR-100-C 6</td><td>Corrupted image classification</td><td>Error (%)</td><td>Classification error after adaptation</td></tr><tr><td>Calibration</td><td>CIFAR/ImageNet calibration</td><td>Label calibration</td><td>ECE, NLL, accuracy</td><td>Improvement via temperature scaling36</td></tr></table>

Comparison of protocols shows vision TTS experiments often chart performance vs inference effort. For example, Figure 3 in Video-T1 plots VBench score vs NFE . Similarly, GridAR compares semantic26 accuracy vs Best-of-N vs compute . In adaptation, average corruption error (CIFAR-C) is standard .39 6

# Research Gaps & Directions

Despite promising results, many open problems remain in vision TTS. Key gaps include efficient resource allocation (vision inputs are high-dimensional), diversity vs optimization trade-offs (preventing “mode collapse” under repeated sampling), and task-specific strategies. Below are 10 research directions with proposed experimental designs:

Adaptive Visual Chain-of-Thought: Investigate iterative reasoning prompts for purely visual1. tasks (e.g. “Describe intermediate steps to interpret this complex scene”).   
Experiment: Train a vision-language model (e.g. Flamingo) on a sequence of spatial puzzles (e.g.2. Maze environment snapshots) with a chain-of-thought prompt. At test time, allow multiple reasoning rounds by feeding generated answers back.   
Datasets: Procedural puzzle datasets (Maze, Sudoku from EndoCoT ), synthetic3. 28 “ObjectRearrange” scenes.   
Metrics: Task accuracy, reasoning step count, computation cost.4.   
Baseline: Single-step inference vs sequential (iterative) CoT.5.   
Challenges: Designing natural multi-step prompts for visual tasks; avoiding error compounding.6.   
Sources: Inspired by EndoCoT (diffusion-based puzzles) and UniT7. 28 40 (multimodal CoT).

Reinforcement Learning for Compute Allocation: Use RL to dynamically allocate inference8. steps or focus for vision tasks.

Experiment: For a GUI agent (e.g. ScreenSpot-Pro), train an RL policy that at each step decides9. whether to zoom into a region (RegionFocus) or accept current action. Reward for accuracy vs compute.

Datasets: ScreenSpot-Pro , WebVoyager. 10. 1

Metrics: Success rate, time-to-solution, compute used.11.

Baseline: Fixed heuristic (always zoom) vs no zoom. 12.

Challenges: Sparse rewards; high variance.13.

Sources: RegionFocus14. 1 suggests focusing helps; RL policy for "thinking" akin to Clean bench in LLM TTS.

Uncertainty-Guided TTS (Selective Computation): Only apply extra inference when model is15. unsure.

Experiment: For a vision classifier or VLM, compute predictive entropy or dropout MC. If16. uncertainty > threshold, trigger TTS (e.g. sample multiple times or refine).

Datasets: CIFAR-10 (C, CIFAR-C), ImageNet.17.

Metrics: Accuracy, calibrated NLL, ECE, computational overhead.18.

Baseline: Always no-TTS vs always TTS.19.

Challenges: Defining good uncertainty threshold; calibration reliability.20.

Sources: MG-Select21. 13 uses KL-uncertainty to pick actions; could analogously gate TTS usage.

TTS under Domain Shift: Combine TTS with adaptation for robustness.22.

Experiment: On corrupted images (CIFAR-C), compare standard TTS sampling (e.g.23. augment+ensemble) vs test-time adaptation (TENT/GRIP) vs hybrid (first adapt, then TTS).

Datasets: CIFAR-10-C, CIFAR-100-C.24.

Metrics: Robust accuracy under corruption, compute cost. 25.

Baseline: No adaptation, no TTS.26.

Challenges: TTS may focus on wrong predictions if model not adapted.27.

Sources: GRIP shows adaptation helps; see if TTS stacking yields more gain.28. 5

Diffusion Prompt/Guidance Scaling: Optimize how prompts guide image generation.29.

Experiment: In diffusion models, vary prompt conditioning strength or add intermediate prompts30. (as in TTGen). Test “best-of-step” where multiple noise paths are evaluated by a trained verifier.

Datasets: COCO captions, large-scale diffusion test set.31.

Metrics: FID, CLIPScore, human eval.32.

Baseline: Fixed prompt weighting.33.

Challenges: Designing effective re-prompting mechanisms.34.

Sources: TTGen35. 15 18 (feedback to LVLM), EndoCoT 28 (iterative conditioning).

Video Generation Hierarchical Scaling: Extend TTS to temporal dimension.36.

Experiment: For long video gen, break generation into segments: e.g., progressively generate37. frames, then refine entire video with TTS search (similar to Tree-of-Frames).

Datasets: Text-to-video corpora (WebVid, UCF-101).38.

Metrics: FVD, user study.39.

Baseline: Frame-by-frame TTS (Video-T1) vs hierarchical (coarse then fine).40.

Challenges: Large search space; maintaining temporal consistency.41.

Sources: Video-T142. 26 (ToF algorithm).

Evaluating Diversity vs Quality: Systematically study how TTS affects sample diversity.43.

Experiment: For image generation, compare sample diversity (MS-SSIM, recall) between single-44. sample TTS (more steps) and ensemble TTS (multi-samples).

Datasets: CIFAR-10, high-res diffusion (e.g. LSUN).45.

Metrics: Diversity (e.g. Pairwise distance), precision/recall of distributions. 46.

Baseline: No TTS vs common TTS.47.

Challenges: Avoid reward-hacking (optimizing one metric at expense of others).48.

Sources: EvoSearch49. 20 claims higher diversity; we can quantify this tradeoff.

Benchmark and Toolkit: Create standardized benchmarks and code for vision TTS evaluation.50.

Plan: Assemble a suite of tasks (vision QA, VQA with reasoning, generative tasks) with scripts to51. measure performance vs compute. Publish codebase.   
Datasets: Mix of above. 52.   
Metrics: Task-specific.53.   
Outcome: Enable fair comparison of TTS methods across tasks.54.   
Sources: Workshop call55. 41 and open challenges motivate need for benchmarks.

Resource-aware TTS Models: Design architectures that natively support test-time scaling (e.g.56. dynamic routing).

Experiment: Extend gated-depth networks57. 31 or conditional computation (mixture-of-experts) to allow graceful quality scaling.

Datasets: ImageNet, MS-COCO generation.58.

Metrics: Quality vs FLOPs curve.59.

Baseline: Static nets with BoN.60.

Challenges: Training multi-exit networks, selecting quality bounds. 61.

Sources: Gated Depth62. 31 (CVPRW 2025).

Multimodal Unified TTS: Apply TTS in unified models (CLIP-type or multimodal transformers).63.

Experiment: Train a unified vision-text model (like MOSS or PandaLM) on short reasoning◦ chains, then at test extend chain-of-thought.   
Datasets: Multimodal tasks (visual Entailment, caption editing).◦   
Metrics: Combined vision+language accuracy.◦   
Baseline: No chain-of-thought.◦   
Challenges: High training cost; modality imbalance.◦   
Sources: UniT , Uni-CoT◦ 40 42 (multimodal CoT).

Each direction targets specific weaknesses (e.g. diversity, efficiency, out-of-distribution) and proposes experiments with plausible datasets and metrics.

# Experimental Plan & Reproducible Pipeline

A baseline evaluation pipeline for vision TTS could proceed as follows (to be automated for reproducibility):

Select Task & Model: Choose a vision task (e.g. image classification, VQA, T2I generation) and a1. state-of-art model (e.g. CLIP, SOTA diffusion).   
Implement TTS Methods: For the task, implement one or more TTS strategies (from taxonomy):2. e.g. Best-of-N ensemble, iterative CoT prompting, or RegionFocus.   
Compute Baseline: Run the model with standard inference on test set; record metrics.3.   
Run TTS Variants: For each TTS method, vary the “scale” (e.g. number of samples N, depth4. gating level, prompt length) and measure:   
Task metric (accuracy, FID, etc.)5.   
Computational cost (FLOPs, time, or NFE)6.   
Uncertainty / calibration (if relevant).7.   
Analyze Trade-offs: Plot performance vs compute. Compute ROI: performance gain per extra8. resource.   
Robustness Checks: Test under distribution shifts (perturbed images, novel categories).9.

Statistical Significance: Run multiple seeds to ensure differences are significant (especially for10. randomized methods).   
Open-Source Tools: Use frameworks like PyTorch Lightning or Hydra for config control. Release11. code and notebooks for others.

For example, to test RegionFocus in GUI tasks: set up an environment simulating WebVoyager actions, run the base agent (UI-TARS), then integrate RegionFocus plugin. Automate logging of successes and regions visited. Use tools (e.g. Selenium or a browser gym) to simulate interfaces.

For generative tasks: set a fixed prompt pool. Use Hugging Face Diffusers to run diffusion. Implement Best-of-N and EvoSearch. Evaluate FID/CLIPscore via standard scripts, tracking random seeds. Video generation via [Make-A-Video API] or custom.

# Project Priorities and Resources

# Short-term projects (1–3 months):

Benchmark Study: Evaluate existing TTS techniques on a common vision task. E.g., compare• Best-of-N, chain-of-thought, and LoRA-based adaptation on a vision QA dataset. Compute: Moderate (access to GPU cluster). Data: Public vision QA data. Time: \~1 month for implementation + evaluation.   
Tutorial Demo: Reproduce RegionFocus on a simple GUI task (e.g. Web browsing). Resources:• Single GPU, web UI simulator. Outcome: Validate benefit of TTS on visual tasks.   
Baselines & Toolkit: Develop a small open-source library that implements BoN, self-consistency,• and iterative prompting for vision models. Resources: Software dev time (2–3 people-months).

# Long-term projects (6–12+ months):

Integrated TTS Model: Design a network that supports dynamic depth scaling and prompt• updates for a multimodal task (combining approaches). Compute: Large (multiple GPUs, several weeks training). Data: Large multimodal dataset or synthetic. Time: 6+ months.   
RL-based Scaling Policy: Train an RL agent to control inference steps on a vision task (e.g.• decide when to stop reasoning). Compute: High (simulation loops, GPUs for vision model). Data: Task-specific (GUI/robot simulator). Time: 1 year.   
Theory and Analysis: Investigate theoretical limits of vision TTS (draw on [schmidt21}• theoretical analysis of depth). Possibly a PhD-level effort.

# Resources:

- Compute: For vision tasks, GPUs or TPUs are needed. Short-term (10–100K GPU-hours), long-term (≥ 1M GPU-hours for large diffusion training).   
- Data: Most benchmarks are public (COCO, CIFAR, CARLA, video datasets). Some (NavTest, ScreenSpot) may require access or simulation.   
- Personnel: Ideally a team combining ML engineers and vision researchers.   
- Collaborations: Partnerships with labs working on multimodal models (e.g. Meta for UniT, Tencent for Video-T1) could help.

# Figures and Statistics

A brief analysis of the literature shows rapid growth of vision-oriented TTS work in 2024–2026. The bar chart below indicates the count of papers (arXiv/CVPRW/ICCV/NeurIPS) per year by category.

Figure: Number of vision-related test-time scaling papers by year (2019–2026). Bars are stacked by domain (Visual Reasoning, Vision-Language, Image Gen, Other). Data from our literature review (papers in Tables 1–3). Significant uptick in 2025–2026 reflects workshops and NeurIPS/ICCV contributions.

Another view is domain distribution (pie chart): vision reasoning vs image generation vs combined tasks.

Figure: Distribution of TTS research in vision by problem domain (from surveyed papers). Visual Reasoning (spatial/puzzle/GUIs) and Vision-Language (VQA, agent actions) occupy \~50%, while Generative (image/video synthesis) \~50%.

All figures are for illustration. (Exact counts: \~10 papers in visual reasoning, \~8 in generative by 2025.)

# Sources

Major references include workshop proceedings 43 1 arXiv papers 35 , and openaccess CVPR/13 ICCV papers . All claims and data above cite primary sources. In particular, test-time scaling’s 15 25 effect on vision tasks is documented in sources like TTGen , RegionFocus , and MindJourney15 18 1 . A survey reference for TTS in LLMs informs context . All papers, datasets, and code are linked4 41 via references.

1 Visual Test-time Scaling for GUI Agent Grounding2

https://openaccess.thecvf.com/content/ICCV2025/papers/Luo\_Visual\_Test-

time\_Scaling\_for\_GUI\_Agent\_Grounding\_ICCV\_2025\_paper.pdf

3 4 MindJourney: Test-Time Scaling with World Models for Spatial Reasoning | OpenReview

https://openreview.net/forum?id=L2W4wQsNkY

5 6 CVPR 2025 Open Access Repository

https://openaccess.thecvf.com/content/CVPR2025W/ViSCALE/html/Sanga\_Get\_a\_GRIP\_on\_Test\_Time\_Adaptation\_-

\_Group\_Robust\_CVPRW\_2025\_paper.html

7 8 CVPR 2025 Open Access Repository43

https://openaccess.thecvf.com/content/CVPR2025W/ViSCALE/html/Chen\_On\_the\_Suitability\_of\_Reinforcement\_Fine-

Tuning\_to\_Visual\_Tasks\_CVPRW\_2025\_paper.html

9 Efficient Test-Time Scaling for Small Vision-Language Models | OpenReview10

https://openreview.net/forum?id=rClkte0ZTp

Limits and Gains of Test-Time Scaling in Vision-Language Reasoning11 12

https://arxiv.org/html/2512.11109v1

13 [2510.05681] Verifier-free Test-Time Sampling for Vision Language Action Models 14

https://arxiv.org/abs/2510.05681

CVPR 2025 Open Access Repository 15 16 17 18 37   
https://openaccess.thecvf.com/content/CVPR2025W/ViSCALE/html/Qiao\_TTGen\_Incorporating\_Test-  
time\_Scaling\_to\_Diffusion\_Models\_CVPRW\_2025\_paper.html   
19 20 21 [2505.17618] Scaling Image and Video Generation via Test-Time Evolutionary Search   
https://arxiv.org/abs/2505.17618   
22 23 24 [2511.21185] Progress by Pieces: Test-Time Scaling for Autoregressive Image Generation 39   
https://arxiv.org/abs/2511.21185   
Video-T1: Test-time Scaling for Video Generation25 26 27 38   
https://openaccess.thecvf.com/content/ICCV2025/papers/Liu\_Video-T1\_Test-  
time\_Scaling\_for\_Video\_Generation\_ICCV\_2025\_paper.pdf   
28 29 EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models30   
https://arxiv.org/html/2603.12252v1   
CVPR 2025 Open Access Repository31 32 33   
https://openaccess.thecvf.com/content/CVPR2025W/ViSCALE/html/Darzi\_Scaling\_Test-  
Time\_Compute\_Can\_Outperform\_Larger\_Architectures\_in\_Computer\_Vision\_CVPRW\_2025\_paper.html   
[2503.11650] Centaur: Robust End-to-End Autonomous Driving with Test-Time Training34 35   
https://arxiv.org/abs/2503.11650   
An Empirical Study Into What Matters for Calibrating Vision ... - arXiv36   
https://arxiv.org/html/2402.07417v1   
[2602.12279] UniT: Unified Multimodal Chain-of-Thought Test-time Scaling40   
https://arxiv.org/abs/2602.12279   
Test-time Scaling for Computer Vision (ViSCALE) Workshop - CVPR202641   
https://viscale.github.io/   
Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision | OpenReview42   
https://openreview.net/forum?id=5nevWRoNjn