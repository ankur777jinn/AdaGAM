# Related work and how AdaGAM differs

One line each: what the work does, and how AdaGAM differs. Venues were checked against the
papers' own pages where possible; anything not re-checked is marked *(verify)*.

| Work | What it does | How AdaGAM differs |
|---|---|---|
| **FiLM** — Perez et al., AAAI 2018 | Predicts per-channel scale γ and shift β from an *external* conditioning input (e.g. a question embedding) and applies γ·x + β to CNN feature maps. | Same primitive (affine modulation). AdaGAM is *self-conditioned*: the condition is the neighbourhood's own pooled statistics, it sits inside PointMLP's geometric affine, and it is a zero-initialised *residual* on the learned α, β (α(1+Δα), β+Δβ). |
| **Conditional BN** — de Vries et al., NeurIPS 2017 | Predicts changes to BatchNorm's γ, β from a language embedding. | Also residual on γ/β, but conditioned externally and applied to BN. AdaGAM keeps PointMLP's normalisation (anchor-centred, one σ per sample) and conditions on the input itself. |
| **AdaIN** — Huang & Belongie, ICCV 2017 | Replaces a content feature's channel mean/std with those of a *style* image. | AdaIN copies another input's statistics; AdaGAM *learns* a correction from the same input's statistics, per neighbourhood, with an identity starting point. |
| **Squeeze-and-Excitation** — Hu et al., CVPR 2018 | Global-pooled features → bottleneck MLP → sigmoid gates in [0,1] that rescale channels. | Same pooled-bottleneck shape. SE only rescales, with bounded gates and no shift; AdaGAM predicts an unbounded (or optionally tanh-bounded) scale *and* a shift, per local neighbourhood rather than per sample. |
| **Self-Modulation** — Chen et al., ICLR 2019 | GAN generator modulates its own BN parameters from the latent code z (no labels needed). | Closest in spirit ("self" conditioning). It conditions on the generator's input noise; AdaGAM conditions on local point-cloud geometry inside a classifier and targets corruption robustness. |
| **PointMLP** — Ma et al., ICLR 2022 | Residual-MLP point-cloud classifier; its Geometric Affine Module normalises each kNN group (anchor-centred, one σ per sample) and applies learned α, β. | AdaGAM is a drop-in change to that module only: α, β become input-dependent. At init the network is *exactly* PointMLP (max diff 0.0, `smoke_test.py`). Overhead +1.35% params / +0.11% FLOPs (pointMLP, r=8). |
| **ModelNet-C** — Ren et al., ICML 2022 | Benchmark: 7 corruptions × 5 severities on the ModelNet40 test set; mCE normalised to DGCNN; forbids training on similar corruptions. | We use it as the evaluation protocol; headline = clean-trained runs only. |
| **RPC** — Ren et al., ICML 2022 (same paper) | A robust classifier assembled from the architectural choices the paper's analysis found most robust *(verify the exact components in the paper)*. | RPC is an architecture search over known blocks; AdaGAM is one adaptive block added to an unchanged backbone, analysed with a counterfactual switch. |
| **WOLFMix** — Ren et al., ICML 2022 (same paper); **PointWOLF** — Kim et al., ICCV 2021 | PointWOLF: locally weighted random transformations as augmentation; WOLFMix: PointWOLF + RSMix mixing, the paper's strongest augmentation. | Data-side robustness with a *static* network. AdaGAM is architecture-side; our `_aug` runs use ModelNet-C-like corruptions with a consistency loss and are reported separately as not leaderboard-comparable. |
| **PointCAT** — Huang et al., arXiv 2209.07788 (2022) *(verify venue)* | Contrastive adversarial training that pulls features of perturbed clouds toward clean ones; uses t-SNE to show the effect. | Our consistency term (1 − cos(f_corrupt, sg(f_clean))) is of this family but uses corruption views, not adversarial ones, and is optional; the architectural change works without it. |
| **RSCNN** — Liu et al., CVPR 2019; **PAConv** — Xu et al., CVPR 2021 | Data-dependent kernels: weights predicted from local geometric relations (RSCNN) or a learned mix of weight banks (PAConv). | ModelNet-C found these *more* corruption-sensitive. AdaGAM only predicts 2·D affine numbers per neighbourhood (not kernel weights), starts at identity, and we report its per-corruption CE and |Δα| to check whether it inherits that sensitivity. |
| **CloudFixer** — Shim et al., ECCV 2024 | Test-time adaptation: a diffusion model "repairs" corrupted inputs by optimisation at test time. | AdaGAM adapts in a single forward pass with no test-time optimisation or extra model. We do not quote latency ratios we have not measured. |
| **TENT** — Wang et al., ICLR 2021 | Test-time entropy minimisation that updates BN affine parameters online. | TENT changes the affine by gradient steps at test time (and needs batches); AdaGAM predicts the affine change per input in one pass. |

**Honest novelty statement.** AdaGAM combines known primitives — PointMLP's geometric affine, FiLM-style
modulation, an SE-style pooled bottleneck, and ReZero/ControlNet-style zero initialisation. It is an
incremental architectural extension, not a new paradigm. What is specifically ours: (1) self-conditioning
the affine *inside* PointMLP's local grouper with exact identity at init and a same-weights counterfactual
switch; (2) the corruption-type linear probe on the modulator's context; (3) the single-pass per-point
adaptation saliency; (4) the ModelNet-C analysis of what the adaptation does per corruption. We have not
found prior point-cloud work that self-conditions the grouper's affine, but this search is not exhaustive
— check again before claiming novelty in public.
