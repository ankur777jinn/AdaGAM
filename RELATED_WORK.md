# Related work and where AdaGAM fits

AdaGAM predicts a per-neighbourhood correction (Δα, Δβ) to PointMLP's geometric affine from the neighbourhood's own statistics. The primitive (predict a scale and shift) is well known; what is specific to AdaGAM is where it sits, what conditions it, and the robustness question it is used for.

## Conditional scaling and shifting (the mechanism's family)

| Work | What it does | How AdaGAM differs |
|---|---|---|
| **FiLM** (Perez et al., AAAI 2018) | Predicts a per-channel scale γ and shift β from an external conditioning input (e.g. a question) and applies γ·x + β to feature maps. | AdaGAM's condition is the data itself (self-conditioned), it acts inside PointMLP's normalisation step per local neighbourhood, and it is a residual correction α(1+Δα), β+Δβ initialised to exactly zero. |
| **Conditional BatchNorm** (de Vries et al., NeurIPS 2017) | Predicts changes to BatchNorm's scale and shift from a language embedding. | Same idea of predicting a normalisation's affine parameters; AdaGAM predicts them from local geometry, not another modality, and per group rather than per image. |
| **AdaIN** (Huang & Belongie, ICCV 2017) | Replaces a feature map's mean and std with those of a style input. | AdaGAM keeps PointMLP's statistics and learns a small correction; no second input. |
| **Squeeze-and-Excitation** (Hu et al., CVPR 2018) | Pools a feature map globally and rescales channels with sigmoid gates in [0, 1]. | AdaGAM also shifts, its scale is unbounded (1 + Δα), and it works per local neighbourhood, not per whole input. |
| **Self-modulation** (Chen et al., ICLR 2019) | GAN generator modulates its own normalisation layers from its input latent. | Closest in spirit ("self" conditioning); different domain (point clouds), different input (local geometry) and goal (corruption robustness). |

## Point-cloud backbones

| Work | Relevance |
|---|---|
| **PointNet / PointNet++** (Qi et al., CVPR 2017 / NeurIPS 2017) | Shared MLPs with symmetric pooling; set abstraction with sampling + grouping that PointMLP builds on. |
| **DGCNN** (Wang et al., ACM TOG 2019) | Dynamic graph convolution; the reference model ModelNet-C normalises its metrics to (mCE = 1). |
| **RS-CNN** (Liu et al., CVPR 2019), **PAConv** (Xu et al., CVPR 2021) | Data-dependent / adaptive kernels. ModelNet-C reports such adaptive designs can be *more* corruption-sensitive, a counter-argument AdaGAM must answer with evidence (our jitter result is a concrete instance; corruption training resolves it). |
| **PointMLP** (Ma et al., ICLR 2022) | Our backbone. Its geometric affine module applies a learned but fixed α, β to every normalised neighbourhood. AdaGAM makes that step input-dependent. |

## Robustness benchmarks and methods

| Work | What it does | Relation to our study |
|---|---|---|
| **ModelNet-C** (Ren, Pan & Liu, ICML 2022) | 7 corruptions × 5 severities on ModelNet40, mCE/RmCE normalised to DGCNN; forbids training on similar corruptions; also proposes RPC (an architecture) and WOLFMix (augmentation). | Our benchmark and protocol. Headline results use clean training only; corruption-trained runs are reported separately. |
| **ModelNet40-C** (Sun et al., 2022) | A different benchmark with 15 corruption types. | Not used; often confused with ModelNet-C. |
| **PointWOLF** (Kim et al., ICCV 2021), **RSMix** (Lee et al., CVPR 2021) | Point-cloud data augmentation (local deformations; mixing). WOLFMix combines them. | Augmentation methods change the data; AdaGAM changes the architecture. They could be combined. |
| **PointCAT** (Huang et al., 2022) | Contrastive adversarial training for robust point-cloud recognition; aligns clean and perturbed features. | Our corruption-trained runs use a related clean/corrupt feature-consistency loss. |
| **BN-adapt** (Schneider et al., NeurIPS 2020), **TENT** (Wang et al., ICLR 2021) | Test-time adaptation: re-estimate (BN-adapt) or optimise (TENT) normalisation statistics on unlabelled test batches. | BN-adapt is our upper-bound baseline (mCE 0.707 vs 1.131 for PointMLP, clean-trained). It needs a batch of same-corruption test data; AdaGAM adapts per object with nothing extra at test time. |
| **CloudFixer** (Shim et al., ECCV 2024) | Test-time adaptation for point clouds using a diffusion model to repair the input. | Repairs the input at test time with an extra generative model; AdaGAM adapts the features inside the network in one forward pass. |

## Explainability methods we use

| Work | Use |
|---|---|
| **Point-cloud saliency maps** (Zheng et al., ICCV 2019) | Per-point importance s_i = −(∂L/∂r_i)·r_i²; our saliency analysis and faithfulness (point-dropping) check. |
| **Linear CKA** (Kornblith et al., ICML 2019) | Similarity of clean vs corrupted representations per stage; our feature-drift analysis. |

## Novelty claim (as we state it)

The scale-and-shift primitive is FiLM-like and we say so. Our contribution is (1) making PointMLP's geometric affine self-conditioned on each neighbourhood's own statistics, as a zero-initialised residual that costs +0.11% FLOPs; (2) a protocol-compliant ModelNet-C evaluation; and (3) counterfactual (adaptation on/off) and feature-drift analyses showing exactly when the adaptation helps (outliers) and hurts (jitter, when trained only on clean data), and that a training signal removes the failure.

We have not done an exhaustive literature search for prior point-cloud work that conditions this specific affine step on local statistics. Check before claiming "first".
