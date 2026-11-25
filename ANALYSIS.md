# ANALYSIS — novelty check and code audit

*25 Nov 2026. Line numbers refer to this commit (`classification_ModelNet40/` unless noted).*

This repository was rebuilt from scratch from the project handoffs (the earlier `AdaGAM_v2` working copy was not
available to the rebuild), so every audit item below is "the bug that the audit listed → how this code avoids it",
with a pointer to the line that does it and the test that checks it.

## 1. What is (and is not) new

**Claim we make.** AdaGAM is an incremental architectural extension built from known primitives:
PointMLP's geometric affine + FiLM-style modulation + an SE-style pooled bottleneck + ReZero/ControlNet-style zero
initialisation. It is not a new mathematical paradigm. The parts that are ours:

1. **Self-conditioned, zero-init residual FiLM inside PointMLP's `LocalGrouper`.** `α(1+Δα)·x̃ + (β+Δβ)` with `fc2 = 0`
   at init (`models/adagam.py:73`), so the untrained network equals PointMLP bit for bit (smoke test [1]:
   max diff `0.00e+00` for both `pointMLP` and `pointMLPElite`), and `adapt=False` gives a same-weights counterfactual
   (smoke test [5]: output equals the static model exactly).
2. **Corruption-type linear probe** (`probe_corruption.py`): do the modulator's context `g` and hidden `h` encode
   which corruption the input has, without ever being told? Balanced classes, object-grouped split.
3. **Single-pass per-point adaptation saliency** (`saliency_map.py`): stage-1 `|Δα|` spread back to points through
   kNN membership; no backward pass.
4. **ModelNet-C analysis** of the adaptation per corruption and severity (`|Δα|` logged by `eval_modelnet_c.py`).

**Answer to "isn't this just FiLM?"** The primitive is FiLM-like. The contribution is where it sits (the geometric
affine in the local grouper), what conditions it (the neighbourhood's own statistics; no external input), the
identity/counterfactual design, and the corruption analysis. See `docs/RELATED_WORK.md`.

**Known counter-argument.** ModelNet-C found data-dependent kernels (RSCNN, PAConv) *more* corruption-sensitive.
We answer with our own numbers (per-corruption CE vs PointMLP, `|Δα|` per corruption/severity) once the runs exist.
If clean-trained AdaGAM does not beat PointMLP, the honest story is "the adaptive affine needs a training signal",
tested by the `_aug` runs.

**Open:** we have not found prior point-cloud work that self-conditions the grouper's affine, but the search is not
exhaustive.

## 2. Corrections to the mid-term claims

| Mid-term claim | What is true (measured or checked) |
|---|---|
| PointMLP normalises per channel, per patch | Anchor-centred, divided by one σ per sample (`models/pointmlp.py`, `LocalGrouper.forward`). Per-patch kept as ablation `--gam_norm patch` (`models/pointmlp.py:194`). |
| "ModelNet-C has 15 corruptions" | 7 corruptions × 5 severities; the 15-corruption set is ModelNet40-C (a different benchmark). |
| "94.5% → ~55% on ModelNet-C" | Not traceable; not quoted. We measure PointMLP ourselves. |
| "< 2% params with r = 4" | r = 4 gives +2.65% (13.5888M); we use r = 8: +1.35% (13.4167M). |
| "15.8 GFLOPs", "12.6M params" | 31.212 GFLOPs (FLOPs = 2 × MACs) and 13.2385M params, measured on our re-implementation (`audit_overhead.py`). |
| ScanObjectNN zero-shot | Dropped: 40 vs 15 classes, not directly scoreable. |
| Targets 68–72% OA, mCE ≤ 78 | Goals, not results. All results are TBD until the GPU runs. |

## 3. Audit: bug → fix → check

### Models (`models/`)
| Audit item | Fix | Check |
|---|---|---|
| `norm="patch"` ignored the anchor and pooled raw coordinates | Always anchor-centred; the context comes from the PointMLP-normalised tensor; patch mode re-normalises per channel/patch only the modulated tensor (`models/pointmlp.py:194`) | smoke [1] patch parity |
| No output bound | Optional `tanh` bound, `max_scale=1`, `max_shift=2`; `tanh(0)=0` keeps identity (`models/adagam.py:98`) | smoke [1] |
| No min/std options, biased-std ambiguity | `pool_context` with `use_min`, `use_std` (`unbiased=False`) (`models/adagam.py:26`); argument validation in `__init__` | smoke [2] permutation invariance with both on |
| Default `reduction=4` vs r = 8 in docs | Default is 8 everywhere | `audit_overhead.py` numbers |
| FPS called `torch.randint` at every eval forward | `deterministic=not self.training` (`models/pointmlp.py:173`); deterministic FPS starts at 0, like the CUDA op | smoke [2]: two eval forwards bit-identical |
| Slow / un-vectorised FPS fallback | Batched FPS, one loop over picks (`models/pointmlp.py:85`); kNN/normalisation in float32 under AMP (`models/pointmlp.py:170`) | smoke [3] |
| No shape checks | `min_points` (`models/pointmlp.py:337`), checks in `square_distance`, FPS, `knn_point` (`models/pointmlp.py:122`), `LocalGrouper` | smoke [3]: 150-point cloud rejected with a clear error |
| Module flags mutated for stats (not thread-safe) | `forward(..., adapt=, return_aux=)` returns per-stage aux dicts; module flags kept only for compatibility | used by eval, probe, saliency, demo |

### Training (`main.py`, `utils/`)
| Audit item | Fix | Check |
|---|---|---|
| P0 default `--model PointNet` → KeyError | Default `pointMLPElite` (`main.py:39`); choices limited to valid keys (Section 0.0 makes Elite primary) | `main.py --help` |
| P0 resume: `initial_lr` KeyError, epoch repeated, `torch.load` args | Fresh scheduler + `load_state_dict` (`main.py:278`), `start_epoch = epoch + 1` (`main.py:292`), `map_location` + `weights_only=False` (`utils/misc.py:72`), scheduler/scaler saved (`utils/misc.py:52`), atomic save | e2e: killed-then-resumed run continues at the next epoch with a continuous LR schedule |
| Logger fails on resume without file; trailing tab | `utils/logger.py:16` | e2e |
| Forced `DataParallel` | Only if `device_count() > 1` (`main.py:258`) | – |
| `cudnn.benchmark=True` overrides determinism | `set_seed(seed, deterministic)` sets one or the other (`utils/misc.py:45`, `main.py:240`) | – |
| `--aug corrupt` doubled CE and mixed BN statistics | Separate forward passes per view (`main.py:152`), CE averaged (`main.py:155`) | e2e `adagam_aug` 1-epoch run |
| LR logged after `scheduler.step()` | Read before stepping (`main.py:306`) | `log.txt` epoch 0 shows lr 0.1 |
| 6 GB GPU | `--amp`, auto batch halving (`main.py:106`), `--num_workers` (0 on Windows), `--max_batches`, EPOCHS recommendation after epoch 1 (`main.py:210`) | dry runs |

### Data and corruptions
| Audit item | Fix | Check |
|---|---|---|
| P0 POSIX `wget/unzip/mv/rm`, race between runs | `urllib` + `zipfile` + `shutil`, lock file (`data.py:47`, `data.py:96`), runner downloads once first (`run_experiments.py:118`) | concurrent download test (two processes, one download) |
| P1 translate before corrupt | `corrupted = translate(corrupt(raw))`, same translation for both views (`data.py:240`) | smoke [4] |
| `_refill` duplicated points (shrinks kNN radius) | No refill: one corruption + severity per batch keeps batches rectangular with the real point count (`data.py:220`) | smoke [4] |
| `add_local` crashed for N < 500 | Buffer sized by the points added (`corruptions.py:128`) | smoke [4] (300-point cloud) |
| `drop_*` vs `dropout_*` names | Alias map (`corruptions.py:24`); everything is stored with ModelNet-C names | smoke [4] |
| Approximate corruption parameters | Same definitions as the official build script, with explicit seeded RNG (for the demo) | smoke [4] sizes match ModelNet-C |

### Evaluation and summary
| Audit item | Fix | Check |
|---|---|---|
| `|Δα|` batch means unweighted | Weighted by batch size (`eval_modelnet_c.py:74`) | smoke [6] |
| `DGCNN_OA` typed from memory | **Verified** against the official `modelnetc_utils/eval.py` (`eval_modelnet_c.py:32`); README table confirms DGCNN mCE 1.000 / clean 0.926. Per-severity DGCNN numbers are not published there. | – |
| Eval not resumable | Per-split cache invalidated if the checkpoint epoch changes (`eval_modelnet_c.py:100`) | smoke [6] second call reuses the cache |
| Hard-coded seen/unseen; float bolding | Seen set from each run's `args.json` (`summarize.py:31`); bolding on rounded values (`summarize.py:87`) | smoke [6], e2e |
| 4 jobs launched concurrently on 1 GPU | Sequential by default, `PARALLEL=1` optional (`run_experiments.py:131`); `.sh` and `.ps1` wrap the same Python runner | `bash -n`, dry run order |

### Diagnostics
| Audit item | Fix | Check |
|---|---|---|
| `audit_overhead.py` counted static α/β as overhead; could hang | Per-stage count = AdaGAM module only (`audit_overhead.py:83`); `--no_latency`, time cap (`audit_overhead.py:50`) | reproduces 13.2385M/31.212 G and 13.4167M (+1.35%)/31.245 G |
| Probe: 5:1 imbalance, object leakage, 4 stages hard-coded, no static parity | Same objects for every class, round-robin severities (`probe_corruption.py:48`), split by object id (`probe_corruption.py:44`), stages from the model (`probe_corruption.py:69`), parity pooling for static runs (`probe_corruption.py:105`) | e2e with `samples_per_class` 3 and 4 |
| Saliency: unsorted indices crash h5py; docstring said max | `np.sort` (`saliency_map.py:37`); docstring says mean | e2e |
| Plot saliency: no alignment check | Indices and labels checked against the clean file (`plot_saliency.py:69`); titles show label, prediction, and prediction with adaptation off | e2e |
| `dump_features.py`: wrong dim in docstring, overwrites | Dim read from the model; one file per split (`dump_features.py:50`) | e2e |
| `plot_tsne.py`: independent fits, fixed perplexity, no class colours | One fit per model over clean+corrupted (or `--mode joint`), z-scored, `--perplexity`; colour = class, marker = second cue; embedding-independent scores in the title (`plot_tsne.py:42`, `plot_tsne.py:98`) | e2e |

## 4. Still to do (needs a GPU)

All four training runs; ModelNet-C evaluation; GPU latency; probe, saliency and t-SNE on real checkpoints;
final figures. No result in this repository is a measured accuracy yet.
