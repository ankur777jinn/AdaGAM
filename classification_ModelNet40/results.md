# ModelNet-C results

OA = overall accuracy (higher is better). mCE / RmCE are normalised to DGCNN = 1.000 (lower is better).
Per-corruption columns are OA averaged over the 5 severities. Checkpoint: the one named in `which` (use `last`; `best` is picked on the test set).

| Run | Model | GAM | Train | Ckpt | Clean OA | mCE | RmCE | Scale | Jitter | Rotate | Drop-G | Drop-L | Add-G | Add-L | Seen OA | Unseen OA | mCE vs base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pointMLPElite-base-42 | pointMLPElite | static | clean | last@39 | 0.918 | **1.283** | **1.351** | 0.896 | **0.651** | **0.794** | 0.675 | 0.708 | 0.484 | **0.630** | **0.676** | **0.711** (ref split) | – |
| pointMLPElite-adagam-42 | pointMLPElite | adagam | clean | last@39 | **0.920** | 1.307 | 1.391 | **0.899** | 0.635 | 0.783 | **0.676** | **0.716** | **0.485** | 0.592 | 0.674 | 0.697 (ref split) | 1.020 |

**Headline (protocol-compliant):** clean-trained `base` vs clean-trained `adagam`.

⚠ Runs marked *clean+corrupt* were trained with ModelNet-C-style corruptions, which the ModelNet-C protocol forbids for leaderboard numbers. They are **not leaderboard-comparable**; for them only the *Unseen OA* column (corruptions never seen in training) is a generalisation result.

Seen / unseen sets per run:
- `pointMLPElite-base-42`: seen = jitter, dropout_global, add_global, scale; unseen = rotate, dropout_local, add_local (reference split; this run trained on clean data only)
- `pointMLPElite-adagam-42`: seen = jitter, dropout_global, add_global, scale; unseen = rotate, dropout_local, add_local (reference split; this run trained on clean data only)

One seed per run: treat OA differences below ~0.5% as within run-to-run noise.
