"""Training and checkpointing helpers."""

import os
import shutil
import sys
import time
import torch
import torch.nn.functional as F

_last_time = time.time()
_begin_time = _last_time


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def cal_loss(pred, gold, smoothing=True):
    """Cross-entropy loss with optional label smoothing (eps=0.2)."""
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        return -(one_hot * log_prb).sum(dim=1).mean()
    return F.cross_entropy(pred, gold, reduction="mean")


def save_model(net, epoch, path, acc, is_best, **kwargs):
    state = {"net": net.state_dict(), "epoch": epoch, "acc": acc, **kwargs}
    filepath = os.path.join(path, "last_checkpoint.pth")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, "best_checkpoint.pth"))


def save_args(args):
    with open(os.path.join(args.checkpoint, "args.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")


def progress_bar(current, total, msg=None):
    global _last_time, _begin_time
    if current == 0:
        _begin_time = time.time()
    cur_time = time.time()
    tot_time = cur_time - _begin_time
    _last_time = cur_time
    bar_len = 30
    filled = int(bar_len * (current + 1) / max(1, total))
    bar = "=" * filled + "." * (bar_len - filled)
    line = f" [{bar}] {current + 1}/{total} | {tot_time:.1f}s"
    if msg:
        line += f" | {msg}"
    sys.stdout.write(line + ("\n" if current >= total - 1 else "\r"))
    sys.stdout.flush()
