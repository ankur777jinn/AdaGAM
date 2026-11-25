"""Loss, seeding and checkpoint helpers."""
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def cal_loss(pred: torch.Tensor, gold: torch.Tensor, smoothing: float = 0.2) -> torch.Tensor:
    """Cross-entropy with label smoothing (as in upstream PointMLP / DGCNN)."""
    gold = gold.contiguous().view(-1)
    if smoothing > 0:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        log_prb = F.log_softmax(pred.float(), dim=1)
        return -(one_hot * log_prb).sum(dim=1).mean()
    return F.cross_entropy(pred.float(), gold)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def save_model(path: str, net, epoch: int, optimizer=None, scheduler=None, scaler=None,
               best_acc: float = 0.0, extra: Optional[dict] = None) -> None:
    net_state = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    ckpt = {
        "epoch": epoch,
        "net": net_state,
        "best_test_acc": best_acc,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if extra:
        ckpt.update(extra)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)          # atomic: an interrupted save never corrupts the checkpoint


def load_checkpoint(path: str, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location=map_location)


def device_report(device: torch.device) -> str:
    if device.type == "cuda":
        i = device.index or 0
        p = torch.cuda.get_device_properties(i)
        return (f"CUDA device {i}: {p.name} | {p.total_memory / 2**30:.1f} GB VRAM | "
                f"sm_{p.major}{p.minor} | torch {torch.__version__} (CUDA {torch.version.cuda})")
    return f"CPU | torch {torch.__version__}"
