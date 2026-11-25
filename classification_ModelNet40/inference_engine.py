"""Thread-safe inference for the live demo.

* One forward pass per model per request: the static model once, the AdaGAM model once (with
  adaptation on, or off for the counterfactual). The AdaGAM pass returns logits, per-stage |d_alpha|
  and the stage-1 kNN indices together (return_aux), so saliency needs no extra pass.
* No nn.Module attribute is ever changed per request: the counterfactual is the `adapt=False`
  argument of forward(), so concurrent Gradio requests cannot interfere. Inference runs under a
  lock anyway (one GPU, small model) and results are cached.
* Without checkpoints the engine builds "demo weights" (random, untrained) so the app runs anywhere;
  the app shows a banner saying predictions are meaningless in that mode.
"""
import collections
import hashlib
import json
import os
import threading
import time
from typing import Dict, Optional

import numpy as np
import torch

import models
from data import SHAPE_NAMES
from saliency_map import point_saliency
from utils import load_checkpoint, strip_module_prefix


def _load_run(path: str, device):
    """Accepts a run folder (with args.json) or a .pth file next to an args.json."""
    run_dir = path if os.path.isdir(path) else os.path.dirname(path)
    pth = os.path.join(path, "last_checkpoint.pth") if os.path.isdir(path) else path
    args = {}
    if os.path.exists(os.path.join(run_dir, "args.json")):
        with open(os.path.join(run_dir, "args.json")) as f:
            args = json.load(f)
    ckpt = load_checkpoint(pth, map_location="cpu")
    state = strip_module_prefix(ckpt.get("net", ckpt))
    gam = args.get("gam") or ("adagam" if any(".adagam." in k for k in state) else "static")
    model_name = args.get("model")
    if model_name is None:                                 # infer from the embedding width
        w = state.get("embedding.net.0.weight")
        model_name = "pointMLP" if w is not None and w.shape[0] == 64 else "pointMLPElite"
    net = models.build_model(model_name, gam=gam, reduction=args.get("reduction", 8), use_std=args.get("use_std", False),
                             use_min=args.get("use_min", False), gam_norm=args.get("gam_norm", "pointmlp"),
                             bound=args.get("bound", False))
    net.load_state_dict(state)
    return net.to(device).eval(), {"model": model_name, "gam": gam, "path": pth, "epoch": ckpt.get("epoch")}


class InferenceEngine:
    def __init__(self, adagam_ckpt: Optional[str] = None, base_ckpt: Optional[str] = None,
                 model: str = "pointMLPElite", device: Optional[str] = None, seed: int = 0, cache_size: int = 256):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lock = threading.Lock()
        self.cache: "collections.OrderedDict[str, Dict]" = collections.OrderedDict()
        self.cache_size = cache_size
        self.demo_weights = adagam_ckpt is None
        if adagam_ckpt:
            self.adagam, self.adagam_info = _load_run(adagam_ckpt, self.device)
            if self.adagam_info["gam"] != "adagam":
                raise ValueError(f"{adagam_ckpt} is not an AdaGAM run (gam={self.adagam_info['gam']})")
        else:
            torch.manual_seed(seed)
            self.adagam = models.build_model(model, gam="adagam").to(self.device).eval()
            for m in self.adagam.adagam_modules():        # non-zero so the demo shows *some* adaptation
                torch.nn.init.normal_(m.fc2.weight, std=0.02)
            self.adagam_info = {"model": model, "gam": "adagam", "path": "demo weights (untrained)", "epoch": None}
        if base_ckpt:
            self.base, self.base_info = _load_run(base_ckpt, self.device)
            self.base_is_counterfactual = False
        else:
            # no baseline checkpoint: static model = AdaGAM's own backbone with adaptation removed
            self.base = models.build_model(self.adagam_info["model"], gam="static",
                                           gam_norm=self.adagam.local_grouper_list[0].gam_norm).to(self.device).eval()
            self.base.load_state_dict({k: v for k, v in self.adagam.state_dict().items() if ".adagam." not in k})
            self.base_info = {"model": self.adagam_info["model"], "gam": "static",
                              "path": "AdaGAM weights with adaptation removed", "epoch": self.adagam_info["epoch"]}
            self.base_is_counterfactual = True
        self.min_points = max(self.adagam.min_points, self.base.min_points)
        self.class_names = SHAPE_NAMES

    def describe(self) -> str:
        dev = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        return (f"AdaGAM: {self.adagam_info['model']} · {self.adagam_info['path']} · "
                f"static: {self.base_info['path']} · device: {dev}")

    def _top5(self, logits: torch.Tensor) -> Dict[str, float]:
        p = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        idx = np.argsort(-p)[:5]
        return {self.class_names[i]: float(p[i]) for i in idx}

    @torch.inference_mode()
    def predict(self, points: np.ndarray, adapt: bool = True) -> Dict:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError("points must be a finite [N, 3] array")
        if pts.shape[0] < self.min_points:
            raise ValueError(f"cloud has {pts.shape[0]} points; the model needs at least {self.min_points}")
        key = hashlib.sha1(pts.tobytes()).hexdigest() + f"|{int(adapt)}"
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return {**self.cache[key], "cached": True}
            x = torch.from_numpy(pts).to(self.device).T.unsqueeze(0)     # [1, 3, N]
            t0 = time.perf_counter()
            logits_s = self.base(x)
            t1 = time.perf_counter()
            logits_a, _, aux = self.adagam(x, adapt=adapt, return_aux=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t2 = time.perf_counter()
            sal, _ = point_saliency(aux[0]["dalpha_group"], aux[0]["knn_idx"], pts.shape[0])
            out = {
                "static_top5": self._top5(logits_s),
                "adagam_top5": self._top5(logits_a),
                "saliency": sal[0].cpu().numpy(),
                "stage_dalpha": [float(a["dalpha_group"].mean()) for a in aux],
                "stage_dbeta": [float(a["dbeta_group"].mean()) for a in aux],
                "ms_static": 1000 * (t1 - t0), "ms_adagam": 1000 * (t2 - t1),
                "n_points": int(pts.shape[0]), "adapt": bool(adapt), "cached": False,
            }
            self.cache[key] = out
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return out
