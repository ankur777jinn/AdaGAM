"""Train PointMLP / PointMLP+AdaGAM on ModelNet40.

Examples (Windows: replace `python` with the Python310 interpreter path):
    python main.py --model pointMLPElite --gam static --msg base
    python main.py --model pointMLPElite --gam adagam --reduction 8 --msg adagam
    python main.py --model pointMLPElite --gam adagam --aug corrupt --consistency 1.0 --msg adagam_aug
    python main.py --epoch 1 --msg speedtest --gam adagam        # timing -> recommended EPOCHS
    python main.py --msg adagam --gam adagam --resume            # continue an interrupted run

Valid models: pointMLP, pointMLPElite (default; ~0.72M params, the primary model on a 6 GB laptop GPU).
"""
import argparse
import datetime
import json
import math
import os
import platform
import time

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import models
from corruptions import SEEN_DEFAULT, canonical
from data import (DATA_DIR, CorruptCollate, ModelNet40, ModelNet40Pair, download, worker_init_fn)
from utils import AverageMeter, Logger, cal_loss, device_report, load_checkpoint, save_model, set_seed

IS_WINDOWS = platform.system() == "Windows"
MATRIX_COST = {"base": 1, "adagam": 1, "base_aug": 2, "adagam_aug": 2}   # relative cost per epoch


def parse_args(argv=None):
    p = argparse.ArgumentParser("AdaGAM-3D training")
    p.add_argument("--model", default="pointMLPElite", choices=sorted(models.MODELS))
    p.add_argument("--msg", type=str, default=None, help="run name; checkpoint dir = <model>-<msg>-<seed>")
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--batch_size", type=int, default=32, help="halved automatically on OOM (see --no_auto_batch)")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--epoch", type=int, default=100, help="upstream uses 300; time 1 epoch first (see README)")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--min_lr", type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=2e-4)
    p.add_argument("--smoothing", type=float, default=0.2, help="label smoothing (upstream: 0.2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", "--workers", dest="num_workers", type=int, default=0 if IS_WINDOWS else 4)
    p.add_argument("--amp", action="store_true", help="mixed precision (autocast + GradScaler)")
    p.add_argument("--amp_dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--deterministic", action="store_true", help="cudnn deterministic (slower)")
    p.add_argument("--fps", default="auto", choices=["auto", "torch", "cuda"], help="FPS backend")
    p.add_argument("--no_auto_batch", action="store_true")
    p.add_argument("--no_dp", action="store_true", help="never wrap in DataParallel")
    # AdaGAM
    p.add_argument("--gam", default="static", choices=["static", "adagam"])
    p.add_argument("--reduction", type=int, default=8)
    p.add_argument("--use_std", action="store_true")
    p.add_argument("--use_min", action="store_true")
    p.add_argument("--bound", action="store_true", help="tanh-bounded d_alpha / d_beta")
    p.add_argument("--gam_norm", default="pointmlp", choices=["pointmlp", "patch"])
    # corruption-aware training (_aug runs; NOT leaderboard-comparable on ModelNet-C)
    p.add_argument("--aug", default="none", choices=["none", "corrupt"])
    p.add_argument("--train_corruptions", default=",".join(SEEN_DEFAULT),
                   help="comma list; drop_global/drop_local accepted as aliases")
    p.add_argument("--max_severity", type=int, default=4)
    p.add_argument("--consistency", type=float, default=1.0, help="lambda; 0 disables the consistency term")
    p.add_argument("--cons_type", default="feat", choices=["feat", "jsd"])
    # bookkeeping
    p.add_argument("--resume", action="store_true", help="resume from <run>/last_checkpoint.pth if present")
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--max_batches", type=int, default=0, help="debug: cap train/val batches per epoch")
    p.add_argument("--budget_hours", type=float, default=8.0, help="for the EPOCHS recommendation")
    p.add_argument("--download_only", action="store_true")
    p.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    return p.parse_args(argv)


def make_amp(args, device):
    enabled = bool(args.amp) and device.type == "cuda"
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = enabled and dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def ctx():
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    return ctx, scaler


def consistency_loss(kind, logits_c, logits_x, feat_c, feat_x):
    if kind == "feat":
        return (1 - F.cosine_similarity(feat_x.float(), feat_c.detach().float(), dim=-1)).mean()
    p_c = F.softmax(logits_c.float(), dim=1)
    p_x = F.softmax(logits_x.float(), dim=1)
    m = torch.clamp((p_c + p_x) / 2, 1e-7, 1).log()
    return 0.5 * (F.kl_div(m, p_c, reduction="batchmean") + F.kl_div(m, p_x, reduction="batchmean"))


def probe_batch_size(net, args, device, amp_ctx):
    """Halve the batch size until one train step fits in GPU memory. Restores all weights/BN stats."""
    if device.type != "cuda" or args.no_auto_batch:
        return args.batch_size
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    bs = args.batch_size
    views = 2 if args.aug == "corrupt" else 1
    while bs >= 4:
        try:
            net.train()
            x = torch.randn(bs, 3, args.num_points, device=device)
            for _ in range(views):
                with amp_ctx():
                    out = net(x)
                out.float().sum().backward()
            net.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            break
        except torch.cuda.OutOfMemoryError:
            net.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"[mem] batch {bs} does not fit -> trying {bs // 2}")
            bs //= 2
    net.load_state_dict(state)
    torch.cuda.empty_cache()
    if bs < 4:
        raise RuntimeError("even batch 4 does not fit in GPU memory; use --model pointMLPElite and --amp")
    if bs != args.batch_size:
        print(f"[mem] using --batch_size {bs} (requested {args.batch_size})")
    return bs


def train_epoch(net, loader, optimizer, scaler, amp_ctx, args, device):
    net.train()
    loss_m, ce_m, cons_m = AverageMeter(), AverageMeter(), AverageMeter()
    preds, trues = [], []
    for b, batch in enumerate(loader):
        if args.max_batches and b >= args.max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        if args.aug == "corrupt":
            clean, corrupt, label, _, _ = batch
            clean = clean.to(device, non_blocking=True).permute(0, 2, 1)
            corrupt = corrupt.to(device, non_blocking=True).permute(0, 2, 1)
            label = label.to(device, non_blocking=True)
            with amp_ctx():
                # two separate forward passes -> BatchNorm never mixes clean and corrupted statistics
                logits_c, feat_c = net(clean, return_feat=True)
                logits_x, feat_x = net(corrupt, return_feat=True)
                ce = 0.5 * (cal_loss(logits_c, label, args.smoothing) + cal_loss(logits_x, label, args.smoothing))
                cons = (consistency_loss(args.cons_type, logits_c, logits_x, feat_c, feat_x)
                        if args.consistency > 0 else torch.zeros((), device=device))
                loss = ce + args.consistency * cons
            logits = logits_c
        else:
            data, label = batch
            data = data.to(device, non_blocking=True).permute(0, 2, 1)
            label = label.to(device, non_blocking=True)
            with amp_ctx():
                logits = net(data)
                ce = cal_loss(logits, label, args.smoothing)
                cons = torch.zeros((), device=device)
                loss = ce
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at batch {b}; try without --amp or with --bound")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        n = label.size(0)
        loss_m.update(loss.item(), n)
        ce_m.update(ce.item(), n)
        cons_m.update(float(cons.item()), n)
        preds.append(logits.detach().argmax(dim=1).cpu().numpy())
        trues.append(label.cpu().numpy())
    if not preds:
        raise RuntimeError("no training batches: dataset smaller than --batch_size (drop_last=True)")
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return {"loss": loss_m.avg, "ce": ce_m.avg, "cons": cons_m.avg,
            "acc": 100.0 * metrics.accuracy_score(trues, preds),
            "acc_avg": 100.0 * metrics.balanced_accuracy_score(trues, preds), "batches": len(preds)}


@torch.no_grad()
def validate(net, loader, amp_ctx, args, device):
    net.eval()
    loss_m = AverageMeter()
    preds, trues = [], []
    for b, (data, label) in enumerate(loader):
        if args.max_batches and b >= args.max_batches:
            break
        data = data.to(device).permute(0, 2, 1)
        label = label.to(device)
        with amp_ctx():
            logits = net(data)
        loss_m.update(cal_loss(logits, label, args.smoothing).item(), label.size(0))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(label.cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return {"loss": loss_m.avg, "acc": 100.0 * metrics.accuracy_score(trues, preds),
            "acc_avg": 100.0 * metrics.balanced_accuracy_score(trues, preds)}


def print_epoch_recommendation(run_dir, args, t_train, n_batches, full_batches, t_val):
    t_epoch = t_train / max(n_batches, 1) * full_batches + t_val
    unit = t_epoch / (2.0 if args.aug == "corrupt" else 1.0)       # cost of one clean-run epoch
    total_units = sum(MATRIX_COST.values())                       # base + adagam + 2x(base_aug, adagam_aug)
    rec = max(1, min(300, int(args.budget_hours * 3600 / (unit * total_units))))   # 300 = upstream schedule
    lines = [f"[timing] measured epoch time {t_epoch:.1f}s ({args.model}, gam={args.gam}, aug={args.aug}, "
             f"batch {args.batch_size}, amp={args.amp}) -> clean-run epoch ~ {unit:.1f}s",
             f"[timing] 4-run matrix (base, adagam, base_aug x2, adagam_aug x2) = {total_units} clean-epoch units per epoch",
             f"[timing] budget {args.budget_hours:.1f} h -> recommended EPOCHS = {rec}",
             "[timing] EPOCHS | per clean run | per _aug run | whole matrix"]
    for e in sorted({50, 100, 150, 200, 300, rec}):
        lines.append(f"[timing] {e:6d} | {e * unit / 3600:9.2f} h | {2 * e * unit / 3600:9.2f} h | {e * unit * total_units / 3600:9.2f} h")
    print("\n".join(lines))
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump({"epoch_seconds": t_epoch, "clean_epoch_seconds": unit, "recommended_epochs": rec,
                   "budget_hours": args.budget_hours, "batch_size": args.batch_size, "amp": args.amp}, f, indent=2)


def main(argv=None):
    args = parse_args(argv)
    if args.download_only:
        download(args.data_dir)
        return None
    args.train_corruptions = [canonical(c) for c in args.train_corruptions.split(",") if c.strip()]
    if args.msg is None:
        args.msg = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"==> {device_report(device)}")
    if device.type == "cpu":
        print("==> WARNING: training on CPU will be very slow; this is only meant for smoke tests.")
    set_seed(args.seed, args.deterministic)
    models.set_fps_backend(args.fps)
    print(f"==> FPS backend: {'pointnet2_ops (CUDA)' if models.HAS_POINTNET2_OPS and args.fps != 'torch' and device.type == 'cuda' else 'pure PyTorch'}")

    run_dir = os.path.join(args.checkpoint_root, f"{args.model}-{args.msg}-{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last_checkpoint.pth")
    resuming = args.resume and os.path.exists(last_path)

    download(args.data_dir)
    net = models.build_model(args.model, num_classes=40, gam=args.gam, reduction=args.reduction,
                             use_std=args.use_std, use_min=args.use_min, gam_norm=args.gam_norm,
                             bound=args.bound).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"==> {args.model} gam={args.gam} r={args.reduction} params={n_params / 1e6:.4f}M -> {run_dir}")

    amp_ctx, scaler = make_amp(args, device)
    args.batch_size = probe_batch_size(net, args, device, amp_ctx)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and not args.no_dp:
        net = nn.DataParallel(net)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params, "torch": torch.__version__}, f, indent=2)

    workers = max(0, args.num_workers)
    common = dict(num_workers=workers, worker_init_fn=worker_init_fn, pin_memory=device.type == "cuda",
                  persistent_workers=workers > 0)
    if args.aug == "corrupt":
        train_set = ModelNet40Pair(args.num_points, "train", args.data_dir)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  collate_fn=CorruptCollate(args.train_corruptions, args.max_severity), **common)
    else:
        train_set = ModelNet40(args.num_points, "train", args.data_dir)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(ModelNet40(args.num_points, "test", args.data_dir), batch_size=args.eval_batch_size,
                             shuffle=False, drop_last=False, **common)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.min_lr)   # fresh: no 'initial_lr' KeyError
    start_epoch, best_acc, best_avg = 0, 0.0, 0.0
    if resuming:
        ckpt = load_checkpoint(last_path, map_location=device)
        (net.module if hasattr(net, "module") else net).load_state_dict(ckpt["net"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
            if scheduler.T_max != args.epoch:
                print(f"[resume] note: checkpoint schedule T_max={scheduler.T_max}, --epoch={args.epoch}; keeping T_max={args.epoch}")
                scheduler.T_max = args.epoch
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1                       # next epoch, not a repeat
        best_acc = float(ckpt.get("best_test_acc", 0.0))
        best_avg = float(ckpt.get("best_test_acc_avg", 0.0))
        print(f"==> resumed from epoch {ckpt['epoch']} (best acc {best_acc:.2f}); continuing at {start_epoch}")
    elif args.resume:
        print("==> --resume given but no checkpoint found; starting fresh")

    logger = Logger(os.path.join(run_dir, "log.txt"), title="ModelNet40-" + args.model, resume=resuming)
    logger.set_names(["Epoch-Num", "Learning-Rate", "Train-Loss", "Train-acc-B", "Train-acc", "Consistency",
                      "Valid-Loss", "Valid-acc-B", "Valid-acc", "Seconds"])

    full_batches = len(train_loader)
    timed = False
    for epoch in range(start_epoch, args.epoch):
        lr = optimizer.param_groups[0]["lr"]          # logged BEFORE scheduler.step()
        t0 = time.time()
        tr = train_epoch(net, train_loader, optimizer, scaler, amp_ctx, args, device)
        t_train = time.time() - t0
        do_val = (epoch + 1) % args.val_every == 0 or epoch == args.epoch - 1
        t1 = time.time()
        te = validate(net, test_loader, amp_ctx, args, device) if do_val else {"loss": float("nan"), "acc": float("nan"), "acc_avg": float("nan")}
        t_val = time.time() - t1
        scheduler.step()
        is_best = do_val and te["acc"] > best_acc
        if is_best:
            best_acc, best_avg = te["acc"], te["acc_avg"]
        extra = {"best_test_acc_avg": best_avg, "args": vars(args)}
        save_model(last_path, net, epoch, optimizer, scheduler, scaler, best_acc, extra)
        if is_best:
            save_model(os.path.join(run_dir, "best_checkpoint.pth"), net, epoch, None, None, None, best_acc, extra)
        logger.append([epoch, lr, tr["loss"], tr["acc_avg"], tr["acc"], tr["cons"],
                       te["loss"], te["acc_avg"], te["acc"], time.time() - t0])
        print(f"Epoch {epoch + 1}/{args.epoch} lr {lr:.4f} | train loss {tr['loss']:.3f} acc {tr['acc']:.2f} "
              f"cons {tr['cons']:.4f} | test acc {te['acc']:.2f} (best {best_acc:.2f}) | {time.time() - t0:.1f}s")
        if not timed:
            print_epoch_recommendation(run_dir, args, t_train, tr["batches"], full_batches, t_val)
            timed = True
    logger.close()
    print(f"==> done. best test acc (upstream-style, test-set selected) {best_acc:.2f}; "
          f"evaluate the LAST checkpoint: python eval_modelnet_c.py --run {run_dir} --which last")
    return run_dir


if __name__ == "__main__":
    main()
