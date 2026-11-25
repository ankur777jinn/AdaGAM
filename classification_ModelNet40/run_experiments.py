"""Cross-platform experiment runner (Windows, Linux, Colab). run_experiments.sh / .ps1 call this.

    python run_experiments.py train  --epochs 100 --gpu 0          # 4 runs, one after another
    python run_experiments.py eval   --epochs 100                   # ModelNet-C eval (last ckpt) + summarize
    python run_experiments.py all    --epochs 100                   # train then eval
    python run_experiments.py train  --dry_run                      # print the commands only

Environment variables give the defaults (so the shell wrappers stay tiny):
    GPU (0), EPOCHS (100), R (8), MODEL (pointMLPElite), SEED (42), BATCH (32), AMP (1),
    PARALLEL (0), WORKERS (0 on Windows, 4 elsewhere), MODELNET_C (data/modelnet_c)

Runs are SEQUENTIAL by default (one ~6 GB GPU). Every training command passes --resume, so
re-running the same command continues interrupted runs instead of starting over.
"""
import argparse
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = platform.system() == "Windows"

RUNS = {  # headline runs first (clean-trained), then the separately-reported _aug runs
    "base": ["--gam", "static"],
    "adagam": ["--gam", "adagam", "--reduction", "{R}"],
    "base_aug": ["--gam", "static", "--aug", "corrupt", "--consistency", "1.0"],
    "adagam_aug": ["--gam", "adagam", "--reduction", "{R}", "--aug", "corrupt", "--consistency", "1.0"],
}
ABLATIONS = {  # optional; enable with --ablations
    "adagam_aug_nocons": ["--gam", "adagam", "--reduction", "{R}", "--aug", "corrupt", "--consistency", "0"],
    "adagam_std": ["--gam", "adagam", "--reduction", "{R}", "--use_std"],
    "adagam_patch": ["--gam", "adagam", "--reduction", "{R}", "--gam_norm", "patch"],
}


def env(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v


def parse(argv=None):
    p = argparse.ArgumentParser("AdaGAM-3D experiment runner")
    p.add_argument("mode", choices=["train", "eval", "all"])
    p.add_argument("--gpu", default=env("GPU", "0"))
    p.add_argument("--epochs", type=int, default=int(env("EPOCHS", "100")))
    p.add_argument("--r", type=int, default=int(env("R", "8")))
    p.add_argument("--model", default=env("MODEL", "pointMLPElite"))
    p.add_argument("--seed", type=int, default=int(env("SEED", "42")))
    p.add_argument("--batch_size", type=int, default=int(env("BATCH", "32")))
    p.add_argument("--amp", type=int, default=int(env("AMP", "1")))
    p.add_argument("--workers", type=int, default=int(env("WORKERS", "0" if IS_WINDOWS else "4")))
    p.add_argument("--parallel", type=int, default=int(env("PARALLEL", "0")), help="1 = launch all at once (not recommended on one GPU)")
    p.add_argument("--runs", default=None, help="comma list of run names (default: the 4 main runs)")
    p.add_argument("--ablations", action="store_true", help="also run the optional ablations")
    p.add_argument("--modelnet_c", default=env("MODELNET_C", os.path.join(HERE, "data", "modelnet_c")))
    p.add_argument("--checkpoint_root", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--data_dir", default=env("DATA_DIR", ""), help="ModelNet40 data folder (default: data/)")
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_batches", type=int, default=0, help="debug: cap batches per epoch (dry runs)")
    p.add_argument("--max_samples", type=int, default=0, help="debug: cap clouds per ModelNet-C file")
    p.add_argument("--dry_run", action="store_true", help="print commands without running them")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args(argv)


def selected_runs(a):
    table = dict(RUNS)
    if a.ablations:
        table.update(ABLATIONS)
    names = a.runs.split(",") if a.runs else list(table)
    unknown = [n for n in names if n not in {**RUNS, **ABLATIONS}]
    if unknown:
        raise SystemExit(f"unknown run(s) {unknown}; valid: {list(RUNS) + list(ABLATIONS)}")
    return [(n, [x.replace("{R}", str(a.r)) for x in {**RUNS, **ABLATIONS}[n]]) for n in names]


def run_dir(a, name):
    return os.path.join(a.checkpoint_root, f"{a.model}-{name}-{a.seed}")


def _env(a):
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    e["PYTHONUNBUFFERED"] = "1"
    return e


def _launch(cmd, log_path, a, wait=True):
    print(("[dry-run] " if a.dry_run else "") + " ".join(cmd) + f"   > {os.path.relpath(log_path, HERE)}")
    if a.dry_run:
        return None
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=HERE, env=_env(a), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    if not wait:
        def drain():
            for line in proc.stdout:
                log.write(line)
            log.close()
        import threading
        threading.Thread(target=drain, daemon=True).start()
        return proc
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
        log.flush()
    proc.wait()
    log.close()
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}); see {log_path}")
    return proc


def train(a):
    # download ModelNet40 ONCE before any run starts (no race between runs)
    data_flag = ["--data_dir", a.data_dir] if a.data_dir else []
    _launch([a.python, "main.py", "--download_only"] + data_flag, os.path.join(HERE, "logs", "download.log"), a)
    procs = []
    for name, flags in selected_runs(a):
        cmd = [a.python, "main.py", "--model", a.model, "--msg", name, "--seed", str(a.seed), "--epoch", str(a.epochs),
               "--batch_size", str(a.batch_size), "--num_workers", str(a.workers), "--resume",
               "--checkpoint_root", a.checkpoint_root] + data_flag + flags
        if a.amp:
            cmd.append("--amp")
        if a.max_batches:
            cmd += ["--max_batches", str(a.max_batches)]
        t0 = time.time()
        p = _launch(cmd, os.path.join(HERE, "logs", f"{a.model}-{name}-{a.seed}.log"), a, wait=not a.parallel)
        if p is not None and a.parallel:
            procs.append((name, p))
        elif not a.dry_run:
            print(f"[runner] {name} finished in {(time.time() - t0) / 3600:.2f} h")
    for name, p in procs:
        p.wait()
        print(f"[runner] {name} exited with {p.returncode}")


def evaluate(a):
    names = [n for n, _ in selected_runs(a)]
    base = run_dir(a, "base")
    ordered = (["base"] if "base" in names else []) + [n for n in names if n != "base"]
    for name in ordered:
        rd = run_dir(a, name)
        if not a.dry_run and not os.path.exists(os.path.join(rd, "last_checkpoint.pth")):
            print(f"[runner] skip {name}: no last_checkpoint.pth in {rd}")
            continue
        cmd = [a.python, "eval_modelnet_c.py", "--run", rd, "--data", a.modelnet_c, "--which", "last",
               "--eval_batch_size", str(a.eval_batch_size)]
        if name != "base" and (a.dry_run or os.path.exists(os.path.join(base, "modelnet_c.json"))):
            cmd += ["--ref", base]
        if a.max_samples:
            cmd += ["--max_samples", str(a.max_samples)]
        _launch(cmd, os.path.join(HERE, "logs", f"eval-{a.model}-{name}-{a.seed}.log"), a)
    _launch([a.python, "summarize.py", "--root", a.checkpoint_root, "--out_md", os.path.join(HERE, "results.md"),
             "--out_csv", os.path.join(HERE, "results.csv")], os.path.join(HERE, "logs", "summarize.log"), a)


def main(argv=None):
    a = parse(argv)
    print(f"[runner] mode={a.mode} model={a.model} epochs={a.epochs} r={a.r} gpu={a.gpu} amp={bool(a.amp)} "
          f"batch={a.batch_size} parallel={bool(a.parallel)} runs={[n for n, _ in selected_runs(a)]}")
    if a.mode in ("train", "all"):
        train(a)
    if a.mode in ("eval", "all"):
        evaluate(a)


if __name__ == "__main__":
    main()
