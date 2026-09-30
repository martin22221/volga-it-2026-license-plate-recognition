#!/usr/bin/env python
"""One entry point for Baseline V1 detector training on Kaggle: smoke test or full run.

Run it straight from the read-only bundle under ``/kaggle/input``::

    python -u /kaggle/input/<bundle>/scripts/kaggle_launch.py --mode smoke --approver "<name>"
    python -u /kaggle/input/<bundle>/scripts/kaggle_launch.py --mode full  --approver "<name>" \\
        --checkpoint-dataset <user>/<slug>

Every stage prints a timestamped line before it starts, every external call has
a timeout, and training output is streamed line by line. Stages:

1. **bundle**   -- the bundle this script sits in: marker, required files, 12,000 images.
2. **machine**  -- Python, CPUs, free disk, ``nvidia-smi -L``.
3. **copy**     -- the bundle to ``--workdir`` (default ``/kaggle/working/baseline_v1``),
   with progress; reused when a complete copy of the same commit is already there.
   A copy of another commit is renamed aside, never deleted (it may hold a run).
4. **packages** -- torch 2.14.0 / torchvision 0.29.0 / numpy 2.5.3 / pillow 11.3.0, as
   pinned in requirements-train.txt; installed (CUDA build) only when they differ.
5. **cuda**     -- a real forward and backward pass on the GPU (GPU 0 only).
6. **weights**  -- the COCO-pretrained detector weights, fetched once with a timeout.
7. **sync**     -- optional: Kaggle token from notebook Secrets, ``kaggle_persist_run.py check``.
   Any failure here only disables checkpoint sync; training still starts.
8. **resume**   -- full mode: a local run, else the latest one in the checkpoint dataset.
   Resumable -> ``--resume``; finished -> nothing to do; incompatible (another
   commit, a smoke run) -> moved aside, fresh start.
9. **train**    -- ``train_baseline.py`` with unbuffered output, a stall watchdog
   (no output for ``--stall-minutes`` -> the run is killed; rerun to resume).
10. **verify**  -- smoke mode: last.pt/best.pt load, carry resume state, finite weights.
   Full mode: the run directory is packed to ``/kaggle/working/baseline_v1_detector_run.tgz``.

Smoke mode trains the detector on 64 images for 2 epochs (``train_baseline.py
--smoke``) and never uploads to the checkpoint dataset (it only checks access),
so it cannot overwrite a real run's checkpoints.

Exit codes: 0 ok (or already finished), 1 a stage failed, 2 training refused,
3 training stalled and was killed.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
PINS = {"torch": "2.14.0", "torchvision": "0.29.0", "numpy": "2.5.3", "PIL": "11.3.0"}
PIP_NAMES = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "PIL": "pillow"}
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
REQUIRED = (
    "TRAINING_BUNDLE.json", "configs/baseline_v1.json", "dataset/meta.csv", "dataset/splits/dataset_v1.json",
    "dataset/splits/dataset_v1_files.csv", "scripts/train_baseline.py", "scripts/kaggle_persist_run.py",
    "scripts/kaggle_checkpoint.py", "src/training/models.py", "src/training/torch_data.py",
)
COPY_MARKER = ".bundle_copy_complete"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_STALLED = 0, 1, 2, 3

_LOG: Path | None = None


def say(message: str) -> None:
    line = f"[launch {datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if _LOG is not None:
        with _LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class StageFailed(RuntimeError):
    pass


def run_quiet(cmd: list[str], timeout: float, env: dict | None = None, cwd: Path | None = None) -> tuple[int, str]:
    """Run to completion with a timeout; (returncode, combined output). 124 on timeout."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd,
                              stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f} s"
    except OSError as exc:
        return 127, str(exc)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def stream(cmd: list[str], *, env: dict, cwd: Path, stall_seconds: float, total_timeout: float | None = None) -> int:
    """Run ``cmd``, echo every line as it arrives, kill it when silent for ``stall_seconds``.

    Returns the exit code, or ``-EXIT_STALLED`` when the watchdog killed it.
    """
    posix = os.name == "posix"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env, cwd=cwd, text=True, bufsize=1, errors="replace", start_new_session=posix)
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    started = last = time.monotonic()
    while True:
        try:
            line = lines.get(timeout=5)
        except queue.Empty:
            line = ""
        if line is None:
            break
        if line:
            last = time.monotonic()
            print(line, end="", flush=True)
            if _LOG is not None:
                with _LOG.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            continue
        now = time.monotonic()
        silent = now - last > stall_seconds
        over = total_timeout is not None and now - started > total_timeout
        if silent or over:
            say(f"no output for {now - last:.0f} s" if silent else f"exceeded {total_timeout:.0f} s")
            say("killing the process (and its data-loader workers)")
            try:
                if posix:
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            return -EXIT_STALLED
    return proc.wait()


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_bundle(bundle: Path) -> dict:
    say(f"stage 1/10 bundle: {bundle}")
    missing = [name for name in REQUIRED if not (bundle / name).is_file()]
    if missing:
        raise StageFailed(f"the bundle lacks {', '.join(missing)}; is this the right dataset version?")
    marker = json.loads((bundle / "TRAINING_BUNDLE.json").read_text(encoding="utf-8"))
    images = bundle / "dataset" / "images" / "synthetic"
    count = sum(1 for e in os.scandir(images) if e.name.endswith(".jpg")) if images.is_dir() else 0
    if count != 12_000:
        raise StageFailed(f"{images} holds {count} images, expected 12,000")
    say(f"  commit {marker['git_commit']}, 12,000 synthetic images, no real images expected "
        f"({marker.get('real_images')})")
    return marker


def stage_machine(workdir: Path, device: str) -> None:
    say("stage 2/10 machine")
    parent = workdir.parent if workdir.parent.exists() else Path.cwd()
    free = shutil.disk_usage(parent).free / 1e9
    say(f"  python {sys.version.split()[0]} ({sys.executable}), {os.cpu_count()} CPU(s), "
        f"{free:.1f} GB free at {parent}")
    if free < 3:
        raise StageFailed(f"only {free:.1f} GB free at {parent}; the copy needs about 1 GB and runs need 1-2 GB")
    if device.startswith("cuda"):
        code, out = run_quiet(["nvidia-smi", "-L"], timeout=30)
        if code != 0:
            raise StageFailed(f"nvidia-smi failed ({out.strip()}); attach a GPU accelerator (T4) to the notebook")
        for line in out.strip().splitlines():
            say(f"  {line}")


def _walk(root: Path) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        files += [Path(dirpath) / f for f in filenames]
    return files


def stage_copy(bundle: Path, workdir: Path, commit: str) -> None:
    say(f"stage 3/10 copy: {bundle} -> {workdir}")
    marker = workdir / COPY_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == commit:
        say("  a complete copy of this commit is already there; reusing it (keeps runs/)")
        return
    if workdir.exists():
        aside = workdir.with_name(f"{workdir.name}.stale-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        say(f"  {workdir} holds another or incomplete copy; moving it to {aside} (nothing is deleted)")
        workdir.rename(aside)
    files = _walk(bundle)
    total = sum(f.stat().st_size for f in files)
    say(f"  copying {len(files)} files, {total / 1e6:.0f} MB")
    started = time.monotonic()
    copied = 0
    for i, source in enumerate(files, 1):
        target = workdir / source.relative_to(bundle)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += source.stat().st_size
        if i % 2000 == 0 or i == len(files):
            say(f"  {i}/{len(files)} files, {copied / 1e6:.0f} MB, {time.monotonic() - started:.0f} s")
    got = _walk(workdir)
    got_bytes = sum(f.stat().st_size for f in got)
    if len(got) != len(files) or got_bytes != total:
        raise StageFailed(f"copy incomplete: {len(got)}/{len(files)} files, {got_bytes}/{total} bytes")
    marker.write_text(commit + "\n", encoding="utf-8")
    say("  copy verified: same file count and byte total (content hashes are checked by the preflight)")


_VERSIONS = (
    "import json, importlib\n"
    "out = {}\n"
    "for m in ('torch', 'torchvision', 'numpy', 'PIL'):\n"
    "    try:\n"
    "        out[m] = importlib.import_module(m).__version__\n"
    "    except Exception as e:\n"
    "        out[m] = None\n"
    "try:\n"
    "    import torch; out['cuda_build'] = torch.version.cuda\n"
    "except Exception:\n"
    "    out['cuda_build'] = None\n"
    "print(json.dumps(out))\n"
)


def _versions(python: str) -> dict:
    code, out = run_quiet([python, "-c", _VERSIONS], timeout=180)
    if code != 0:
        raise StageFailed(f"could not read package versions: {out.strip()[-500:]}")
    return json.loads(out.strip().splitlines()[-1])


def _mismatches(found: dict, need_cuda: bool) -> list[str]:
    bad = [m for m, want in PINS.items() if (found.get(m) or "").split("+")[0] != want]
    if need_cuda and not found.get("cuda_build") and "torch" not in bad:
        bad.append("torch")  # a CPU-only build
    return bad


def stage_packages(python: str, *, device: str, install: bool, env: dict, stall_seconds: float) -> None:
    say("stage 4/10 packages")
    found = _versions(python)
    say(f"  found {found}")
    bad = _mismatches(found, device.startswith("cuda"))
    if not bad:
        say("  pinned versions present")
        return
    if not install:
        raise StageFailed(f"packages differ from requirements-train.txt: {', '.join(bad)} (install disabled)")
    torch_pkgs = [f"{PIP_NAMES[m]}=={PINS[m]}" for m in ("torch", "torchvision") if m in bad]
    other_pkgs = [f"{PIP_NAMES[m]}=={PINS[m]}" for m in ("numpy", "PIL") if m in bad]
    common = ["--progress-bar", "off", "--timeout", "60", "--retries", "3", "--no-input"]
    if torch_pkgs:
        say(f"  installing {' '.join(torch_pkgs)} (CUDA 12.6 build, about 3 GB; several minutes)")
        cmd = [python, "-m", "pip", "install", *common, *torch_pkgs, "--index-url", TORCH_INDEX]
        if stream(cmd, env=env, cwd=Path.cwd(), stall_seconds=stall_seconds, total_timeout=3600) != 0:
            raise StageFailed("pip install of torch/torchvision failed")
    if other_pkgs:
        say(f"  installing {' '.join(other_pkgs)}")
        cmd = [python, "-m", "pip", "install", *common, *other_pkgs]
        if stream(cmd, env=env, cwd=Path.cwd(), stall_seconds=stall_seconds, total_timeout=1800) != 0:
            raise StageFailed("pip install of numpy/pillow failed")
    found = _versions(python)
    say(f"  now {found}")
    still = _mismatches(found, device.startswith("cuda"))
    if still:
        raise StageFailed(f"still not the pinned versions after install: {', '.join(still)}")


_CUDA_CHECK = (
    "import json, torch\n"
    "assert torch.cuda.is_available(), 'torch sees no CUDA device'\n"
    "d = torch.device('cuda:0')\n"
    "conv = torch.nn.Conv2d(3, 16, 3).to(d)\n"
    "x = torch.randn(4, 3, 64, 64, device=d, requires_grad=True)\n"
    "conv(x).square().mean().backward()\n"
    "torch.cuda.synchronize()\n"
    "assert torch.isfinite(conv.weight.grad).all()\n"
    "print(json.dumps({'devices': torch.cuda.device_count(), 'name': torch.cuda.get_device_name(0),\n"
    "                  'capability': list(torch.cuda.get_device_capability(0)), 'cuda': torch.version.cuda}))\n"
)


def stage_cuda(python: str, env: dict, device: str) -> None:
    say("stage 5/10 cuda")
    if not device.startswith("cuda"):
        say(f"  skipped: --device {device}")
        return
    code, out = run_quiet([python, "-c", _CUDA_CHECK], timeout=300, env=env)
    if code != 0:
        raise StageFailed(f"CUDA forward/backward failed: {out.strip()[-800:]}")
    info = json.loads(out.strip().splitlines()[-1])
    say(f"  forward+backward ok on {info['name']} (sm_{''.join(map(str, info['capability']))}, "
        f"CUDA {info['cuda']}); visible to training: GPU 0 only")


_WEIGHTS = (
    "import socket\n"
    "socket.setdefaulttimeout(60)\n"
    "from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights as W\n"
    "sd = W.COCO_V1.get_state_dict(progress=False)\n"
    "print(len(sd))\n"
)


def stage_weights(python: str, env: dict, workdir: Path) -> None:
    say("stage 6/10 weights")
    config = json.loads((workdir / "configs" / "baseline_v1.json").read_text(encoding="utf-8"))
    if not config["detector"].get("pretrained", True):
        say("  skipped: the config trains from scratch")
        return
    code, out = run_quiet([python, "-c", _WEIGHTS], timeout=300, env=env)
    if code != 0:
        raise StageFailed(f"COCO-pretrained weights could not be fetched (Internet on?): {out.strip()[-600:]}")
    say(f"  COCO weights ready ({out.strip().splitlines()[-1]} tensors, cached for training)")


def _load_secrets(env: dict) -> None:
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return
    result: dict = {}

    def fetch() -> None:
        try:
            from kaggle_secrets import UserSecretsClient  # only exists inside Kaggle

            client = UserSecretsClient()
            result["KAGGLE_USERNAME"] = client.get_secret("KAGGLE_USERNAME")
            result["KAGGLE_KEY"] = client.get_secret("KAGGLE_KEY")
        except Exception as exc:  # noqa: BLE001 - any failure only disables sync
            result["error"] = str(exc)

    worker = threading.Thread(target=fetch, daemon=True)
    worker.start()
    worker.join(30)
    if worker.is_alive():
        say("  reading notebook Secrets timed out after 30 s")
    elif "error" in result:
        say(f"  notebook Secrets not available ({result['error']})")
    else:
        env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = result["KAGGLE_USERNAME"], result["KAGGLE_KEY"]
        say("  Kaggle token loaded from notebook Secrets")


def stage_sync(python: str, env: dict, workdir: Path, dataset: str | None) -> bool:
    say("stage 7/10 sync")
    if not dataset:
        say("  no --checkpoint-dataset: checkpoints stay in /kaggle/working only")
        return False
    _load_secrets(env)
    code, out = run_quiet([python, str(workdir / "scripts" / "kaggle_persist_run.py"), "check", "--dataset", dataset],
                          timeout=90, env=env)
    if code != 0:
        say(f"  WARNING: checkpoint dataset not reachable ({out.strip()[-300:]}); sync DISABLED, training continues")
        return False
    say(f"  {dataset}: {out.strip().splitlines()[-1] if out.strip() else 'ok'}; sync after every epoch")
    return True


_INSPECT = (
    "import json, sys, torch\n"
    "c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
    "r = c.get('resume') if isinstance(c, dict) else None\n"
    "o = {'epoch': c.get('epoch') if isinstance(c, dict) else None, 'resumable': isinstance(r, dict)}\n"
    "if isinstance(r, dict):\n"
    "    i = r['identity']\n"
    "    o.update(finished=r['finished'], kind=i.get('kind'), git_commit=i.get('git_commit'),\n"
    "             best_epoch=r['best_epoch'], format=r['format'], optimizer_tensors=len(r['optimizer'].get('state', {})))\n"
    "o['finite'] = all(bool(torch.isfinite(t).all()) for t in c['model'].values() if t.is_floating_point())\n"
    "print(json.dumps(o))\n"
)


def inspect_checkpoint(python: str, path: Path, env: dict) -> dict:
    code, out = run_quiet([python, "-c", _INSPECT, str(path)], timeout=300, env=env)
    if code != 0:
        return {"error": out.strip()[-400:]}
    return json.loads(out.strip().splitlines()[-1])


def choose_start(info: dict | None, commit: str) -> str:
    """fresh | resume | finished | incompatible, for a candidate last.pt."""
    if info is None:
        return "fresh"
    if "error" in info or not info.get("resumable") or info.get("kind") != "baseline" \
            or info.get("git_commit") != commit:
        return "incompatible"
    return "finished" if info.get("finished") else "resume"


def local_runs(workdir: Path) -> list[Path]:
    root = workdir / "runs" / "detector"
    if not root.is_dir():
        return []
    runs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))
            and not d.name.endswith("_smoke") and (d / "last.pt").is_file()]
    return sorted(runs, key=lambda d: (d / "last.pt").stat().st_mtime)


def stage_resume(python: str, env: dict, workdir: Path, commit: str, dataset: str | None,
                 sync_ok: bool) -> tuple[str, Path | None]:
    say("stage 8/10 resume")
    runs = local_runs(workdir)
    if not runs and dataset and not sync_ok:
        say(f"  WARNING: {dataset} is unreachable, so an earlier run stored there cannot be found; "
            "starting FRESH (the stored run is not touched: sync is disabled too)")
    if not runs and dataset and sync_ok:
        say(f"  no local run; fetching the latest version of {dataset}")
        code, out = run_quiet([python, str(workdir / "scripts" / "kaggle_persist_run.py"), "restore",
                               "--dataset", dataset, "--dest-root", str(workdir / "runs")], timeout=1000, env=env)
        say(f"  restore: exit {code}: {out.strip().splitlines()[-1] if out.strip() else ''}")
        runs = local_runs(workdir)
    if not runs:
        say("  no earlier run: fresh start")
        return "fresh", None
    run = runs[-1]
    info = inspect_checkpoint(python, run / "last.pt", env)
    decision = choose_start(info, commit)
    say(f"  {run.name}: {info} -> {decision}")
    if decision == "incompatible":
        aside = workdir / "runs" / "detector" / f"_incompatible_{run.name}_{datetime.now().strftime('%H%M%S')}"
        run.rename(aside)
        say(f"  moved to {aside.name}; fresh start")
        return "fresh", None
    return decision, run / "last.pt"


def stage_train(python: str, env: dict, workdir: Path, args, *, resume: Path | None, sync: bool) -> int:
    say("stage 9/10 train")
    cmd = [python, "-u", "scripts/train_baseline.py", "--component", "detector",
           "--i-have-approval", args.approver, "--device", args.device, "--workers", str(args.workers)]
    if args.mode == "smoke":
        cmd.append("--smoke")
    if resume is not None:
        cmd += ["--resume", str(resume)]
    if sync:
        push = f'"{python}" "{workdir / "scripts" / "kaggle_persist_run.py"}" push --dataset {args.checkpoint_dataset}'
        cmd += ["--sync-command", push]
    say("  " + " ".join(c if " " not in c else repr(c) for c in cmd))
    return stream(cmd, env=env, cwd=workdir, stall_seconds=args.stall_minutes * 60)


def stage_verify_smoke(python: str, env: dict, workdir: Path) -> None:
    say("stage 10/10 verify smoke checkpoints")
    runs = sorted((workdir / "runs" / "detector").glob("*_smoke"), key=lambda d: d.stat().st_mtime)
    if not runs:
        raise StageFailed("no smoke run directory was written")
    run = runs[-1]
    info = inspect_checkpoint(python, run / "last.pt", env)
    best = inspect_checkpoint(python, run / "best.pt", env)
    say(f"  last.pt {info}")
    say(f"  best.pt {best}")
    problems = []
    if not (info.get("resumable") and info.get("format") == 1 and info.get("epoch") == 2 and info.get("finished")):
        problems.append("last.pt lacks the expected resume state (format 1, epoch 2, finished)")
    if not info.get("optimizer_tensors"):
        problems.append("last.pt has no optimizer state")
    if not (info.get("finite") and best.get("finite")):
        problems.append("non-finite weights")
    if "error" in best:
        problems.append(f"best.pt unreadable: {best['error']}")
    metrics = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if (run / "metrics.jsonl").is_file() else []
    if len(metrics) != 2:
        problems.append(f"metrics.jsonl has {len(metrics)} lines, expected 2")
    if problems:
        raise StageFailed("; ".join(problems))
    say(f"  SMOKE PASSED: data loading, model build, CUDA training, checkpoint write/read ({run})")


def pack_run(workdir: Path) -> Path | None:
    runs = local_runs(workdir)
    if not runs:
        return None
    run = runs[-1]
    out = workdir.parent / "baseline_v1_detector_run.tgz"
    with tarfile.open(out, "w:gz") as tar:
        for name in ("run.json", "metrics.jsonl", "train.log", "sync.log", "best.pt", "last.pt"):
            if (run / name).is_file():
                tar.add(run / name, arcname=f"runs/detector/{run.name}/{name}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    global _LOG
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--approver", required=True, help="the person who authorised this run")
    parser.add_argument("--checkpoint-dataset", default=None, help="<user>/<slug>; optional, full mode syncs to it")
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working/baseline_v1"))
    parser.add_argument("--bundle", type=Path, default=BUNDLE_ROOT, help="default: the bundle holding this script")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--no-install", action="store_true", help="only check package versions, never pip install")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    python = sys.executable
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    if args.device.startswith("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU: the trainer uses one, and this keeps it unambiguous
    say(f"Baseline V1 detector launcher, mode {args.mode}, device {args.device}")
    try:
        marker = stage_bundle(args.bundle.resolve())
        stage_machine(args.workdir, args.device)
        args.workdir.parent.mkdir(parents=True, exist_ok=True)
        stage_copy(args.bundle.resolve(), args.workdir, marker["git_commit"])
        _LOG = args.workdir / "launcher.log"
        say(f"  launcher log: {_LOG}")
        stage_packages(python, device=args.device, install=not args.no_install, env=env,
                       stall_seconds=args.stall_minutes * 60)
        stage_cuda(python, env, args.device)
        stage_weights(python, env, args.workdir)
        sync = stage_sync(python, env, args.workdir, args.checkpoint_dataset)
        if args.mode == "smoke":
            for old in (args.workdir / "runs" / "detector").glob("*_smoke"):
                shutil.rmtree(old)  # smoke runs are disposable; a fresh one must not be refused
            say("stage 8/10 resume: skipped in smoke mode (and smoke never uploads)")
            code = stage_train(python, env, args.workdir, args, resume=None, sync=False)
        else:
            decision, resume = stage_resume(python, env, args.workdir, marker["git_commit"],
                                            args.checkpoint_dataset, sync)
            if decision == "finished":
                say("detector training already FINISHED for this commit; nothing to train")
                packed = pack_run(args.workdir)
                say(f"run packed: {packed}")
                return EXIT_OK
            code = stage_train(python, env, args.workdir, args, resume=resume, sync=sync)
    except StageFailed as exc:
        say(f"FAILED: {exc}")
        return EXIT_FAILED
    if code == -EXIT_STALLED:
        say("training STALLED and was killed; last.pt holds the last finished epoch -- rerun this cell to resume")
        return EXIT_STALLED
    if code != 0:
        say(f"training exited with {code}; see the lines above and {args.workdir / 'launcher.log'}")
        return EXIT_REFUSED if code == 2 else EXIT_FAILED
    if args.mode == "smoke":
        try:
            stage_verify_smoke(python, env, args.workdir)
        except StageFailed as exc:
            say(f"FAILED: {exc}")
            return EXIT_FAILED
        return EXIT_OK
    say("stage 10/10 pack")
    say(f"run packed: {pack_run(args.workdir)}")
    say("detector training FINISHED")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
