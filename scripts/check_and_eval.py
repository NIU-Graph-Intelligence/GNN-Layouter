"""
check_and_eval.py - poll training completion and run incremental eval.
Designed to be called from ScheduleWakeup; safe to call multiple times.
Detects training completion via progress log "done." marker (not just ckpt existence).
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET_PATH = os.path.join(ROOT, "data/processed/comm_5k_v2_with_encodings.pt")
PYTHON = os.path.join(ROOT, ".venv/bin/python")

MODELS = ["phi1", "mpnn", "gmpnn", "forgetnet"]
SEEDS  = [7, 2024]

GPU_FOR_SEED = {7: "cuda:0", 2024: "cuda:1"}

def training_done(model, seed):
    """Return True if training completed (done marker in progress log)."""
    log = os.path.join(ROOT, f"checkpoints/{model}_seed{seed}_progress.log")
    if not os.path.exists(log):
        return False
    with open(log) as f:
        return any(f"=== {model}_seed{seed} done" in line for line in f)

def ckpt_path(model, seed):
    return os.path.join(ROOT, f"checkpoints/{model}_seed{seed}/executor_best.pt")

def npz_path(model, seed):
    return os.path.join(ROOT, f"eval/predictions/{model}_seed{seed}.npz")

def json_path(model, seed):
    return os.path.join(ROOT, f"eval/results/eval_{model}_seed{seed}.json")

def extrap_path(model, seed):
    return os.path.join(ROOT, f"checkpoints/{model}_seed{seed}/extrapolation.txt")

def run(cmd, **kw):
    print(f"  + {' '.join(cmd)[:100]}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[-400:]}")
    else:
        print(r.stdout[-300:] if r.stdout.strip() else "  (no stdout)")
    return r.returncode == 0


def main():
    print(f"\n=== check_and_eval.py ===")
    
    completed_training = []
    pending_training   = []
    
    for model in MODELS:
        for seed in SEEDS:
            if training_done(model, seed):
                completed_training.append((model, seed))
            else:
                pending_training.append((model, seed))
    
    print(f"\nTraining status:")
    for m, s in completed_training:
        print(f"  DONE    {m}_seed{s}")
    for m, s in pending_training:
        # Show progress if log exists
        log = os.path.join(ROOT, f"checkpoints/{m}_seed{s}_progress.log")
        if os.path.exists(log):
            import subprocess as sp
            last = sp.run(["tail", "-1", log], capture_output=True, text=True).stdout.strip()
        else:
            last = "(not started)"
        print(f"  WAITING {m}_seed{s}  — {last}")
    
    # Step: dump predictions for newly-completed training runs
    for model, seed in completed_training:
        npz = npz_path(model, seed)
        if not os.path.exists(npz):
            dev = GPU_FOR_SEED.get(seed, "cuda:0")
            ckpt = ckpt_path(model, seed)
            print(f"\nDumping {model}_seed{seed}...")
            run([PYTHON, "eval/dump_executor_predictions.py",
                 "--checkpoint", ckpt,
                 "--name", f"{model}_seed{seed}",
                 "--device", dev,
                 "--dataset_path", DATASET_PATH])
        
    # Step: score any unscored npz
    for model, seed in completed_training:
        jpath = json_path(model, seed)
        npz   = npz_path(model, seed)
        if not os.path.exists(jpath) and os.path.exists(npz):
            print(f"\nScoring {model}_seed{seed}...")
            run([PYTHON, "eval/score_predictions.py",
                 "--pred", npz,
                 "--dataset_path", DATASET_PATH])
    
    # Step: step_extrapolation for unextrapolated models
    for model, seed in completed_training:
        ep = extrap_path(model, seed)
        ckpt = ckpt_path(model, seed)
        if not os.path.exists(ep) or os.path.getsize(ep) == 0:
            dev = GPU_FOR_SEED.get(seed, "cuda:0")
            print(f"\nStep extrapolation {model}_seed{seed}...")
            r = subprocess.run(
                [PYTHON, "-m", "philayouter.executor.evaluate",
                 "--checkpoint", ckpt,
                 "--dataset_path", DATASET_PATH,
                 "--device", dev],
                cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                with open(ep, "w") as f:
                    f.write(r.stdout)
                print(r.stdout[-400:])
            else:
                print(f"  ERROR: {r.stderr[-300:]}")
    
    # Summary
    print(f"\n--- Summary ---")
    all_done_eval = True
    for model, seed in completed_training:
        j = json_path(model, seed)
        e = extrap_path(model, seed)
        j_ok = os.path.exists(j)
        e_ok = os.path.exists(e) and os.path.getsize(e) > 0
        flag = "OK" if (j_ok and e_ok) else "PARTIAL"
        print(f"  {model}_seed{seed}: json={j_ok}, extrap={e_ok}  [{flag}]")
        if not (j_ok and e_ok):
            all_done_eval = False
    
    if pending_training:
        print(f"\n  Still training: {len(pending_training)} run(s)")
        all_done_eval = False
    
    # If everything is ready: write §12
    if all_done_eval and len(completed_training) == 8:
        print("\nAll 8 runs evaluated — running eval_and_write_sec12.py")
        r = subprocess.run([PYTHON, "scripts/eval_and_write_sec12.py"],
                           cwd=ROOT, capture_output=True, text=True)
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print(f"ERROR: {r.stderr[-500:]}")
    else:
        remaining = 8 - len(completed_training)
        print(f"\n  {remaining} training run(s) still pending. Schedule another wakeup.")
    
    return all_done_eval and len(completed_training) == 8


if __name__ == "__main__":
    done = main()
    sys.exit(0 if done else 1)
