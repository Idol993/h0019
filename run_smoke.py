#!/usr/bin/env python3
"""
SMOKE integration test for the continual-learning LoRA pipeline.

Covers:
  1. Task stream training with mixed num_labels (cola=2, mrpc=2, mnli=3)
  2. Gradient signature / task similarity computation across different label spaces
  3. Similar-Task initialization with mixed-rank LoRA merge + prune
  4. Training progress snapshots (training_progress.json)
  5. Evaluation -> task matrix + forgetting curves PNG generation

Usage:
  python run_smoke.py [--no-cuda]
"""
import os
import sys
import json
import shutil
import argparse
import subprocess


def check_file(path: str, desc: str) -> bool:
    ok = os.path.exists(path) and os.path.getsize(path) > 0
    print(f"  [{'OK' if ok else 'FAIL'}] {desc}: {path}" + ("" if ok else " (MISSING/EMPTY)"))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--max-train", type=int, default=48, help="Max train samples per task (SMOKE)")
    parser.add_argument("--max-val", type=int, default=32, help="Max val samples per task (SMOKE)")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep smoke_loras / smoke_eval after test")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    lora_dir = os.path.join(project_root, "smoke_loras")
    eval_dir = os.path.join(project_root, "smoke_eval")

    if not args.keep_artifacts:
        for d in (lora_dir, eval_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)

    base_flags = ["--config", "configs_smoke.yaml",
                  "--task-stream", "tasks_smoke.txt",
                  "--seed", "42"]
    if args.no_cuda:
        base_flags.append("--no-cuda")

    train_flags = list(base_flags) + [
        "--max-train-samples", str(args.max_train),
        "--max-val-samples", str(args.max_val),
    ]

    print("\n" + "=" * 70)
    print("SMOKE PHASE 1/2: cl_solver.py (train 3 tasks with mixed num_labels)")
    print("=" * 70)
    train_cmd = [sys.executable, "cl_solver.py"] + train_flags
    print("Running:", " ".join(train_cmd))
    rc = subprocess.call(train_cmd, cwd=project_root)
    if rc != 0:
        print(f"\n[FATAL] Training phase failed with exit code {rc}")
        sys.exit(rc)

    print("\n" + "=" * 70)
    print("SMOKE PHASE 2/2: eval.py (task matrix + forgetting curves)")
    print("=" * 70)
    eval_cmd = [sys.executable, "eval.py"] + list(base_flags) + [
        "--max-val-samples", str(args.max_val),
    ]
    print("Running:", " ".join(eval_cmd))
    rc = subprocess.call(eval_cmd, cwd=project_root)
    if rc != 0:
        print(f"\n[FATAL] Eval phase failed with exit code {rc}")
        sys.exit(rc)

    print("\n" + "=" * 70)
    print("SMOKE: artifact validation")
    print("=" * 70)
    checks = []
    checks.append(check_file(
        os.path.join(lora_dir, "training_progress.json"),
        "training progress snapshot (row=after training step i, col=task j accuracy)",
    ))
    for task in ("smoke_cola", "smoke_mrpc", "smoke_mnli"):
        checks.append(check_file(
            os.path.join(lora_dir, task, "lora_weights.pt"), f"LoRA weights for {task}",
        ))
        checks.append(check_file(
            os.path.join(lora_dir, task, "metadata.json"), f"LoRA metadata for {task}",
        ))
    checks.append(check_file(
        os.path.join(lora_dir, "registry_index.json"), "LoRA registry index",
    ))
    checks.append(check_file(
        os.path.join(eval_dir, "task_matrix.png"), "Task matrix heatmap PNG",
    ))
    checks.append(check_file(
        os.path.join(eval_dir, "forgetting_curves.png"), "Forgetting curves PNG",
    ))
    checks.append(check_file(
        os.path.join(eval_dir, "avg_accuracy.png"), "Avg accuracy over steps PNG",
    ))
    checks.append(check_file(
        os.path.join(eval_dir, "task_matrix.npy"), "Task matrix .npy",
    ))
    checks.append(check_file(
        os.path.join(eval_dir, "summary.json"), "Eval summary JSON",
    ))

    progress_path = os.path.join(lora_dir, "training_progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            prog = json.load(f)
        rows = prog.get("progress_rows", [])
        order = prog.get("task_order", [])
        print(f"\n[OK] training_progress.json: task_order={order}, {len(rows)} rows")
        for i, r in enumerate(rows):
            print(f"    After step {i} ({order[i]}): { {k: round(v, 4) for k, v in r.items()} }")

        if len(rows) == len(order) == 3:
            print("\n[OK] Mixed-num_labels flow verified: cola(2) -> mrpc(2) -> mnli(3) produced 3 rows.")
        else:
            print("[WARN] Expected 3 tasks, got", len(rows), "rows and", len(order), "task names")

    summary_path = os.path.join(eval_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            s = json.load(f)
        print(f"\n[OK] Eval summary:")
        print(f"    avg_forgetting         = {s.get('avg_forgetting', None)}")
        print(f"    avg_accuracy_seen_tasks= {s.get('avg_accuracy_seen_tasks', None)}")

    all_ok = all(checks)
    print("\n" + "=" * 70)
    print(f"SMOKE TEST: {'PASSED' if all_ok else 'FAILED'}  "
          f"({sum(checks)}/{len(checks)} artifact checks passed)")
    print("=" * 70)

    if not args.keep_artifacts and all_ok:
        print("Cleaning up smoke artifacts (--keep-artifacts to preserve)...")
        for d in (lora_dir, eval_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
