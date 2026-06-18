from __future__ import annotations

import os
import sys
import argparse
import yaml
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def _lazy_import_training_stack():
    """Lazy-import heavy deps (torch/transformers/peft/datasets/lora_manager) only for re-eval."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoConfig,
        DataCollatorWithPadding,
    )
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
    from datasets import load_dataset
    from lora_manager import LoRAManager, _to_json_serializable
    return {
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "AutoTokenizer": AutoTokenizer,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoConfig": AutoConfig,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "set_peft_model_state_dict": set_peft_model_state_dict,
        "TaskType": TaskType,
        "load_dataset": load_dataset,
        "LoRAManager": LoRAManager,
        "_to_json_serializable": _to_json_serializable,
    }


def _to_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_json_serializable(v) for v in sorted(obj, key=lambda x: str(x))]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return obj


def set_seed(seed: int = 42):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_task_stream(task_stream_path: str) -> List[Dict[str, Any]]:
    tasks = []
    with open(task_stream_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                task_name, dataset_path = parts[0], parts[1]
                epochs = int(parts[2]) if len(parts) > 2 else 5
                num_labels = int(parts[3]) if len(parts) > 3 else 2
                tasks.append({
                    "name": task_name,
                    "dataset": dataset_path,
                    "epochs": epochs,
                    "num_labels": num_labels,
                })
    return tasks


def load_task_dataset(
    dataset_spec: str,
    tokenizer,
    max_seq_length: int,
    num_labels: int,
    max_val_samples: Optional[int] = None,
):
    if ":" in dataset_spec:
        ds_name, ds_subset = dataset_spec.split(":", 1)
    else:
        ds_name, ds_subset = dataset_spec, None

    if ds_subset:
        dataset = load_dataset(ds_name, ds_subset)
    else:
        dataset = load_dataset(ds_name)

    def determine_keys(split):
        keys = list(dataset[split].features.keys())
        sent_keys = [k for k in keys if k in ("sentence", "sentence1", "question", "premise", "text", "context", "hypothesis")]
        label_key = "label" if "label" in keys else "labels" if "labels" in keys else keys[-1]
        return sent_keys, label_key

    def preprocess(examples, sent_keys, label_key):
        if len(sent_keys) == 1:
            texts = examples[sent_keys[0]]
            texts = [str(t) if t is not None else "" for t in texts]
            encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=max_seq_length)
        elif len(sent_keys) >= 2:
            text1 = [str(t) if t is not None else "" for t in examples[sent_keys[0]]]
            text2 = [str(t) if t is not None else "" for t in examples[sent_keys[1]]]
            encoded = tokenizer(text1, text2, truncation=True, padding="max_length", max_length=max_seq_length)
        else:
            encoded = {}
        if label_key in examples:
            encoded["labels"] = examples[label_key]
        return encoded

    val_key = None
    if "validation" in dataset:
        val_key = "validation"
    elif "validation_matched" in dataset:
        val_key = "validation_matched"
    elif "test" in dataset:
        val_key = "test"

    if val_key is None:
        return None

    val_split = dataset[val_key]
    if max_val_samples and max_val_samples < len(val_split):
        val_split = val_split.select(range(max_val_samples))

    val_sent_keys, val_label_key = determine_keys(val_key)
    val_ds = val_split.map(
        lambda x: preprocess(x, val_sent_keys, val_label_key),
        batched=True,
        remove_columns=[c for c in val_split.column_names if c not in ("input_ids", "attention_mask", "labels")],
    )
    return val_ds


def compute_accuracy(preds: List[int], labels: List[int]) -> float:
    if len(preds) == 0:
        return 0.0
    return float((np.array(preds) == np.array(labels)).mean())


def evaluate_single_task(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: str,
) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "labels" not in batch:
                continue
            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            labels = batch["labels"].cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return compute_accuracy(all_preds, all_labels)


def build_task_matrix(
    task_infos: List[Dict[str, Any]],
    task_order: List[str],
    accuracies: Dict[str, Dict[str, float]],
) -> np.ndarray:
    n = len(task_order)
    matrix = np.zeros((n, n))
    for i, train_task in enumerate(task_order):
        for j, test_task in enumerate(task_order):
            matrix[i, j] = accuracies.get(train_task, {}).get(test_task, 0.0)
    return matrix


def plot_task_matrix(
    matrix: np.ndarray,
    task_order: List[str],
    save_path: str,
):
    n = len(task_order)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), max(6, n * 0.9)))
    im = ax.imshow(matrix, cmap=cm.YlGnBu, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(task_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(task_order, fontsize=9)
    ax.set_xlabel("Test Task (column)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Trained After Task # (row)", fontsize=11, fontweight="bold")
    ax.set_title("Task Accuracy Matrix (rows=training progress, cols=evaluation)", fontsize=12, fontweight="bold", pad=15)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", rotation=270, labelpad=15)

    ax.axhline(y=-0.5, color="black", linewidth=1.5)
    ax.axhline(y=n - 0.5, color="black", linewidth=1.5)
    ax.axvline(x=-0.5, color="black", linewidth=1.5)
    ax.axvline(x=n - 0.5, color="black", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Task matrix plot saved to {save_path}")


def plot_forgetting_curves(
    matrix: np.ndarray,
    task_order: List[str],
    save_path: str,
):
    n = len(task_order)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = cm.tab10(np.linspace(0, 1, max(10, n)))
    step_indices = np.arange(n)

    forgetting_measures = {}
    avg_forgetting = 0.0
    count = 0

    for j in range(n):
        perf_curve = []
        for i in range(n):
            if i >= j:
                perf_curve.append(matrix[i, j])
            else:
                perf_curve.append(np.nan)

        valid_curve = [x for x in perf_curve if not np.isnan(x)]
        if len(valid_curve) >= 2:
            initial_perf = valid_curve[0]
            final_perf = valid_curve[-1]
            forgetting = initial_perf - final_perf
            forgetting_measures[task_order[j]] = forgetting
            avg_forgetting += forgetting
            count += 1

        line, = ax.plot(
            step_indices, perf_curve,
            marker="o", markersize=7, linewidth=2,
            color=colors[j % len(colors)],
            label=f"{task_order[j]} (forget={forgetting_measures.get(task_order[j], 0):.3f})"
            if j in forgetting_measures else f"{task_order[j]}"
        )

        if not np.isnan(perf_curve[j]):
            ax.annotate(
                f"{perf_curve[j]:.2f}",
                (j, perf_curve[j]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, color=colors[j % len(colors)],
                fontweight="bold",
            )

    if count > 0:
        avg_forgetting /= count

    ax.set_xlabel("Training Step (Task # Completed)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy on Test Task", fontsize=11, fontweight="bold")
    ax.set_title(f"Forgetting Curves\n(Avg Forgetting = {avg_forgetting:.3f})", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, framealpha=0.9)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(step_indices)
    ax.set_xticklabels(task_order, rotation=45, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Forgetting curves saved to {save_path}")
    return forgetting_measures, avg_forgetting


def plot_avg_accuracy(
    matrix: np.ndarray,
    task_order: List[str],
    save_path: str,
):
    n = len(task_order)
    avg_per_step = []
    for i in range(n):
        seen = matrix[i, : (i + 1)]
        avg_per_step.append(seen.mean())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        range(n), avg_per_step,
        marker="s", markersize=9, linewidth=2.5,
        color="#2c3e50", markerfacecolor="#3498db", markeredgecolor="white",
        markeredgewidth=1.5,
    )
    for i, v in enumerate(avg_per_step):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold", color="#2c3e50")

    ax.set_xlabel("Training Step (Task # Completed)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Accuracy (seen tasks)", fontsize=11, fontweight="bold")
    ax.set_title("Average Accuracy Over All Seen Tasks", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(range(n))
    ax.set_xticklabels(task_order, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(max(0.0, min(avg_per_step) - 0.1), 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Average accuracy plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Continual Learning with LoRA - Evaluation")
    parser.add_argument("--task-stream", type=str, default=None, help="Path to task stream file")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to config YAML")
    parser.add_argument("--lora-dir", type=str, default=None, help="LoRA save directory (override config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (override config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--re-eval", action="store_true",
                        help="Force re-evaluation from LoRA modules (ignore training_progress.json)")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Limit validation samples per task (for smoke tests)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Decide if we will need the full training stack (torch/transformers/peft/datasets).
    # We can load matrix directly from training_progress.json without heavy imports.
    _tmp_cfg = load_config(args.config)
    _tmp_lora_dir = args.lora_dir or _tmp_cfg["output"]["lora_save_dir"]
    _tmp_progress = os.path.join(_tmp_lora_dir, "training_progress.json")
    _need_training_stack = args.re_eval or not os.path.exists(_tmp_progress)

    _ts = None
    torch = None
    if _need_training_stack:
        print("[Eval] Re-eval mode: loading torch/transformers/peft/datasets (may take a moment)...")
        _ts = _lazy_import_training_stack()
        torch = _ts["torch"]
        # _to_json_serializable has a local definition (lora_manager-free); don't overwrite
        globals().update({k: _ts[k] for k in _ts if k not in ("torch", "_to_json_serializable")})

    device = "cpu" if args.no_cuda or (torch is not None and not torch.cuda.is_available()) else "cpu"
    print(f"[Eval] Using device: {device}")

    config = load_config(args.config)

    if args.task_stream and os.path.exists(args.task_stream):
        tasks = parse_task_stream(args.task_stream)
    else:
        tasks = config.get("tasks", [])

    if len(tasks) == 0:
        print("[Eval] ERROR: No tasks defined!")
        sys.exit(1)

    task_name_to_info = {t["name"]: t for t in tasks}
    all_task_names = [t["name"] for t in tasks]

    model_cfg = config["model"]
    training_cfg = config["training"]
    output_cfg = config["output"]

    for k in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"):
        if k in training_cfg and isinstance(training_cfg[k], str):
            training_cfg[k] = float(training_cfg[k])
    for k in ("batch_size", "max_seq_length", "gradient_accumulation_steps"):
        if k in training_cfg and isinstance(training_cfg[k], str):
            training_cfg[k] = int(training_cfg[k])

    lora_save_dir = args.lora_dir or output_cfg["lora_save_dir"]
    eval_output_dir = args.output_dir or output_cfg.get("eval_output_dir", "./eval_results")
    os.makedirs(eval_output_dir, exist_ok=True)

    task_order: List[str] = []
    task_matrix: Optional[np.ndarray] = None
    progress_source = ""

    progress_path = os.path.join(lora_save_dir, "training_progress.json")
    if os.path.exists(progress_path) and not args.re_eval:
        try:
            with open(progress_path, "r") as f:
                prog = json.load(f)
            task_order = prog.get("task_order", [])
            rows = prog.get("progress_rows", [])
            if len(task_order) > 0 and len(rows) == len(task_order):
                n = len(task_order)
                task_matrix = np.zeros((n, n), dtype=np.float64)
                for i, train_task in enumerate(task_order):
                    row = rows[i]
                    for j, test_task in enumerate(task_order):
                        v = row.get(test_task, float("nan"))
                        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                            task_matrix[i, j] = 0.0
                        else:
                            task_matrix[i, j] = float(v)
                progress_source = f"training_progress.json ({len(task_order)} rows)"
                print(f"[Eval] Loaded progress matrix from {progress_path}")
        except Exception as e:
            print(f"[Eval] Warning: failed to load progress json: {e}, will re-eval from LoRAs")
            task_matrix = None

    if task_matrix is None:
        print(f"[Eval] Loading LoRA modules from {lora_save_dir} to re-evaluate...")
        print(f"[Eval] Loading tokenizer and base model: {model_cfg['name_or_path']}")
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], use_fast=False)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["name_or_path"],
            num_labels=model_cfg["num_labels"],
            ignore_mismatched_sizes=True,
        ).to(device)

        lora_manager = LoRAManager(base_model, save_dir=lora_save_dir, device=device)
        loaded = lora_manager.load_all_saved_loras()
        print(f"[Eval] Loaded {len(loaded)} LoRA modules: {loaded}")

        if len(loaded) == 0:
            print("[Eval] ERROR: No LoRA modules found!")
            sys.exit(1)

        registered = lora_manager.get_registered_tasks()
        task_order = [t for t in all_task_names if t in registered]
        if len(task_order) == 0:
            task_order = sorted(registered)
        print(f"[Eval] Task order: {task_order}")

        print(f"\n[Eval] Loading validation datasets for {len(task_order)} tasks...")
        val_dataloaders: Dict[str, DataLoader] = {}
        collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="max_length",
            max_length=training_cfg["max_seq_length"],
        )

        for task_name in tqdm(task_order, desc="Loading val datasets"):
            if task_name not in task_name_to_info:
                continue
            task_info = task_name_to_info[task_name]
            num_labels = task_info.get("num_labels", model_cfg["num_labels"])
            try:
                val_ds = load_task_dataset(
                    task_info["dataset"], tokenizer,
                    training_cfg["max_seq_length"], num_labels,
                    max_val_samples=args.max_val_samples,
                )
                if val_ds is not None:
                    val_ds.set_format("torch")
                    val_dataloaders[task_name] = DataLoader(
                        val_ds,
                        batch_size=training_cfg["batch_size"],
                        shuffle=False,
                        collate_fn=collator,
                    )
            except Exception as e:
                print(f"\n[Eval] Warning: Failed to load val dataset for '{task_name}': {e}")

        if len(val_dataloaders) == 0:
            print("[Eval] ERROR: No validation datasets available!")
            sys.exit(1)

        print(f"\n[Eval] Running evaluation: {len(task_order)} training_steps x {len(val_dataloaders)} tasks...")
        n = len(task_order)
        task_matrix = np.zeros((n, n), dtype=np.float64)

        for i, train_task in enumerate(tqdm(task_order, desc="After training step")):
            if train_task not in registered:
                continue
            task_info = task_name_to_info.get(train_task, tasks[0])
            num_labels = task_info.get("num_labels", model_cfg["num_labels"])

            eval_cfg = AutoConfig.from_pretrained(model_cfg["name_or_path"], num_labels=num_labels)
            eval_model = AutoModelForSequenceClassification.from_pretrained(
                model_cfg["name_or_path"], config=eval_cfg, ignore_mismatched_sizes=True,
            ).to(device)

            lora_module = lora_manager.registry[train_task]
            peft_model = get_peft_model(eval_model, lora_module.lora_config)
            if lora_module.state_dict is not None:
                try:
                    set_peft_model_state_dict(peft_model, lora_module.state_dict)
                except Exception as e:
                    print(f"[Eval] Warn: load state for {train_task} failed: {e}")
            peft_model.eval()

            for j, test_task in enumerate(task_order):
                if test_task not in val_dataloaders:
                    task_matrix[i, j] = 0.0
                    continue
                test_info = task_name_to_info.get(test_task, tasks[0])
                test_labels = test_info.get("num_labels", model_cfg["num_labels"])
                if test_labels == num_labels:
                    ref = peft_model
                else:
                    tmp_cfg = AutoConfig.from_pretrained(model_cfg["name_or_path"], num_labels=test_labels)
                    tmp_m = AutoModelForSequenceClassification.from_pretrained(
                        model_cfg["name_or_path"], config=tmp_cfg, ignore_mismatched_sizes=True,
                    ).to(device)
                    ref = get_peft_model(tmp_m, lora_module.lora_config)
                    if lora_module.state_dict is not None:
                        try:
                            set_peft_model_state_dict(ref, lora_module.state_dict)
                        except Exception:
                            pass
                    ref.eval()

                task_matrix[i, j] = evaluate_single_task(ref, val_dataloaders[test_task], device)

                if test_labels != num_labels:
                    del ref
                    torch.cuda.empty_cache() if device == "cuda" else None

            del peft_model, eval_model
            torch.cuda.empty_cache() if device == "cuda" else None

        progress_source = f"Re-evaluated from {len(loaded)} LoRA modules"

    print(f"\n{'='*60}")
    print(f"[Eval] Task Accuracy Matrix ({progress_source})")
    print(f"  ROW i = after completing training on task_order[i]")
    print(f"  COL j = accuracy evaluated on task_order[j] validation set")
    print(f"{'='*60}")
    header = f"{'After Step':<15} " + " ".join(f"{t:>10}" for t in task_order)
    print(header)
    print("-" * len(header))
    for i, train_task in enumerate(task_order):
        row_str = f"{'  '+train_task:<15} " + " ".join(
            f"{task_matrix[i, j]:>10.4f}" for j in range(len(task_order))
        )
        print(row_str)

    matrix_path = os.path.join(eval_output_dir, "task_matrix.npy")
    np.save(matrix_path, task_matrix)
    accuracies: Dict[str, Dict[str, float]] = {}
    for i, tr in enumerate(task_order):
        accuracies[tr] = {}
        for j, te in enumerate(task_order):
            accuracies[tr][te] = float(task_matrix[i, j])

    json_path = os.path.join(eval_output_dir, "accuracies.json")
    with open(json_path, "w") as f:
        json.dump(_to_json_serializable({
            "task_order": task_order,
            "accuracies": accuracies,
            "progress_source": progress_source,
        }), f, indent=2)

    print(f"\n[Eval] Numerical results saved to:")
    print(f"  - Matrix (npy): {matrix_path}")
    print(f"  - JSON:          {json_path}")

    plot_task_matrix(task_matrix, task_order, os.path.join(eval_output_dir, "task_matrix.png"))
    forgetting_dict, avg_forgetting = plot_forgetting_curves(
        task_matrix, task_order, os.path.join(eval_output_dir, "forgetting_curves.png")
    )
    plot_avg_accuracy(task_matrix, task_order, os.path.join(eval_output_dir, "avg_accuracy.png"))

    print(f"\n{'='*60}")
    print("[Eval] Forgetting Measure per Task (init_perf - final_perf):")
    for tn, f_val in forgetting_dict.items():
        print(f"  - {tn}: forgetting = {f_val:.4f}")
    print(f"\n  Average Forgetting across tasks: {avg_forgetting:.4f}")
    print(f"{'='*60}")

    summary = {
        "task_order": task_order,
        "avg_forgetting": float(avg_forgetting),
        "forgetting_per_task": {k: float(v) for k, v in forgetting_dict.items()},
        "avg_accuracy_final": float(np.nanmean(task_matrix[-1, :]) if task_matrix.shape[0] > 0 else 0.0),
        "avg_accuracy_seen_tasks": [
            float(np.nanmean(task_matrix[i, : i + 1])) for i in range(len(task_order))
        ],
        "progress_source": progress_source,
    }
    summary_path = os.path.join(eval_output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(_to_json_serializable(summary), f, indent=2)
    print(f"\n[Eval] Summary saved to {summary_path}")
    print("[Eval] Done!")


if __name__ == "__main__":
    main()
