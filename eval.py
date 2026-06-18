import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType

from datasets import load_dataset

from lora_manager import LoRAManager


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    val_sent_keys, val_label_key = determine_keys(val_key)
    val_ds = dataset[val_key].map(
        lambda x: preprocess(x, val_sent_keys, val_label_key),
        batched=True,
        remove_columns=[c for c in dataset[val_key].column_names if c not in ("input_ids", "attention_mask", "labels")],
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
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

    model_cfg = config["model"]
    training_cfg = config["training"]
    output_cfg = config["output"]

    lora_save_dir = args.lora_dir or output_cfg["lora_save_dir"]
    eval_output_dir = args.output_dir or output_cfg.get("eval_output_dir", "./eval_results")
    os.makedirs(eval_output_dir, exist_ok=True)

    print(f"[Eval] Loading base model: {model_cfg['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])

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
    task_order = [t["name"] for t in tasks if t["name"] in registered]
    if len(task_order) == 0:
        task_order = sorted(registered)
    print(f"[Eval] Task order for evaluation: {task_order}")

    print(f"\n[Eval] Loading validation datasets for {len(task_order)} tasks...")
    val_datasets = {}
    val_dataloaders = {}
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=training_cfg["max_seq_length"])

    for task_name in tqdm(task_order, desc="Loading datasets"):
        if task_name not in task_name_to_info:
            continue
        task_info = task_name_to_info[task_name]
        num_labels = task_info.get("num_labels", model_cfg["num_labels"])
        try:
            val_ds = load_task_dataset(
                task_info["dataset"], tokenizer,
                training_cfg["max_seq_length"], num_labels,
            )
            if val_ds is not None:
                val_ds.set_format("torch")
                val_datasets[task_name] = val_ds
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

    print(f"\n[Eval] Running evaluation: {len(task_order)} LoRA x {len(val_dataloaders)} tasks...")
    accuracies: Dict[str, Dict[str, float]] = {}

    for train_task in tqdm(task_order, desc="Evaluating trained models"):
        if train_task not in registered:
            continue
        task_info = task_name_to_info.get(train_task, tasks[0])
        num_labels = task_info.get("num_labels", model_cfg["num_labels"])

        eval_config = AutoConfig.from_pretrained(model_cfg["name_or_path"], num_labels=num_labels)
        eval_model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["name_or_path"],
            config=eval_config,
            ignore_mismatched_sizes=True,
        ).to(device)

        lora_module = lora_manager.registry[train_task]
        peft_model = get_peft_model(eval_model, lora_module.lora_config)
        if lora_module.state_dict is not None:
            set_peft_model_state_dict(peft_model, lora_module.state_dict)
        peft_model.eval()

        accuracies[train_task] = {}
        for test_task in task_order:
            if test_task not in val_dataloaders:
                accuracies[train_task][test_task] = 0.0
                continue
            acc = evaluate_single_task(peft_model, val_dataloaders[test_task], device)
            accuracies[train_task][test_task] = acc

        del peft_model, eval_model
        torch.cuda.empty_cache() if device == "cuda" else None

    task_matrix = build_task_matrix(tasks, task_order, accuracies)

    print(f"\n{'='*60}")
    print("[Eval] Task Accuracy Matrix (rows=training step, cols=test task)")
    print(f"{'='*60}")
    header = f"{'Train\\Test':<15} " + " ".join(f"{t:>10}" for t in task_order)
    print(header)
    print("-" * len(header))
    for i, train_task in enumerate(task_order):
        row = f"{train_task:<15} " + " ".join(f"{task_matrix[i, j]:>10.4f}" for j in range(len(task_order)))
        print(row)

    matrix_path = os.path.join(eval_output_dir, "task_matrix.npy")
    np.save(matrix_path, task_matrix)
    json_path = os.path.join(eval_output_dir, "accuracies.json")
    with open(json_path, "w") as f:
        json.dump({"task_order": task_order, "accuracies": accuracies}, f, indent=2)

    print(f"\n[Eval] Numerical results saved to:")
    print(f"  - Matrix: {matrix_path}")
    print(f"  - JSON:   {json_path}")

    plot_task_matrix(task_matrix, task_order, os.path.join(eval_output_dir, "task_matrix.png"))
    forgetting_dict, avg_forgetting = plot_forgetting_curves(
        task_matrix, task_order, os.path.join(eval_output_dir, "forgetting_curves.png")
    )
    plot_avg_accuracy(task_matrix, task_order, os.path.join(eval_output_dir, "avg_accuracy.png"))

    print(f"\n{'='*60}")
    print("[Eval] Forgetting Measure per Task:")
    for tn, f_val in forgetting_dict.items():
        print(f"  - {tn}: forgetting = {f_val:.4f}")
    print(f"\n  Average Forgetting: {avg_forgetting:.4f}")
    print(f"{'='*60}")

    summary = {
        "task_order": task_order,
        "avg_forgetting": avg_forgetting,
        "forgetting_per_task": forgetting_dict,
        "avg_accuracy_final": float(task_matrix[-1, :].mean()),
        "avg_accuracy_seen_tasks": [float(task_matrix[i, : i + 1].mean()) for i in range(len(task_order))],
    }
    summary_path = os.path.join(eval_output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Eval] Summary saved to {summary_path}")
    print("[Eval] Done!")


if __name__ == "__main__":
    main()
