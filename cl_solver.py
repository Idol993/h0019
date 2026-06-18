import os
import sys
import argparse
import yaml
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_scheduler,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType

from datasets import load_dataset, concatenate_datasets

from lora_manager import LoRAManager, _to_json_serializable


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    max_train_samples: Optional[int] = None,
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
        sent_keys = [k for k in keys if k in ("sentence", "sentence1", "question", "premise", "text", "context")]
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

    if "validation" in dataset:
        val_key = "validation"
    elif "validation_matched" in dataset:
        val_key = "validation_matched"
    else:
        val_key = None

    train_split = dataset["train"]
    if max_train_samples and max_train_samples < len(train_split):
        train_split = train_split.select(range(max_train_samples))

    train_sent_keys, train_label_key = determine_keys("train")
    train_ds = train_split.map(
        lambda x: preprocess(x, train_sent_keys, train_label_key),
        batched=True,
        remove_columns=[c for c in train_split.column_names if c not in ("input_ids", "attention_mask", "labels")],
    )

    val_ds = None
    if val_key:
        val_split = dataset[val_key]
        if max_val_samples and max_val_samples < len(val_split):
            val_split = val_split.select(range(max_val_samples))
        val_sent_keys, val_label_key = determine_keys(val_key)
        val_ds = val_split.map(
            lambda x: preprocess(x, val_sent_keys, val_label_key),
            batched=True,
            remove_columns=[c for c in val_split.column_names if c not in ("input_ids", "attention_mask", "labels")],
        )

    return train_ds, val_ds


class ReplayDatasetWrapper(Dataset):
    def __init__(self, replay_samples: List[Dict[str, Any]]):
        self.samples = replay_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if not isinstance(item, dict):
            return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([]), "labels": torch.tensor(0)}
        result = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                result[k] = v
            elif isinstance(v, list):
                try:
                    result[k] = torch.tensor(v, dtype=torch.long if k in ("input_ids", "labels") else torch.float32)
                except Exception:
                    result[k] = v
            else:
                result[k] = v
        return result


def compute_accuracy(preds: List[int], labels: List[int]) -> float:
    if len(preds) == 0:
        return 0.0
    return float((np.array(preds) == np.array(labels)).mean())


def extract_lora_gradients(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: str,
    max_batches: int = 5,
) -> List[np.ndarray]:
    model.eval()
    model.zero_grad()

    lora_params: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            lora_params.append((name, p))

    if len(lora_params) == 0:
        print("[extract_lora_gradients] WARNING: no LoRA params found, falling back to all trainable params")
        for name, p in model.named_parameters():
            if p.requires_grad:
                lora_params.append((name, p))

    accumulated: List[torch.Tensor] = [
        torch.zeros_like(p.detach(), device="cpu") for _, p in lora_params
    ]
    num_valid = 0

    loss_fn = nn.CrossEntropyLoss()
    batch_idx = 0

    with torch.enable_grad():
        for batch in val_dataloader:
            if batch_idx >= max_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "labels" not in batch:
                continue

            model.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            loss.backward()

            valid = True
            for i, (_, p) in enumerate(lora_params):
                if p.grad is None:
                    valid = False
                    break
                accumulated[i] += p.grad.detach().cpu()

            if valid:
                num_valid += 1
            batch_idx += 1

    if num_valid == 0:
        return []
    return [(g / num_valid).cpu().numpy() for g in accumulated]


def evaluate_all_tasks_from_model(
    peft_model: nn.Module,
    task_name_to_val_loader: Dict[str, DataLoader],
    device: str,
) -> Dict[str, float]:
    peft_model.eval()
    results: Dict[str, float] = {}
    for test_task_name, loader in task_name_to_val_loader.items():
        all_preds: List[int] = []
        all_labels: List[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if "labels" not in batch:
                    continue
                outputs = peft_model(**batch)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                labels = batch["labels"].cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
        results[test_task_name] = compute_accuracy(all_preds, all_labels)
    return results


def train_one_task(
    task_name: str,
    model,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    optimizer,
    lr_scheduler,
    num_epochs: int,
    device: str,
    max_grad_norm: float = 1.0,
) -> Tuple[nn.Module, float]:
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(train_dataloader, desc=f"  Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "labels" not in batch:
                continue

            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, num_batches)
        val_acc = 0.0

        if val_dataloader:
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
            val_acc = compute_accuracy(all_preds, all_labels)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

        print(f"  [Epoch {epoch+1}/{num_epochs}] Loss={avg_loss:.4f}, Val-Acc={val_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Continual Learning with LoRA - Training Scheduler")
    parser.add_argument("--task-stream", type=str, default=None, help="Path to task stream file (tasks.txt)")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to config YAML file")
    parser.add_argument("--resume-from", type=int, default=None, help="Resume from task index (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--max-train-samples", type=int, default=None, help="SMOKE: limit train samples per task")
    parser.add_argument("--max-val-samples", type=int, default=None, help="SMOKE: limit val samples per task")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"[CL-Solver] Using device: {device}")

    config = load_config(args.config)
    print(f"[CL-Solver] Loaded config from {args.config}")

    if args.task_stream and os.path.exists(args.task_stream):
        tasks = parse_task_stream(args.task_stream)
        print(f"[CL-Solver] Loaded {len(tasks)} tasks from task stream file")
    else:
        tasks = config.get("tasks", [])
        print(f"[CL-Solver] Using {len(tasks)} tasks from config")

    if len(tasks) == 0:
        print("[CL-Solver] ERROR: No tasks defined!")
        sys.exit(1)

    model_cfg = config["model"]
    training_cfg = config["training"]
    lora_cfg = config["lora"]
    cl_cfg = config["continual_learning"]
    output_cfg = config["output"]
    resume_cfg = config.get("resume", {})

    # --- normalize numeric configs (YAML parser can leave them as str on some versions) ---
    for k in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"):
        if k in training_cfg and isinstance(training_cfg[k], str):
            training_cfg[k] = float(training_cfg[k])
    for k in ("batch_size", "max_seq_length", "gradient_accumulation_steps"):
        if k in training_cfg and isinstance(training_cfg[k], str):
            training_cfg[k] = int(training_cfg[k])
    for k in ("initial_rank", "lora_alpha"):
        if k in lora_cfg and isinstance(lora_cfg[k], str):
            lora_cfg[k] = int(lora_cfg[k])
    for k in ("lora_dropout",):
        if k in lora_cfg and isinstance(lora_cfg[k], str):
            lora_cfg[k] = float(lora_cfg[k])
    for k in ("top_k_similar", "replay_buffer_size_per_task"):
        if k in cl_cfg and isinstance(cl_cfg[k], str):
            cl_cfg[k] = int(cl_cfg[k])
    for k in ("prune_energy_threshold", "singular_value_threshold", "replay_ratio"):
        if k in cl_cfg and isinstance(cl_cfg[k], str):
            cl_cfg[k] = float(cl_cfg[k])

    lora_save_dir = output_cfg["lora_save_dir"]
    os.makedirs(lora_save_dir, exist_ok=True)
    eval_output_dir = output_cfg.get("eval_output_dir", "./eval_results")
    os.makedirs(eval_output_dir, exist_ok=True)

    print(f"\n[CL-Solver] Loading tokenizer: {model_cfg['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], use_fast=False)

    print(f"\n[CL-Solver] Pre-loading validation datasets for ALL tasks...")
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=training_cfg["max_seq_length"],
    )

    all_task_val_loaders: Dict[str, DataLoader] = {}
    all_task_info: Dict[str, Dict[str, Any]] = {}
    for t in tasks:
        tname = t["name"]
        num_labels = t.get("num_labels", model_cfg["num_labels"])
        all_task_info[tname] = {
            "dataset": t["dataset"],
            "num_labels": num_labels,
            "epochs": t["epochs"],
        }
        try:
            _, val_ds = load_task_dataset(
                t["dataset"], tokenizer,
                training_cfg["max_seq_length"], num_labels,
                max_val_samples=args.max_val_samples,
            )
            if val_ds is not None:
                if args.max_val_samples:
                    val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))
                val_ds.set_format("torch")
                all_task_val_loaders[tname] = DataLoader(
                    val_ds,
                    batch_size=training_cfg["batch_size"],
                    shuffle=False,
                    collate_fn=collator,
                )
                print(f"  - {tname}: {len(val_ds)} val samples, num_labels={num_labels}")
            else:
                print(f"  - {tname}: no val split found")
        except Exception as e:
            print(f"  - {tname}: FAILED to load ({e})")

    print(f"[CL-Solver] Preloaded val loaders for {len(all_task_val_loaders)} tasks")

    tmp_config = AutoConfig.from_pretrained(model_cfg["name_or_path"])
    dummy_base = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name_or_path"],
        num_labels=model_cfg["num_labels"],
        ignore_mismatched_sizes=True,
    )
    lora_manager = LoRAManager(dummy_base, save_dir=lora_save_dir, device=device)
    loaded_existing = lora_manager.load_all_saved_loras()
    if loaded_existing:
        print(f"[CL-Solver] Loaded {len(loaded_existing)} existing LoRA modules: {loaded_existing}")
    lora_manager.load_replay_buffer()
    del dummy_base

    resume_from_task = args.resume_from if args.resume_from is not None else resume_cfg.get("resume_from_task", -1)
    if resume_from_task == -1:
        registered = lora_manager.get_registered_tasks()
        start_idx = 0
        for t_idx, t in enumerate(tasks):
            if t["name"] in registered:
                start_idx = t_idx + 1
    else:
        start_idx = max(0, resume_from_task)

    print(f"[CL-Solver] Starting from task index {start_idx} (total {len(tasks)} tasks)")

    progress_rows: List[Dict[str, float]] = []
    task_order: List[str] = []
    progress_path = os.path.join(lora_save_dir, "training_progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r") as f:
                prev = json.load(f)
            progress_rows = prev.get("progress_rows", [])
            task_order = prev.get("task_order", [])
            print(f"[CL-Solver] Resumed progress: {len(progress_rows)} rows already saved")
        except Exception:
            pass

    for t_idx in range(start_idx, len(tasks)):
        task_info = tasks[t_idx]
        task_name = task_info["name"]
        dataset_spec = task_info["dataset"]
        num_epochs = task_info["epochs"]
        num_labels = task_info.get("num_labels", model_cfg["num_labels"])

        print(f"\n{'='*60}")
        print(f"[CL-Solver] Task {t_idx}/{len(tasks)}: '{task_name}' (dataset={dataset_spec}, epochs={num_epochs}, num_labels={num_labels})")
        print(f"{'='*60}")
        task_order.append(task_name)

        try:
            train_ds, val_ds = load_task_dataset(
                dataset_spec, tokenizer,
                training_cfg["max_seq_length"],
                num_labels,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
            )
        except Exception as e:
            print(f"[CL-Solver] ERROR loading dataset for '{task_name}': {e}")
            continue

        if train_ds is None:
            print(f"[CL-Solver] ERROR: no train split for '{task_name}'")
            continue

        if args.max_train_samples:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
            print(f"[CL-Solver] SMOKE: limited train samples to {len(train_ds)}")

        train_ds.set_format("torch")
        if val_ds:
            if args.max_val_samples:
                val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))
            val_ds.set_format("torch")

        existing_task_names = lora_manager.get_registered_tasks()
        similar_tasks: List[Tuple[str, float]] = []
        initialized_from_history = False

        if len(existing_task_names) > 0 and val_ds is not None:
            tmp_model = AutoModelForSequenceClassification.from_pretrained(
                model_cfg["name_or_path"],
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            ).to(device)

            tmp_lora_config = LoraConfig(
                r=lora_cfg["initial_rank"],
                lora_alpha=lora_cfg.get("lora_alpha", lora_cfg["initial_rank"] * 2),
                lora_dropout=lora_cfg.get("lora_dropout", 0.1),
                target_modules=lora_cfg.get("target_modules", ["query", "value"]),
                bias=lora_cfg.get("bias", "none"),
                task_type=TaskType.SEQ_CLS,
            )
            tmp_peft = get_peft_model(tmp_model, tmp_lora_config)

            tmp_val_loader = DataLoader(
                val_ds, batch_size=training_cfg["batch_size"],
                collate_fn=collator, shuffle=False,
            )

            print(f"[CL-Solver] Computing LoRA gradient signature for '{task_name}'...")
            new_grads = extract_lora_gradients(tmp_peft, tmp_val_loader, device, max_batches=3)
            del tmp_model, tmp_peft
            torch.cuda.empty_cache() if device == "cuda" else None

            if len(new_grads) > 0:
                similar_tasks = lora_manager.compute_task_similarity(
                    task_name, new_grads, top_k=cl_cfg["top_k_similar"]
                )
                if len(similar_tasks) > 0:
                    print(f"[CL-Solver] Top-{min(cl_cfg['top_k_similar'], len(similar_tasks))} similar tasks:")
                    for st_name, st_score in similar_tasks:
                        print(f"    - {st_name}: cosine_sim={st_score:.4f}")
                    initialized_from_history = True
                else:
                    print("[CL-Solver] No comparable historical signatures found -> init from scratch")
            else:
                print("[CL-Solver] Could not extract LoRA gradients -> init from scratch")

        current_model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["name_or_path"],
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        ).to(device)

        base_lora_config = LoraConfig(
            r=lora_cfg["initial_rank"],
            lora_alpha=lora_cfg.get("lora_alpha", lora_cfg["initial_rank"] * 2),
            lora_dropout=lora_cfg.get("lora_dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", ["query", "value"]),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.SEQ_CLS,
        )

        init_state_dict = OrderedDict()
        if initialized_from_history:
            print(f"[CL-Solver] Initializing LoRA from similar tasks...")
            _, init_state_dict = lora_manager.init_lora_from_similar_tasks(base_lora_config, similar_tasks)

        peft_model = get_peft_model(current_model, base_lora_config)
        if len(init_state_dict) > 0:
            try:
                set_peft_model_state_dict(peft_model, init_state_dict)
                print(f"[CL-Solver] Successfully applied historical LoRA initialization")
            except Exception as e:
                print(f"[CL-Solver] Warning: Failed to apply init state, falling back to scratch: {e}")

        peft_model.print_trainable_parameters()

        all_train_ds = train_ds
        if len(lora_manager.replay_buffer) > 0 and cl_cfg.get("replay_ratio", 0) > 0:
            replay_samples = lora_manager.replay_buffer.get_all_samples()
            replay_count = int(max(1, len(train_ds)) * cl_cfg["replay_ratio"])
            replay_count = min(replay_count, len(replay_samples))
            if replay_count > 0:
                replay_indices = np.random.choice(len(replay_samples), replay_count, replace=False)
                selected_replays = [replay_samples[i] for i in replay_indices]
                replay_ds = ReplayDatasetWrapper(selected_replays)
                all_train_ds = ConcatDataset([train_ds, replay_ds])
                print(f"[CL-Solver] Mixed {replay_count} replay samples (total={len(all_train_ds)})")

        train_loader = DataLoader(
            all_train_ds,
            batch_size=training_cfg["batch_size"],
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
        )
        val_loader = None
        if val_ds:
            val_loader = DataLoader(
                val_ds,
                batch_size=training_cfg["batch_size"],
                shuffle=False,
                collate_fn=collator,
            )

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg.get("weight_decay", 0.01),
        )

        num_update_steps_per_epoch = max(1, len(train_loader) // training_cfg.get("gradient_accumulation_steps", 1))
        total_train_steps = num_epochs * num_update_steps_per_epoch
        warmup_steps = int(total_train_steps * training_cfg.get("warmup_ratio", 0.06))
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

        print(f"[CL-Solver] Training '{task_name}' for {num_epochs} epochs (steps={total_train_steps})")
        peft_model, final_val_acc = train_one_task(
            task_name, peft_model, train_loader, val_loader,
            optimizer, lr_scheduler, num_epochs, device,
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        )
        print(f"[CL-Solver] '{task_name}' best Val-Acc on itself: {final_val_acc:.4f}")

        task_grads_np: List[np.ndarray] = []
        if val_loader is not None:
            task_grads_np = extract_lora_gradients(peft_model, val_loader, device, max_batches=3)

        lora_manager.register_lora(task_name, base_lora_config, metadata={
            "val_accuracy": final_val_acc,
            "num_epochs": num_epochs,
            "dataset": dataset_spec,
            "num_labels": num_labels,
            "similar_tasks_init": [(n, float(s)) for n, s in similar_tasks],
        })
        lora_manager.save_lora(task_name, peft_model)

        if len(task_grads_np) > 0:
            sig = lora_manager._gradient_signature_projection(task_grads_np).tolist()
            lora_manager.registry[task_name].metadata["gradient_signature"] = sig
            lora_manager.save_lora(task_name)

        old_rank, new_rank = lora_manager.prune_lora(
            task_name,
            energy_threshold=cl_cfg["prune_energy_threshold"],
            singular_value_threshold=cl_cfg["singular_value_threshold"],
        )
        lora_manager.save_lora(task_name)
        print(f"[CL-Solver] '{task_name}' rank: {old_rank} -> {new_rank}")

        if val_ds is not None:
            lora_manager.add_to_replay_buffer(
                val_ds, task_name,
                num_samples=cl_cfg["replay_buffer_size_per_task"],
            )
            lora_manager.save_replay_buffer()

        lora_manager.save_registry_index()

        print(f"\n[CL-Solver] === PROGRESS SNAPSHOT: evaluating '{task_name}' on all {len(all_task_val_loaders)} tasks ===")
        snapshot_row: Dict[str, float] = {}
        for test_task_name in [t["name"] for t in tasks]:
            if test_task_name not in all_task_val_loaders:
                snapshot_row[test_task_name] = float("nan")
                continue
            test_num_labels = all_task_info[test_task_name]["num_labels"]
            if test_num_labels == num_labels:
                eval_model_ref = peft_model
            else:
                eval_cfg = AutoConfig.from_pretrained(model_cfg["name_or_path"], num_labels=test_num_labels)
                tmp_eval_model = AutoModelForSequenceClassification.from_pretrained(
                    model_cfg["name_or_path"],
                    config=eval_cfg,
                    ignore_mismatched_sizes=True,
                ).to(device)
                lora_mod = lora_manager.registry[task_name]
                eval_model_ref = get_peft_model(tmp_eval_model, lora_mod.lora_config)
                if lora_mod.state_dict is not None:
                    try:
                        set_peft_model_state_dict(eval_model_ref, lora_mod.state_dict)
                    except Exception as e:
                        print(f"    [WARN] load LoRA to {test_task_name} failed: {e}")

            acc = 0.0
            try:
                all_preds: List[int] = []
                all_labels: List[int] = []
                eval_model_ref.eval()
                with torch.no_grad():
                    for batch in all_task_val_loaders[test_task_name]:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        if "labels" not in batch:
                            continue
                        outputs = eval_model_ref(**batch)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                        labels = batch["labels"].cpu().numpy().tolist()
                        all_preds.extend(preds)
                        all_labels.extend(labels)
                acc = compute_accuracy(all_preds, all_labels)
            except Exception as e:
                print(f"    [WARN] eval on {test_task_name} failed: {e}")
            snapshot_row[test_task_name] = acc
            print(f"    {task_name} -> {test_task_name}: acc={acc:.4f}")

            if test_num_labels != num_labels:
                del eval_model_ref
                torch.cuda.empty_cache() if device == "cuda" else None

        progress_rows.append(snapshot_row)

        with open(progress_path, "w") as f:
            json.dump(_to_json_serializable({
                "task_order": task_order,
                "progress_rows": progress_rows,
                "all_task_info": all_task_info,
                "device": device,
                "seed": args.seed,
            }), f, indent=2, allow_nan=True)

        del peft_model, current_model
        torch.cuda.empty_cache() if device == "cuda" else None

        print(f"\n[CL-Solver] Registry status:")
        for t_name in lora_manager.get_registered_tasks():
            mod = lora_manager.registry[t_name]
            print(f"    - {t_name}: rank={mod.current_rank}, acc={mod.metadata.get('val_accuracy', 'N/A')}")

    print(f"\n{'='*60}")
    print(f"[CL-Solver] All tasks completed! Total registered: {len(lora_manager.get_registered_tasks())}")
    print(f"[CL-Solver] Training progress saved to {progress_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
