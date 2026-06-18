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

from lora_manager import LoRAManager


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

    if "test" in dataset:
        test_key = "test"
    elif "test_matched" in dataset:
        test_key = "test_matched"
    else:
        test_key = None

    train_sent_keys, train_label_key = determine_keys("train")
    train_ds = dataset["train"].map(
        lambda x: preprocess(x, train_sent_keys, train_label_key),
        batched=True,
        remove_columns=[c for c in dataset["train"].column_names if c not in ("input_ids", "attention_mask", "labels")],
    )

    val_ds = None
    if val_key:
        val_sent_keys, val_label_key = determine_keys(val_key)
        val_ds = dataset[val_key].map(
            lambda x: preprocess(x, val_sent_keys, val_label_key),
            batched=True,
            remove_columns=[c for c in dataset[val_key].column_names if c not in ("input_ids", "attention_mask", "labels")],
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


def extract_gradients(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: str,
    max_batches: int = 5,
) -> List[torch.Tensor]:
    model.eval()
    model.zero_grad()

    params_for_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and "classifier" in name:
            params_for_grad.append(p)
        elif p.requires_grad and ("score" in name and "lora" not in name):
            params_for_grad.append(p)

    if len(params_for_grad) == 0:
        for name, p in model.named_parameters():
            if p.requires_grad and "lora" in name:
                params_for_grad.append(p)

    accumulated_grads = [torch.zeros_like(p.detach(), device="cpu") for p in params_for_grad]
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
            for i, p in enumerate(params_for_grad):
                if p.grad is not None:
                    accumulated_grads[i] += p.grad.detach().cpu()
                else:
                    valid = False

            if valid:
                num_valid += 1
            batch_idx += 1

    if num_valid > 0:
        accumulated_grads = [g / num_valid for g in accumulated_grads]

    return [g for g in accumulated_grads if g.abs().sum() > 0]


def gradients_to_list(grads: List[torch.Tensor]) -> List[List[float]]:
    return [g.cpu().numpy().tolist() for g in grads]


def list_to_gradients(grad_list: List[List[float]]) -> List[torch.Tensor]:
    return [torch.tensor(np.array(g), dtype=torch.float32) for g in grad_list]


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

    lora_save_dir = output_cfg["lora_save_dir"]
    os.makedirs(lora_save_dir, exist_ok=True)

    print(f"\n[CL-Solver] Loading base model: {model_cfg['name_or_path']}")
    hf_config = AutoConfig.from_pretrained(model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name_or_path"],
        num_labels=model_cfg["num_labels"],
        ignore_mismatched_sizes=True,
    ).to(device)

    lora_manager = LoRAManager(base_model, save_dir=lora_save_dir, device=device)
    loaded_existing = lora_manager.load_all_saved_loras()
    if loaded_existing:
        print(f"[CL-Solver] Loaded {len(loaded_existing)} existing LoRA modules: {loaded_existing}")
    lora_manager.load_replay_buffer()

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

    performance_history: Dict[str, List[float]] = {}
    task_order = []

    for t_idx in range(start_idx, len(tasks)):
        task_info = tasks[t_idx]
        task_name = task_info["name"]
        dataset_spec = task_info["dataset"]
        num_epochs = task_info["epochs"]
        num_labels = task_info.get("num_labels", model_cfg["num_labels"])

        print(f"\n{'='*60}")
        print(f"[CL-Solver] Task {t_idx}/{len(tasks)}: '{task_name}' (dataset={dataset_spec}, epochs={num_epochs})")
        print(f"{'='*60}")
        task_order.append(task_name)

        try:
            train_ds, val_ds = load_task_dataset(
                dataset_spec, tokenizer,
                training_cfg["max_seq_length"],
                num_labels,
            )
        except Exception as e:
            print(f"[CL-Solver] ERROR loading dataset for '{task_name}': {e}")
            continue

        train_ds.set_format("torch")
        if val_ds:
            val_ds.set_format("torch")

        existing_task_names = lora_manager.get_registered_tasks()
        similar_tasks = []
        initialized_from_history = False

        if len(existing_task_names) > 0 and val_ds is not None:
            tmp_config = AutoConfig.from_pretrained(model_cfg["name_or_path"], num_labels=num_labels)
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

            collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=training_cfg["max_seq_length"])
            tmp_val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"], collate_fn=collator, shuffle=False)

            print(f"[CL-Solver] Computing gradient signature for '{task_name}'...")
            new_grads = extract_gradients(tmp_peft, tmp_val_loader, device)
            del tmp_model, tmp_peft
            torch.cuda.empty_cache() if device == "cuda" else None

            similar_tasks = lora_manager.compute_task_similarity(
                task_name, new_grads, top_k=cl_cfg["top_k_similar"]
            )

            if len(similar_tasks) > 0:
                print(f"[CL-Solver] Top-{cl_cfg['top_k_similar']} similar tasks:")
                for st_name, st_score in similar_tasks:
                    print(f"    - {st_name}: cosine_sim={st_score:.4f}")
                initialized_from_history = True

        current_config = AutoConfig.from_pretrained(model_cfg["name_or_path"], num_labels=num_labels)
        current_model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["name_or_path"],
            config=current_config,
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
                print(f"[CL-Solver] Warning: Failed to apply init state: {e}")

        peft_model.print_trainable_parameters()

        all_train_ds = train_ds
        if len(lora_manager.replay_buffer) > 0 and cl_cfg.get("replay_ratio", 0) > 0:
            replay_samples = lora_manager.replay_buffer.get_all_samples()
            replay_count = int(len(train_ds) * cl_cfg["replay_ratio"])
            replay_count = min(replay_count, len(replay_samples))
            if replay_count > 0:
                replay_indices = np.random.choice(len(replay_samples), replay_count, replace=False)
                selected_replays = [replay_samples[i] for i in replay_indices]
                replay_ds = ReplayDatasetWrapper(selected_replays)
                all_train_ds = ConcatDataset([train_ds, replay_ds])
                print(f"[CL-Solver] Mixed {replay_count} replay samples into training data (total={len(all_train_ds)})")

        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=training_cfg["max_seq_length"])
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

        print(f"[CL-Solver] Training '{task_name}' for {num_epochs} epochs (total steps={total_train_steps})")
        peft_model, final_val_acc = train_one_task(
            task_name, peft_model, train_loader, val_loader,
            optimizer, lr_scheduler, num_epochs, device,
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        )
        print(f"[CL-Solver] '{task_name}' best Val-Acc: {final_val_acc:.4f}")

        performance_history[task_name] = [final_val_acc]

        task_grads = None
        if val_loader is not None:
            task_grads = extract_gradients(peft_model, val_loader, device)

        lora_manager.register_lora(task_name, base_lora_config, metadata={
            "val_accuracy": final_val_acc,
            "num_epochs": num_epochs,
            "dataset": dataset_spec,
        })
        lora_manager.save_lora(task_name, peft_model)

        if task_grads is not None:
            lora_manager.registry[task_name].metadata["gradients"] = gradients_to_list(task_grads)
            lora_manager.save_lora(task_name)

        old_rank, new_rank = lora_manager.prune_lora(
            task_name,
            energy_threshold=cl_cfg["prune_energy_threshold"],
            singular_value_threshold=cl_cfg["singular_value_threshold"],
        )
        lora_manager.save_lora(task_name)

        if val_ds is not None:
            lora_manager.add_to_replay_buffer(
                val_ds, task_name,
                num_samples=cl_cfg["replay_buffer_size_per_task"],
            )
            lora_manager.save_replay_buffer()

        lora_manager.save_registry_index()

        del peft_model, current_model
        torch.cuda.empty_cache() if device == "cuda" else None

        print(f"\n[CL-Solver] Current registry: {lora_manager.get_registered_tasks()}")
        for t_name in lora_manager.get_registered_tasks():
            mod = lora_manager.registry[t_name]
            print(f"    - {t_name}: rank={mod.current_rank}, acc={mod.metadata.get('val_accuracy', 'N/A')}")

    print(f"\n{'='*60}")
    print(f"[CL-Solver] All tasks completed! Total registered: {len(lora_manager.get_registered_tasks())}")
    print(f"{'='*60}")

    history_path = os.path.join(lora_save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "task_order": task_order,
            "performance_history": performance_history,
            "device": device,
            "seed": args.seed,
        }, f, indent=2, default=str)
    print(f"[CL-Solver] Training history saved to {history_path}")


if __name__ == "__main__":
    main()
