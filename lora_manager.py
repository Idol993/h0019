import os
import json
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict as get_peft_state


class ReplayBuffer(Dataset):
    def __init__(self):
        self.samples: List[Dict[str, Any]] = []
        self.task_names: List[str] = []

    def add_samples(self, samples: List[Dict[str, Any]], task_name: str):
        self.samples.extend(samples)
        self.task_names.extend([task_name] * len(samples))

    def sample(self, n: int) -> List[Dict[str, Any]]:
        if len(self.samples) == 0:
            return []
        indices = np.random.choice(len(self.samples), min(n, len(self.samples)), replace=False)
        return [self.samples[i] for i in indices]

    def get_all_samples(self) -> List[Dict[str, Any]]:
        return list(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LoRAModule:
    def __init__(
        self,
        task_name: str,
        lora_config: LoraConfig,
        state_dict: Optional[OrderedDict] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_name = task_name
        self.lora_config = lora_config
        self.state_dict = state_dict
        self.metadata = metadata or {}
        self.current_rank = lora_config.r

    def to(self, device: str):
        if self.state_dict is not None:
            self.state_dict = OrderedDict({
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in self.state_dict.items()
            })
        return self


class LoRAManager:
    def __init__(
        self,
        base_model: nn.Module,
        save_dir: str = "./loras",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.base_model = base_model
        self.save_dir = save_dir
        self.device = device
        self.registry: Dict[str, LoRAModule] = OrderedDict()
        self.replay_buffer = ReplayBuffer()
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_task_save_path(self, task_name: str) -> str:
        return os.path.join(self.save_dir, task_name)

    def register_lora(
        self,
        task_name: str,
        lora_config: LoraConfig,
        state_dict: Optional[OrderedDict] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LoRAModule:
        lora_module = LoRAModule(task_name, lora_config, state_dict, metadata)
        self.registry[task_name] = lora_module
        return lora_module

    def save_lora(self, task_name: str, peft_model: Optional[PeftModel] = None):
        if task_name not in self.registry:
            raise ValueError(f"Task {task_name} not registered in LoRA registry")

        task_path = self._get_task_save_path(task_name)
        os.makedirs(task_path, exist_ok=True)

        lora_module = self.registry[task_name]

        if peft_model is not None:
            state_dict = get_peft_state(peft_model)
            lora_module.state_dict = state_dict

        if lora_module.state_dict is None:
            raise ValueError(f"No state dict available for task {task_name}")

        config_path = os.path.join(task_path, "lora_config.json")
        state_path = os.path.join(task_path, "lora_weights.pt")
        meta_path = os.path.join(task_path, "metadata.json")

        with open(config_path, "w") as f:
            json.dump(lora_module.lora_config.to_dict(), f, indent=2)

        state_dict_cpu = OrderedDict({
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in lora_module.state_dict.items()
        })
        torch.save(state_dict_cpu, state_path)

        with open(meta_path, "w") as f:
            json.dump(lora_module.metadata, f, indent=2, default=str)

        print(f"[LoRAManager] Saved LoRA for task '{task_name}' -> {task_path}")

    def load_lora(self, task_name: str) -> LoRAModule:
        task_path = self._get_task_save_path(task_name)
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"LoRA for task '{task_name}' not found at {task_path}")

        config_path = os.path.join(task_path, "lora_config.json")
        state_path = os.path.join(task_path, "lora_weights.pt")
        meta_path = os.path.join(task_path, "metadata.json")

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        lora_config = LoraConfig(**config_dict)

        state_dict = torch.load(state_path, map_location=self.device)
        state_dict = OrderedDict(state_dict)

        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        lora_module = LoRAModule(task_name, lora_config, state_dict, metadata)
        self.registry[task_name] = lora_module
        return lora_module

    def load_all_saved_loras(self) -> List[str]:
        loaded = []
        if not os.path.exists(self.save_dir):
            return loaded
        for task_name in sorted(os.listdir(self.save_dir)):
            task_path = os.path.join(self.save_dir, task_name)
            if os.path.isdir(task_path) and task_name not in self.registry:
                try:
                    self.load_lora(task_name)
                    loaded.append(task_name)
                except Exception as e:
                    print(f"[LoRAManager] Failed to load task '{task_name}': {e}")
        return loaded

    def get_registered_tasks(self) -> List[str]:
        return list(self.registry.keys())

    def apply_lora_to_model(self, task_name: str, model: Optional[nn.Module] = None) -> PeftModel:
        if task_name not in self.registry:
            raise ValueError(f"Task {task_name} not in registry")

        target_model = model if model is not None else self.base_model
        lora_module = self.registry[task_name]

        peft_model = get_peft_model(target_model, lora_module.lora_config)
        if lora_module.state_dict is not None:
            set_peft_model_state_dict(peft_model, lora_module.state_dict)
        return peft_model

    def merge_loras(
        self,
        task_weights: Dict[str, float],
        merged_name: str = "merged",
    ) -> Tuple[LoraConfig, OrderedDict]:
        if len(task_weights) == 0:
            raise ValueError("No tasks specified for merging")

        valid_tasks = {k: v for k, v in task_weights.items() if k in self.registry}
        if len(valid_tasks) == 0:
            raise ValueError("None of the specified tasks are in the registry")

        ref_task = next(iter(valid_tasks.keys()))
        ref_module = self.registry[ref_task]
        base_config_dict = ref_module.lora_config.to_dict()

        total_weight = sum(valid_tasks.values())
        normalized_weights = {k: v / total_weight for k, v in valid_tasks.items()}

        merged_state: OrderedDict = OrderedDict()
        parameter_keys = None

        for task_name, weight in normalized_weights.items():
            lora_module = self.registry[task_name]
            sd = lora_module.state_dict
            if sd is None:
                continue

            if parameter_keys is None:
                parameter_keys = list(sd.keys())

            for key in parameter_keys:
                if key not in sd:
                    continue
                param = sd[key]
                if not isinstance(param, torch.Tensor):
                    continue

                if "lora_A" in key:
                    scaled = param * weight
                elif "lora_B" in key:
                    scaled = param  # B 不缩放，A 缩放后与 B 相乘等价于整体缩放
                else:
                    scaled = param * weight

                scaled = scaled.to(self.device).float()

                if key not in merged_state:
                    merged_state[key] = torch.zeros_like(scaled)
                merged_state[key] = merged_state[key] + scaled

        ref_rank = ref_module.lora_config.r
        merged_config = LoraConfig(**{**base_config_dict, "r": ref_rank})

        merged_module = LoRAModule(merged_name, merged_config, merged_state, {
            "type": "merged",
            "task_weights": normalized_weights,
        })
        self.registry[merged_name] = merged_module

        return merged_config, merged_state

    def prune_lora(
        self,
        task_name: str,
        energy_threshold: float = 0.95,
        singular_value_threshold: float = 0.05,
    ) -> Tuple[int, int]:
        if task_name not in self.registry:
            raise ValueError(f"Task {task_name} not in registry")

        lora_module = self.registry[task_name]
        state_dict = lora_module.state_dict
        if state_dict is None:
            return lora_module.current_rank, lora_module.current_rank

        old_rank = lora_module.current_rank

        lora_pairs = self._group_lora_params(state_dict)
        rank_scores_per_pair = []

        for base_name, (lora_A, lora_B) in lora_pairs.items():
            W_approx = (lora_B.float() @ lora_A.float()).cpu().numpy()
            try:
                U, S, Vh = np.linalg.svd(W_approx, full_matrices=False)
            except np.linalg.LinAlgError:
                rank_scores_per_pair.append(np.ones(old_rank))
                continue

            S_squared = S ** 2
            total_energy = S_squared.sum() + 1e-12
            singular_energy_ratios = S_squared / total_energy

            cumulative = np.cumsum(singular_energy_ratios)
            keep_by_energy = min(len(S), int(np.searchsorted(cumulative, energy_threshold) + 1))

            keep_by_threshold = np.sum(singular_energy_ratios >= singular_value_threshold)
            keep_count = min(keep_by_energy, len(S))

            scores = singular_energy_ratios[:old_rank]
            if len(scores) < old_rank:
                scores = np.pad(scores, (0, old_rank - len(scores)), mode="constant")
            rank_scores_per_pair.append(scores)

        if len(rank_scores_per_pair) == 0:
            return old_rank, old_rank

        avg_scores = np.mean(np.stack(rank_scores_per_pair, axis=0), axis=0)
        cumulative_avg = np.cumsum(avg_scores / (avg_scores.sum() + 1e-12))
        new_rank = max(1, int(np.searchsorted(cumulative_avg, energy_threshold) + 1))
        new_rank = min(new_rank, old_rank)

        if new_rank < old_rank:
            new_state_dict = self._prune_state_dict(state_dict, lora_pairs, new_rank)
            lora_module.state_dict = new_state_dict
            lora_module.lora_config = LoraConfig(**{
                **lora_module.lora_config.to_dict(),
                "r": new_rank,
            })
            lora_module.current_rank = new_rank
            lora_module.metadata["pruned_rank"] = new_rank
            lora_module.metadata["original_rank"] = old_rank
            print(f"[LoRAManager] Pruned task '{task_name}': rank {old_rank} -> {new_rank}")
        else:
            print(f"[LoRAManager] Task '{task_name}': keep rank {old_rank}")

        return old_rank, new_rank

    def _group_lora_params(
        self, state_dict: OrderedDict
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                continue
            parts = key.split(".")
            lora_idx = None
            for i, p in enumerate(parts):
                if p in ("lora_A", "lora_B"):
                    lora_idx = i
                    break
            if lora_idx is None:
                continue
            base_name = ".".join(parts[:lora_idx])
            ab_type = parts[lora_idx]
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name][ab_type] = param

        result = {}
        for base_name, d in lora_pairs.items():
            if "lora_A" in d and "lora_B" in d:
                result[base_name] = (d["lora_A"], d["lora_B"])
        return result

    def _prune_state_dict(
        self,
        state_dict: OrderedDict,
        lora_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        new_rank: int,
    ) -> OrderedDict:
        new_state = OrderedDict()
        for key, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                new_state[key] = param
                continue
            is_lora_param = any("lora_A" in key or "lora_B" in key for _ in [key])
            if not is_lora_param:
                new_state[key] = param
                continue

            parts = key.split(".")
            lora_idx = None
            for i, p in enumerate(parts):
                if p in ("lora_A", "lora_B"):
                    lora_idx = i
                    break
            if lora_idx is None:
                new_state[key] = param
                continue

            base_name = ".".join(parts[:lora_idx])
            ab_type = parts[lora_idx]

            if base_name not in lora_pairs:
                new_state[key] = param
                continue

            lora_A, lora_B = lora_pairs[base_name]
            W_approx = (lora_B.float() @ lora_A.float()).cpu().numpy()
            try:
                U, S, Vh = np.linalg.svd(W_approx, full_matrices=False)
            except np.linalg.LinAlgError:
                new_state[key] = param
                continue

            k = min(new_rank, len(S))
            S_k = np.sqrt(S[:k] + 1e-12)

            if ab_type == "lora_A":
                new_A = (np.diag(S_k) @ Vh[:k, :]).astype(np.float32)
                new_state[key] = torch.from_numpy(new_A).to(param.device, param.dtype)
            else:
                new_B = (U[:, :k] @ np.diag(S_k)).astype(np.float32)
                new_state[key] = torch.from_numpy(new_B).to(param.device, param.dtype)

        return new_state

    def add_to_replay_buffer(
        self,
        dataset: Dataset,
        task_name: str,
        num_samples: int = 50,
    ):
        n = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), n, replace=False).tolist()
        samples = []
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict):
                samples.append({k: v for k, v in item.items()})
            else:
                samples.append({"data": item})
        self.replay_buffer.add_samples(samples, task_name)
        print(f"[LoRAManager] Added {n} samples to replay buffer from '{task_name}' (total={len(self.replay_buffer)})")

    def save_replay_buffer(self, path: Optional[str] = None):
        save_path = path or os.path.join(self.save_dir, "replay_buffer.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({
                "samples": self.replay_buffer.samples,
                "task_names": self.replay_buffer.task_names,
            }, f)

    def load_replay_buffer(self, path: Optional[str] = None):
        load_path = path or os.path.join(self.save_dir, "replay_buffer.pkl")
        if not os.path.exists(load_path):
            return False
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.replay_buffer.samples = data["samples"]
        self.replay_buffer.task_names = data["task_names"]
        print(f"[LoRAManager] Loaded replay buffer with {len(self.replay_buffer)} samples")
        return True

    def save_registry_index(self):
        index = {}
        for task_name, module in self.registry.items():
            index[task_name] = {
                "rank": module.current_rank,
                "metadata": module.metadata,
            }
        index_path = os.path.join(self.save_dir, "registry_index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, default=str)

    def compute_task_similarity(
        self,
        new_task_name: str,
        new_gradients: List[torch.Tensor],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        similarities = []
        new_grad_flat = torch.cat([g.flatten() for g in new_gradients if g is not None])
        new_norm = torch.norm(new_grad_flat) + 1e-12

        for task_name in self.registry.keys():
            if task_name == new_task_name:
                continue
            if "gradients" not in self.registry[task_name].metadata:
                continue

            hist_grads = self.registry[task_name].metadata["gradients"]
            hist_flat = torch.cat([
                torch.tensor(g) if not isinstance(g, torch.Tensor) else g.flatten()
                for g in hist_grads
            ])
            hist_norm = torch.norm(hist_flat) + 1e-12

            cosine = torch.dot(new_grad_flat, hist_flat) / (new_norm * hist_norm)
            similarity = float(cosine.cpu().numpy())
            similarities.append((task_name, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def init_lora_from_similar_tasks(
        self,
        base_lora_config: LoraConfig,
        similar_tasks: List[Tuple[str, float]],
    ) -> Tuple[LoraConfig, OrderedDict]:
        if len(similar_tasks) == 0:
            return base_lora_config, OrderedDict()

        total_sim = sum(max(0.0, s) for _, s in similar_tasks) + 1e-12
        weights = {task: max(0.0, sim) / total_sim for task, sim in similar_tasks}

        merged_config, merged_state = self.merge_loras(weights, merged_name="init_merge_tmp")

        if merged_config.r > base_lora_config.r:
            merged_config = LoraConfig(**{**merged_config.to_dict(), "r": base_lora_config.r})
            merged_state = self._pad_or_truncate_state(merged_state, base_lora_config.r)
        elif merged_config.r < base_lora_config.r:
            merged_state = self._pad_or_truncate_state(merged_state, base_lora_config.r)
            merged_config = LoraConfig(**{**merged_config.to_dict(), "r": base_lora_config.r})

        if "init_merge_tmp" in self.registry:
            del self.registry["init_merge_tmp"]

        return merged_config, merged_state

    def _pad_or_truncate_state(self, state_dict: OrderedDict, target_rank: int) -> OrderedDict:
        new_state = OrderedDict()
        lora_pairs = self._group_lora_params(state_dict)

        for key, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                new_state[key] = param
                continue

            parts = key.split(".")
            lora_idx = None
            for i, p in enumerate(parts):
                if p in ("lora_A", "lora_B"):
                    lora_idx = i
                    break
            if lora_idx is None:
                new_state[key] = param
                continue

            base_name = ".".join(parts[:lora_idx])
            ab_type = parts[lora_idx]

            if base_name not in lora_pairs:
                new_state[key] = param
                continue

            if ab_type == "lora_A":
                current_rank = param.shape[0]
                if current_rank == target_rank:
                    new_state[key] = param
                elif current_rank > target_rank:
                    new_state[key] = param[:target_rank, :].clone()
                else:
                    padding = torch.zeros(
                        target_rank - current_rank, param.shape[1],
                        dtype=param.dtype, device=param.device
                    )
                    new_state[key] = torch.cat([param, padding], dim=0)
            else:
                current_rank = param.shape[1]
                if current_rank == target_rank:
                    new_state[key] = param
                elif current_rank > target_rank:
                    new_state[key] = param[:, :target_rank].clone()
                else:
                    padding = torch.zeros(
                        param.shape[0], target_rank - current_rank,
                        dtype=param.dtype, device=param.device
                    )
                    new_state[key] = torch.cat([param, padding], dim=1)

        return new_state
