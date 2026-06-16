"""
SmolVLM Dataset

Dataset classes for SmolVLM-VLA training.
Supports:
  - Single-frame samples [V, C, H, W]            (all encoders)
  - Temporal clip samples [V, T, C, H, W]         (V-JEPA 2 with num_frames > 1)
  - Data-fraction subsampling at trajectory level  (controlled experiment)
"""

from __future__ import annotations
from typing import Dict, Iterable, List
import io
import json
import random as _random_mod
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_config import DATA_WEIGHTS
from .domain_handler.registry import get_handler_cls


# ------------------------------------------------------------------ #
# Deterministic trajectory-level subsampling                          #
# ------------------------------------------------------------------ #

def _subsample_datalist(datalist: list, frac: float, seed: int) -> list:
    """
    Deterministically subsample `frac` fraction of trajectories.

    Selection is at the trajectory level (not frame level) to avoid
    distribution shift: each selected trajectory contributes all its
    frames. Randomisation is seeded so the same frac+seed always picks
    the same subset across encoder conditions — the controlled-experiment
    invariant requires this.

    Args:
        datalist: full list of trajectory entries from the meta JSON.
        frac:     fraction in (0, 1]. Values ≥ 1.0 return the full list.
        seed:     integer seed for reproducible selection.

    Returns:
        Subsampled list (sorted by original index to maintain file-read order).
    """
    if frac >= 1.0:
        return datalist
    n = max(1, int(len(datalist) * frac))
    rng = _random_mod.Random(seed)
    indices = list(range(len(datalist)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n])
    return [datalist[i] for i in chosen]


# ------------------------------------------------------------------ #
# Main Dataset                                                         #
# ------------------------------------------------------------------ #

class SmolVLMDataReader(IterableDataset):
    """
    Infinite data reader for SmolVLM-VLA training.

    Output sample (single-frame, num_frames=1):
      {
        'language_instruction': str,
        'image_input':  FloatTensor[V, C, H, W],
        'image_mask':   BoolTensor[V],
        'proprio':      FloatTensor[dim_proprio],
        'action':       FloatTensor[T_action, dim_action],
      }

    Output sample (temporal clip, num_frames=T):
      {
        'image_input':  FloatTensor[V, T, C, H, W],   # ← only shape differs
        ...
      }

    Data-fraction subsampling
    -------------------------
    When data_frac < 1.0 the datalist is deterministically reduced to
    `ceil(N * data_frac)` trajectories using `data_seed`. The same
    frac + seed always selects the same subset, guaranteeing comparability
    across encoder conditions.
    """

    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        metas_path: str,
        num_actions: int = 10,
        num_views: int = 3,
        training: bool = True,
        action_mode: str = "galaxea_joint",
        image_size: int = 384,
        num_frames: int = 1,
        data_frac: float = 1.0,
        data_seed: int = 42,
    ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.image_size = image_size
        self.num_frames = num_frames
        self.data_frac = data_frac
        self.data_seed = data_seed
        self.metas: Dict[str, dict] = {}

        print(f"[SmolVLM Dataset] image_size={image_size}  num_frames={num_frames}  "
              f"data_frac={data_frac}  data_seed={data_seed}")

        # Load metadata
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(
                metas_path, suffix=".json", recursive=True, list_dir=False
            )
            root = metas_path
        elif metas_path.endswith(".json"):
            try:
                with open(metas_path) as f:
                    content = json.load(f)
                if isinstance(content, list):
                    meta_files = content
                    root = ""
                else:
                    meta_files, root = [metas_path], ""
            except Exception:
                meta_files, root = [metas_path], ""
        else:
            meta_files, root = [metas_path], ""

        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f:
                meta = json.load(f)

            # ── Deterministic trajectory subsampling ──
            original_n = len(meta["datalist"])
            meta["datalist"] = _subsample_datalist(
                meta["datalist"], data_frac, data_seed
            )
            sampled_n = len(meta["datalist"])
            print(
                f"== dataset {meta['dataset_name']}: "
                f"{sampled_n}/{original_n} trajs "
                f"(frac={data_frac}, seed={data_seed})"
            )
            self.metas[meta["dataset_name"]] = meta

        self.image_aug = self._build_image_transforms(training)

    def _build_image_transforms(self, training: bool) -> transforms.Compose:
        transform_list = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
        if training:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
                )
            )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD, inplace=True),
        ])
        return transforms.Compose(transform_list)

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        if self.training:
            _random_mod.shuffle(traj_indices)

        Handler = get_handler_cls(dataset_name)
        handler = Handler(meta=meta, num_views=self.num_views)

        for traj_idx in traj_indices:
            try:
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                    action_mode=self.action_mode,
                    num_frames=self.num_frames,
                ):
                    idx_for_delta = meta.get("idx_for_delta", [])
                    has_proprio = "proprio" in sample
                    slice_result = action_slice(sample["abs_trajectory"], idx_for_delta)

                    if has_proprio:
                        sample["action"] = slice_result["action"]
                    else:
                        sample.update(slice_result)
                    del sample["abs_trajectory"]

                    yield sample
            except Exception:
                continue

        if self.training:
            yield from self._iter_one_dataset(dataset_name)

    def __iter__(self):
        names = list(self.metas.keys())
        if not self.training:
            for n in names:
                yield from self._iter_one_dataset(n)
        else:
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws)
            ws = [w / s for w in ws]
            while True:
                i = _random_mod.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])


# ------------------------------------------------------------------ #
# Padded variant (unchanged from original)                            #
# ------------------------------------------------------------------ #

class SmolVLMDataReaderWithPadding(SmolVLMDataReader):
    """SmolVLM data reader with reflection-padding for small images."""

    PADDING_MODE = "reflect"

    def _build_image_transforms(self, training: bool) -> transforms.Compose:
        class SmartResize:
            def __init__(self, target_size: int):
                self.target_size = target_size

            def __call__(self, img):
                from PIL import Image as _PIL
                import numpy as _np
                w, h = img.size
                if w < self.target_size // 2 and h < self.target_size // 2:
                    result = _PIL.new("RGB", (self.target_size, self.target_size))
                    paste_x = (self.target_size - w) // 2
                    paste_y = (self.target_size - h) // 2
                    result.paste(img, (paste_x, paste_y))
                    result_np = _np.array(result)
                    if paste_x > 0:
                        result_np[:, :paste_x] = _np.flip(
                            result_np[:, paste_x:paste_x * 2], axis=1
                        )[:, :paste_x]
                        result_np[:, paste_x + w:] = _np.flip(
                            result_np[:, paste_x + w - paste_x:paste_x + w], axis=1
                        )[:, :self.target_size - paste_x - w]
                    return _PIL.fromarray(result_np)
                return img.resize((self.target_size, self.target_size), _PIL.BICUBIC)

        transform_list = [SmartResize(self.image_size)]
        if training:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
                )
            )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD, inplace=True),
        ])
        return transforms.Compose(transform_list)


# ------------------------------------------------------------------ #
# Factory                                                              #
# ------------------------------------------------------------------ #

def create_smolvlm_dataloader(
    batch_size: int,
    metas_path: str,
    num_actions: int,
    training: bool,
    action_mode: str,
    num_workers: int = 4,
    image_size: int = 384,
    num_frames: int = 1,
    data_frac: float = 1.0,
    data_seed: int = 42,
    use_smart_padding: bool = False,
):
    """
    Create dataloader for SmolVLM-VLA training.

    Parameters
    ----------
    num_frames : int
        1 (default) → single-frame samples [V, C, H, W].
        N > 1       → temporal clip samples [V, N, C, H, W]; passes num_frames
                      to the domain handler so it returns real consecutive frames.
                      Only VJEPA2Encoder uses all N frames; other encoders take
                      the last frame (current observation) in their forward().
    data_frac : float
        Fraction of trajectories to use. Deterministic by (data_frac, data_seed).
    data_seed : int
        Seed for trajectory subsampling (must be fixed across encoder conditions).
    """
    from torch.utils.data import DataLoader

    def worker_init_fn(worker_id: int):
        base_seed = torch.initial_seed() % (2 ** 32)
        import random
        np.random.seed(base_seed)
        random.seed(base_seed)
        torch.manual_seed(base_seed)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass

    import os

    DatasetClass = SmolVLMDataReaderWithPadding if use_smart_padding else SmolVLMDataReader

    dataset = DatasetClass(
        metas_path=metas_path,
        num_actions=num_actions,
        training=training,
        action_mode=action_mode,
        image_size=image_size,
        num_frames=num_frames,
        data_frac=data_frac,
        data_seed=data_seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "SmolVLMDataReader",
    "SmolVLMDataReaderWithPadding",
    "create_smolvlm_dataloader",
]
