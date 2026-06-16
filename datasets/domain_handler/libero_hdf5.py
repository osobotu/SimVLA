"""
LIBERO HDF5 Data Handler

LIBERO dataset format (original HDF5):
- data/demo_X/actions: [T, 7] - delta actions (xyz_delta(3) + euler_delta(3) + gripper(1))
- data/demo_X/obs/agentview_rgb: [T, 128, 128, 3] - third-person view image
- data/demo_X/obs/eye_in_hand_rgb: [T, 128, 128, 3] - wrist view image
- data/demo_X/obs/ee_pos: [T, 3] - end-effector position
- data/demo_X/obs/ee_ori: [T, 3] - end-effector orientation (euler in HDF5)
- data/demo_X/obs/gripper_states: [T, 2] - gripper states
- data/demo_X/obs/joint_states: [T, 7] - joint states

Actions range: [-1, 1]

Output format (libero_joint mode):
- state (proprio): 8-dim [ee_pos(3) + axis_angle(3) + gripper(2)]  <-- converted to axis-angle!
- actions: 7-dim [delta_xyz(3) + delta_euler(3) + gripper_cmd(1)]

Temporal clip output (num_frames > 1):
  image_input: [V, T, C, H, W]  — T consecutive frames ending at the current step.
               Frames before episode start are padded with frame 0.
  Only VJEPA2Encoder uses all T frames; other encoders take the last frame.
"""

from __future__ import annotations

import io
import random
import glob
import os
import re
from typing import Optional, Tuple, Iterable, Sequence, Any, Dict, List

import numpy as np
import h5py
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from .base import DomainHandler


def _quat2axisangle_single(quat: np.ndarray) -> np.ndarray:
    """
    Convert single quaternion [x,y,z,w] to axis-angle.

    Follows robosuite implementation for consistency with inference.
    """
    import math
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def euler_to_axisangle(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (XYZ) to axis-angle representation.

    Converts euler -> quat -> axis-angle

    Args:
        euler: [T, 3] euler angles (roll, pitch, yaw)

    Returns:
        axis_angle: [T, 3] axis-angle representation
    """
    rot = R.from_euler('xyz', euler)
    quats = rot.as_quat()  # [T, 4] as [x, y, z, w]

    if quats.ndim == 1:
        return _quat2axisangle_single(quats)

    axis_angles = np.zeros((len(quats), 3), dtype=np.float32)
    for i in range(len(quats)):
        axis_angles[i] = _quat2axisangle_single(quats[i])
    return axis_angles


class LiberoHDF5Handler(DomainHandler):
    """
    LIBERO original HDF5 data handler.

    Directly reads LIBERO official HDF5 file format.
    Supports libero_10, libero_90, libero_goal, libero_object, libero_spatial.

    Temporal clip mode (num_frames > 1):
      Each sample yields image_input of shape [V, T, C, H, W] where T=num_frames
      consecutive frames ending at timestep t are stacked along dim 1.
      Frames before t=0 are padded by repeating frame 0.
      This provides real motion context for video encoders (e.g. VJEPA2Encoder).
      Single-frame encoders (SmolVLM, DINOv2, I-JEPA) discard all but the
      last frame inside their own forward() methods.
    """
    dataset_name = "libero_hdf5"

    # Data frame rate and prediction duration
    FREQ = 10.0  # Hz
    QDUR = 1.0   # seconds

    def __init__(self, meta: dict, num_views: int = 3) -> None:
        super().__init__(meta, num_views)
        self.data_dir = meta.get("data_dir", "")
        self.h5_files: List[str] = []
        self.task_names: List[str] = []

        # Get HDF5 file list from datalist
        if "datalist" in meta:
            for item in meta["datalist"]:
                if isinstance(item, dict):
                    self.h5_files.append(item["path"])
                    self.task_names.append(item.get("task", ""))
                else:
                    self.h5_files.append(item)
                    self.task_names.append(self._parse_task_from_filename(item))

    def _parse_task_from_filename(self, filepath: str) -> str:
        """Parse task description from filename."""
        base = os.path.basename(filepath)
        # e.g., KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
        task = re.sub(r"_demo\.hdf5$", "", base)
        m = re.search(r"SCENE\d+_", task)
        if m:
            task = task[m.end():]
        task = task.replace("_", " ")
        return task

    def _open_h5(self, path: str) -> h5py.File:
        """Open HDF5 file."""
        return h5py.File(path, "r")

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        training: bool = True,
        image_aug=None,
        action_mode: str = "libero_joint",
        lang_aug_map: dict | None = None,
        num_frames: int = 1,
        **kwargs
    ) -> Iterable[dict]:
        """
        Iterate over all samples in an episode.

        Args:
            traj_idx:    trajectory index
            num_actions: action chunk length
            training:    whether in training mode
            image_aug:   image augmentation transform (applied to each frame)
            action_mode: action mode (libero_joint)
            lang_aug_map: language augmentation mapping
            num_frames:  number of consecutive frames to include in each sample.
                         1 → single frame [V, C, H, W] (default, backward compat).
                         N → temporal clip [V, N, C, H, W] ending at timestep t,
                             padded with frame 0 at the start of episodes.
        """
        h5_path = self.h5_files[traj_idx]
        task_instruction = self.task_names[traj_idx]

        with self._open_h5(h5_path) as f:
            if "data" not in f:
                return
            data_grp = f["data"]

            # Get all demo keys and shuffle during training
            demo_keys = list(data_grp.keys())
            if training:
                random.shuffle(demo_keys)

            for demo_key in demo_keys:
                demo = data_grp[demo_key]

                # Check required keys exist
                required_keys = ["actions", "obs/agentview_rgb", "obs/eye_in_hand_rgb"]
                if not all(k in demo or f"obs/{k.split('/')[-1]}" in demo.get("obs", {})
                          for k in required_keys if "/" not in k):
                    continue

                try:
                    yield from self._iter_demo(
                        demo,
                        task_instruction,
                        num_actions=num_actions,
                        training=training,
                        image_aug=image_aug,
                        action_mode=action_mode,
                        lang_aug_map=lang_aug_map,
                        num_frames=num_frames,
                    )
                except Exception as e:
                    print(f"Error processing {h5_path}/{demo_key}: {e}")
                    continue

    def _iter_demo(
        self,
        demo: h5py.Group,
        task_instruction: str,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode: str,
        lang_aug_map: dict | None,
        num_frames: int = 1,
    ) -> Iterable[dict]:
        """Process single demo."""

        # Load data
        actions = np.array(demo["actions"])  # [T, 7]
        agentview_rgb = np.array(demo["obs/agentview_rgb"])  # [T, H, W, 3]
        wrist_rgb = np.array(demo["obs/eye_in_hand_rgb"])     # [T, H, W, 3]

        # Load proprio state
        ee_pos = np.array(demo["obs/ee_pos"])  # [T, 3]
        ee_ori_euler = np.array(demo["obs/ee_ori"])  # [T, 3] - HDF5 stores euler
        gripper_states = np.array(demo["obs/gripper_states"])  # [T, 2]

        # Convert Euler to axis-angle
        ee_ori_axisangle = euler_to_axisangle(ee_ori_euler)  # [T, 3]

        T = min(len(actions), len(agentview_rgb), len(wrist_rgb))

        # Build proprio: [ee_pos(3), axis_angle(3), gripper(2)] = 8-dim
        proprio = np.concatenate([
            ee_pos[:T],
            ee_ori_axisangle[:T],  # Using axis-angle, not euler
            gripper_states[:T]
        ], axis=-1).astype(np.float32)

        # Actions: [T, 7]
        actions = actions[:T].astype(np.float32)

        # Candidate indices
        indices = list(range(max(0, T - num_actions)))
        if training:
            random.shuffle(indices)

        # Image mask: agentview + wrist valid, optional third view padded
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:2] = True  # agentview + wrist

        for idx in indices:
            # Get action chunk
            action_chunk = self._get_action_chunk(actions, idx, num_actions)

            # Language augmentation
            instruction = task_instruction
            if training and lang_aug_map and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])

            if num_frames == 1:
                image_input = self._get_single_frame(
                    agentview_rgb, wrist_rgb, idx, image_aug
                )
            else:
                image_input = self._get_temporal_clip(
                    agentview_rgb, wrist_rgb, idx, num_frames, image_aug
                )

            yield {
                "language_instruction": instruction,
                "image_input": image_input,
                "image_mask": image_mask,
                "proprio": torch.tensor(proprio[idx], dtype=torch.float32),
                "abs_trajectory": torch.tensor(action_chunk, dtype=torch.float32),
            }

    def _get_single_frame(
        self,
        agentview_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        idx: int,
        image_aug,
    ) -> torch.Tensor:
        """
        Build image_input [V, C, H, W] for a single timestep.
        (Original single-frame path — unchanged from baseline.)
        """
        imgs = []

        # Agentview (third-person) — rotate 180 degrees for consistency
        img_data = agentview_rgb[idx][::-1, ::-1].copy()
        img = Image.fromarray(img_data)
        if image_aug:
            img = image_aug(img)
        imgs.append(img)

        # Wrist — also rotate 180 degrees
        wrist_data = wrist_rgb[idx][::-1, ::-1].copy()
        wrist_img = Image.fromarray(wrist_data)
        if image_aug:
            wrist_img = image_aug(wrist_img)
        imgs.append(wrist_img)

        # Pad empty views
        while len(imgs) < self.num_views:
            imgs.append(torch.zeros_like(imgs[0]))

        return torch.stack(imgs, dim=0)  # [V, C, H, W]

    def _get_temporal_clip(
        self,
        agentview_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        idx: int,
        num_frames: int,
        image_aug,
    ) -> torch.Tensor:
        """
        Build image_input [V, T, C, H, W] — a T-frame clip ending at idx.

        Frame window: [idx - num_frames + 1, ..., idx].
        Frames before t=0 are padded by repeating frame 0 (edge padding).

        image_aug is applied independently to each frame. ColorJitter uses
        independent random parameters per frame, which is intentional: slight
        per-frame colour variation acts as implicit temporal augmentation.
        """
        # Frame indices, clamped to [0, T_ep-1]
        frame_indices = [max(0, idx - num_frames + 1 + t) for t in range(num_frames)]

        view_clips: list[torch.Tensor] = []

        for raw_frames, _ in [(agentview_rgb, "agent"), (wrist_rgb, "wrist")]:
            frame_tensors: list[torch.Tensor] = []
            for fi in frame_indices:
                frame_data = raw_frames[fi][::-1, ::-1].copy()  # rotate 180°
                frame_img = Image.fromarray(frame_data)
                if image_aug:
                    frame_img = image_aug(frame_img)  # PIL → tensor [C, H, W]
                frame_tensors.append(frame_img)
            # [T, C, H, W]
            view_clips.append(torch.stack(frame_tensors, dim=0))

        # Pad to num_views
        while len(view_clips) < self.num_views:
            view_clips.append(torch.zeros_like(view_clips[0]))

        return torch.stack(view_clips, dim=0)  # [V, T, C, H, W]

    def _get_action_chunk(
        self,
        actions: np.ndarray,
        start_idx: int,
        num_actions: int
    ) -> np.ndarray:
        """
        Get action chunk, pad with last frame if out of range.

        Returns:
            [num_actions+1, action_dim] - includes current state + num_actions future actions
        """
        T, action_dim = actions.shape
        chunk = np.zeros((num_actions + 1, action_dim), dtype=np.float32)

        for i in range(num_actions + 1):
            t = min(start_idx + i, T - 1)
            chunk[i] = actions[t]

        return chunk


def create_libero_meta(
    data_dir: str,
    subsets: List[str] = None,
    output_path: str = None
) -> dict:
    """
    Create LIBERO dataset meta configuration.

    Args:
        data_dir: LIBERO dataset root directory
        subsets: List of subsets to include, e.g., ["libero_10", "libero_goal"]
                 Default includes all subsets
        output_path: Optional path to save meta JSON

    Returns:
        meta dictionary
    """
    import json

    if subsets is None:
        subsets = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]

    datalist = []

    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"Warning: {subset_dir} does not exist, skipping")
            continue

        h5_files = sorted(glob.glob(os.path.join(subset_dir, "*.hdf5")))
        for h5_path in h5_files:
            # Parse task description
            base = os.path.basename(h5_path)
            task = re.sub(r"_demo\.hdf5$", "", base)
            m = re.search(r"SCENE\d+_", task)
            if m:
                task = task[m.end():]
            task = task.replace("_", " ")

            datalist.append({
                "path": h5_path,
                "task": task,
                "subset": subset,
            })

    meta = {
        "dataset_name": "libero_hdf5",
        "data_dir": data_dir,
        "datalist": datalist,
        "num_episodes": len(datalist),
        "observation_key": ["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        "action_key": "actions",
        "state_dim": 8,
        "action_dim": 7,
        "fps": 10,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta to {output_path}")

    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="LIBERO dataset directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output meta JSON path")
    args = parser.parse_args()

    meta = create_libero_meta(
        args.data_dir,
        output_path=args.output
    )
    print(f"Found {meta['num_episodes']} episodes")
