# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Datapaser for sdfstudio formatted data"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import torch
import numpy as np

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []

@dataclass
class SDFStudioDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: SDFStudio)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: int = 1
    """How much to downscale images."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal """
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    train_val_no_overlap: bool = False
    """remove selected / sampled validation images from training set"""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_highres: bool = True
    """load high resolution images from dataset (cannot be True when include_mono_prior is True)"""
    # include_foreground_mask: bool = False
    # """whether or not to load foreground mask"""

@dataclass
class SDFStudio(DataParser):
    """SDFStudio Dataset"""

    config: SDFStudioDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # load meta data
        meta = load_from_json(self.config.data / "meta_data.json")

        indices = list(range(len(meta["frames"])))
        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]

        image_filenames = []
        depth_filenames = []
        normal_filenames = []
        # mask_filenames = []
        transform = None
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []

        for frame in meta["frames"]:
            
            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            if self.config.load_highres:
                image_filename = self.config.data  / "image" /  frame["rgb_path"].replace("_rgb", "")
                intrinsics[:2, :] *= 1200 / 384.0
                intrinsics[0, 2] += 200
                height, width = 1200, 1600
                meta["height"], meta["width"] = height, width
                depth_filename = None
                normal_filename = None
                # mask_filename = self.config.data  / "mask" /  ...
            else:
                image_filename = self.config.data / frame["rgb_path"]
                depth_filename = frame.get("mono_depth_path")
                normal_filename = frame.get("mono_normal_path")
                # mask_filename = frame.get("mask_path")

            # append data
            image_filenames.append(image_filename)
            if depth_filename is not None:
                depth_filenames.append(self.config.data / depth_filename)
            if normal_filename is not None:
                normal_filenames.append(self.config.data / normal_filename)
            # if mask_filename is not None:
            #     mask_filenames.append(self.config.data / mask_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        c2w_colmap = torch.stack(camera_to_worlds)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        camera_to_worlds = torch.from_numpy(np.array(camera_to_worlds).astype(np.float32))
        camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        camera_to_worlds[:, :3, 3] *= scale_factor

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fx = fx[indices]
        fy = fy[indices]
        cx = cx[indices]
        cy = cy[indices]
        camera_to_worlds = camera_to_worlds[indices]

        height, width = meta["height"], meta["width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)
        
        if self.config.include_mono_prior:
            assert meta["has_mono_prior"], f"no mono prior in {self.config.data}"

        dataparser_outputs = DataparserOutputs(
            image_filenames=filter_list(image_filenames, indices),
            cameras=cameras,
            scene_box=scene_box,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "normal_filenames": normal_filenames if len(normal_filenames) > 0 else None,
                "transform": transform,
                # required for normal maps, these are in colmap format so they require c2w before conversion
                "camera_to_worlds": c2w_colmap if len(c2w_colmap) > 0 else None,
                "include_mono_prior": self.config.include_mono_prior,
            },
        )

        return dataparser_outputs
