import copy
import csv
import io
import json
import os
import os.path as osp
import tarfile
from collections import defaultdict

import numpy as np
import torch
import trimesh
from loguru import logger
from PIL import Image
from pytorch3d.renderer.cameras import PerspectiveCameras


def get_splits(split_path):
    forbidden_buildings = ["mosquito", "tansboro"]

    # The 'small-image' dataset has some misssing buildings. Either we can ignore those buildings or we can regenerate the dataset.
    forbidden_buildings += [
        "newfields",
        "ludlowville",
        "goodyear",
        "castroville",
    ]  # missing RGB
    # forbidden_buildings = ['mosquito', 'tansboro', 'tomkins', 'darnestown', 'brinnon']
    # We do not have the rgb data for tomkins, darnestown, brinnon

    forbidden_buildings += ["rough"]  # Contains some wrong view-points
    forbidden_buildings += [
        "woodbine",
        "darnestown",
        "gough",
        "willow",
        "winooski",
        "wyatt",
        "yankeetown",
        "tomkins",
        "brinnon",
        "wyldwood",
        "yadkinville",
        "yscloskey",
    ]  # json not found FIXME
    forbidden_buildings += ["german"]
    with open(split_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")

        train_list = []
        val_list = []
        test_list = []

        for row in readCSV:
            name, is_train, is_val, is_test = row
            if name in forbidden_buildings:
                continue
            if is_train == "1":
                train_list.append(name)
            if is_val == "1":
                val_list.append(name)
            if is_test == "1":
                test_list.append(name)
    return {
        "train": sorted(train_list),
        "val": sorted(val_list),
        "test": sorted(test_list),
    }


def get_taskonomy_split():
    split_f = "./dataset/omnidata/taskonomy/"
    # Taskonomy
    # subsets = ['debug', 'tiny', 'medium', 'full', 'fullplus']
    subsets = ["debug", "tiny", "medium"]
    taskonomy_split_files = {
        s: os.path.join(split_f, "train_val_test_{}.csv".format(s.lower()))
        for s in subsets
    }

    taskonomy_split_to_buildings = {
        s: get_splits(taskonomy_split_files[s]) for s in subsets
    }

    taskonomy_flat_split_to_buildings = {}
    for subset in taskonomy_split_to_buildings:
        for split, buildings in taskonomy_split_to_buildings[subset].items():
            taskonomy_flat_split_to_buildings[subset + "-" + split] = buildings

    # assertion
    for i in np.arange(1, len(subsets) - 1):
        for phase in ["train", "val", "test"]:
            if not set(
                taskonomy_flat_split_to_buildings[subsets[i] + "-" + phase]
            ).issubset(
                set(taskonomy_flat_split_to_buildings[subsets[i + 1] + "-" + phase])
            ):
                breakpoint()
                pass
    return taskonomy_flat_split_to_buildings


class OmniData:
    def __init__(self, cfg):
        self.tar_root = cfg.DATALOADER.TAR_ROOT
        self.index_root = cfg.DATALOADER.INDEX_ROOT
        self.all_tars = defaultdict(lambda: defaultdict(object))
        self.all_indices = defaultdict(lambda: defaultdict(object))

    def read_tar_header(self, house_id, data_type):
        tar_root = self.tar_root
        if house_id in self.all_tars.keys():
            if data_type in self.all_tars[house_id].keys():
                return
        if house_id not in self.all_tars.keys():
            self.all_tars[house_id] = {}
            self.all_indices[house_id] = {}
        tar_name = osp.join(tar_root, f"{data_type}__taskonomy__{house_id}.tar")
        index_name = osp.join(
            self.index_root, f"{data_type}__taskonomy__{house_id}.tar"
        )

        self.all_tars[house_id][data_type] = open(tar_name, "rb")
        self.all_indices[house_id][data_type] = open(index_name, "r")  # .readlines()

    def lookup(self, path, house_id, data_type, tar_name):
        if "_raw" in data_type:
            data_type = data_type.split("_raw")[0]
        self.read_tar_header(house_id, data_type)
        if data_type not in self.all_indices[house_id].keys():
            print(
                f"error in dataloader {data_type} not in self.all_indices[{house_id}]"
            )
            with tarfile.open(tar_name) as tf:
                try:
                    data = tf.extractfile(path)
                    data = data.read()
                except:
                    breakpoint()
                    tf.getmembers()
            return data
        if hasattr(self.all_indices[house_id][data_type], "readlines"):
            self.all_indices[house_id][data_type] = self.all_indices[house_id][
                data_type
            ].readlines()

        for line in self.all_indices[house_id][data_type]:
            m = line[:-1].rsplit(" ", 2)
            if path == m[0]:
                self.all_tars[house_id][data_type].seek(int(m[1]))
                buffer = self.all_tars[house_id][data_type].read(int(m[2]))
                return buffer

    def get_from_tar(
        self, house_id, dp_id, data_type, original_fov_deg=None, new_fov_deg=None
    ):
        tar_root = self.tar_root
        if data_type in ["depth_zbuffer", "depth_zbuffer_raw"]:
            tar_name = osp.join(tar_root, f"depth_zbuffer__taskonomy__{house_id}.tar")
            data_name = f"depth_zbuffer/{dp_id}_depth_zbuffer.png"

        elif data_type in ["rgb", "rgb_raw"]:
            tar_name = osp.join(tar_root, f"rgb__taskonomy__{house_id}.tar")
            data_name = f"rgb/{dp_id}_rgb.png"

        elif data_type == "mask_valid":
            tar_name = osp.join(tar_root, f"mask_valid__taskonomy__{house_id}.tar")
            data_name = f"./mask_valid/{house_id}/{dp_id}_depth_zbuffer.png"

        elif data_type == "point_info":
            tar_name = osp.join(tar_root, f"point_info__taskonomy__{house_id}.tar")
            data_name = f"point_info/{dp_id}_point_info.json"
        elif data_type == "segment_semantic":
            tar_name = osp.join(
                tar_root, f"segment_semantic__taskonomy__{house_id}.tar"
            )
            data_name = f"segment_semantic/{dp_id}_segmentsemantic.png"
        elif data_type == "edge_occlusion_raw":
            tar_name = osp.join(tar_root, f"edge_occlusion__taskonomy__{house_id}.tar")
            data_name = f"edge_occlusion/{dp_id}_edge_occlusion.png"

        elif data_type == "principal_curvature":
            tar_name = osp.join(
                tar_root, f"principal_curvature__taskonomy__{house_id}.tar"
            )
            data_name = f"principal_curvature/{dp_id}_principal_curvature.png"

        else:
            raise NotImplementedError

        data = self.lookup(data_name, house_id, data_type, tar_name)
        if data_type == "point_info":
            data = json.loads(data)
        elif data_type == "principal_curvature":
            data = Image.open(io.BytesIO(data))
        elif data_type == "rgb_raw":
            # data = data.read()
            data = Image.open(io.BytesIO(data))
        elif data_type in ["depth_zbuffer_raw"]:
            # data = data.read()
            data = Image.open(io.BytesIO(data))
            data = np.array(data) / 512
        elif data_type in ["edge_occlusion_raw"]:
            # data = data.read()
            data = Image.open(io.BytesIO(data))
            data = np.array(data) / (2**16 - 1)
        elif data_type in ["segment_semantic", "mask_valid"]:
            # data = data.read()
            data = Image.open(io.BytesIO(data))
            data = np.array(data)
        else:
            raise NotImplementedError

        if original_fov_deg is not None:
            # crop to fixed fov
            if data_type in ["depth_zbuffer_raw", "mask_valid"]:
                h, w = data.shape
                assert h == w
                rel_focal = 1 / 2 / np.tan(np.radians(original_fov_deg) / 2)
                rel_half_crop_size = rel_focal * np.tan(np.radians(new_fov_deg / 2))
                half_crop_size = int(h * rel_half_crop_size)
                h_start = int(h / 2 - half_crop_size)
                w_start = int(w / 2 - half_crop_size)
                data = data[
                    h_start : half_crop_size * 2 + h_start,
                    w_start : half_crop_size * 2 + w_start,
                ]
            elif data_type in ["rgb_raw"]:
                data = np.array(data)
                h, w, c = data.shape
                assert h == w
                rel_focal = 1 / 2 / np.tan(np.radians(original_fov_deg) / 2)
                rel_half_crop_size = rel_focal * np.tan(np.radians(new_fov_deg / 2))

                half_crop_size = int(h * rel_half_crop_size)
                h_start = int(h / 2 - half_crop_size)
                w_start = int(w / 2 - half_crop_size)
                data = data[
                    h_start : half_crop_size * 2 + h_start,
                    w_start : half_crop_size * 2 + w_start,
                    :,
                ]
                data = Image.fromarray(data)
            elif data_type in ["ray_intersection"]:
                # this is already generated by fixed fov camera (size is 256x256)
                pass
            else:
                breakpoint()
        return data
