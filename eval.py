import argparse
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fires.configs.default import get_cfg_defaults
from fires.data.omnidataloader import SpDRDFMapper, load_taskonomy_pkl
from fires.eval.eval_scene import ConsistencyRepository, FscoreRepository
from fires.model.spnet import load_model
from fires.utils.geometry_utils import get_point_image_colors


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cfg-path", type=str, help="main config path")
    parser.add_argument(
        "--ckpt-path", type=str, default="", help="path to the checkpoint"
    )
    parser.add_argument("--output-dir", type=str, default="", help="output folder")
    parser.add_argument(
        "--dataset-name", type=str, default="", help="test dataset name"
    )
    parser.add_argument("--test-view", type=int, default=3, help="eval number of views")

    return parser.parse_args()


def to_cpu(d):
    """
    Recursively converts all tensors and numpy arrays in a dictionary to CPU.
    """
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.cpu()
        elif isinstance(v, np.ndarray):
            continue
        elif isinstance(v, dict):
            d[k] = to_cpu(v)
    return d


class Single_Dataloader:
    def __init__(self, dataset_dicts, mapper):
        self.dataset_dicts = dataset_dicts
        self.mapper = mapper

    def __getitem__(self, idx):
        return self.mapper(self.dataset_dicts[idx])

    def __len__(self):
        return len(self.dataset_dicts)


if __name__ == "__main__":
    args = parse_args()
    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_path)
    model_path = args.ckpt_path
    device = "cuda"
    model = load_model(model_path, cfg, device)
    model.eval()

    dataset_dicts = load_taskonomy_pkl(args.dataset_name)
    mapper = SpDRDFMapper(cfg, is_train=False)
    mapper.test_view = [args.test_view]
    dataloader = Single_Dataloader(dataset_dicts, mapper)

    evaluator_fscore = FscoreRepository()
    evaluator_consistency = ConsistencyRepository()
    for idx in tqdm(range(len(dataloader))):
        dataset_dict = dataloader[idx]
        with torch.inference_mode():
            predictions = model([dataset_dict])[0]

        visibility, valid_mask = mapper.extract_point_visibility_pred(
            predictions["pcd_world"], dataset_dict, predictions
        )

        predictions["scene_visibility"] = visibility
        predictions["scene_valid_mask"] = valid_mask
        predictions = to_cpu(predictions)

        mesh_pts, visibility, valid_mask = mapper.extract_point_visibility_gt(
            dataset_dict, num_samples=100000
        )
        gt_dict = {}
        gt_dict["scene_pcd"] = mesh_pts
        gt_dict["scene_visibility"] = visibility
        gt_dict["scene_valid_mask"] = valid_mask
        gt_dict = to_cpu(gt_dict)

        uuid = (
            dataset_dict["dataset_dict"]["house_id"]
            + "_"
            + dataset_dict["dataset_dict"]["set_id"].replace("#", "_")
            + ".pkl"
        )
        evaluator_fscore.eval_single(uuid, predictions, gt_dict)
        evaluator_consistency.eval_single(idx, predictions, dataset_dict)
    eval_result_lst, combined_result = evaluator_fscore.summary()
    print("")
    eval_result_consistency = evaluator_consistency.summary()
