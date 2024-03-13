import argparse
import os
import os.path as osp

import cv2
import torch

from fires.configs.default import get_cfg_defaults
from fires.data.omnidataloader import SpDRDFMapper, load_taskonomy_pkl
from fires.model.spnet import load_model
from fires.utils.geometry_utils import save_scene_as_glb, get_point_image_colors


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cfg-path", type=str, help="main config path")
    parser.add_argument(
        "--ckpt-path", type=str, default="", help="path to the checkpoint"
    )
    parser.add_argument("--output-dir", type=str, default="", help="output folder")

    return parser.parse_args()


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

    dataset_dicts = load_taskonomy_pkl(cfg.DATASETS.TEST[0])
    dataloader = SpDRDFMapper(cfg, is_train=False)

    dataset_dict = dataloader(dataset_dicts[0])
    with torch.inference_mode():
        predictions = model([dataset_dict])[0]

    pcd, color = get_point_image_colors(
        predictions["pcd_world"],
        predictions["rayid"],
        predictions["visibility"],
        dataset_dict["camera_torch"],
        dataset_dict["rgbs"],
        dataset_dict["num_ray_per_img"],
        device,
    )
    pred = {
        "pcd": pcd,
        "color": color,
    }

    for i in range(len(dataset_dict["rgbs"])):
        cv2.imwrite(
            osp.join(save_folder, f"rgb_{i}.jpg"),
            dataset_dict["rgbs"][i]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)[..., ::-1],
        )

    # save individual glb
    for i in range(len(dataset_dict["rgbs"])):
        save_scene_as_glb(
            pred["pcd"][i == predictions["imgid"]],
            pred["color"][i == predictions["imgid"]],
            dataset_dict["camera_torch"][i],
            osp.join(save_folder, f"pred_{i}.glb"),
            highres=cfg.DATALOADER.HIGH_RES_OUTPUT,
            camid=i,
        )
    save_scene_as_glb(
        pred["pcd"],
        pred["color"],
        dataset_dict["camera_torch"],
        osp.join(save_folder, "pred.glb"),
        highres=cfg.DATALOADER.HIGH_RES_OUTPUT,
    )
