import copy
import os.path as osp
import pickle
import random
from collections import defaultdict

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange, repeat
from loguru import logger
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.implicit.utils import (RayBundle,
                                               ray_bundle_to_ray_points)
from pytorch3d.structures import Meshes
from tqdm import tqdm

from fires.data.omnidata import OmniData, get_taskonomy_split
from fires.utils.ray_bundle_sampler import RayBundleSampler
from fires.utils.geometry_utils import (GroundTruthRayDistance, drdf2depth,
                                   get_signed_distance_to_closest_torch,
                                   project_ndc_depth, smallest_k_values)


def load_taskonomy_pkl(dataset_name):

    SPLITS_TASKONOMY = {
        "train_set3": ("dataset/omnidata_medium_set_3_train.pkl"),
        "val_set3": ("dataset/omnidata_medium_set_3_val.pkl"),
        "test_set3": ("dataset/omnidata_medium_set_3_test.pkl"),
        "test_set5": ("dataset/omnidata_medium_set_5_test.pkl"),
    }
    pkl_file = SPLITS_TASKONOMY[dataset_name]
    with open(pkl_file, "rb") as f:
        summary = pickle.load(f)
    dataset_dicts = summary["data"]
    logger.info(f"len({dataset_name}): {len(dataset_dicts)}")
    return dataset_dicts


class SpDRDFMapper:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.
    Note that for our existing models, mean/std normalization is done by the model instead of here.
    According to cfg.MODEL.PIXEL_MEAN, the output image is 0-255
    """

    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        # fmt: off
        self.resize                 = cfg.DATALOADER.RESIZE
        self.znear                  = cfg.DATASET_GENERATE.ZNEAR
        self.zfar                   = cfg.DATASET_GENERATE.ZFAR
        self.ray_sample_resolution  = cfg.DATASET_GENERATE.RAY_SAMPLE_RESOLUTION
        self.max_hit                = cfg.DATASET_GENERATE.MAX_HIT
        self.train_view             = cfg.DATALOADER.TRAIN_VIEW
        self.test_view              = cfg.DATALOADER.TEST_VIEW
        assert len(self.train_view) > 0
        assert len(self.test_view) > 0
        self.ray_per_img            = cfg.DATALOADER.NUM_RAY_PER_IMG_SAMPLE
        self.sample_gaussian_on     = cfg.DATALOADER.SAMPLE_GAUSSIAN_ON
        if self.sample_gaussian_on:
            self.num_gaussian_pt    = cfg.DATALOADER.NUM_GAUSSIAN_PT
            self.num_uniform_pt     = cfg.DATALOADER.NUM_UNIFORM_PT
            self.gaussian_std       = cfg.DATALOADER.GAUSSIAN_STD
        else:
            self.num_uniform_pt     = int((self.zfar - self.znear) // self.ray_sample_resolution + 1)
        self.depth_only_on          = cfg.MODEL.DEPTH_ONLY
        self.loss_decay_on          = cfg.MODEL.DECAY_LOSS_ON
        self.adaptive_sampling_on   = cfg.DATALOADER.ADAPTIVE_SAMPLING
        self.fixed_fov_deg          = cfg.DATASET_GENERATE.FIX_FOV_DEG
        self.camera_mode = cfg.DATALOADER.CAMERA_MODE
        # fmt: on
        self.dataloader = OmniData(cfg)
        self.is_train = is_train
        self.init_aug()
        self.cache_all_mesh()
        self.init_ray_sampler()

    def init_aug(self):
        self.aug = A.Compose([A.Resize(self.resize, self.resize)])

    def cache_all_mesh(self):
        version_root = self.cfg.DATALOADER.CACHE_PATH
        subset = "medium"
        house_ids = {}
        for phase in ["test", "val", "train"]:
            taskonomy_flat_split_to_buildings = get_taskonomy_split()
            for house_id in taskonomy_flat_split_to_buildings[subset + "-" + phase]:
                house_ids[house_id] = phase
        logger.info("Loading per view roomid")

        per_view_f = osp.join(version_root, "per_view_roomid.pkl")

        if True and osp.exists(per_view_f):
            logger.info(f"loading cached {per_view_f}...")
            with open(per_view_f, "rb") as f:
                per_view_roomid = pickle.load(f)
        else:
            logger.info(f"{per_view_f} not found, generating...")
            per_view_roomid = defaultdict(dict)
            for house_id in tqdm(house_ids.keys()):
                per_view_mesh_info_root = osp.join(version_root, "per_view_mesh_info")
                per_view_mesh_info_f = osp.join(
                    per_view_mesh_info_root, house_id + ".pkl"
                )
                if not osp.exists(per_view_mesh_info_f):
                    logger.info(f"{per_view_mesh_info_f} does not exist!")
                    continue
                with open(
                    osp.join(per_view_mesh_info_root, house_id + ".pkl"), "rb"
                ) as f:
                    cameras = pickle.load(f)
                    for cam in cameras:
                        per_view_roomid[cam["house_id"]][cam["img_id"]] = cam["mesh"][
                            "roomids"
                        ]
            with open(per_view_f, "wb") as f:
                pickle.dump(per_view_roomid, f)
        self.per_view_roomid = per_view_roomid

        logger.info("loading house id")

        houses2mesh_pkl = osp.join(version_root, "cached_mesh.pkl")
        logger.info(f"loading cached {houses2mesh_pkl}...")
        with open(houses2mesh_pkl, "rb") as f:
            houses2mesh = pickle.load(f)

        self.houses2mesh = houses2mesh

    def init_ray_sampler(self):
        if self.is_train and self.adaptive_sampling_on:
            ray_per_img = min(128**2, 10 * self.ray_per_img)
        else:
            ray_per_img = self.ray_per_img
        self.ray_sampler = RayBundleSampler(
            ray_per_cam=ray_per_img,
            num_uniform_pt=self.num_uniform_pt,
            znear=self.znear,
            zfar=self.zfar,
            orthographic_size=2,
            multinomial_size=64,
            high_res=self.cfg.DATALOADER.HIGH_RES_OUTPUT,
        )

    def _get_face_idxs_by_room_ids(self, room_idxs, house_id):
        face_2_roomid = self.houses2mesh[house_id]["face_2_roomid"]
        valid_face_ids = []
        for roomid in room_idxs:
            valid_face_ids.extend(np.where(face_2_roomid == roomid)[0])
        valid_face_ids = torch.tensor(valid_face_ids)
        return valid_face_ids

    def extract_mesh(self, dataset_dict, origin=None, select_idx=None):
        imgids = dataset_dict["set_id"].split("#")[:-1]
        if select_idx is not None:
            imgids = [imgids[i] for i in select_idx]

        roomids = []
        for imgid in imgids:
            roomids.extend(self.per_view_roomid[dataset_dict["house_id"]][imgid])
        roomids = np.unique(roomids)
        valid_face_ids = self._get_face_idxs_by_room_ids(
            roomids, dataset_dict["house_id"]
        )
        mesh = self.houses2mesh[dataset_dict["house_id"]]
        if origin is not None:
            vertices = mesh["vertices"] - origin.numpy()
        else:
            vertices = mesh["vertices"]

        mesh_pytorch3d = Meshes(
            verts=[torch.FloatTensor(vertices)], faces=[torch.tensor(mesh["faces"])]
        )
        mesh_pytorch3d = mesh_pytorch3d.submeshes([valid_face_ids.unsqueeze(0)])
        return mesh_pytorch3d

    def load_rgb(self, dataset_dict):
        """loading rgb"""
        rgbs = []
        for idx, camera in enumerate(dataset_dict["cameras"]):
            rgb = np.asarray(
                self.dataloader.get_from_tar(
                    camera["house_id"],
                    camera["img_id"],
                    "rgb_raw",
                    original_fov_deg=np.degrees(camera["field_of_view_rads"]),
                    new_fov_deg=self.fixed_fov_deg,
                )
            )
            rgb = self.augmentation(rgb).transpose(2, 0, 1)
            rgbs.append(rgb)
        rgbs = torch.as_tensor(np.array(rgbs).astype("float32"))
        return rgbs

    def load_depth(self, dataset_dict):
        """loading depth"""
        depths = []
        for idx, camera in enumerate(dataset_dict["cameras"]):
            depth = np.asarray(
                self.dataloader.get_from_tar(
                    camera["house_id"],
                    camera["img_id"],
                    "depth_zbuffer_raw",
                    original_fov_deg=np.degrees(camera["field_of_view_rads"]),
                    new_fov_deg=self.fixed_fov_deg,
                )
            )
            depth = self.augmentation(depth[..., None]).transpose(2, 0, 1)
            depths.append(depth)
        depths = torch.as_tensor(np.array(depths).astype("float32"))
        return depths

    def load_camera(self, dataset_dict, origin):
        """
        update camera center in dataset_dict and create PerspectiveCameras
        """
        for idx, camera in enumerate(dataset_dict["cameras"]):
            camera["camera_torch"].T = (
                -torch.inverse(camera["camera_torch"].R)
                @ (
                    -(camera["camera_torch"].R @ camera["camera_torch"].T.T).squeeze(2)
                    - origin
                ).T
            ).squeeze(2)
        Rs = torch.cat(
            [camera["camera_torch"].R for camera in dataset_dict["cameras"]], axis=0
        )
        Ts = torch.cat(
            [camera["camera_torch"].T for camera in dataset_dict["cameras"]], axis=0
        )
        # reference for not dividing 2 https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/camera_conversions.py#L36
        if self.fixed_fov_deg:
            focals = torch.cat(
                [
                    1.0
                    / torch.tan(torch.FloatTensor([np.radians(self.fixed_fov_deg)]) / 2)
                    for camera in dataset_dict["cameras"]
                ],
                axis=0,
            ).unsqueeze(1)
        else:
            focals = torch.cat(
                [
                    1.0
                    / torch.tan(torch.FloatTensor([camera["field_of_view_rads"]]) / 2)
                    for camera in dataset_dict["cameras"]
                ],
                axis=0,
            ).unsqueeze(1)
        cameras_all = PerspectiveCameras(focal_length=focals, R=Rs, T=Ts, image_size=1)
        return cameras_all, dataset_dict

    def load_hit2origin(self, gt_mesh, ray_bundle):
        tmesh = trimesh.Trimesh(
            vertices=gt_mesh.verts_packed().numpy(),
            faces=gt_mesh.faces_packed().numpy(),
        )
        intersector = GroundTruthRayDistance(tmesh)

        locations, index_ray, index_tri = intersector.get_all_intersections(
            ray_bundle.origins.reshape(-1, 3).numpy(),
            ray_bundle.directions.reshape(-1, 3).numpy(),
        )
        distances = np.linalg.norm(
            locations - ray_bundle.origins.reshape(-1, 3).numpy()[index_ray], axis=1
        )
        # hit2origin
        if len(ray_bundle.origins.shape) == 3:
            total_num_ray = ray_bundle.origins.shape[0] * ray_bundle.origins.shape[1]
        elif len(ray_bundle.origins.shape) == 4:
            total_num_ray = (
                ray_bundle.origins.shape[0]
                * ray_bundle.origins.shape[1]
                * ray_bundle.origins.shape[2]
            )
        else:
            breakpoint()
        hit2origin = smallest_k_values(
            index_ray, distances, k=self.max_hit + 1, total_num_ray=total_num_ray
        )

        hit2origin = torch.FloatTensor(hit2origin)
        if self.is_train and self.adaptive_sampling_on and (not self.depth_only_on):
            n_cam, adaptive_ray_per_image, _ = ray_bundle.directions.shape
            assert len(hit2origin.shape) == 2
            hit2origin = rearrange(
                hit2origin, "(c r) d -> c r d", c=n_cam, r=adaptive_ray_per_image
            )
            selected_indices_list = []
            for idx in range(n_cam):
                hit2origin_tmp = hit2origin[idx]
                occ_rate = (
                    ~torch.isinf(hit2origin_tmp[:, 1])
                ).sum() / hit2origin_tmp.shape[0]
                occ_threshold = 0.8
                if 0 < occ_rate < occ_threshold:
                    selected_indices = []
                    # sample more points with occlusions
                    selected_indices.extend(
                        np.random.choice(
                            np.where(hit2origin_tmp[:, 1].numpy() < self.zfar)[0],
                            int(self.ray_per_img * occ_threshold),
                        )
                    )
                    selected_indices.extend(
                        np.random.choice(
                            np.where(hit2origin_tmp[:, 1].numpy() > self.zfar)[0],
                            self.ray_per_img - len(selected_indices),
                        )
                    )

                    selected_indices = np.array(selected_indices)
                else:
                    selected_indices = random.sample(
                        np.arange(len(hit2origin_tmp)).tolist(), self.ray_per_img
                    )
                selected_indices_list.append(selected_indices)
            selected_indices_list = torch.tensor(
                np.array(selected_indices_list)
            ).unsqueeze(-1)
            hit2origin = torch.gather(
                hit2origin, 1, selected_indices_list.expand(-1, -1, hit2origin.size(2))
            )
            hit2origin = rearrange(hit2origin, "c r d -> (c r) d")
            ray_bundle = RayBundle(
                directions=torch.gather(
                    ray_bundle.directions,
                    1,
                    selected_indices_list.expand(-1, -1, ray_bundle.directions.size(2)),
                ),
                lengths=torch.gather(
                    ray_bundle.lengths,
                    1,
                    selected_indices_list.expand(-1, -1, ray_bundle.lengths.size(2)),
                ),
                origins=torch.gather(
                    ray_bundle.origins,
                    1,
                    selected_indices_list.expand(-1, -1, ray_bundle.origins.size(2)),
                ),
                xys=torch.gather(
                    ray_bundle.xys,
                    1,
                    selected_indices_list.expand(-1, -1, ray_bundle.xys.size(2)),
                ),
            )
        del intersector
        return hit2origin, ray_bundle

    def sample_points_along_ray_uniform(
        self, num_ray, jitter, znear, zfar, sample_points=0
    ):
        points_uniform = torch.arange(znear, zfar, self.ray_sample_resolution).repeat(
            num_ray, 1
        )
        if sample_points > 0:
            select_idx = torch.multinomial(
                torch.ones_like(points_uniform), sample_points
            )
            points_uniform = torch.gather(points_uniform, 1, select_idx)

        if jitter:
            points_uniform += torch.normal(
                0 * torch.ones_like(points_uniform),
                self.ray_sample_resolution * torch.ones_like(points_uniform),
            )
        return points_uniform

    def sample_points_along_ray_gaussion(self, hit2cam, samples_per_points, std):
        # sample gaussian points along ray
        num_ray, num_hit = hit2cam.shape
        sample_shape = (*hit2cam.shape, samples_per_points)
        points_gaussian = hit2cam[..., None] + torch.normal(
            0 * torch.ones(sample_shape), std * torch.ones(sample_shape)
        )
        points_gaussian = points_gaussian.view(num_ray, num_hit * samples_per_points)
        points_gaussian_sampled = []
        for row in points_gaussian:
            p = (row != torch.inf).numpy() * 1
            if p.sum() == 0:
                points_gaussian_sampled.append(np.ones(samples_per_points) * np.inf)
                continue
            p = p / p.sum()
            points_gaussian_sampled.append(
                np.random.choice(row.numpy(), samples_per_points, p=p)
            )

        points_gaussian_sampled = torch.FloatTensor(np.stack(points_gaussian_sampled))
        return points_gaussian_sampled

    def prepare_query(self, ray_bundle, hit2cam):
        if self.is_train and self.sample_gaussian_on:
            ray_distance = self.sample_points_along_ray_uniform(
                hit2cam.shape[0],
                jitter=False,
                znear=self.znear,
                zfar=self.zfar,
                sample_points=self.num_uniform_pt,
            )
            sampled_pts_gaussian = self.sample_points_along_ray_gaussion(
                hit2cam, self.num_gaussian_pt, self.gaussian_std
            )
            ray_distance = torch.cat((sampled_pts_gaussian, ray_distance), dim=1)
            ray_distance, _ = torch.sort(ray_distance, dim=1)
            ray_distance = ray_distance.clamp(self.znear, self.zfar)
            ray_bundle = RayBundle(
                origins=ray_bundle.origins,
                directions=ray_bundle.directions,
                lengths=ray_distance.reshape(*ray_bundle.lengths.shape[:2], -1),
                xys=ray_bundle.xys,
            )

        c, r, _ = ray_bundle.xys.shape

        gt_drdf, loss_weight, first_hit_mask = self.get_gt_drdf(ray_bundle, hit2cam)
        gt_drdf = gt_drdf.reshape(c, r, -1)
        loss_weight = loss_weight.reshape(c, r, -1)
        first_hit_mask = first_hit_mask.reshape(c, r, -1)

        return gt_drdf, loss_weight, first_hit_mask, ray_bundle

    def augmentation(self, img):
        h, w, _ = img.shape
        transformed = self.aug(
            image=img,
        )
        img_aug = transformed["image"]
        return img_aug

    def get_gt_drdf(self, raybundle, hit2cam):
        assert len(hit2cam.shape) == 2
        if self.depth_only_on:
            hit2cam[:, 1:] = np.inf
        pts = rearrange(ray_bundle_to_ray_points(raybundle), "c r p d -> (c r) p d")
        ray_distance = rearrange(raybundle.lengths, "c r p -> (c r) p")
        drdf = get_signed_distance_to_closest_torch(ray_distance, hit2cam)
        i_th_hit = (
            ray_distance[..., None]
            > (
                hit2cam[:, None, : self.max_hit]
                + hit2cam[:, None, 1 : self.max_hit + 1]
            )
            / 2
        ).sum(2)
        first_hit_mask = i_th_hit == 0
        valid_hit_mask = i_th_hit < self.max_hit
        if self.depth_only_on:
            loss_weights = first_hit_mask.float()
        else:
            if self.loss_decay_on:
                loss_weights = torch.exp(-(i_th_hit**2) / self.max_hit)
            else:
                loss_weights = torch.ones_like(i_th_hit)
        loss_weights[~valid_hit_mask] = 0
        loss_weights[hit2cam[:, 0] == np.inf] = 0
        return drdf, loss_weights, first_hit_mask

    def extract_point_visibility_pred(self, pcd, dataset_dict, predictions):
        camera_torch = dataset_dict["camera_torch"]
        if not isinstance(pcd, torch.Tensor):
            pcd = torch.FloatTensor(pcd)
        if pcd.is_cuda:
            device = pcd.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pcd = pcd.to(device)

        points_ndc = project_ndc_depth(
            pcd,
            camera_torch.to(device),
            znear=None,
            zfar=None,
        )
        if len(points_ndc.shape) == 2:
            points_ndc = points_ndc.unsqueeze(0)
        valid_mask = (
            (points_ndc[..., 0] < 1)
            & (points_ndc[..., 0] > -1)
            & (points_ndc[..., 1] < 1)
            & (points_ndc[..., 1] > -1)
            & (points_ndc[..., 2] < self.zfar)
            & (points_ndc[..., 2] > self.znear)
        ).permute(1, 0)
        if predictions is not None:
            depth_layer = 3
            ray_distances = dataset_dict["ray_bundle"].lengths
            ray_distances = rearrange(ray_distances, "c r f -> (c r) f")
            pred_depths = []
            ray_per_cam = dataset_dict["num_ray_per_img"]
            for cam_id in range(len(dataset_dict["camera_torch"])):
                with torch.inference_mode():
                    pred_depth = drdf2depth(
                        drdf=predictions["pred_drdf"][
                            ray_per_cam * cam_id : ray_per_cam * (cam_id + 1), ...
                        ]
                        .detach()
                        .cpu(),
                        ray_distance_query=ray_distances[
                            ray_per_cam * cam_id : ray_per_cam * (cam_id + 1), ...
                        ],
                        camera=dataset_dict["camera_torch"][cam_id].to("cpu"),
                        depth_layer=depth_layer,
                    )
                pred_depths.append(pred_depth)
            # use predicted depth
            depths = (
                torch.cat(
                    [torch.FloatTensor(depth[..., :1]) for depth in pred_depths],
                    dim=-1,
                )
                .permute(2, 0, 1)
                .unsqueeze(1)
                .to(device)
            )
        else:
            depths = self.load_depth(dataset_dict["dataset_dict"]).to(device)
        points_depths = (
            F.grid_sample(
                depths,
                -points_ndc[:, :, :2].unsqueeze(1),
                align_corners=True,
            )
            .squeeze(2)
            .permute(0, 2, 1)
        )
        tol = 0.2
        visibility = (
            (points_depths > (points_ndc[..., 2:] - tol)).squeeze(-1).permute(1, 0)
        )
        visibility[valid_mask == False] = False
        visibility = visibility.cpu()
        valid_mask = valid_mask.cpu()
        return visibility, valid_mask

    def extract_point_visibility_gt(self, dataset_dict, num_samples):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mesh = dataset_dict["gt_mesh"]
        pts_sampled = sample_points_from_meshes(mesh, num_samples)[0]
        points_ndc = project_ndc_depth(
            pts_sampled.to(device),
            dataset_dict["camera_torch"].to(device),
            znear=None,
            zfar=None,
        )
        if len(points_ndc.shape) == 2:
            points_ndc = points_ndc.unsqueeze(0)
        valid_mask = (
            (points_ndc[..., 0] < 1)
            & (points_ndc[..., 0] > -1)
            & (points_ndc[..., 1] < 1)
            & (points_ndc[..., 1] > -1)
            & (points_ndc[..., 2] < self.zfar)
            & (points_ndc[..., 2] > self.znear)
        ).permute(1, 0)
        pts_sampled = pts_sampled[valid_mask.sum(1) > 0]
        points_ndc = points_ndc[:, valid_mask.sum(1) > 0, :]
        valid_mask = valid_mask[valid_mask.sum(1) > 0]
        depths = self.load_depth(dataset_dict["dataset_dict"]).to(device)
        points_depths = (
            F.grid_sample(
                depths,
                -points_ndc[:, :, :2].unsqueeze(1),
                align_corners=True,
            )
            .squeeze(2)
            .permute(0, 2, 1)
        )
        tol = 0.2
        visibility = (
            (points_depths > (points_ndc[..., 2:] - tol)).squeeze(-1).permute(1, 0)
        )
        visibility[valid_mask == False] = False
        pts_sampled = pts_sampled.cpu()
        visibility = visibility.cpu()
        valid_mask = valid_mask.cpu()
        return pts_sampled, visibility, valid_mask

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        if self.is_train:
            sample_view = np.random.choice(self.train_view)
            select_idx = np.random.choice(
                np.arange(len(dataset_dict["cameras"])), sample_view, replace=False
            )
        else:
            sample_view = np.random.choice(self.test_view)
            select_idx = np.arange(sample_view)

        rtn_dicts = defaultdict(list)
        dataset_dict["cameras"] = [dataset_dict["cameras"][i] for i in select_idx]
        if self.is_train:
            dataset_dict.pop("room_ids_union")
            dataset_dict.pop("relation_graph")
            dataset_dict.pop("cameras_selected")
            dataset_dict.pop("invisible_face_area_ratio")

        origin = torch.FloatTensor(np.array([dataset_dict["cameras"][0]["c2w"][:3, 3]]))
        gt_mesh = self.extract_mesh(dataset_dict, origin, select_idx)
        rtn_dicts["rgbs"] = self.load_rgb(dataset_dict)
        rtn_dicts["camera_torch"], dataset_dict = self.load_camera(dataset_dict, origin)

        ray_bundle, camera_mode = self.ray_sampler.sample(
            rtn_dicts["camera_torch"],
            camera_mode=self.camera_mode,
            train_on=self.is_train,
        )
        try:
            hit2origin, ray_bundle = self.load_hit2origin(gt_mesh, ray_bundle)
        except:
            print("Err in load_hit2origin")
            print(dataset_dict["cameras"])
            return self.prev_rtn_dict
        gt_drdf, loss_weight, first_hit_mask, ray_bundle = self.prepare_query(
            ray_bundle, hit2origin
        )

        rtn_dicts.update(
            {
                "ray_bundle": ray_bundle,
                "gt_drdf": gt_drdf,
                "dataset_dict": dataset_dict,
                "height": self.resize,
                "width": self.resize,
                "num_ray_per_img": ray_bundle.xys.shape[1],
                "origin": origin,
                "loss_weights": loss_weight,
                "first_hit_masks": first_hit_mask,
                "gt_mesh": gt_mesh,
                "hit2origin": hit2origin,
                "ray_camera_mode": camera_mode,
                "vis_camera_id": [
                    i for i in range(len(rtn_dicts["camera_torch"]))
                ],  # index for camera to be visualized in tensorboard
            }
        )
        self.prev_rtn_dict = copy.deepcopy(rtn_dicts)
        return rtn_dicts


class Dataloader3DFire:
    def __init__(self, dataset_dicts, mapper):
        self.dataset_dicts = dataset_dicts
        self.mapper = mapper

    def __getitem__(self, idx):
        return self.mapper(self.dataset_dicts[idx])

    def __len__(self):
        return len(self.dataset_dicts)


def collate_fn(examples):
    return examples


def get_dataloader(cfg):
    tr_dataset = Dataloader3DFire(
        dataset_dicts=load_taskonomy_pkl("train_set3"),
        mapper=SpDRDFMapper(cfg, is_train=True),
    )
    te_dataset = Dataloader3DFire(
        dataset_dicts=load_taskonomy_pkl("val_set3"),
        mapper=SpDRDFMapper(cfg, is_train=False),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    from fires.configs.default import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file("ckpts/config.yaml")
    dataset_dicts = load_taskonomy_pkl(cfg.DATASETS.TRAIN[0])
    dataloader = SpDRDFMapper(cfg)
    dataloader(dataset_dicts[0])
