import numpy as np
import torch
from einops import rearrange, repeat
from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from fires.model.midas.model_loader import load_midas_model
from fires.model.resnetfc import PositionalEncoding, ResnetFC
from fires.utils.geometry_utils import (apply_log_transform, drdf2pcd,
                                   project_ndc_depth)


def compute_loss(pred, gt, weights, first_hit_masks):
    base_loss_fn = torch.nn.L1Loss(reduction="none")
    loss_base = base_loss_fn(
        apply_log_transform(pred.reshape(-1)), apply_log_transform(gt.reshape(-1))
    ) * weights.view(-1)
    losses = {
        "drdf_regression_loss_1st_hit": loss_base[first_hit_masks].sum()
        / weights.view(-1)[first_hit_masks].sum(),
        "drdf_regression_loss_occ_hit": loss_base[~first_hit_masks].sum()
        / weights.view(-1)[~first_hit_masks].sum(),
    }
    if first_hit_masks.sum() == 0 or weights.view(-1)[first_hit_masks].sum() == 0:
        losses["drdf_regression_loss_1st_hit"] = (loss_base * 0.0).sum()
    if (~first_hit_masks).sum() == 0 or weights.view(-1)[~first_hit_masks].sum() == 0:
        losses["drdf_regression_loss_occ_hit"] = (loss_base * 0.0).sum()
    return losses


class Projector:
    def __init__(self):
        pass

    def inbound(self, point_ndc):
        """
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 3]
        :return: mask, bool, [...]
        """
        threshold = 1

        return (
            (point_ndc[..., 0] < threshold)
            & (point_ndc[..., 0] > -threshold)
            & (point_ndc[..., 1] < threshold)
            & (point_ndc[..., 1] > -threshold)
            & (point_ndc[..., 2] < threshold)
            & (point_ndc[..., 2] > -threshold)
        )

    def compute_angle(self, pts, cameras, ray_dirs):
        """
        :param pts: [n_ray, n_pt_per_ray, 3]
        :param cameras: [n_cam]
        :param ray_dirs: [n_ray, 3]
        :return: ray_diff: [n_cam, n_pt, 4]; The first 3 channels are unit-length vector of the difference between
               query and target ray directions, the last channel is the inner product of the two directions.
        """
        n_ray, n_pt_per_ray, _3 = pts.shape
        assert ray_dirs.shape == (n_ray, 3)
        # Repeat ray_dirs without reshaping
        pt2tar_pose = -ray_dirs.unsqueeze(1).unsqueeze(0)
        # Compute pt2ref_pose without reshaping
        pt2ref_pose = F.normalize(
            cameras.get_camera_center().unsqueeze(1).unsqueeze(2) - pts.unsqueeze(0),
            dim=-1,
        )

        ray_diff = pt2ref_pose - pt2tar_pose
        # ray_diff_dir = F.normalize(ray_diff, dim=-1)

        # Compute ray_diff_dot without keepdim=True
        ray_diff_dot = torch.sum(pt2ref_pose * pt2tar_pose, dim=-1).unsqueeze(-1)

        # Concatenate ray_diff_dir and ray_diff_dot
        ray_diff = torch.cat([ray_diff, ray_diff_dot], dim=-1)
        return ray_diff

    def compute(self, pts, cameras, images, featmaps, ray_dirs):
        """
        :param pts: [n_ray, n_pt, 3]
        :param camers: [n_cam]
        :param images: [n_cam, 3, im_h, im_w]
        :param featmaps: [n_cam, d_feat, feat_h, feat_w]
        :param ray_dirs: [n_ray, 3]
        :return: rgb_feat_sampled_list: List([n_camera, n_rays, 3+n_feat]),
                 valid_mask_list: List([n_camera, n_rays, 1]),
                 ray_diff_list: List([n_camera, n_rays, 4]),
                 pt_ndc_list: List([n_camera, n_rays, 3])
        """
        # input sanity check
        assert len(pts.shape) == 3
        assert len(ray_dirs.shape) == 2
        assert cameras.in_ndc()
        point_ndc = project_ndc_depth(
            pts.view(-1, 3),
            cameras,
            znear=0,
            zfar=8,
        )
        if len(cameras) == 1:
            point_ndc = point_ndc.unsqueeze(0)
        valid_mask = self.inbound(point_ndc)[..., None]
        # negate axis for grid sampling
        normalized_pixel_locations = -point_ndc[:, :, :2].unsqueeze(1)
        normalized_pixel_locations = torch.clamp(normalized_pixel_locations, -1, 1)

        # rgb + feature sampling
        feat_sampled = (
            F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True)
            .squeeze(2)
            .permute(0, 2, 1)
        )
        # ray angles
        ray_diff = self.compute_angle(pts, cameras, ray_dirs)

        n_cam, n_ray, n_pt_per_ray, _4 = ray_diff.shape
        feat_sampled = feat_sampled.view(n_cam, n_ray, n_pt_per_ray, -1)
        valid_mask = valid_mask.view(n_cam, n_ray, n_pt_per_ray, -1)
        point_ndc = point_ndc.view(n_cam, n_ray, n_pt_per_ray, -1)
        return feat_sampled, valid_mask, ray_diff, point_ndc


class ScaledTanh(nn.Module):
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        return self.max_value * torch.tanh(x)


class DRDF_MLP(nn.Module):
    def __init__(self, cfg, in_feat_ch=32, **kwargs):
        super(DRDF_MLP, self).__init__()
        self.cfg = cfg
        self.cam_attention_on = cfg.MODEL.DRDF_MLP.CAM_ATTENTION_ON
        self.ray_attention_on = cfg.MODEL.DRDF_MLP.RAY_ATTENTION_ON
        activation_func = nn.ReLU(inplace=False)
        self.pt_ndc_emb = PositionalEncoding(
            num_freqs=cfg.MODEL.DRDF_MLP.POSITIONAL_ENCODING_FREQ, d_in=3
        )
        self.ray_dir_emb = PositionalEncoding(
            num_freqs=cfg.MODEL.DRDF_MLP.POSITIONAL_ENCODING_FREQ, d_in=4
        )

        self.feature_fc = ResnetFC(
            d_in=self.pt_ndc_emb.d_out + self.ray_dir_emb.d_out + in_feat_ch,
            n_blocks=3,
            d_out=cfg.MODEL.DRDF_MLP.MLP_FEATURE_DIM,
            activation=activation_func,
            use_batch_norm=cfg.MODEL.DRDF_MLP.MLP_BATCH_NORM,
        )

        mlp_input_size = self.pt_ndc_emb.d_out + self.ray_dir_emb.d_out + in_feat_ch

        self.feature_fc_weights = ResnetFC(
            d_in=mlp_input_size,
            n_blocks=3,
            d_out=cfg.MODEL.DRDF_MLP.MLP_FEATURE_DIM,
            activation=activation_func,
            use_batch_norm=cfg.MODEL.DRDF_MLP.MLP_BATCH_NORM,
        )
        if self.cam_attention_on:
            self.cam_attention_torch = torch.nn.MultiheadAttention(
                cfg.MODEL.DRDF_MLP.MLP_FEATURE_DIM, 4
            )
        if self.ray_attention_on:
            self.ray_attention_torch = torch.nn.MultiheadAttention(
                cfg.MODEL.DRDF_MLP.MLP_FEATURE_DIM, 4
            )

        self.weights_mlp_mid = ResnetFC(
            d_in=cfg.MODEL.DRDF_MLP.MLP_FEATURE_DIM,
            n_blocks=3,
            d_out=cfg.MODEL.DRDF_MLP.MLP_FEATURE_DIM,
            activation=nn.Sigmoid(),
            use_batch_norm=cfg.MODEL.DRDF_MLP.MLP_BATCH_NORM,
        )
        self.out_geometry_fc = ResnetFC(
            d_in=cfg.MODEL.DRDF_MLP.MLP_FEATURE_DIM,
            d_out=1,
            last_op=ScaledTanh(max_value=1),
            activation=None,  # it will be ReLU since beta > 0
            use_batch_norm=cfg.MODEL.DRDF_MLP.MLP_BATCH_NORM,
        )

    def forward(self, rgb_feat, valid_mask, ray_diff, point_ndc):
        """
        :param rgb_feat: rgbs and image features [bn, n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [bn, n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [bn, n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [bn, n_rays, n_samples, 4]
        """
        n_cam, n_ray, n_pt_per_ray, _1 = valid_mask.shape
        assert rgb_feat.shape[:-1] == (n_cam, n_ray, n_pt_per_ray)
        assert ray_diff.shape[:-1] == (n_cam, n_ray, n_pt_per_ray)
        assert point_ndc.shape[:-1] == (n_cam, n_ray, n_pt_per_ray)

        direction_feat = self.ray_dir_emb(
            ray_diff.reshape(n_cam * n_ray * n_pt_per_ray, 4) / 2.0
        ).view(n_cam, n_ray, n_pt_per_ray, -1)
        pt_ndc_feat = self.pt_ndc_emb(
            torch.clamp(
                point_ndc,
                -1,
                1,
            ).reshape(n_cam * n_ray * n_pt_per_ray, 3)
        ).view(n_cam, n_ray, n_pt_per_ray, -1)

        total_feat = torch.cat(
            [rgb_feat, direction_feat, pt_ndc_feat], dim=-1
        )  # n_cam, n_ray, n_pt_per_ray, feat_dim

        drdf_feat = self.feature_fc(total_feat)  # n_cam, n_ray, n_pt_per_ray, feat_dim
        x = self.feature_fc_weights(total_feat)  # n_cam, n_ray, n_pt_per_ray, feat_dim
        x = x * valid_mask

        if self.cam_attention_on:
            x = x.reshape(
                n_cam, n_ray * n_pt_per_ray, -1
            )  # n_cam, (n_ray * n_pt_per_ray), feat_dim
            x, _ = self.cam_attention_torch(x, x, x, need_weights=False)
            x = x.reshape(n_cam, n_ray, n_pt_per_ray, -1)

        weights = self.weights_mlp_mid(x)  # (n_ray * n_pt_per_ray), 3
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-5)
        drdf_feat = weights * drdf_feat
        drdf_feat = drdf_feat.sum(dim=0)
        if self.ray_attention_on:
            drdf_feat = rearrange(drdf_feat, "nr npt d->npt nr d")
            drdf_feat, _ = self.ray_attention_torch(
                drdf_feat, drdf_feat, drdf_feat, need_weights=False
            )
            drdf_feat = rearrange(drdf_feat, "npt nr d->nr npt d")
        drdf_refined = self.out_geometry_fc(drdf_feat)
        return drdf_refined


class SpDRDFNet(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(
        self,
        cfg,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()

        self.backbone, normalization, input_w, input_h = load_midas_model(
            model_type=cfg.MODEL.BACKBONE.NAME
        )
        pixel_mean = [m * 255.0 for m in normalization._mean]
        pixel_std = [s * 255.0 for s in normalization._std]
        self.max_hit = cfg.DATASET_GENERATE.MAX_HIT
        self.projector = Projector()

        self.drdf_mlp = DRDF_MLP(cfg, in_feat_ch=self.backbone.d_out)
        self.drdf_tau = cfg.MODEL.DRDF_TAU
        self.camera_mode = cfg.DATALOADER.CAMERA_MODE
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.freeze = cfg.MODEL.FREEZE
        self.cfg = cfg
        self.input_w = input_w
        self.input_h = input_h
        self.znear = cfg.DATASET_GENERATE.ZNEAR
        self.zfar = cfg.DATASET_GENERATE.ZFAR

        for layers in self.freeze:
            layer = layers.split(".")
            final = self
            for l in layer:
                final = getattr(final, l)
            for params in final.parameters():
                params.requires_grad = False

    @property
    def device(self):
        return self.pixel_mean.device

    def query(
        self,
        xyz_world,
        ray_dirs,
        cameras,
        images,
        featmaps,
        is_train,
        unit_per_chunk=2000,
    ):
        if is_train:
            num_chunks = 1
        else:
            num_chunks = max(len(xyz_world) // unit_per_chunk, 1)
        pts_chunks = torch.chunk(xyz_world, num_chunks, dim=0)
        ray_dirs_chunks = torch.chunk(ray_dirs, num_chunks, dim=0)
        results = []

        num_chunks = len(pts_chunks)
        if num_chunks > 1 or not is_train:
            iterations = range(num_chunks)
        else:
            iterations = range(num_chunks)

        for i in iterations:
            rgb_feat_padded, valid_mask_padded, ray_diff_padded, point_ndc_padded = (
                self.projector.compute(
                    pts_chunks[i],
                    cameras,
                    images,
                    featmaps,
                    ray_dirs_chunks[i],
                )
            )
            drdf_refined = self.drdf_mlp(
                rgb_feat_padded, valid_mask_padded, ray_diff_padded, point_ndc_padded
            )

            results.append(drdf_refined)
        drdf = torch.cat(results, dim=0)
        return drdf

    def forward(self, batched_inputs):
        assert len(batched_inputs) == 1
        batched_inputs = batched_inputs[0]
        images_packed = batched_inputs["rgbs"].to(self.device)

        if self.input_w:
            assert images_packed.shape[-1] == self.input_w
        if self.input_h:
            assert images_packed.shape[-2] == self.input_h

        images_packed = (images_packed - self.pixel_mean) / self.pixel_std
        featmaps_packed = self.backbone(
            images_packed
        )  # (n_img, feat_dim, feat_h, feat_w)

        xyz_world = ray_bundle_to_ray_points(batched_inputs["ray_bundle"]).to(
            self.device
        )
        ray_origins = batched_inputs["ray_bundle"].origins.to(self.device)
        ray_dirs = F.normalize(batched_inputs["ray_bundle"].directions, dim=-1).to(
            self.device
        )
        cameras = batched_inputs["camera_torch"].to(self.device)

        xyz_world = rearrange(xyz_world, "c r p f -> (c r) p f")
        ray_dirs = rearrange(ray_dirs, "c r f -> (c r) f")
        ray_origins = rearrange(ray_origins, "c r f -> (c r) f")
        drdf = self.query(
            xyz_world,  # ray, pt, 3
            ray_dirs,  # ray, 3
            cameras,
            images_packed,
            featmaps_packed,
            is_train=self.training,
        )

        if self.training:
            pred_drdf = drdf.squeeze(-1)
            gt_drdf = torch.clamp(
                batched_inputs["gt_drdf"].to(self.device), -self.drdf_tau, self.drdf_tau
            )
            loss_weights = batched_inputs["loss_weights"].to(self.device)
            first_hit_masks = batched_inputs["first_hit_masks"].to(self.device).view(-1)

            losses = compute_loss(pred_drdf, gt_drdf, loss_weights, first_hit_masks)

            for k, x in losses.items():
                if torch.any(torch.isnan(x)):
                    print(f"{k} is nan")
                    breakpoint()
                    pass
            return losses
        pcd_world, visibility, rayid, ptid = drdf2pcd(
            drdf=drdf,
            ray_dirs=ray_dirs.to(self.device),
            ray_pts=xyz_world.to(self.device),
        )
        # inference
        postprocessed = [
            {
                "pred_drdf": drdf,
                "pcd_world": pcd_world,
                "visibility": visibility,
                "rayid": rayid,
                "ptid": ptid,
                "imgid": rayid // batched_inputs["num_ray_per_img"],
            }
        ]
        return postprocessed


def load_model(model_path, cfg, device):
    net = SpDRDFNet(cfg)
    print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    net.load_state_dict(ckpt["model"], strict=False)
    return net.to(device)
