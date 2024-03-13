import numpy as np
import plotly.graph_objects as go
import pytorch3d
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch3d.renderer import ray_bundle_to_ray_points
from pytorch3d.renderer.cameras import OrthographicCameras, PerspectiveCameras
from pytorch3d.renderer.implicit.raysampling import (MonteCarloRaysampler,
                                                     MultinomialRaysampler,
                                                     RayBundle)
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix

from fires.utils.geometry_utils import unproject_ndc_depth


def concat_raybundle(bundle_list):
    """
    Concatenate multiple RayBundles into a single RayBundle.

    Args:
        bundle_list (list): A list of RayBundle objects to be concatenated.

    Returns:
        RayBundle: A single RayBundle object containing the concatenated data.
    """

    # Check if the list is empty
    if not bundle_list:
        raise ValueError("The provided bundle list is empty.")

    # Initialize lists to hold concatenated data
    origins = []
    directions = []
    lengths = []
    xys = []

    # Loop through each bundle and append data to lists
    for bundle in bundle_list:
        origins.append(bundle.origins)
        directions.append(bundle.directions)
        lengths.append(bundle.lengths)
        xys.append(bundle.xys)

    # Concatenate lists to form new tensors
    origins = torch.cat(origins, dim=1)
    directions = torch.cat(directions, dim=1)
    lengths = torch.cat(lengths, dim=1)
    xys = torch.cat(xys, dim=1)

    # Return a new RayBundle
    return RayBundle(origins=origins, directions=directions, lengths=lengths, xys=xys)


class RayBundleSampler:
    def __init__(
        self,
        ray_per_cam,
        num_uniform_pt,
        znear,
        zfar,
        orthographic_size,
        multinomial_size,
        high_res=False,
    ):
        self.ray_per_cam = ray_per_cam
        self.num_uniform_pt = num_uniform_pt
        self.znear = znear
        self.zfar = zfar
        self.orthographic_size = orthographic_size  # meter
        self.multinomial_size = multinomial_size
        self.high_res = high_res
        self.camera_modes = [
            "perspective",
            "orthographic",
            "orthographic_right",
            "orthographic_up",
            "random",
        ]
        pass

    def sample(self, camera, camera_mode, train_on, pt_per_side=None):
        assert camera_mode in self.camera_modes
        if train_on:
            ray_sampler = MonteCarloRaysampler(
                min_x=-1.0,
                max_x=1.0,
                min_y=-1.0,
                max_y=1.0,
                n_rays_per_image=self.ray_per_cam,
                n_pts_per_ray=self.num_uniform_pt,
                min_depth=self.znear,
                max_depth=self.zfar,
            )
        else:
            # designed for inferencing, because all points are sampled uniformly.
            if pt_per_side is None:
                pt_per_side = 512 if self.high_res else 128
            resolution = 2 / (pt_per_side + 1)
            ray_sampler = MultinomialRaysampler(
                min_x=1.0 - resolution,
                max_x=-1.0 + resolution,
                min_y=1.0 - resolution,
                max_y=-1.0 + resolution,
                image_height=pt_per_side,
                image_width=pt_per_side,
                n_pts_per_ray=256,  # self.num_uniform_pt,
                min_depth=self.znear,
                max_depth=self.zfar,
            )
        if camera_mode == "random":
            camera_mode = np.random.choice(self.camera_modes[:-1], 1)
        if camera_mode == "perspective":
            ray_bundle = ray_sampler(camera)
        elif camera_mode == "orthographic":
            camera_ortho = OrthographicCameras(
                focal_length=torch.ones(len(camera), 1) * 2 / self.orthographic_size,
                R=camera.R,
                T=camera.T,
            )
            ray_bundle = ray_sampler(camera_ortho)
        elif "orthographic_" in camera_mode:
            camera_ortho_origin = OrthographicCameras(
                focal_length=torch.ones(len(camera), 1) * 2 / self.orthographic_size,
                R=camera.R,
                T=camera.T,
            )
            if "right" in camera_mode:
                volumn_center = unproject_ndc_depth(
                    torch.FloatTensor([[0, 0, 0]]), camera_ortho_origin, 0, self.zfar
                ).squeeze(1)
                volumn_up = unproject_ndc_depth(
                    torch.FloatTensor([[0, 1, 0]]), camera_ortho_origin, 0, self.zfar
                ).squeeze(1)
                axis_angle_quat = (
                    F.normalize(volumn_up - volumn_center, dim=1) * np.pi / 2
                )
                rotation_m = pytorch3d.transforms.axis_angle_to_matrix(axis_angle_quat)
            elif "up" in camera_mode:
                volumn_center = unproject_ndc_depth(
                    torch.FloatTensor([[0, 0, 0]]), camera_ortho_origin, 0, self.zfar
                ).squeeze(1)
                volumn_right = unproject_ndc_depth(
                    torch.FloatTensor([[-1, 0, 0]]), camera_ortho_origin, 0, self.zfar
                ).squeeze(1)
                axis_angle_quat = (
                    F.normalize(volumn_right - volumn_center, dim=1) * -np.pi / 2
                )
                rotation_m = pytorch3d.transforms.axis_angle_to_matrix(axis_angle_quat)
            else:
                breakpoint()

            transformation = Translate(-volumn_center).compose(
                Rotate(R=rotation_m).compose(
                    Translate(volumn_center)
                    .compose(Rotate(camera.R))
                    .compose(Translate(camera.T))
                )
            )
            R_pytorch = transformation.get_matrix()[:, :3, :3]
            T_pytorch = transformation.get_matrix()[:, 3, :3]
            camera_ortho_right = OrthographicCameras(
                focal_length=torch.ones(len(camera), 1) * 2 / self.orthographic_size,
                R=R_pytorch,
                T=T_pytorch,
            )
            ray_bundle = ray_sampler(camera_ortho_right)
        else:
            breakpoint()
        if not train_on:
            ray_bundle = RayBundle(
                origins=rearrange(ray_bundle.origins, "c s1 s2 d -> c (s1 s2) d"),
                directions=rearrange(ray_bundle.directions, "c s1 s2 d -> c (s1 s2) d"),
                lengths=rearrange(ray_bundle.lengths, "c s1 s2 d -> c (s1 s2) d"),
                xys=rearrange(ray_bundle.xys, "c s1 s2 d -> c (s1 s2) d"),
            )
        # normalize ray
        ray_bundle = RayBundle(
            origins=ray_bundle.origins,
            directions=F.normalize(ray_bundle.directions, dim=-1),
            lengths=ray_bundle.lengths
            * torch.norm(ray_bundle.directions, dim=-1)[..., None],
            xys=ray_bundle.xys,
        )
        return ray_bundle, camera_mode


def vis_query_ray(ray_bundle, name):
    bid = 0
    rays_points = ray_bundle_to_ray_points(ray_bundle)[bid].reshape(-1, 3)
    vectors = ray_bundle.origins + ray_bundle.directions
    vectors = vectors[bid].numpy().reshape(-1, 3)

    # Create lines from origin to the vectors
    lines = []
    for ray_id, vec in enumerate(vectors):
        ray_origin = ray_bundle.origins[bid].reshape(-1, 3)[ray_id]

        lines.append(
            go.Scatter3d(
                x=[ray_origin[0], vec[0]],
                y=[ray_origin[1], vec[1]],
                z=[ray_origin[2], vec[2]],
                mode="lines",
                line=dict(width=6),
            )
        )

    # Create a 3D scatter plot
    fig = go.Figure(data=lines)

    # Add markers at the end of the vectors
    fig.add_trace(
        go.Scatter3d(
            x=rays_points[:, 0],
            y=rays_points[:, 1],
            z=rays_points[:, 2],
            mode="markers",
            marker=dict(size=5, opacity=0.8),
        )
    )

    # Set axis titles
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    # Save as HTML
    fig.write_html(f"html/v31_ray_sampler/{name}.html")
    print(f"saving to html/v31_ray_sampler/{name}.html")


def main():
    ray_sampler = RayBundleSampler()
    Rs = torch.cat([torch.eye(3)[None] for _ in range(3)])
    Ts = torch.cat([torch.zeros(3)[None] for _ in range(3)])
    focals = torch.cat(
        [1.0 / torch.tan(torch.FloatTensor([np.radians(64)]) / 2) for _ in range(3)],
        axis=0,
    ).unsqueeze(1)
    camera = PerspectiveCameras(focal_length=focals, R=Rs, T=Ts, image_size=1)
    train_on = True
    bundle_list = []
    for name in [
        "orthographic",
        "orthographic_right",
        "orthographic_up",
        "perspective",
    ]:
        ray_bundle, camera_mode = ray_sampler.sample(camera, name, train_on)
        vis_query_ray(ray_bundle, name + f"_train{train_on}")
        bundle_list.append(ray_bundle)

    vis_query_ray(concat_raybundle(bundle_list), f"merge_{train_on}")


if __name__ == "__main__":
    main()
