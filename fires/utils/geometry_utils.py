import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pytorch3d
import torch
import torch.nn as nn
import trimesh
from numba import njit
from numpy.core.fromnumeric import clip
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes, join_meshes_as_scene
from torch import nn
from torch.nn import functional as F
from trimesh.ray.ray_pyembree import RayMeshIntersector

from fires.utils.camera import camera_mesh_from_fov_camera
from fires.utils.colormap import colormap as d2colormap


def to_numpy(input_data):
    if isinstance(input_data, np.ndarray):
        return input_data
    elif isinstance(input_data, torch.Tensor):
        return input_data.detach().cpu().numpy()
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor")


def convert_points_to_mesh_pytorch3d(points, color, radius, highres=False):
    points = to_numpy(points)
    if highres:
        icosphere = trimesh.creation.icosahedron()
        radius = 0.02
    else:
        icosphere = trimesh.creation.box()
    icosphere.vertices *= radius
    faces = icosphere.faces
    vertices = icosphere.vertices
    faces_offset = np.arange(0, len(points), dtype=np.int32)
    faces_offset = len(vertices) * faces_offset[:, None] * np.ones((1, len(faces)))

    new_vertices = vertices[None,] + points[:, None, :]
    new_vertices = new_vertices.reshape(-1, 3)
    new_faces = faces_offset[:, :, None] + faces[None,]
    new_faces = new_faces.reshape(-1, 3)
    vertex_colors = torch.FloatTensor(
        np.tile(color[:, None, :], (1, len(vertices), 1)).reshape(-1, 3)
    )

    textures = TexturesVertex(verts_features=[vertex_colors])
    new_vertices = torch.tensor(new_vertices.copy(), dtype=torch.float32)
    new_faces = torch.tensor(new_faces.copy(), dtype=torch.int64)
    try:
        return Meshes(
            verts=[torch.FloatTensor(new_vertices)],
            faces=[new_faces],
            textures=textures,
        )
    except:
        breakpoint()


def save_pytorch3d_mesh_as_glb(pytorch3d_mesh, glb_path, face_visibilities=None):
    if face_visibilities is not None:
        o3d_mesh = o3d.t.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(pytorch3d_mesh.verts_packed().cpu().numpy()),
            o3d.utility.Vector3iVector(pytorch3d_mesh.faces_packed().cpu().numpy()),
        )
        cm = d2colormap(rgb=True, maximum=1)[: face_visibilities.shape[1], :]
        face_visibilities = F.normalize(face_visibilities.float(), p=1, dim=1).numpy()
        blended_face_colors = face_visibilities @ cm
        # Ensure the computed colors are in [0, 1]
        blended_face_colors = np.clip(blended_face_colors, 0, 1)

        face_indices = pytorch3d_mesh.faces_packed().cpu().numpy()

        # Create a vertex_colors array
        vertex_colors = np.zeros((pytorch3d_mesh.verts_packed().shape[0], 3))

        # Compute vertex colors using vectorized operations
        # Add up colors from each face and count occurrences of each vertex
        np.add.at(vertex_colors, face_indices.ravel(), blended_face_colors)
        flag = o3d.t.io.write_triangle_mesh(
            glb_path, o3d_mesh, write_triangle_uvs=face_visibilities is not None
        )

    else:
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(pytorch3d_mesh.verts_packed().cpu().numpy()),
            o3d.utility.Vector3iVector(pytorch3d_mesh.faces_packed().cpu().numpy()),
        )
        flag = o3d.io.write_triangle_mesh(
            glb_path, o3d_mesh, write_vertex_normals=False, write_vertex_colors=False
        )

    return flag


def save_scene_as_glb(
    points, color, camera_torch, glb_path, radius=0.05, highres=False, camid=0
):
    if len(points) == 0:
        points = np.zeros((1, 3))
        color = np.zeros((1, 3))
        print("len(points) = 0")
    cm = d2colormap(rgb=True, maximum=1)
    meshes_list = []
    meshes_list.append(convert_points_to_mesh_pytorch3d(points, color, radius, highres))
    for idx in range(len(camera_torch)):
        meshes_list.append(
            camera_mesh_from_fov_camera(
                camera_torch[idx].to(torch.device("cpu")), color=cm[idx + camid]
            )
        )

    joined_mesh = join_meshes_as_scene(meshes_list)
    verts = joined_mesh.verts_packed().cpu().numpy()
    faces = joined_mesh.faces_packed().cpu().numpy()
    vertex_colors = joined_mesh.textures.verts_features_packed().cpu().numpy()

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    )
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    o3d.io.write_triangle_mesh(
        glb_path, o3d_mesh, write_vertex_normals=False, write_vertex_colors=True
    )
    return True


class VisibilityCategory:
    def __init__(self):
        self.categories = [
            # 'single',
            "single-visible",
            "single-hidden",
            "multi-all-hidden",
            "multi-all-visible",
            "multi-mixed",
            "all-hidden",
            "any-visible",
        ]

    def classify(self, visibility, valid_mask):
        category_masks = {}
        for cat in self.categories:
            # if cat == 'single':
            #     category_masks[cat] = valid_mask.sum(1) == 1
            if cat == "single-visible":
                category_masks[cat] = (valid_mask.sum(1) == 1) & (
                    visibility.sum(1) == 1
                )
            elif cat == "single-hidden":
                category_masks[cat] = (valid_mask.sum(1) == 1) & (
                    visibility.sum(1) == 0
                )
            elif cat == "multi-all-hidden":
                category_masks[cat] = (valid_mask.sum(1) > 1) & (visibility.sum(1) == 0)
            elif cat == "multi-all-visible":
                category_masks[cat] = (valid_mask.sum(1) > 1) & (
                    visibility.sum(1) == valid_mask.sum(1)
                )
            elif cat == "multi-mixed":
                category_masks[cat] = (
                    (valid_mask.sum(1) > 1)
                    & (visibility.sum(1) > 0)
                    & (visibility.sum(1) < valid_mask.sum(1))
                )
            elif cat == "all-hidden":
                category_masks[cat] = (valid_mask.sum(1) >= 1) & (
                    visibility.sum(1) == 0
                )
            elif cat == "any-visible":
                category_masks[cat] = (valid_mask.sum(1) >= 1) & (visibility.sum(1) > 0)
            else:
                raise NotImplementedError()
        return category_masks


def project_ndc_depth(xyz_world, camera_torch, znear, zfar):
    """
    if znear=zfar=None, z is depth.
    otherwise, [znear, zfar] is mapped to [-1, 1]
    """
    depth = camera_torch.get_world_to_view_transform().transform_points(xyz_world)[
        ..., 2:
    ]
    xy_ndc = camera_torch.transform_points(xyz_world)[..., :2]
    if znear is not None:
        if zfar is not None:
            m = 2.0 / (zfar - znear)
            b = -2.0 * znear / (zfar - znear) - 1
        else:
            assert 0
    else:
        m = 1
        b = 0
    z_ndc = depth * m + b
    return torch.cat((xy_ndc, z_ndc), dim=-1)


def unproject_ndc_depth(xyz_ndc, camera_torch, znear, zfar):
    """unproject points from ndc from to world frame.
    ndc definition:
        left to right: x from +1 to -1
        top to bottom: y from +1 to -1
        zbuffer from znear to zfar: z from -1 to 1.
    Args:
        xyz_ndc (_type_): _description_
        camera_torch (_type_): _description_
        znear (_type_): _description_
        zfar (_type_): _description_
    """
    xy_ndc = xyz_ndc[..., :2]
    z_ndc = xyz_ndc[..., 2:]
    if znear is not None:
        if zfar is not None:
            m = 2.0 / (zfar - znear)
            b = -2.0 * znear / (zfar - znear) - 1
        else:
            assert 0
    else:
        m = 1
        b = 0
    depth = (z_ndc - b) / m
    xy_depth = torch.cat((xy_ndc, depth), dim=-1)
    xyz_world = camera_torch.unproject_points(
        xy_depth,
        world_coordinates=True,
    )
    return xyz_world


def to_numpy(input_data):
    if isinstance(input_data, np.ndarray):
        return input_data
    elif isinstance(input_data, torch.Tensor):
        return input_data.detach().cpu().numpy()
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor")


def zbuf2raydistance(zbuf, camera_torch):
    intHeight, intWidth = zbuf.shape[:2]
    assert camera_torch.focal_length[0][0] == camera_torch.focal_length[0][1]
    fltFocal = (
        camera_torch.focal_length[0][0].item() * intHeight / 2
    )  # divide 2 since pytorch3d assumes different convension
    npyImageplaneX = (
        np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)
        .reshape(1, intWidth)
        .repeat(intHeight, 0)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneY = (
        np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight)
        .reshape(intHeight, 1)
        .repeat(intWidth, 1)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    distance = zbuf * np.linalg.norm(npyImageplane, 2, 2)[..., None] / fltFocal
    return distance


def raydistance2zbuf(raydistance, camera_torch):
    intHeight, intWidth = raydistance.shape[:2]
    assert camera_torch.focal_length[0][0] == camera_torch.focal_length[0][1]
    fltFocal = (
        camera_torch.focal_length[0][0].item() * intHeight / 2
    )  # divide 2 since pytorch3d assumes different convension
    npyImageplaneX = (
        np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)
        .reshape(1, intWidth)
        .repeat(intHeight, 0)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneY = (
        np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight)
        .reshape(intHeight, 1)
        .repeat(intWidth, 1)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    # distance = zbuf * np.linalg.norm(npyImageplane, 2, 2)[...,None] / fltFocal
    zbuf = raydistance / np.linalg.norm(npyImageplane, 2, 2)[..., None] * fltFocal
    return zbuf


@njit
def smallest_k_values(index_ray, d_hit2cam, k, total_num_ray):
    result = np.full((total_num_ray, k), np.inf)

    for i in range(len(index_ray)):
        ray_id = index_ray[i]
        value = d_hit2cam[i]
        for j in range(k):
            if value < result[ray_id, j]:
                result[ray_id, j + 1 :] = result[ray_id, j:-1]
                result[ray_id, j] = value
                break

    return result


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


class DRDFIntersectionFinder(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        self.hanning_smoother = torch.nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 5),
            stride=1,
            bias=False,
            padding="same",
        )
        self.sign_detector = torch.nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 2),
            stride=1,
            bias=False,
            padding="same",
        )

        hanning_weights = np.hanning(window_size)
        hanning_weights = torch.FloatTensor(hanning_weights).to(
            self.hanning_smoother.weight.dtype
        )
        hanning_weights = hanning_weights.reshape(self.hanning_smoother.weight.shape)
        self.hanning_smoother.weight = torch.nn.Parameter(hanning_weights)

        sign_detector_weights = torch.FloatTensor(np.array([1, -1])).to(
            self.sign_detector.weight.dtype
        )
        sign_detector_weights = sign_detector_weights.reshape(
            self.sign_detector.weight.shape
        )
        self.sign_detector.weight = torch.nn.Parameter(sign_detector_weights)
        return

    def forward(self, distance_func):
        x = self.hanning_smoother(distance_func.unsqueeze(0))
        x = torch.sign(x)
        x = self.sign_detector(x)
        return x


def drdf_intersections_via_convolution(distance_func):
    drdf_detector = DRDFIntersectionFinder(window_size=5)
    device = distance_func.device
    drdf_detector.to(device)
    signs = drdf_detector.forward(distance_func[None,]).squeeze(0).squeeze(0)
    intersection_inds = torch.where(signs == 2)
    return_tuple = (None, None)
    if len(intersection_inds) > 0 and len(intersection_inds[0]) > 0:
        return_tuple = [k.cpu().numpy() for k in intersection_inds]
    return return_tuple, signs


def drdf2pcd_novisibility(drdf, ray_dirs, ray_pts):
    """
    drdf: (ray, points, 1)
    ray_dirs: (ray, 3)
    ray_pts: (ray, points, 3)

    pcd_world: (ray*points, 3)
    visibility: (ray*points): number of hit along the ray, if not occluded, = 0.
    """

    (rayid, ptid), signs = drdf_intersections_via_convolution(drdf[:, :, 0])
    if rayid is None:
        rayid = ptid = np.arange(1)
    pcd_world = drdf[rayid, ptid] * ray_dirs[rayid] + ray_pts[rayid, ptid]
    return pcd_world, rayid, ptid


def drdf2pcd(drdf, ray_dirs, ray_pts):
    """
    drdf: (ray, points, 1)
    ray_dirs: (ray, 3)
    ray_pts: (ray, points, 3)

    pcd_world: (ray*points, 3)
    visibility: (ray*points): number of hit along the ray, if not occluded, = 0.
    """
    pcd_world, rayid, ptid = drdf2pcd_novisibility(drdf, ray_dirs, ray_pts)

    count = 0
    tmp_ray_id = 0
    visibility = []

    for i in range(len(rayid)):
        if tmp_ray_id == rayid[i]:
            visibility.append(count)
        else:
            tmp_ray_id = rayid[i]
            count = 0
            visibility.append(count)
        count += 1

    visibility = torch.tensor(visibility)
    return pcd_world, visibility, rayid, ptid


def drdf2depth(drdf, ray_distance_query, camera, depth_layer=1):
    """
    drdf: (ray, points, 1)
    ray_dirs: (ray, 3)
    ray_pts: (ray, points, 3)

    pcd_world: (ray*points, 3)
    visibility: (ray*points): number of hit along the ray, if not occluded, = 0.
    """
    hit2cam = drdf2hit2cam(drdf, ray_distance_query, depth_layer=depth_layer)
    depth_size = int(np.sqrt(len(hit2cam)))
    hit2cam = hit2cam.reshape(depth_size, depth_size, depth_layer)
    depth = raydistance2zbuf(to_numpy(hit2cam), camera)
    return depth


def drdf2hit2cam(drdf, ray_distance_query, depth_layer):
    assert len(drdf.shape) == 3
    assert drdf.shape[-1] == 1
    (rayid, ptid), signs = drdf_intersections_via_convolution(drdf[:, :, 0])

    if rayid is None:
        rayid = ptid = np.arange(depth_layer)
    ray_distance = (ray_distance_query + drdf.squeeze(-1)).numpy()[rayid, ptid]
    hit2cam = smallest_k_values(
        rayid,
        ray_distance,
        depth_layer,
        len(ray_distance_query),
    )
    return hit2cam


def compute_normals(pcd, center=np.array([0, 0, 0])):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=15)
    )
    normals = np.asarray(pcd.normals)
    normals = align_normals(np.asarray(pcd.points), normals, center)
    normals = np.clip(normals, a_min=-1, a_max=1)
    return normals


def align_normals(points, normals, center):
    direction = center[None, 0:3] - points
    dot_p = np.sign(np.sum(normals * direction, axis=1))
    normals = normals * dot_p[:, None]
    return normals


def get_pcd_w_normals_w_visibility(
    pcl,
    visibility,
    image_point_colors,
):
    pcl = to_numpy(pcl)
    visibility = to_numpy(visibility)
    image_point_colors = to_numpy(image_point_colors)

    world_frame_pcd = o3d.geometry.PointCloud()
    world_frame_pcd.points = o3d.utility.Vector3dVector(pcl)
    normals = compute_normals(world_frame_pcd)

    normals = normals[:, (2, 0, 1)]

    normals[:, 1] = -1 * normals[:, 1]
    normals[:, 2] = -1 * normals[:, 2]
    normal_colors = (normals + 1) / 2

    first_hits = visibility[:, None] == 0  ## has the intersection index.
    pcl_colors = (
        first_hits * image_point_colors[:, 0:3] / 255
        + np.logical_not(first_hits) * normal_colors
    )
    pcl_colors = pcl_colors.astype(np.float32)
    return pcl, pcl_colors


def get_point_image_colors(
    pcd_world, rayid, visibility, cameras, images, ray_per_image, device
):
    """
    pcd_world: (ray*points, 3)
    rayid: (ray*points, 1)
    visibility: (ray*points)
    cameras: N
    image: N
    ray_per_image: ray // 3
    """
    pcd_world = pcd_world.to(device)
    cameras = cameras.to(device)
    images = images.to(device)
    pt_colors = torch.zeros((len(visibility), 3)).to(device)
    for i in range(len(cameras)):
        mask = rayid < ((i + 1) * ray_per_image)
        mask = np.logical_and(mask, rayid >= (i) * ray_per_image)
        point_ndc_masked = cameras[i].transform_points(pcd_world[mask].to(device))
        normalized_pixel_locations = -point_ndc_masked[:, :2].unsqueeze(0).unsqueeze(0)
        rgb_sampled = (
            F.grid_sample(
                images[i][None], normalized_pixel_locations, align_corners=True
            )
            .squeeze(2)
            .squeeze(0)
            .permute(1, 0)
        )

        # cv2.imwrite(f"debug/rgb-{i}.jpg", rgb_sampled.reshape(256, 256, 3).numpy().transpose())
        pt_colors[mask] = rgb_sampled
    pcd, colors = get_pcd_w_normals_w_visibility(pcd_world, visibility, pt_colors)
    return pcd, colors


def get_signed_distance_to_closest_torch(arr, compare_arr):
    """
    arr: [num_ray, num_pt], points along rays
    compare_arr: #ray, max_hit (hit2cam)
    """
    num_ray, _ = arr.shape
    assert compare_arr.shape[0] == num_ray
    # Calculate the absolute difference between the arrays
    diff = torch.abs(arr[..., None] - compare_arr.unsqueeze(1))

    # Find the index of the minimum value in each row
    _, min_index = torch.min(diff, dim=2)

    # Get the signed distance by using the minimum index to index into
    # the compare array
    signed_dist = torch.gather(compare_arr, 1, min_index) - arr
    return signed_dist


class GroundTruthRayDistance:
    def __init__(self, mesh):
        self.mesh_intersector = RayMeshIntersector(mesh)

    def get_all_intersections(self, points, rays):
        locations, index_ray, index_tri = self.mesh_intersector.intersects_location(
            points, rays, multiple_hits=True
        )
        return locations, index_ray, index_tri
