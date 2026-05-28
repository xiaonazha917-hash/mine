import argparse
import os
import shutil
import sys
import time
import pandas as pd
from importlib.machinery import SourceFileLoader
from scipy.spatial.transform import Rotation as R_scipy
sys.path.append('/usr/local/lib')
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)


import gc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from sgs_datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)
from sgs_utils.common_utils import seed_everything, save_params_ckpt, save_params
from sgs_utils.eval_helpers import report_loss, report_progress, eval
from sgs_utils.keyframe_selection import keyframe_selection_overlap
from sgs_utils.recon_helpers import setup_camera
from sgs_utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from sgs_utils.slam_external import calc_ssim, build_rotation, prune_gaussians,prune_gaussians1, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from dbow2 import BowVector
import dbow2
import open3d as o3d
import g2o
import math
from raft import RAFT
from core.utils.utils import InputPadder



def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, mask=None,
                   compute_mean_sq_dist=False, mean_sq_dist_method="projective", device="cuda",
                   load_semantics=False, semantic_id=None, semantic_color=None):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).to(device).float(), 
                                    torch.arange(height).to(device).float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).to(device).float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    
    # Concat semantic label if load_semantics=True
    if load_semantics:
        semantic_id = torch.permute(semantic_id, (1, 2, 0)).reshape(-1, 1) # (1, H, W) -> (H, W, 1) -> (H * W, 1)
        semantic_color = torch.permute(semantic_color, (1, 2, 0)).reshape(-1, 3) # (3, H, W) -> (H, W, 3) -> (H * W, 3)
        point_cld = torch.cat((point_cld, semantic_id, semantic_color), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld
# ✅ 完整重写后的 get_pointcloud 函数：在生成点云前就剔除动态像素

# ✅ 完整重写后的 get_pointcloud 函数：在生成点云前就剔除动态像素，并修复 mask 尺寸不匹配错误

# ✅ 完整重写后的 get_pointcloud 函数：在生成点云前就剔除动态像素，并修复 mask 尺寸不匹配错误

# def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, mask=None,
#                    compute_mean_sq_dist=False, mean_sq_dist_method="projective", device="cuda",
#                    load_semantics=False, semantic_id=None, semantic_color=None,
#                    dynamic_class_ids=[0]):

#     width, height = color.shape[2], color.shape[1]
#     CX = intrinsics[0][2]
#     CY = intrinsics[1][2]
#     FX = intrinsics[0][0]
#     FY = intrinsics[1][1]

#     # Compute indices of pixels
#     x_grid, y_grid = torch.meshgrid(torch.arange(width).to(device).float(),
#                                     torch.arange(height).to(device).float(),
#                                     indexing='xy')
#     xx = (x_grid - CX)/FX
#     yy = (y_grid - CY)/FY
#     xx = xx.reshape(-1)
#     yy = yy.reshape(-1)
#     depth_z = depth[0].reshape(-1)  # [H*W]

#     # ========== ✅ Step 1: 提前过滤动态像素 ==========
#     keep_mask = torch.ones_like(depth_z, dtype=torch.bool)
#     semantic_id_flat, semantic_color_flat = None, None

#     if load_semantics and dynamic_class_ids is not None and semantic_id is not None:
#         semantic_id_flat = semantic_id[0].reshape(-1)  # [H*W]
#         dyn_ids = torch.tensor(dynamic_class_ids, device=device)
#         dynamic_mask = torch.isin(semantic_id_flat.long(), dyn_ids)  # [H*W]
#         keep_mask = ~dynamic_mask
#         # 计算删除了多少动态像素点
#         num_total = dynamic_mask.shape[0]
#         num_dynamic = dynamic_mask.sum().item()
#         num_kept = keep_mask.sum().item()
#         print(f"-------------------------[Info] Removed dynamic pixels: {num_dynamic} / {num_total} "
#       f"({100 * num_dynamic / num_total:.2f}%), kept: {num_kept}")

#         if semantic_color is not None:
#             semantic_color_flat = torch.permute(semantic_color, (1, 2, 0)).reshape(-1, 3)[keep_mask]  # [N, 3]
#         semantic_id_flat = semantic_id_flat[keep_mask].reshape(-1, 1)  # [N, 1]

#     # 应用 keep_mask 到所有像素属性
#     xx = xx[keep_mask]
#     yy = yy[keep_mask]
#     depth_z = depth_z[keep_mask]
#     color_z = torch.permute(color, (1, 2, 0)).reshape(-1, 3)[keep_mask]  # [N, 3]
#     if mask is not None:
#         mask = mask[keep_mask]  # ✅ 同步过滤外部 mask

#     # ========== Step 2: 生成点云 ==========
#     pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)

#     if transform_pts:
#         pix_ones = torch.ones(pts_cam.shape[0], 1).to(device).float()
#         pts4 = torch.cat((pts_cam, pix_ones), dim=1)
#         c2w = torch.inverse(w2c)
#         pts = (c2w @ pts4.T).T[:, :3]
#     else:
#         pts = pts_cam

#     # Step 3: mean squared distance for Gaussian scale
#     if compute_mean_sq_dist:
#         if mean_sq_dist_method == "projective":
#             scale_gaussian = depth_z / ((FX + FY)/2)
#             mean3_sq_dist = scale_gaussian**2
#         else:
#             raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

#     # Step 4: RGB颜色拼接
#     cols = color_z  # [N, 3]
#     point_cld = torch.cat((pts, cols), -1)

#     # Step 5: 拼接语义信息（如果有）
#     if load_semantics and semantic_id_flat is not None and semantic_color_flat is not None:
#         if semantic_id_flat.shape[0] == point_cld.shape[0]:
#             point_cld = torch.cat((point_cld, semantic_id_flat, semantic_color_flat), -1)
#         else:
#             print("[Warning] semantic_id and point cloud size mismatch, skipping semantic append")

#     # Step 6: 应用额外 mask（如边缘、视野）
#     if mask is not None:
#         assert mask.shape[0] == point_cld.shape[0], "Mask and point cloud size mismatch"
#         point_cld = point_cld[mask]
#         if compute_mean_sq_dist:
#             mean3_sq_dist = mean3_sq_dist[mask]

#     if compute_mean_sq_dist:
#         return point_cld, mean3_sq_dist
#     else:
#         return point_cld



def rot_to_vec(R: torch.Tensor) -> torch.Tensor:
    """
    输入:
      R: 3x3旋转矩阵，torch.float32张量
    输出:
      omega: 3维旋转向量，方向是旋转轴，长度是旋转角（弧度）
    """
    assert R.shape == (3,3)

    # 计算旋转角度 theta = arccos((trace(R)-1)/2)
    cos_theta = (torch.trace(R) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 防止数值误差越界
    theta = torch.acos(cos_theta)

    # 当旋转角度接近0时，旋转向量接近0向量
    if torch.isclose(theta, torch.tensor(0.0), atol=1e-6):
        return torch.zeros(3, dtype=R.dtype, device=R.device)

    # 旋转轴方向
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = torch.tensor([rx, ry, rz], dtype=R.dtype, device=R.device) / (2 * torch.sin(theta))

    # 旋转向量 = 旋转角度 * 旋转轴
    omega = theta * axis
    return omega
def make_renderer(fx, fy, cx, cy, W, H, params,near, far, device='cuda'):
    # 1) 在工厂里初始化渲染器实例
    rasterizer = Renderer(
        
        fx=fx, fy=fy,
        cx=cx, cy=cy,
        H=H,W=W,
        near=near,
        far=far,
        num_channels=3,
        device=device
    )
    rasterizer.set_scene(
    means3D=params['means3D'],
    features=params['rgb_colors'],
    scales=torch.exp(params['log_scales']),         # 从 log 还原
    rotations=params['unnorm_rotations'],
    opacities=torch.sigmoid(params['logit_opacities'])  # 从 logit 还原
)
    # 2) 返回真正用的 renderer 函数
    def renderer(cam_w2c: torch.Tensor) -> torch.Tensor:
        # 把 GPU 上的 cam_w2c -> CPU numpy
        w2c = cam_w2c.detach().cpu().numpy().astype(np.float32)
        # 求 camera->world（渲染器要的姿态）
        c2w = np.linalg.inv(w2c)
        # 渲染并转回 torch
        img = rasterizer.render(c2w)  # 返回 (H,W,3) numpy
        return torch.from_numpy(img).to(device=cam_w2c.device)
    
    return renderer    
def se3_to_mat(x):
    """把 6-d vector (tx,ty,tz, rx,ry,rz) 转成 4x4 SE3 矩阵（Rodrigues）"""
    t = x[:3]                # 平移
    omega = x[3:]            # 旋转向量
    theta = torch.norm(omega)
    # Rodrigues 公式
    if theta.item() < 1e-8:
        R = torch.eye(3, device=x.device)
    else:
        k = omega / theta
        K = torch.tensor([[    0, -k[2],  k[1]],
                          [ k[2],     0, -k[0]],
                          [-k[1],  k[0],     0]], device=x.device)
        R = torch.eye(3, device=x.device) + \
            torch.sin(theta)*K + \
            (1-torch.cos(theta))*(K @ K)
    T = torch.eye(4, device=x.device)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def photometric_refine(keyframes, optimized_w2c, renderer,
                       num_iters=50, sample_n=500, lr=1e-2):
    """
    keyframes: list of dict, 每个 dict 含 'rgb'(H×W×3 Tensor), 'depth'(H×W Tensor)
    optimized_w2c: list of np.ndarray (4×4)
    renderer: function(T_cam2world) -> H×W×3 Tensor 的渲染器
    """
    device = torch.device("cuda")
    N = len(keyframes)
    # 1) 初始化 poses 参数
    poses = []
    for w2c in optimized_w2c:
        T = torch.from_numpy(w2c).to(device, dtype=torch.float32)
        # 分离平移 + Rodrigues 逆
        t = T[:3,3]
        R = T[:3,:3]
        # 这里假设你已有从 R 到 omega 的函数 rot_to_vec
        omega = rot_to_vec(R)  # 返回 3-vector
        x = torch.cat([t, omega], dim=0).clone().detach().requires_grad_()
        poses.append(x)
    optimizer = optim.Adam(poses, lr=lr)

    # 2) 采样像素索引 (一次性固定)
    H, W = keyframes[0]['color'].shape[:2]
    ys = torch.randint(0, H, (sample_n,), device=device)
    xs = torch.randint(0, W, (sample_n,), device=device)

    # 3) 优化循环
    for it in range(num_iters):
        optimizer.zero_grad()
        loss = 0.0
        # 对每个关键帧，只跟邻帧或回环帧做光度对齐
        for i in range(N-1):
            # 渲染 i+1 从帧 i 的视角
            T_i = se3_to_mat(poses[i])
            T_j = se3_to_mat(poses[i+1])
            
            I_syn = renderer(T_i)         # H×W×3
            I_obs = keyframes[i]['color'].to(device)
            # 在采样点上比较
            sampled_syn = I_syn[ys, xs]   # sample_n×3
            sampled_obs = I_obs[ys, xs]
            loss = loss + torch.mean((sampled_syn - sampled_obs)**2)
        # 也可以对回环对再加一次 photometric
        # for (src, tgt, _) in loop_closures: ...
        loss.backward()
        optimizer.step()
        if it % 10 == 0:
            print(f"[PhotoBA] iter {it} loss {loss.item():.6f}")

    # 4) 提取优化后 SE3 写回
    refined_w2c = []
    for x in poses:
        T = se3_to_mat(x).detach().cpu().numpy()
        refined_w2c.append(T)
    return refined_w2c
def save_params_to_kitti_txt(params, filename):
    """
    将 params 中的相机位姿保存为 KITTI 格式，每行为一个4x4位姿矩阵的前12项。
    
    params:
        - cam_unnorm_rots: (1, 4, N)，四元数 [w, x, y, z]（未归一化或半归一化存储）
        - cam_trans:      (1, 3, N)
    """
    cam_rots  = params['cam_unnorm_rots']  # (1, 4, N)
    cam_trans = params['cam_trans']        # (1, 3, N)
    num_frames = cam_rots.shape[2]

    with open(filename, 'w') as f:
        for i in range(num_frames):
            # 1. 取出原始四元数 & 平移
            q = cam_rots[0, :, i].detach().cpu()      # Tensor([w, x, y, z])
            t = cam_trans[0, :, i].detach().cpu()     # Tensor([tx, ty, tz])

            # 2. 先正规化四元数
            q = q / q.norm()

            # 3. 转 rotation matrix
            R = quaternion_to_rotation_matrix(q.numpy())  # 3x3 numpy

            # 4. 构建 w2c → 再 inv 得到 c2w
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R
            w2c[:3, 3] = t.numpy()
            c2w = np.linalg.inv(w2c)

            # 5. 提取前 3x4，flatten 写入
            Rt_flat = c2w[:3, :].reshape(-1)
            line = ' '.join(f"{v:.18e}" for v in Rt_flat)
            f.write(line + "\n")

    print(f"[INFO] 已保存位姿为 KITTI 格式：{filename}")

def quaternion_to_rotation_matrix(q):
    """
    将单位四元数 [w, x, y, z] 转换为 3x3 旋转矩阵
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)
# def save_params_to_kitti_txt(params, filename):
#     """
#     将 params 中的相机位姿保存为 KITTI 格式，每行为一个4x4位姿矩阵的前12项。
    
#     params:
#         - cam_unnorm_rots: (1, 4, N)，四元数 [w, x, y, z]
#         - cam_trans: (1, 3, N)
#     """
#     cam_rots = params['cam_unnorm_rots']  # (1, 4, N)
#     cam_trans = params['cam_trans']      # (1, 3, N)
#     num_frames = cam_rots.shape[2]

#     with open(filename, 'w') as f:
#         for i in range(num_frames):
#             q = cam_rots[0, :, i]  # w, x, y, z
#             t = cam_trans[0, :, i]

#             # 转换为旋转矩阵
#             R = quaternion_to_rotation_matrix(q.cpu().detach().numpy()) # 3x3


#             # 拼成 3x4 的矩阵
#             # Rt = np.hstack([R, t.cpu().detach().numpy().reshape(3, 1)])  # 3x4
#   # 3x4
#             # 构建 4x4 的 w2c 矩阵
#             w2c = np.eye(4)
#             w2c[:3, :3] = R
#             w2c[:3, 3] = t.cpu().detach().numpy()

#             # 转成 c2w：相机到世界
#             c2w = np.linalg.inv(w2c)

#            # 取出前 3x4
#             Rt = c2w[:3, :]
#             # 转成12个数写入
#             Rt_flat = Rt.flatten()
#             line = ' '.join(f'{v:.6f}' for v in Rt_flat.tolist())
#             f.write(line + '\n')

#     print(f"[INFO] 已保存位姿为 KITTI 格式：{filename}")

# def quaternion_to_rotation_matrix(q):
#     """
#     将四元数 [w, x, y, z] 转换为 3x3 旋转矩阵
#     """
#     w, x, y, z = q
#     R = np.array([
#         [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
#         [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
#         [2*x*z - 2*y*w,         2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
#     ])
#     return R
def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, device, load_semantics=False):
    num_pts = init_pt_cld.shape[0]
    # channel 0-2 for 3d axis
    means3D = init_pt_cld[:, :3]
    # channel 3-5 for rgb colors
    rgb_colors = init_pt_cld[:, 3:6]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device=device)
    
    params = {
        'means3D': means3D,
        'rgb_colors': rgb_colors,
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }

    params_opt_exclude = set()
    if load_semantics:
        # Exclude semantic_ids from gradient
        params_opt_exclude.add('semantic_ids')
        # channel =6 for semantic id
        params['semantic_ids'] = init_pt_cld[:, 6]
        # Channel 7-9 for semantic colors
        params['semantic_colors'] = init_pt_cld[:, 7:10]

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))
    
    for k, v in params.items():
        if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).to(device).float()}

    return params, variables, params_opt_exclude


def initialize_optimizer(params, params_opt_exclude, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k not in params_opt_exclude]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, device="cuda",
                              densify_dataset=None, load_semantics=False):
    # Get RGB-D Data & Camera Parameters
    if load_semantics:
        color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[0]
    else:
        color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
    
    if load_semantics:
        semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        semantic_color = semantic_color.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
    else:
        semantic_id = None
        semantic_color = None
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                       w2c.detach().cpu().numpy(), device=device)

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                   w2c.detach().cpu().numpy(), device=device)
    else:
        densify_intrinsics = intrinsics
    
    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics,
                                                w2c, mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                load_semantics=load_semantics, semantic_id=semantic_id,
                                                semantic_color=semantic_color)

    # Initialize Parameters
    params, variables, params_opt_exclude = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, device,
                                                              load_semantics)
    # initialize_first_timestep 函数中，原来初始化 means3D 等空张量的地方：
 
    # 新增这一行：与高斯点一一对应，存储它们的来源帧 ID
    params_opt_exclude.add("gaussian_frame_ids")
    empty_ids = torch.zeros((0,), dtype=torch.long, device=device)
    
   
    params['gaussian_frame_ids'] = empty_ids
    N = params['means3D'].shape[0]
    new_ids = torch.full((N,), 0, dtype=torch.long, device=device)
    params['gaussian_frame_ids'] = torch.cat([params['gaussian_frame_ids'], new_ids], dim=0)
    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude

#以下这个单向掩码可以用，但是我想要双向，先注释
# def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
#              use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
#              plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False):
#     # Initialize Loss Dictionary
#     losses = {}

#     if tracking:
#         # Get current frame Gaussians, where only the camera pose gets gradient
#         transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
#                                              camera_grad=True, device=device)
#     elif mapping:
#         if do_ba:
#             # Get current frame Gaussians, where both camera pose and Gaussians get gradient
#             transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                                  camera_grad=True, device=device)
#         else:
#             # Get current frame Gaussians, where only the Gaussians get gradient
#             transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                                  camera_grad=False, device=device)
#     else:
#         # Get current frame Gaussians, where only the Gaussians get gradient
#         transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                              camera_grad=False, device=device)

#     # Initialize Render Variables
#     rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
#     depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
#                                                                  transformed_pts, device=device)
#     # RGB Rendering
#     rendervar['means2D'].retain_grad()
#     im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
#     variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

#     # Depth & Silhouette Rendering
#     depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
#     depth = depth_sil[0, :, :].unsqueeze(0)
#     silhouette = depth_sil[1, :, :]
#     presence_sil_mask = (silhouette > sil_thres)
#     depth_sq = depth_sil[2, :, :].unsqueeze(0)
#     uncertainty = depth_sq - depth**2
#     uncertainty = uncertainty.detach()

#     # Semantic colors Rendering
#     if load_semantics:
#         semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
#         rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)

#     # Mask with valid depth values (accounts for outlier depth values)

#     # nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
#     # if ignore_outlier_depth_loss:
#     #     depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
#     #     mask = (depth_error < 10*depth_error.median())
#     #     mask = mask & (curr_data['depth'] > 0)
#     # else:
#     #     mask = (curr_data['depth'] > 0)
#     # mask = mask & nan_mask
#     # # Mask with presence silhouette mask (accounts for empty space)
#     # if tracking and use_sil_for_loss:
#     #     mask = mask & presence_sil_mask

# #修改MAD的做法，下是原MAD做法
#     # # 原有 mask
#     # nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
#     # if ignore_outlier_depth_loss:
#     #    depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
#     #    mask = (depth_error < 10*depth_error.median())
#     #    mask = mask & (curr_data['depth'] > 0)
#     # else:
#     #    mask = (curr_data['depth'] > 0)
#     # mask = mask & nan_mask

#     # # Mask with presence silhouette mask (accounts for empty space)
#     # if tracking and use_sil_for_loss:
#     #    mask = mask & presence_sil_mask

#     # # ✅ 新增：屏蔽动态物体
#     # if 'mask' in curr_data:
#     #     mask = mask & curr_data['mask'].reshape(mask.shape)

#     # ------------------- 核心优化 1：语义引导的统计兜底 -------------------
#     nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    
#     # 提前获取语义掩码 (假设 True 为静态背景，False 为被 YOLO 剔除的动态人/车)
#     semantic_mask = curr_data['mask'].reshape(curr_data['depth'].shape) if 'mask' in curr_data else None

#     if ignore_outlier_depth_loss:
#         # 计算全局深度误差
#         depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        
#         # 1. 计算纯净中位数 (Pure Median)
#         if semantic_mask is not None:
#             # 仅提取被 YOLO 判定为“静态背景”且深度有效的像素误差
#             static_error_region = depth_error[semantic_mask & (curr_data['depth'] > 0)]
#             total_valid_depth_pixels = (curr_data['depth'] > 0).sum().item()
#             min_static_pixels_required = max(2000, int(total_valid_depth_pixels * 0.01))
#             # 确保静态区域有足够的像素点来计算中位数（防止 YOLO 遮挡面积过大导致全图变 False，发生除零错误）
#             if static_error_region.numel() > min_static_pixels_required:
#                 pure_median = static_error_region.median()
#             else:
#                 pure_median = depth_error.median() # 极端情况：退化为全局中位数兜底
#         else:
#             pure_median = depth_error.median()
            
#         # 2. 使用纯净中位数作为绝对标尺，去抓全图的漏网之鱼
#         # 注意：因为现在的中位数非常纯净准确，我们可以把原本死板的 10 倍稍微收紧到 8 倍（甚至 5 倍，可根据效果调参），过滤动态物体的效果会更猛！
#         mad_multiplier = 8.0 
#         mask = (depth_error < mad_multiplier * pure_median)
#         mask = mask & (curr_data['depth'] > 0)
#     else:
#         mask = (curr_data['depth'] > 0)
        
#     mask = mask & nan_mask

#     # Mask with presence silhouette mask (accounts for empty space)
#     if tracking and use_sil_for_loss:
#         mask = mask & presence_sil_mask

#     # 最后，依然要把 YOLO 明确识别出来的动态物体彻底从 mask 中交集剔除
#     if semantic_mask is not None:
#         mask = mask & semantic_mask
#     # -------------------------------------------------------------------------


#     # Depth loss
#     if use_l1:
#         mask = mask.detach()
#         if tracking:
#             losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
#         else:
#             losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
#     # RGB Loss
#     if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
#         color_mask = torch.tile(mask, (3, 1, 1))
#         color_mask = color_mask.detach()
#         losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
#         if load_semantics:
#             losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[color_mask].sum()
#     elif tracking:
#         losses['im'] = torch.abs(curr_data['im'] - im).sum()
#         if load_semantics:
#             losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg).sum()
#     else:
#         losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
#         if load_semantics:
#             losses['seg'] = 0.8 * l1_loss_v1(rendered_seg, curr_data['semantic_color']) \
#                 + 0.2 * (1.0 - calc_ssim(rendered_seg, curr_data['semantic_color']))

#     # Visualize the Diff Images
#     if tracking and visualize_tracking_loss:
#         fig, ax = plt.subplots(2, 4, figsize=(12, 6))
#         weighted_render_im = im * color_mask
#         weighted_im = curr_data['im'] * color_mask
#         weighted_render_depth = depth * mask
#         weighted_depth = curr_data['depth'] * mask
#         diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
#         diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
#         viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
#         ax[0, 0].imshow(viz_img)
#         ax[0, 0].set_title("Weighted GT RGB")
#         viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
#         ax[1, 0].imshow(viz_render_img)
#         ax[1, 0].set_title("Weighted Rendered RGB")
#         ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
#         ax[0, 1].set_title("Weighted GT Depth")
#         ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
#         ax[1, 1].set_title("Weighted Rendered Depth")
#         ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
#         ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
#         ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
#         ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
#         ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
#         ax[0, 3].set_title("Silhouette Mask")
#         ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
#         ax[1, 3].set_title("Loss Mask")
#         # Turn off axis
#         for i in range(2):
#             for j in range(4):
#                 ax[i, j].axis('off')
#         # Set Title
#         fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
#         # Figure Tight Layout
#         fig.tight_layout()
#         os.makedirs(plot_dir, exist_ok=True)
#         plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
#         plt.close()
#         plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
#         cv2.imshow('Diff Images', plot_img)
#         cv2.waitKey(1)
#         ## Save Tracking Loss Viz
#         # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
#         # os.makedirs(save_plot_dir, exist_ok=True)
#         # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
#         # plt.close()

    # weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    # loss = sum(weighted_losses.values())

    # seen = radius > 0
    # variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    # variables['seen'] = seen
    # weighted_losses['loss'] = loss

    # return loss, variables, weighted_losses

# def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
#              use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
#              plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False):
#     # Initialize Loss Dictionary
#     losses = {}

#     if tracking:
#         # Get current frame Gaussians, where only the camera pose gets gradient
#         transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
#                                              camera_grad=True, device=device)
#     elif mapping:
#         if do_ba:
#             # Get current frame Gaussians, where both camera pose and Gaussians get gradient
#             transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                                  camera_grad=True, device=device)
#         else:
#             # Get current frame Gaussians, where only the Gaussians get gradient
#             transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                                  camera_grad=False, device=device)
#     else:
#         # Get current frame Gaussians, where only the Gaussians get gradient
#         transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                              camera_grad=False, device=device)

#     # Initialize Render Variables
#     rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
#     depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
#                                                                  transformed_pts, device=device)
#     # RGB Rendering
#     rendervar['means2D'].retain_grad()
#     im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
#     variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

#     # Depth & Silhouette Rendering
#     depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
#     depth = depth_sil[0, :, :].unsqueeze(0)
#     silhouette = depth_sil[1, :, :]
#     presence_sil_mask = (silhouette > sil_thres)
#     depth_sq = depth_sil[2, :, :].unsqueeze(0)
#     uncertainty = depth_sq - depth**2
#     uncertainty = uncertainty.detach()

#     # Semantic colors Rendering
#     rendered_seg = None
#     if load_semantics:
#         semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
#         rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)


#     # =========================================================================================
#     # ✅ 核心优化：解耦的双向语义掩码策略 (Decoupled Bi-directional Masking)
#     # =========================================================================================
#     nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    
#     # 1. 提取【真实观测掩码】(YOLO 提供的视野状态)
#     obs_mask = curr_data['mask'].reshape(curr_data['depth'].shape) if 'mask' in curr_data else None

#     # 2. 提取【地图渲染掩码】(系统脑海中的记忆状态)
#     # rendered_mask = None
#     # # print("==22222=======================rendered_seg=================================",rendered_seg)
#     # if load_semantics and rendered_seg is not None and 'dynamic_class_ids' in curr_data:
#     #     dyn_ids = torch.tensor(curr_data['dynamic_class_ids'], dtype=torch.long, device=device)
#     #     print("==进入第一层=================================")
#     #     # 防御性形状检查
       
#     #     if rendered_seg.shape[0] > 1:
#     #         rendered_class_id = rendered_seg.argmax(dim=0)
#     #         print("-22222--------------------------情况A------------------------------------")   
#     #     else:
#     #         rendered_class_id = rendered_seg.squeeze(0)
#     #         print("-22222-------------------情况B--------------------")    
#     #     print("-进入第二层----------------------------出结果---------------------")    
#     #     rendered_dynamic = torch.isin(rendered_class_id.long(), dyn_ids)
#     #     rendered_mask = ~rendered_dynamic.reshape(curr_data['depth'].shape)

    

#     rendered_mask = None
#     if load_semantics and rendered_seg is not None and 'dynamic_class_ids' in curr_data:
#         dyn_ids = torch.tensor(curr_data['dynamic_class_ids'], dtype=torch.long, device=device)
        
#         if rendered_seg.shape[0] > 1:
#             # =========================================================================
#             # 🚀【完美修复：RGB 黑背景拦截机制】
#             # 因为 rendered_seg 是 3通道的 RGB 图！argmax 只会输出 0(红), 1(绿), 2(蓝)。
#             # =========================================================================
#             max_color_value, _ = rendered_seg.max(dim=0)
            
#             # 原本的错误逻辑：直接取 argmax
#             rendered_class_id = rendered_seg.argmax(dim=0)
            
#             # 严谨纠偏：如果连 0.3 的亮度都达不到，说明是黑色背景或渲染噪点
#             is_background = (max_color_value < 0.3)
            
#             # 强行改为绝对安全的背景 ID (255)
#             rendered_class_id[is_background] = 255  
#             # =========================================================================
#         else:
#             rendered_class_id = rendered_seg.squeeze(0)
            
#         rendered_dynamic = torch.isin(rendered_class_id.long(), dyn_ids)
#         rendered_mask = ~rendered_dynamic.reshape(curr_data['depth'].shape)    

#     # 3. 合并掩码：根据 Tracking 还是 Mapping 采取截然不同的哲学！
#     if tracking:
#         # 【Tracking 定位阶段】：惹不起躲得起！
#         # 只要现实或者记忆里有一方判定是动态，就一票否决，坚决不用它算位姿。
#         if obs_mask is not None and rendered_mask is not None:
#             semantic_mask = obs_mask & rendered_mask
#             print("__________这个开始工作___________")
#         elif obs_mask is not None:
#             semantic_mask = obs_mask
#         elif rendered_mask is not None:
#             semantic_mask = rendered_mask
#         else:
#             semantic_mask = None
#     else:
#         # 【Mapping 建图阶段】：铁面无私，消灭幽灵！
#         # 绝不包庇地图里的“渲染幽灵”。只看现实视野 (obs_mask) 是否被挡住。
#         # 如果现实是干净的墙，强行计算误差，产生巨大梯度去洗刷地图记忆里的幽灵！
#         semantic_mask = obs_mask

#     # 4. MAD 异常深度过滤 (带语义兜底保障)
#     if ignore_outlier_depth_loss:
#         depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        
#         if semantic_mask is not None:
#             # 在极其纯净的静态区域计算中位数
#             static_error_region = depth_error[semantic_mask & (curr_data['depth'] > 0)]
#             total_valid_depth_pixels = (curr_data['depth'] > 0).sum().item()
#             min_static_pixels_required = max(2000, int(total_valid_depth_pixels * 0.01))
#             print("--------min_static_pixels_required----------",min_static_pixels_required)
#             if static_error_region.numel() > min_static_pixels_required:
#                 pure_median = static_error_region.median()
#             else:
#                 pure_median = depth_error.median() 
#         else:
#             pure_median = depth_error.median()
            
#         mad_multiplier = 8.0 
#         mask = (depth_error < mad_multiplier * pure_median)
#         mask = mask & (curr_data['depth'] > 0)
#     else:
#         mask = (curr_data['depth'] > 0)
        
#     mask = mask & nan_mask

#     # 5. 加上空洞屏蔽 (仅Tracking阶段)
#     if tracking and use_sil_for_loss:
#         mask = mask & presence_sil_mask

#     # 6. 最后一道防线：强行套上我们刚才精心设计的语义掩码
#     if semantic_mask is not None:
#         mask = mask & semantic_mask
#     # =========================================================================================

#     # Depth loss
#     if use_l1:
#         mask = mask.detach()
#         if tracking:
#             losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
#         else:
#             losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
#     # RGB Loss & Semantic Seg Loss
#     if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
#         color_mask = torch.tile(mask, (3, 1, 1))
#         color_mask = color_mask.detach()
#         losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
#         if load_semantics and rendered_seg is not None:
#             losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[color_mask].sum()
#     elif tracking:
#         losses['im'] = torch.abs(curr_data['im'] - im).sum()
#         if load_semantics and rendered_seg is not None:
#             losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg).sum()
#     # else:
#     #     losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
#     #     if load_semantics and rendered_seg is not None:
#     #         losses['seg'] = 0.8 * l1_loss_v1(rendered_seg, curr_data['semantic_color']) \
#     #             + 0.2 * (1.0 - calc_ssim(rendered_seg, curr_data['semantic_color']))

#     else:     # 这是 Mapping 建图分支
#            # 1. 把 1 通道的 mask 变成 3 通道的彩色 mask
#            color_mask = torch.tile(mask, (3, 1, 1)).detach()
        
#             # 2. 核心修复：给渲染图和真实图都“戴上面具”
#             # 动态人物所在的地方，乘以 0 后全部变成纯黑；静态墙壁不受影响
#            masked_im = im * color_mask
#            masked_gt_im = curr_data['im'] * color_mask
        
#            # 3. 在戴了面具的图片上计算 L1 和 SSIM 误差
#            losses['im'] = 0.8 * l1_loss_v1(masked_im, masked_gt_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_gt_im))
        
#            # 4. 如果有语义图，也同样戴上面具处理
#            if load_semantics and rendered_seg is not None:
#               masked_seg = rendered_seg * color_mask
#               masked_gt_seg = curr_data['semantic_color'] * color_mask
#               losses['seg'] = 0.8 * l1_loss_v1(masked_seg, masked_gt_seg) \
#                   + 0.2 * (1.0 - calc_ssim(masked_seg, masked_gt_seg))       

#     # Visualize the Diff Images 
#     if tracking and visualize_tracking_loss:
#         fig, ax = plt.subplots(2, 4, figsize=(12, 6))
#         weighted_render_im = im * color_mask
#         weighted_im = curr_data['im'] * color_mask
#         weighted_render_depth = depth * mask
#         weighted_depth = curr_data['depth'] * mask
#         diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
#         diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
#         viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
#         ax[0, 0].imshow(viz_img)
#         ax[0, 0].set_title("Weighted GT RGB")
#         viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
#         ax[1, 0].imshow(viz_render_img)
#         ax[1, 0].set_title("Weighted Rendered RGB")
#         ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
#         ax[0, 1].set_title("Weighted GT Depth")
#         ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
#         ax[1, 1].set_title("Weighted Rendered Depth")
#         ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
#         ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
#         ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
#         ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
#         ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
#         ax[0, 3].set_title("Silhouette Mask")
#         ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
#         ax[1, 3].set_title("Loss Mask")
#         # Turn off axis
#         for i in range(2):
#             for j in range(4):
#                 ax[i, j].axis('off')
#         # Set Title
#         fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
#         # Figure Tight Layout
#         fig.tight_layout()
#         os.makedirs(plot_dir, exist_ok=True)
#         plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
#         plt.close()
#         plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
#         cv2.imshow('Diff Images', plot_img)
#         cv2.waitKey(1)

#     weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
#     loss = sum(weighted_losses.values())

#     seen = radius > 0
#     variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
#     variables['seen'] = seen
#     weighted_losses['loss'] = loss

#     return loss, variables, weighted_losses

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
             use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
             plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
                                             camera_grad=True, device=device)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=True, device=device)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=False, device=device)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                             camera_grad=False, device=device)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Semantic colors Rendering
    rendered_seg = None
    if load_semantics:
        semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
        rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)

    # =========================================================================================
    # ✅ 核心优化：解耦的双向语义掩码策略 (Decoupled Bi-directional Masking)
    # =========================================================================================
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    
    # 1. 提取【真实观测掩码】(YOLO 提供的视野状态)
    obs_mask = curr_data['mask'].reshape(curr_data['depth'].shape) if 'mask' in curr_data else None

    # 2. 提取【地图渲染掩码】(系统脑海中的记忆状态)
    rendered_mask = None
    if load_semantics and rendered_seg is not None and 'dynamic_class_ids' in curr_data:
        dyn_ids = torch.tensor(curr_data['dynamic_class_ids'], dtype=torch.long, device=device)
       
        if rendered_seg.shape[0] > 1:
            # =========================================================================
            # 🚀【多类别完美兼容版：欧氏距离色彩匹配法】
            # =========================================================================
            # 1. 定义你的全局颜色字典 (请根据你自己的数据集 YOLO 颜色映射进行修改)
            # 格式：{ 类别ID : [R, G, B] }，注意数值是归一化后的 0.0 ~ 1.0
            color_dict = {
               0: [1.0, 0.0, 0.0],    # person: 原BGR[0,0,255] -> RGB[255,0,0] -> [1.0, 0.0, 0.0]
    1: [0.0, 1.0, 0.0],    # bicycle: 原BGR[0,255,0] -> RGB[0,255,0] -> [0.0, 1.0, 0.0]
    2: [0.0, 0.0, 1.0],    # car: 原BGR[255,0,0] -> RGB[0,0,255] -> [0.0, 0.0, 1.0]
    3: [0.0, 1.0, 1.0],    # motorcycle: 原BGR[0,255,255] -> RGB[255,255,0] -> [1.0, 1.0, 0.0]
    4: [1.0, 0.0, 1.0],    # airplane: 原BGR[255,0,255] -> RGB[255,0,255] -> [1.0, 0.0, 1.0]
    5: [0.0, 1.0, 1.0],    # bus: 原BGR[255,255,0] -> RGB[0,255,255] -> [0.0, 1.0, 1.0]
    6: [0.50196, 0.0, 0.50196], # train: 原BGR[128,0,128] -> RGB[128,0,128] -> [0.5, 0.0, 0.5]
    7: [0.50196, 0.50196, 0.0], # truck: 原BGR[0,128,128] -> RGB[128,128,0] -> [0.5, 0.5, 0.0]
    8: [0.0, 0.50196, 0.50196], # boat: 原BGR[128,128,0] -> RGB[0,128,128] -> [0.0, 0.5, 0.5]
    9: [1.0, 0.64706, 0.0],    # traffic light: 原BGR[0,165,255] -> RGB[255,165,0] -> [1.0, 0.65, 0.0]
    10: [0.50196, 0.0, 0.0],   # fire hydrant: 原BGR[0,0,128] -> RGB[128,0,0] -> [0.5, 0.0, 0.0]
    11: [0.50196, 0.50196, 0.50196], # stop sign: 原BGR[128,128,128] -> [0.5, 0.5, 0.5]
    12: [0.75294, 0.75294, 0.75294], # parking meter: 原BGR[192,192,192] -> [0.75, 0.75, 0.75]
    13: [0.0, 0.50196, 0.0],    # bench: 原BGR[0,128,0] -> [0.0, 0.5, 0.0]
    14: [0.29412, 0.0, 0.5098], # bird: 原BGR[130,0,75] -> RGB[75,0,130] -> [0.29, 0.0, 0.51]
    15: [1.0, 0.84314, 0.0],    # cat: 原BGR[0,215,255] -> RGB[255,215,0] -> [1.0, 0.84, 0.0]
    16: [0.13333, 0.5451, 0.13333], # dog: 原BGR[34,139,34] -> [0.13, 0.55, 0.13]
    17: [0.35294, 0.35294, 0.80392], # horse: 原BGR[205,90,90] -> RGB[90,90,205] -> [0.35, 0.35, 0.8]
    18: [0.88235, 0.89412, 1.0],    # sheep: 原BGR[255,228,225] -> RGB[225,228,255] -> [0.88, 0.89, 1.0]
    19: [0.07451, 0.27059, 0.5451], # cow: 原BGR[139,69,19] -> RGB[19,69,139] -> [0.07, 0.27, 0.55]
    20: [0.41176, 0.41176, 0.41176], # elephant: 原BGR[105,105,105] -> [0.41, 0.41, 0.41]
    21: [0.16471, 0.16471, 0.64706], # bear: 原BGR[165,42,42] -> RGB[42,42,165] -> [0.16, 0.16, 0.65]
    22: [0.0, 0.0, 0.0],           # zebra: [0.0, 0.0, 0.0]
    23: [0.0, 0.84314, 1.0],       # giraffe: 原BGR[255,215,0] -> RGB[0,215,255] -> [0.0, 0.84, 1.0]
    24: [0.60392, 0.98039, 0.0],   # backpack: 原BGR[0,250,154] -> RGB[154,250,0] -> [0.6, 0.98, 0.0]
    25: [0.70588, 0.5098, 0.27451],# umbrella: 原BGR[70,130,180] -> RGB[180,130,70] -> [0.7, 0.5, 0.27]
    26: [0.83922, 0.43922, 0.8549],# handbag: 原BGR[218,112,214] -> RGB[214,112,218] -> [0.84, 0.44, 0.85]
    27: [0.54902, 0.90196, 0.94118],# tie: 原BGR[240,230,140] -> RGB[140,230,240] -> [0.55, 0.9, 0.94]
    28: [0.11765, 0.41176, 0.82353],# suitcase: 原BGR[210,105,30] -> RGB[30,105,210] -> [0.12, 0.41, 0.82]
    29: [0.70588, 0.41176, 1.0],    # frisbee: 原BGR[255,105,180] -> RGB[180,105,255] -> [0.7, 0.41, 1.0]
    30: [0.0, 0.54902, 1.0],       # skis: 原BGR[255,140,0] -> RGB[0,140,255] -> [0.0, 0.55, 1.0]
    31: [0.0, 0.27059, 1.0],       # snowboard: 原BGR[255,69,0] -> RGB[0,69,255] -> [0.0, 0.27, 1.0]
    32: [0.27843, 0.38824, 1.0],    # sports ball: 原BGR[255,99,71] -> RGB[71,99,255] -> [0.28, 0.39, 1.0]
    33: [0.44706, 0.50196, 0.98039],# kite: 原BGR[250,128,114] -> RGB[114,128,250] -> [0.45, 0.5, 0.98]
    34: [0.47843, 0.58824, 0.91373],# baseball bat: 原BGR[233,150,122] -> RGB[122,150,233] -> [0.48, 0.59, 0.91]
    35: [0.50196, 0.50196, 0.94118],# baseball glove: 原BGR[240,128,128] -> RGB[128,128,240] -> [0.5, 0.5, 0.94]
    36: [0.36078, 0.36078, 0.80392],# skateboard: 原BGR[205,92,92] -> RGB[92,92,205] -> [0.36, 0.36, 0.8]
    37: [0.23529, 0.07843, 0.86275],# surfboard: 原BGR[220,20,60] -> RGB[60,20,220] -> [0.24, 0.08, 0.86]
    38: [0.13333, 0.13333, 0.69804],# tennis racket: 原BGR[178,34,34] -> RGB[34,34,178] -> [0.13, 0.13, 0.7]
    39: [0.0, 0.0, 0.5451],        # bottle: [0.0, 0.0, 0.55]
    40: [0.0, 0.0, 0.50196],       # wine glass: [0.0, 0.0, 0.5]
    41: [0.94118, 1.0, 1.0],       # cup: [0.94, 1.0, 1.0]
    42: [0.80392, 0.98039, 1.0],    # fork: [0.8, 0.98, 1.0]
    43: [0.84314, 0.92157, 0.98039],# knife: [0.84, 0.92, 0.98]
    44: [0.83529, 0.93725, 1.0],    # spoon: [0.84, 0.94, 1.0]
    45: [0.7098, 0.89412, 1.0],     # bowl: [0.71, 0.89, 1.0]
    46: [0.72549, 0.8549, 1.0],     # banana: [0.73, 0.85, 1.0]
    47: [0.67843, 0.87059, 1.0],    # apple: [0.68, 0.87, 1.0]
    48: [0.76863, 0.89412, 1.0],    # sandwich: [0.77, 0.89, 1.0]
    49: [0.80392, 0.92157, 1.0],    # orange: [0.8, 0.92, 1.0]
    50: [0.86275, 0.97255, 1.0],    # broccoli: [0.86, 0.97, 1.0]
    51: [0.93333, 0.96078, 1.0],    # carrot: [0.93, 0.96, 1.0]
    52: [0.96078, 0.94118, 1.0],    # hot dog: [0.96, 0.94, 1.0]
    53: [0.98039, 0.98039, 1.0],    # pizza: [0.98, 0.98, 1.0]
    54: [0.94118, 1.0, 0.94118],    # donut: [0.94, 1.0, 0.94]
    55: [0.98039, 1.0, 0.96078],    # cake: [0.98, 1.0, 0.96]
    56: [1.0, 1.0, 0.94118],        # chair: [1.0, 1.0, 0.94]
    57: [1.0, 0.94118, 0.94118],    # couch: [1.0, 0.94, 0.94]
    58: [1.0, 0.97255, 0.97255],    # potted plant: [1.0, 0.97, 0.97]
    59: [0.86275, 0.94118, 0.96078],# bed: [0.86, 0.94, 0.96]
    60: [0.90196, 0.96078, 0.99216],# dining table: [0.9, 0.96, 0.99]
    61: [0.94118, 1.0, 1.0],        # toilet: [0.94, 1.0, 1.0]
    62: [0.94118, 1.0, 1.0],        # tv: [0.94, 1.0, 1.0]
    63: [0.84314, 0.92157, 0.98039],# laptop: [0.84, 0.92, 0.98]
    64: [0.84314, 0.92157, 0.98039],# mouse: [0.84, 0.92, 0.98]
    65: [0.83529, 0.93725, 1.0],    # remote: [0.84, 0.94, 1.0]
    66: [0.72549, 0.8549, 1.0],     # keyboard: [0.73, 0.85, 1.0]
    67: [0.72549, 0.8549, 1.0],     # cell phone: [0.73, 0.85, 1.0]
    68: [0.67843, 0.87059, 1.0],    # microwave: [0.68, 0.87, 1.0]
    69: [0.76863, 0.89412, 1.0],    # oven: [0.77, 0.89, 1.0]
    70: [0.80392, 0.92157, 1.0],    # toaster: [0.8, 0.92, 1.0]
    71: [0.86275, 0.97255, 1.0],    # sink: [0.86, 0.97, 1.0]
    72: [0.93333, 0.96078, 1.0],    # refrigerator: [0.93, 0.96, 1.0]
    73: [0.96078, 0.94118, 1.0],    # book: [0.96, 0.94, 1.0]
    74: [0.98039, 0.98039, 1.0],    # clock: [0.98, 0.98, 1.0]
    75: [0.94118, 1.0, 0.94118],    # vase: [0.94, 1.0, 0.94]
    76: [0.98039, 1.0, 0.96078],    # scissors: [0.98, 1.0, 0.96]
    77: [1.0, 1.0, 0.94118],        # teddy bear: [1.0, 1.0, 0.94]
    78: [1.0, 0.94118, 0.94118],    # hair drier: [1.0, 0.94, 0.94]
    79: [1.0, 0.97255, 0.97255],    # toothbrush: [1.0, 0.97, 0.97]
    255: [0.0, 0.0, 0.0]           # background: [0.0, 0.0, 0.0]
            }
            
            # 2. 将字典转换为 GPU Tensor 计算矩阵
            palette_colors = []
            palette_ids = []
            for cid, color in color_dict.items():
                palette_colors.append(color)
                palette_ids.append(cid)
            
            palette_tensor = torch.tensor(palette_colors, device=device, dtype=torch.float32) # [K, 3]
            ids_tensor = torch.tensor(palette_ids, device=device, dtype=torch.long) # [K]
            
            # 3. 将渲染出来的 3DGS 语义图 [3, H, W] 展平为 [H*W, 3] 像素列表
            C, H, W = rendered_seg.shape
            pixels = rendered_seg.view(3, -1).permute(1, 0)
            
            # 4. 计算所有像素到预设字典颜色的距离 (这就是降维打击)
            # torch.cdist 会算出每个像素离字典里哪个颜色最接近
            distances = torch.cdist(pixels, palette_tensor)
            
            # 5. 找出距离最近的颜色索引，并映射回真实的 ID
            closest_idx = distances.argmin(dim=1)
            rendered_class_id = ids_tensor[closest_idx].reshape(H, W)
            # =========================================================================
        else:
            rendered_class_id = rendered_seg.squeeze(0)
            
        rendered_dynamic = torch.isin(rendered_class_id.long(), dyn_ids)
        rendered_mask = ~rendered_dynamic.reshape(curr_data['depth'].shape)

    # 3. 合并掩码：根据 Tracking 还是 Mapping 采取截然不同的哲学！
    if tracking:
        # 【Tracking 定位阶段】：惹不起躲得起！双重保险防线。
        # 第一层保险：如果现实世界告诉我们 100% 干净，直接跳过记忆纠缠，信任现实！
        if obs_mask is not None and obs_mask.all():
            semantic_mask = obs_mask
        # 第二层保险：正常双向合并
        elif obs_mask is not None and rendered_mask is not None:
            semantic_mask = obs_mask & rendered_mask
        elif obs_mask is not None:
            semantic_mask = obs_mask
        elif rendered_mask is not None:
            semantic_mask = rendered_mask
        else:
            semantic_mask = None
    else:
        # 【Mapping 建图阶段】：铁面无私，消灭幽灵！
        # 绝不包庇地图里的“渲染幽灵”。只看现实视野 (obs_mask) 是否被挡住。
        semantic_mask = obs_mask

    # 4. MAD 异常深度过滤 (带语义兜底保障)
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        
        if semantic_mask is not None:
            # 在极其纯净的静态区域计算中位数
            static_error_region = depth_error[semantic_mask & (curr_data['depth'] > 0)]
            total_valid_depth_pixels = (curr_data['depth'] > 0).sum().item()
            min_static_pixels_required = max(2000, int(total_valid_depth_pixels * 0.01))
            
            if static_error_region.numel() > min_static_pixels_required:
                pure_median = static_error_region.median()
            else:
                pure_median = depth_error.median() 
        else:
            pure_median = depth_error.median()
            
        mad_multiplier = 8.0 
        mask = (depth_error < mad_multiplier * pure_median)
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
        
    mask = mask & nan_mask

    # 5. 加上空洞屏蔽 (仅Tracking阶段)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # 6. 最后一道防线：强行套上我们刚才精心设计的语义掩码
    if semantic_mask is not None:
        mask = mask & semantic_mask
    # =========================================================================================

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss & Semantic Seg Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        if load_semantics and rendered_seg is not None:
            losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
        if load_semantics and rendered_seg is not None:
            losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg).sum()

    else:     
          # 【Mapping 建图分支：防伪影修补法】
           color_mask = torch.tile(mask, (3, 1, 1)).detach()
           
           # 1. 克隆一张真实的 Ground Truth 图像
           masked_gt_im = curr_data['im'].clone()
           
           # 2. 无缝替换：把真实图像中“被遮挡（比如有动态人）”的区域，
           #    用系统脑海中渲染出来的纯净背景像素直接替换掉！
           masked_gt_im[~color_mask] = im[~color_mask].detach()
        
           # 3. 直接对完整的图像算 L1 和 SSIM。
           # 因为被遮挡区域的像素在两张图里现在是一模一样的，误差自然为 0。
           # 更重要的是，没有了突兀的黑洞，SSIM 窗口在边缘过渡时会非常平滑，彻底消除伪影！
           losses['im'] = 0.8 * l1_loss_v1(im, masked_gt_im) + 0.2 * (1.0 - calc_ssim(im, masked_gt_im))
        
           # 4. 如果启用了语义渲染，对语义图也做同样的“无缝替换”处理
           if load_semantics and rendered_seg is not None:
              masked_gt_seg = curr_data['semantic_color'].clone()
              masked_gt_seg[~color_mask] = rendered_seg[~color_mask].detach()
              losses['seg'] = 0.8 * l1_loss_v1(rendered_seg, masked_gt_seg) \
                  + 0.2 * (1.0 - calc_ssim(rendered_seg, masked_gt_seg)) 

    # Visualize the Diff Images 
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


# def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
#              use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
#              plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False):
#     # Initialize Loss Dictionary
#     losses = {}

#     if tracking:
#         # Get current frame Gaussians, where only the camera pose gets gradient
#         transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
#                                              camera_grad=True, device=device)
#     elif mapping:
#         if do_ba:
#             # Get current frame Gaussians, where both camera pose and Gaussians get gradient
#             transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                                  camera_grad=True, device=device)
#         else:
#             # Get current frame Gaussians, where only the Gaussians get gradient
#             transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                                  camera_grad=False, device=device)
#     else:
#         # Get current frame Gaussians, where only the Gaussians get gradient
#         transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
#                                              camera_grad=False, device=device)

#     # Initialize Render Variables
#     rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
#     depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
#                                                                  transformed_pts, device=device)
#     # RGB Rendering
#     rendervar['means2D'].retain_grad()
#     im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
#     variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

#     # Depth & Silhouette Rendering
#     depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
#     depth = depth_sil[0, :, :].unsqueeze(0)
#     silhouette = depth_sil[1, :, :]
#     presence_sil_mask = (silhouette > sil_thres)
#     depth_sq = depth_sil[2, :, :].unsqueeze(0)
#     uncertainty = depth_sq - depth**2
#     uncertainty = uncertainty.detach()

#     # Semantic colors Rendering
#     if load_semantics:
#         semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
#         rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)

#     # Mask with valid depth values (accounts for outlier depth values)

#     # nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
#     # if ignore_outlier_depth_loss:
#     #     depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
#     #     mask = (depth_error < 10*depth_error.median())
#     #     mask = mask & (curr_data['depth'] > 0)
#     # else:
#     #     mask = (curr_data['depth'] > 0)
#     # mask = mask & nan_mask
#     # # Mask with presence silhouette mask (accounts for empty space)
#     # if tracking and use_sil_for_loss:
#     #     mask = mask & presence_sil_mask
#     # 原有 mask
#     nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
#     if ignore_outlier_depth_loss:
#        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
#        mask = (depth_error < 10*depth_error.median())
#        mask = mask & (curr_data['depth'] > 0)
#     else:
#        mask = (curr_data['depth'] > 0)
#     mask = mask & nan_mask

#     # Mask with presence silhouette mask (accounts for empty space)
#     if tracking and use_sil_for_loss:
#        mask = mask & presence_sil_mask

#     # ✅ 新增：屏蔽动态物体
#     if 'mask' in curr_data:
#         mask = mask & curr_data['mask'].reshape(mask.shape)

    
#     # Depth loss
#     if use_l1:
#         mask = mask.detach()
#         if tracking:
#             losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
#         else:
#             losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
#     # RGB Loss
#     if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
#         color_mask = torch.tile(mask, (3, 1, 1))
#         color_mask = color_mask.detach()
#         losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
#         if load_semantics:
#             losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[color_mask].sum()
#     elif tracking:
#         losses['im'] = torch.abs(curr_data['im'] - im).sum()
#         if load_semantics:
#             losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg).sum()
#     else:
#         losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
#         if load_semantics:
#             losses['seg'] = 0.8 * l1_loss_v1(rendered_seg, curr_data['semantic_color']) \
#                 + 0.2 * (1.0 - calc_ssim(rendered_seg, curr_data['semantic_color']))

#     # Visualize the Diff Images
#     if tracking and visualize_tracking_loss:
#         fig, ax = plt.subplots(2, 4, figsize=(12, 6))
#         weighted_render_im = im * color_mask
#         weighted_im = curr_data['im'] * color_mask
#         weighted_render_depth = depth * mask
#         weighted_depth = curr_data['depth'] * mask
#         diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
#         diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
#         viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
#         ax[0, 0].imshow(viz_img)
#         ax[0, 0].set_title("Weighted GT RGB")
#         viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
#         ax[1, 0].imshow(viz_render_img)
#         ax[1, 0].set_title("Weighted Rendered RGB")
#         ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
#         ax[0, 1].set_title("Weighted GT Depth")
#         ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
#         ax[1, 1].set_title("Weighted Rendered Depth")
#         ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
#         ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
#         ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
#         ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
#         ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
#         ax[0, 3].set_title("Silhouette Mask")
#         ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
#         ax[1, 3].set_title("Loss Mask")
#         # Turn off axis
#         for i in range(2):
#             for j in range(4):
#                 ax[i, j].axis('off')
#         # Set Title
#         fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
#         # Figure Tight Layout
#         fig.tight_layout()
#         os.makedirs(plot_dir, exist_ok=True)
#         plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
#         plt.close()
#         plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
#         cv2.imshow('Diff Images', plot_img)
#         cv2.waitKey(1)
#         ## Save Tracking Loss Viz
#         # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
#         # os.makedirs(save_plot_dir, exist_ok=True)
#         # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
#         # plt.close()

#     weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
#     loss = sum(weighted_losses.values())

#     seen = radius > 0
#     variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
#     variables['seen'] = seen
#     weighted_losses['loss'] = loss

#     return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=False,
                          params_opt_exclude=None):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device=device)
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    
    if load_semantics:
        params['semantic_ids'] = new_pt_cld[:, 6]
        params['semantic_colors'] = new_pt_cld[:, 7:10]

    for k, v in params.items():
        if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    return params


# def add_new_gaussians(params, params_opt_exclude, variables, curr_data, sil_thres, time_idx,
#                       mean_sq_dist_method, device="cuda", load_semantics=False):
#     # Silhouette Rendering
#     transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False,
#                                          camera_grad=False, device=device)
#     depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
#                                                                  transformed_pts, device=device)
#     depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
#     silhouette = depth_sil[1, :, :]
#     non_presence_sil_mask = (silhouette < sil_thres)
#     # Check for new foreground objects by using GT depth
#     gt_depth = curr_data['depth'][0, :, :]
#     render_depth = depth_sil[0, :, :]
#     depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
#     non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
#     # Determine non-presence mask
#     non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
#     # Flatten mask
#     non_presence_mask = non_presence_mask.reshape(-1)

#     # Get the new frame Gaussians based on the Silhouette
#     if torch.sum(non_presence_mask) > 0:
#         # Get the new pointcloud in the world frame
#         curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
#         curr_cam_tran = params['cam_trans'][..., time_idx].detach()
#         curr_w2c = torch.eye(4).to(device).float()
#         curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
#         curr_w2c[:3, 3] = curr_cam_tran
#         valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
#         non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)

#         if load_semantics:
#             semantic_id = curr_data['semantic_id']
#             semantic_color = curr_data['semantic_color']
#         else:
#             semantic_id = None
#             semantic_color = None

#         new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'],
#                                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
#                                                    mean_sq_dist_method=mean_sq_dist_method, device=device,
#                                                    load_semantics=load_semantics, semantic_id=semantic_id,
#                                                    semantic_color=semantic_color)
#         new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=load_semantics,
#                                            params_opt_exclude=params_opt_exclude)
#         N = new_params['means3D'].shape[0]
#         new_ids = torch.full((N,), time_idx, dtype=torch.long, device=device)
#         params['gaussian_frame_ids'] = torch.cat([params['gaussian_frame_ids'], new_ids], dim=0)
#         for k, v in new_params.items():
#             if k not in params_opt_exclude:
#                 params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
#             else:
#                 params[k] = torch.cat((params[k], v), dim=0)
#         num_pts = params['means3D'].shape[0]
#         variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=device).float()
#         variables['denom'] = torch.zeros(num_pts, device=device).float()
#         variables['max_2D_radius'] = torch.zeros(num_pts, device=device).float()
#         new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device=device).float()
#         variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

#     return params, variables

def add_new_gaussians(params, params_opt_exclude, variables, curr_data, sil_thres, time_idx,
                      mean_sq_dist_method, device="cuda", load_semantics=False):
    # Silhouette Rendering (渲染轮廓，判断哪些地方是已知区域)
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False,
                                         camera_grad=False, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    
    # 基础掩码 1：轮廓外（空白）区域
    non_presence_sil_mask = (silhouette < sil_thres)
    
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    
    # 基础掩码 2：深度误差极大（且真实深度更近）的区域，通常是突然出现的新物体
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    
    # Determine non-presence mask (合并上述两种情况，得到初步的“可建图区域”)
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # 初步检查：如果有需要新建高斯点的地方
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).to(device).float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        
        # 基础掩码 3：深度必须有效（不能是相机的黑边）
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)

        # ✅ 新增：最核心的一步，强行否决动态区域的“出生权”！
        # 如果外层传来了语义静态掩码 (mask)，将动态物体区域彻底剔除
        if 'mask' in curr_data:
            non_presence_mask = non_presence_mask & curr_data['mask'].reshape(-1)

        if load_semantics:
            semantic_id = curr_data['semantic_id']
            semantic_color = curr_data['semantic_color']
        else:
            semantic_id = None
            semantic_color = None
            
        # ✅ 新增二次防崩检查：
        # 经过刚才极其严格的语义过滤后，可能原本要建图的动态区域被全部杀光了（掩码全变 False）。
        # 所以必须再做一次 sum 判断，只有依然有剩余的合法像素，才去执行真实的撒种子操作。
        if torch.sum(non_presence_mask) > 0:
            new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'],
                                                       curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                                       mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                       load_semantics=load_semantics, semantic_id=semantic_id,
                                                       semantic_color=semantic_color)
            
            new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=load_semantics,
                                               params_opt_exclude=params_opt_exclude)
            N = new_params['means3D'].shape[0]
            new_ids = torch.full((N,), time_idx, dtype=torch.long, device=device)
            params['gaussian_frame_ids'] = torch.cat([params['gaussian_frame_ids'], new_ids], dim=0)
            
            for k, v in new_params.items():
                if k not in params_opt_exclude:
                    params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
                else:
                    params[k] = torch.cat((params[k], v), dim=0)
                    
            num_pts = params['means3D'].shape[0]
            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=device).float()
            variables['denom'] = torch.zeros(num_pts, device=device).float()
            variables['max_2D_radius'] = torch.zeros(num_pts, device=device).float()
            new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device=device).float()
            variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables

def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store
def extract_orb_features(image_gray):
    """
    提取 ORB 特征并返回 keypoints 和 descriptors
    """
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    return keypoints, descriptors

def loop_detection(keyframe_list, vocab, current_gray_image, keyframe_bow_list,threshold=0.15,min_loop_interval=150):
    """
    使用 DBoW2 进行回环检测，返回候选匹配 keyframe 对

    参数：
        keyframe_list: 当前关键帧列表
        vocab: DBoW2 词袋模型
        current_gray_image: 当前帧灰度图像
        threshold: 相似度阈值
    
    返回：
        List[(source_idx, target_idx, score)]
    """
    # 提取当前帧 ORB 特征
    _, curr_des = extract_orb_features(current_gray_image)
    if curr_des is None or len(curr_des) == 0:
        return []

    # 转换成 list of cv::Mat（pybind11 绑定中通常直接传 np.array 是可以的）
    curr_des_list = [curr_des[i] for i in range(curr_des.shape[0])]

    # 构建当前帧的 BOW 向量
    curr_bow = BowVector()
    vocab.transform(curr_des_list, curr_bow)
    curr_idx = len(keyframe_list) - 1
    print("当前有：==========================================",curr_idx)
    # 与历史关键帧比较
    loop_candidates = []
    for i, kf in enumerate(keyframe_list[:-1]):  # 不与自己比较
        if curr_idx - i < min_loop_interval:
            continue
        
        # print("+++++++++++++++++++++++++++++++++++++++++++++++")
        # 构建历史帧的 BOW
        kf_bow = keyframe_bow_list[i]
        if kf_bow is None:
            continue
          
        # 计算相似度（使用 L1 或 cosine）
        #score = curr_bow.score(kf_bow)
        score = vocab.score(curr_bow, kf_bow)

        if score > threshold:
            loop_candidates.append((len(keyframe_list)-1, i, score))  # 当前帧是 source
            print("000000000000000000000000000000000000000+score+000000000000000000000000000000000000000",score) 
            print("000000000000000000000+loop_dection中的source_id和target_id+000000000000000",len(keyframe_list)-1, i) 
    return loop_candidates
# ICP 优化函数
# def icp_optimization(source_pc, target_pc, init_transform,
#                               voxel_size=0.03,
#                               max_corr_multiplier=1.2,
#                               nb_neighbors=30,
#                               std_ratio=1.0,
#                               max_iter=60):
#     """
#     使用统计滤波、体素下采样、法线估计，并自动调节 max_correspondence_distance 的 ICP 精配准。

#     参数:pgo
#     返回:
#         refined_transform: ICP 后的变换矩阵
#         info: 包含 fitness 和 rmse
#     """
#     # 1. 转为 Open3D 点云
#     src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pc))
#     tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pc))

#     # 2. 统计离群点移除
#     src, _ = src.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
#     tgt, _ = tgt.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

#     # 3. 体素下采样
#     src_down = src.voxel_down_sample(voxel_size=voxel_size)
#     tgt_down = tgt.voxel_down_sample(voxel_size=voxel_size)

#     # 4. 法线估计（Point-to-Plane 需要）
#     for p in [src_down, tgt_down]:
#         p.estimate_normals(
#             o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
#         )
#         p.normalize_normals()

#     # 5. 运行 ICP
#     max_corr_dist = voxel_size * max_corr_multiplier
#     result = o3d.pipelines.registration.registration_icp(
#         src_down, tgt_down,
#         max_correspondence_distance=max_corr_dist,
#         init=init_transform,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
#     )

#     print(f"1111111111111111111111ICP fitness: {result.fitness:.4f}, 1111111111111111inlier_rmse: {result.inlier_rmse:.4f}")
#     return result.transformation, {
#         "fitness": result.fitness,
#         "rmse": result.inlier_rmse
#     }
# def icp_optimization(source_pc, target_pc, init_transform,
#                               voxel_size=0.05,
#                               max_corr_multiplier=1.5,
#                               nb_neighbors=20,
#                               std_ratio=2.0,
#                               max_iter=50):
#     """
#     带统计滤波、体素下采样、法线估计的 Point-to-Plane ICP 精配准。
#     """
#     # 1. 转为 Open3D 点云
#     src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pc))
#     tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pc))

#     # 2. 统计离群点移除
#     src, _ = src.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
#     tgt, _ = tgt.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

#     # 3. 体素下采样
#     src = src.voxel_down_sample(voxel_size)
#     tgt = tgt.voxel_down_sample(voxel_size)

#     # 4. 法线估计（Point-to-Plane 需要）
#     for p in (src, tgt):
#         p.estimate_normals(
#             o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
#         )
#         p.normalize_normals()

#     # 5. 执行 ICP
#     max_corr = voxel_size * max_corr_multiplier
#     result = o3d.pipelines.registration.registration_icp(
#         src, tgt,
#         max_correspondence_distance=voxel_size*max_corr_multiplier,
#         init=init_transform,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
#     )

#     print(f"[ICP] -------------fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.3f}-------------------------")
#     return result.transformation, {"fitness": result.fitness, "rmse": result.inlier_rmse}

def icp_optimization(source_pc, target_pc, init_transform):
    """
    使用 Open3D 对 source_pc 和 target_pc 执行点云配准（ICP），初始变换为 init_transform。
    
    参数:
        source_pc: N×3 numpy array 的源点云（当前帧）
        target_pc: N×3 numpy array 的目标点云（历史关键帧）
        init_transform: 4×4 初始变换矩阵（numpy）

    返回:
        refined_transform: ICP 精配准后的 4×4 变换矩阵
        result: ICP 的结果信息（包含 fitness、RMSE 等）
    """
    # 转换为 Open3D 点云格式
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pc)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pc)

    # ICP 配准
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=0.07,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
    )
    print("-----------------------",result.fitness,"--------------------------------")
    print("-----------------------",result.inlier_rmse,"--------------------------------")
    return result.transformation, {
        "fitness": result.fitness,
        "rmse": result.inlier_rmse
    }
# PGO 优化函数
# def pgo_optimization(keyframe_poses, loop_closures):
#     """
#     使用 Open3D 对 keyframe_poses 进行 Pose Graph Optimization（PGO）

#     参数:
#         keyframe_poses: List[np.ndarray], 每个关键帧的 4×4 相机 w2c 位姿矩阵
#         loop_closures: List[Tuple[int, int, np.ndarray]], 回环对及其相对变换
        
#     返回:
#         List[np.ndarray]: 优化后的关键帧 w2c 位姿
#     """
#     pose_graph = o3d.pipelines.registration.PoseGraph()

#     # 1. 添加节点（每个关键帧的位姿）
#     for i, w2c in enumerate(keyframe_poses):
#         c2w = np.linalg.inv(w2c)  # Open3D 中 pose 是世界到相机的逆
#         pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(c2w))

#     # 2. 添加连续帧之间的边（强约束）
#     for i in range(1, len(keyframe_poses)):
#         w2c1 = keyframe_poses[i - 1]
#         w2c2 = keyframe_poses[i]
#         rel = np.linalg.inv(w2c1) @ w2c2
#         pose_graph.edges.append(
#             o3d.pipelines.registration.PoseGraphEdge(
#                 i - 1, i, rel,
#                 uncertain=False,  # 相邻帧，约束强
#                 information=np.identity(6)*10
#             )
#         )

#     # # 3. 添加回环边（来自 ICP 的弱约束）
#     # for source_idx, target_idx, transform in loop_closures:
#     #     pose_graph.edges.append(
#     #         o3d.pipelines.registration.PoseGraphEdge(
#     #             target_idx, source_idx,
#     #             transform,
#     #             uncertain=True,  # 回环边不一定可靠
#     #             information=np.identity(6)
#     #         )
#     #     )
#     for source_idx, target_idx, transform in loop_closures:
#         pose_graph.edges.append(
#         o3d.pipelines.registration.PoseGraphEdge(
#             source_idx, target_idx,
#             transform,
#             uncertain=True,
#             information=np.identity(6)*5   # 适度强调信息
#         )
#     )
#     # 4. 优化参数
#     option = o3d.pipelines.registration.GlobalOptimizationOption(
#         max_correspondence_distance=0.5,
#         edge_prune_threshold=0.6,: 6.42

#     # 5. 优化器配置
#     o3d.pipelines.registration.global_optimization(
#         pose_graph,
#         method=o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
#         criteria=o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
#         option=option
#     )

#     # 6. 获取优化后结果
#     optimized_w2c_poses = []
#     for node in pose_graph.nodes:
#         opt_c2w = node.pose
#         opt_w2c = np.linalg.inv(opt_c2w)
#         optimized_w2c_poses.append(opt_w2c)

#     return optimized_w2c_poses


def pgo_optimization(
    keyframe_w2c_list: list[np.ndarray],
    loop_closures: list[tuple[int, int, np.ndarray]]
) -> list[np.ndarray]:
    """
    对一组关键帧位姿（w2c）和回环列表执行 Pose Graph Optimization（PGO）
    """
    # 1. 创建 PoseGraph 对象
    pose_graph = o3d.pipelines.registration.PoseGraph()

    # 2. 转换 w2c 为 c2w (⚠️ 强制转换为 float64，防止 Open3D 内部精度溢出或报错)
    c2w_list = [np.linalg.inv(w2c).astype(np.float64) for w2c in keyframe_w2c_list]
    for c2w in c2w_list:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(c2w))

    num_keyframes = len(c2w_list)

    # 3. 设定信息矩阵 (权重的绝对值稍微缩小，保持相对比例，增加 LM 算法的数值稳定性)
    # sigma_t = 0.01 对应的权重是 10000，数值过大有时会导致矩阵奇异。
    # 这里我们把基准稍微放大，让数值稳定在 100 左右，但回环和里程计的相对信任度依然保持不变。
    sigma_t = 0.1  
    sigma_r = np.deg2rad(5.0) 
    
    information_matrix = np.diag(
        [1.0 / sigma_t**2] * 3 + 
        [1.0 / sigma_r**2] * 3
    ).astype(np.float64)
    
    information_loop = (information_matrix * 0.8).astype(np.float64)

    # 4. 添加相邻帧里程计边（强约束）
    for i in range(1, num_keyframes):
       c2w_prev = c2w_list[i - 1]
       c2w_curr = c2w_list[i]
       rel_transform = (np.linalg.inv(c2w_prev) @ c2w_curr).astype(np.float64)

       edge = o3d.pipelines.registration.PoseGraphEdge(
           source_node_id = i - 1,
           target_node_id = i,
           transformation  = rel_transform,
           uncertain       = False,               # 相邻帧绝对可靠，不参与剪除
           information     = information_matrix   # 连续帧约束
       )
       pose_graph.edges.append(edge)
    
    # 5. 添加回环边（弱约束）
    for (src_idx, tgt_idx, rel_loop) in loop_closures:
        edge = o3d.pipelines.registration.PoseGraphEdge(
            source_node_id = src_idx,
            target_node_id = tgt_idx,
            transformation  = rel_loop.astype(np.float64),
            uncertain       = True,                # 标记为 True，允许优化器剪除假回环
            information     = information_loop     # 回环弱约束
        )
        pose_graph.edges.append(edge)   

    # 6. 配置全局优化选项
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05,  
        edge_prune_threshold=0.25,         
        preference_loop_closure=1.0,       # ⚠️ 新增：显式设定回环偏好权重，防止内部默认值过高
        reference_node=0                  
    )
    
    # 7. 选择优化算法和收敛标准
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = 50 

    # 8. 执行全局优化
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        method=method,
        criteria=criteria,
        option=option
    )

    # 9. 提取结果并转回 w2c
    optimized_w2c_list: list[np.ndarray] = []
    for node in pose_graph.nodes:
        optimized_c2w = node.pose
        optimized_w2c = np.linalg.inv(optimized_c2w)
        optimized_w2c_list.append(optimized_w2c)

    return optimized_w2c_list


#原5.27使用
# def pgo_optimization(
#     keyframe_w2c_list: list[np.ndarray],
#     loop_closures: list[tuple[int, int, np.ndarray]]
# ) -> list[np.ndarray]:
#     """
#     对一组关键帧位姿（w2c）和回环列表执行 Pose Graph Optimization（PGO），
#     返回优化后的关键帧位姿（w2c）。

#     参数:
#         keyframe_w2c_list: List[np.ndarray]
#             每个关键帧的 4×4 相机 w2c 位姿矩阵。
#         loop_closures: List[Tuple[int, int, np.ndarray]]
#             回环列表。每一个元素是 (source_idx, target_idx, transform)，
#             其中 transform 是一个 4×4 矩阵，代表从 source_keyframe 的 c2w 到
#             target_keyframe 的 c2w，即 (c2w_src)^(-1) @ c2w_tgt。

#     返回:
#         List[np.ndarray]:
#             优化后的每个关键帧的 4×4 相机 w2c 位姿矩阵。
#     """

#     # 1. 创建 PoseGraph 对象
#     pose_graph = o3d.pipelines.registration.PoseGraph()

#     # 2. 把输入的 w2c 列表全部转换成 c2w，并作为节点添加到 PoseGraph
#     #    Open3D 中，PoseGraphNode 存储的是 c2w（camera-to-world）。
#     c2w_list = [np.linalg.inv(w2c) for w2c in keyframe_w2c_list]
#     for c2w in c2w_list:
#         node = o3d.pipelines.registration.PoseGraphNode(c2w)
#         pose_graph.nodes.append(node)

#     # 3. 添加相邻帧之间的边（强约束）
#     #    每个边的 transform 要满足：(c2w_i)^(-1) @ c2w_j
#     #    也等价于 w2c_i @ (w2c_j)^(-1)
#     num_keyframes = len(c2w_list)
#     # for i in range(1, num_keyframes):
#     #     # c2w_{i-1}, c2w_i
#     #     c2w_prev = c2w_list[i - 1]
#     #     c2w_curr = c2w_list[i]
#     #     # 计算 (c2w_prev)^(-1) @ c2w_curr
#     #     rel_transform = np.linalg.inv(c2w_prev) @ c2w_curr

#     #     information_matrix = np.identity(6) * 10  # 连续帧约束权重大约 10
#     #     edge = o3d.pipelines.registration.PoseGraphEdge(
#     #         source_node_id=i - 1,
#     #         target_node_id=i,
#     #         transformation=rel_transform,
#     #         uncertain=False,           # 相邻帧约束视为可靠
#     #         information=information_matrix
#     #     )
#     #     pose_graph.edges.append(edge)

#     # # 4. 添加回环边（弱约束）
#     # #    假设 loop_closures 中给出的 transform 已经是 (c2w_src)^(-1) @ c2w_tgt
#     # for (src_idx, tgt_idx, rel_loop) in loop_closures:
#     #     # 如果你确认 rel_loop = (c2w_src)^(-1) @ c2w_tgt，就可直接使用
#     #     information_loop = np.identity(6)*5  # 回环权重可设为 1 或更小
#     #     edge = o3d.pipelines.registration.PoseGraphEdge(
#     #         source_node_id=src_idx,
#     #         target_node_id=tgt_idx,
#     #         transformation=rel_loop,
#     #         uncertain=True,            # 回环约束不一定都精确
#     #         information=information_loop
#     #     )
#     #     pose_graph.edges.append(edge)
#     sigma_t = 0.01
#     sigma_r = np.deg2rad(1.0) 
#     information_matrix = np.diag(
#     [1.0/sigma_t**2]*3 + 
#     [1.0/sigma_r**2]*3
# )   
    
#     information_loop = information_matrix * 0.8
#     for i in range(1, num_keyframes):
#        c2w_prev = c2w_list[i - 1]
#        c2w_curr = c2w_list[i]
#        rel_transform = np.linalg.inv(c2w_prev) @ c2w_curr

#        edge = o3d.pipelines.registration.PoseGraphEdge(
#         source_node_id = i - 1,
#         target_node_id = i,
#         transformation  = rel_transform,
#         uncertain       = False,
#         information     = information_matrix   # 连续帧约束
#     )
       
#        pose_graph.edges.append(edge)
    
#     for (src_idx, tgt_idx, rel_loop) in loop_closures:
#         edge = o3d.pipelines.registration.PoseGraphEdge(
#          source_node_id = src_idx,
#          target_node_id = tgt_idx,
#          transformation  = rel_loop,
#          uncertain       = True,
#          information     = information_loop     # 回环弱约束
#     )
#         pose_graph.edges.append(edge)   
#     # 5. 配置全局优化选项
#     option = o3d.pipelines.registration.GlobalOptimizationOption(
#         max_correspondence_distance=0.05,  # 点云尺度允许的最大对应距离
#         edge_prune_threshold=0.25,         # 若优化后一条边的残差超过 0.6m，则剪除
#         reference_node=0                  # 固定第一个节点为全局坐标的参考
#     )
    
#     # 6. 选择优化算法和收敛标准
#     method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
#     criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
#     criteria.max_iteration    = 50    # 最多迭代 120 次
#     # criteria.relative_fitness = 1e-6   # 当拟合度的改进低于 1e-6 时就认为收敛
#     # criteria.relative_rmse    = 1e-6   # 当 RMSE（残差）改进低于 1e-6 时也认为收敛
#     # option = o3d.pipelines.registration.GlobalOptimizationOption(
#     #     max_correspondence_distance=0.3,
#     #     edge_prune_threshold=0.5,
#     #     reference_node=0
#     # )
#     # criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
#     #     max_iteration=120,        # 最多迭代 50 次
#     #     relative_fitness=1e-6,   # 收敛阈值之一：相对“拟合度”变化
#     #     relative_rmse=1e-6       # 收敛阈值之二：相对“RMSE”变化
#     # )

#     # 7. 执行全局优化
#     o3d.pipelines.registration.global_optimization(
#         pose_graph,
#         method=method,
#         criteria=criteria,
#         option=option
#     )

#     # 8. 从优化后的 pose_graph 中取回 c2w，并转回 w2c
#     optimized_w2c_list: list[np.ndarray] = []
#     for node in pose_graph.nodes:
#         optimized_c2w = node.pose
#         optimized_w2c = np.linalg.inv(optimized_c2w)
#         optimized_w2c_list.append(optimized_w2c)

#     return optimized_w2c_list

# def pgo_optimization(keyframe_poses, loop_closures,
#                      max_trans=0.1,      # 单条回环最大平移 (m)
#                      max_rot_deg=5.0,    # 单条回环最大旋转 (°)
#                      loop_weight=0.1):   # 回环边权重缩放
#     pose_graph = o3d.pipelines.registration.PoseGraph()

#     # 1. 节点
#     for i, w2c in enumerate(keyframe_poses):
#         c2w = np.linalg.inv(w2c)
#         pose_graph.nodes.append(
#             o3d.pipelines.registration.PoseGraphNode(c2w))

#     # 2. 邻帧边：强约束
#     info_strong = np.identity(6)
#     for i in range(1, len(keyframe_poses)):
#         w2c1 = keyframe_poses[i - 1]
#         w2c2 = keyframe_poses[i]
#         rel = np.linalg.inv(w2c1) @ w2c2
#         pose_graph.edges.append(
#             o3d.pipelines.registration.PoseGraphEdge(
#                 i-1, i, rel,
#                 uncertain=False,
#                 information=info_strong
#             )
#         )

#     # 3. 回环边：先筛选、再弱约束
#     info_weak = np.identity(6) * loop_weight
#     for src, tgt, transform in loop_closures:
#         # 3.1 限制平移
#         t = transform[:3, 3]
#         if np.linalg.norm(t) > max_trans:
#             continue
#         # 3.2 限制旋转
#         R = transform[:3, :3]
#         angle = np.degrees(np.arccos((np.trace(R) - 1) * 0.5))
#         if angle > max_rot_deg:
#             continue

#         # 3.3 添加为弱约束
#         pose_graph.edges.append(
#             o3d.pipelines.registration.PoseGraphEdge(
#                 tgt, src, transform,
#                 uncertain=True,
#                 information=info_weak
#             )
#         )

#     # 4. 优化选项：可加一个 preference_loop_closure 权重
#     option = o3d.pipelines.registration.GlobalOptimizationOption(
#         max_correspondence_distance=0.05,
#         edge_prune_threshold=0.25,
#         preference_loop_closure=loop_weight,  # 降低回环整体影响
#         reference_node=0
#     )

#     # 5. 运行优化
#     o3d.pipelines.registration.global_optimization(
#         pose_graph,
#         method=o3d.pipelines.registration.
#                GlobalOptimizationLevenbergMarquardt(),
#         criteria=o3d.pipelines.registration.
#                GlobalOptimizationConvergenceCriteria(),
#         option=option
#     )

#     # 6. 输出结果
#     optimized_w2c = []
#     for node in pose_graph.nodes:
#         opt_c2w = node.pose
#         optimized_w2c.append(np.linalg.inv(opt_c2w))
#     return optimized_w2c

def extract_descriptors(keyframes, vocab):
    descriptors = []
    for kf in keyframes:
        gray = cv2.cvtColor(kf['color'].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
        bow_desc = vocab.transform(gray)
        descriptors.append(bow_desc)
    return descriptors
# def inverse_build_rotation(R: torch.Tensor, old_quat: torch.Tensor) -> torch.Tensor:
#     """
#     将旋转矩阵 R 转换为未归一化四元数（w, x, y, z），用于更新 cam_unnorm_rots。
#     R: torch.Tensor (3x3) 旋转矩阵
#     old_quat: torch.Tensor (4,) 原始未归一化四元数（用于保留模长）
#     return: torch.Tensor (4,) 未归一化新四元数
#     """
#     assert R.shape == (3, 3), "输入必须是 3x3 旋转矩阵"

#     # 转为 numpy 处理
#     R_np = R.detach().cpu().numpy()
#     quat_xyzw = R_scipy.from_matrix(R_np).as_quat()  # [x, y, z, w]
#     quat_wxyz = np.roll(quat_xyzw, 1)  # [w, x, y, z]

#     # 转为 torch 张量
#     quat = torch.tensor(quat_wxyz, device=R.device, dtype=torch.float32)
#     if torch.dot(quat, old_quat) < 0:
#         quat = -quat
#     # 保留原始模长（未归一化）
#     scale = torch.norm(old_quat)
#     return quat * scale  # 返回 shape: (4,)
def inverse_build_rotation(R: torch.Tensor, old_quat: torch.Tensor) -> torch.Tensor:
    """
    把 3x3 旋转矩阵 R 转成 (w,x,y,z) 四元数，并按 old_quat 的模长“反归一化”。
    R:    Tensor, shape (3,3)2、---------------------------params['semantic_ids'].shape[0]----------------------- 4139048
                                                                               -2、---------------------------params['semantic_ids'].shape[0]----------------------- 4139048
Mapping Time Step: 121: 100%|███████████████████| 50/50 [00:04
    old_quat: Tensor, shape (4,), 原来 cam_unnorm_rots 那一列
    return: Tensor, shape (4,)
    """
    # Shoemake 算法，见 https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    # 先正规化矩阵数值在 GPU 上算 trace、分支：
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    else:
        # 找到最大对角元所在行/列
        if (m00 > m11) and (m00 > m22):
            s = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s

    quat = torch.stack([qw, qx, qy, qz], dim=0)

    # 保持与 old_quat 同向，避免 180° 反转
    if torch.dot(quat, old_quat) < 0.0:
        quat = -quat

    # 按 old_quat 的 norm 反归一化
    scale = old_quat.norm()
    return quat * scale

def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    print(f"{config}")

    # Create Output Directories
    output_dir1='./predict.txt'
    output_dir = os.path.join(config["workdir"], config["run_name"])
    output_dir2 = os.path.join(config["workdir"], config["run_name1"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    vocab = dbow2.Vocabulary()
    vocab.load("/home/x/catkin_ws/src/SGS-SLAM/my_vocabulary.yml.gz")
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])
    if config["primary_device"].startswith("cuda:"):
        device_id = int(config["primary_device"].split(':')[1])
        torch.cuda.set_device(device_id)

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    if "load_semantics" not in dataset_config:
        load_semantics = False
        num_semantic_classes = 0
    else:
        load_semantics = dataset_config["load_semantics"]
        num_semantic_classes = dataset_config["num_semantic_classes"]
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        load_semantics=load_semantics,
        num_semantic_classes=num_semantic_classes,
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params1, variables, intrinsics, first_frame_w2c, cam, params_opt_exclude, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        device=device,
                                                                        densify_dataset=densify_dataset,
                                                                       load_semantics=load_semantics)   
        params, variables1, intrinsics1, first_frame_w2c1, cam1, params_opt_exclude1, \
            densify_intrinsics1, densify_cam1 = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        device=device,
                                                                        densify_dataset=densify_dataset,
                                                                       load_semantics=load_semantics)                                                                                                          
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params1, variables, intrinsics, first_frame_w2c, cam, \
            params_opt_exclude = initialize_first_timestep(dataset, num_frames, config['scene_radius_depth_ratio'],
                                                           config['mean_sq_dist_method'], device=device,
                                                        load_semantics=load_semantics)
        params, variables1, intrinsics1, first_frame_w2c1, cam1, \
            params_opt_exclude1 = initialize_first_timestep(dataset, num_frames, config['scene_radius_depth_ratio'],
                                                           config['mean_sq_dist_method'], device=device,
                                                        load_semantics=load_semantics)   
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1],
                                    tracking_intrinsics.cpu().numpy(),
                                    first_frame_w2c.detach().cpu().numpy(), device=device)
    
    # Initialize list to keep track of Keyframes


    flag2=False    
    keyframe_list = []
    # allkeyframe_list=[]
    # allkeyframe_list2=[]
    keyframe_time_indices = []
    timestamp_keyframes = []
    kf_bows_list=[]
    # dess_list=[]
    my_dist={}
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    gt_w2c_all_frames1 = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    mapping_iter_time_sum1 = 0
    mapping_iter_time_count1 = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0
    refined_loop_closures = []
    dynamic_class_ids=[]
    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        for k in params:
            if k not in params_opt_exclude:
                params[k] = torch.tensor(params[k]).to(device).float().requires_grad_(True)
            else:
                params[k] = torch.tensor(params[k]).to(device).float()

        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            if load_semantics:
                color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
            else:
                color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}

                if load_semantics:
                    semantic_id = semantic_id.permute(2, 0, 1)
                    semantic_color = semantic_color.permute(2, 0, 1) / 255
                    curr_keyframe['semantic_id'] = semantic_id
                    curr_keyframe['semantic_color'] = semantic_color
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        if load_semantics:
            color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
        else:
            color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        if time_idx==3:
            print('=====gt_pose====',gt_pose)
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        if load_semantics:
            semantic_id = semantic_id.permute(2, 0, 1)
            semantic_color = semantic_color.permute(2, 0, 1) / 255
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color
            # semantic_id: [H, W], 每个像素的类别 ID
            # dynamic_class_ids: [list], 比如 [0, 1, 2] 表示动态物体类别
            # dyn_ids = torch.tensor(dynamic_class_ids, device=device)
            # dynamic_mask = torch.isin(semantic_id.long(), dyn_ids)  # True 表示动态物体
            # keep_mask = ~dynamic_mask  # True 表示静态像素
            # keep_mask = keep_mask.reshape(-1)
            # curr_data['mask']=keep_mask
            # if dynamic_class_ids: 
            #    curr_data['dynamic_class_ids'] = dynamic_class_ids
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth,
                                  'id': iter_time_idx, 'intrinsics': tracking_intrinsics,
                                  'w2c': first_frame_w2c,'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params1 = initialize_camera_pose(params1, time_idx, forward_prop=config['tracking']['forward_prop'])
        # optimizer1 = initialize_optimizer(params1, params_opt_exclude, config['mapping']['lrs'], tracking=False)
        # with torch.no_grad():
        #  # Prune Gaussians
        #      if config['mapping']['prune_gaussians']:
        #         params1, variables = prune_gaussians1(params1, params_opt_exclude, variables, optimizer, 20, config['mapping']['pruning_dict']) 
        # Step 1: Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params1, params_opt_exclude, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params1['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params1['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params1, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                                                   tracking=True, device=device, plot_dir=eval_dir,
                                                   visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter, load_semantics=load_semantics)
                torch.cuda.empty_cache()
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True, load_semantics=load_semantics)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params1['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params1['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params1, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics, wandb_run=wandb_run, wandb_step=wandb_tracking_step,
                                            wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params1, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics)
                    else:
                        progress_bar.update(1)

                # Update the runtime numbers

                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        torch.cuda.empty_cache()
                        print("过")
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True

                        print("--------------------------继续下一次----------------------")
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                                
                        torch.cuda.empty_cache()           
                    else:
                        torch.cuda.empty_cache()
                        print("没过")
                        break
                torch.cuda.empty_cache()
            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params1['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params1['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                # rel_w2c = curr_gt_w2c[-1]
                # print("----------rel_w2c--------time_idx--------",time_idx,"以及",rel_w2c)
                # rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                # rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                # rel_w2c_tran = rel_w2c[:3, 3].detach()
                # # Update the camera parameters
                # params1['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                # params1['cam_trans'][..., time_idx] = rel_w2c_tran
                rel_w2c = curr_gt_w2c[-1]
                rel_rot = rel_w2c[:3, :3].unsqueeze(0)     # shape (1,3,3)
                rel_trans = rel_w2c[:3, 3]                 # shape (3,)

                 # 旧 quaternion（可能未归一化）
                old_quat = params1['cam_unnorm_rots'][0, :, time_idx]  # shape (4,)

    # 1) 正常把旋转矩阵转成单位四元数
                new_unit_quat = matrix_to_quaternion(rel_rot)         # shape (1,4)
                new_unit_quat = new_unit_quat[0]                      # squeeze to (4,)

    # 2) 反归一化：用旧 quat 的模长恢复幅度
                scale = old_quat.norm()
                new_unnorm_quat = new_unit_quat * scale

    # 3) 赋值回 params
                params1['cam_unnorm_rots'][0, :, time_idx] = new_unnorm_quat
                params1['cam_trans'][0, :, time_idx] = rel_trans
        # Update the runtime get
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params1, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                        tracking=True, device=device, load_semantics=load_semantics, wandb_run=wandb_run, wandb_step=wandb_time_step,
                                        wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params1, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                        tracking=True, device=device, load_semantics=load_semantics)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params1, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')
        # Add frame to normalframe list
        flag1=False
        if time_idx==1:
           print("1--------------params1['cam_unnorm_rots']------------",params1['cam_unnorm_rots'])
        print("=================time_idx=============",time_idx,"======================num_frames============",num_frames)
        ww=time_idx
        if ww==(num_frames-1):

            flag1=True
        # curr_cam_rot1 = F.normalize(params1['cam_unnorm_rots'][..., time_idx].detach())
        # curr_cam_tran1 = params1['cam_trans'][..., time_idx].detach()
        # curr_w2c1 = torch.eye(4).to(device).float()
        # curr_w2c1[:3, :3] = build_rotation(curr_cam_rot1)
        # curr_w2c1[:3, 3] = curr_cam_tran1
        # # Initialize Keyframe Info
        # curr_keyframe1 = {'id': time_idx, 'est_w2c': curr_w2c1, 'color': color, 'depth': depth}
        # if load_semantics:
        #     curr_keyframe1['semantic_id'] = semantic_id
        #     curr_keyframe1['semantic_color'] = semantic_color
        # allkeyframe_list.append(curr_keyframe1)
        # allkeyframe_list2.append(curr_keyframe1)
        # Add frame to keyframe list
        
        #                     torch.cuda.empty_cache() 
        #                      # （A） PGO 完成后，先清空所有 Gaussian
        #                     print("`````````````````````````````````````````````````allkeyframe_list,keyframe_list",len(allkeyframe_list),len(keyframe_list))
        #                     allkeyframe_poses = [kf['est_w2c'].cpu().numpy() for kf in allkeyframe_list]
        #                     allkeyframe2_poses = [kf['est_w2c'].cpu().numpy() for kf in allkeyframe_list2]
        #                     orig = {i: torch.tensor(pose, device=device, dtype=torch.float32)
        #                            for i, pose in enumerate(allkeyframe2_poses)}
        #                     opt  = {i: torch.tensor(pose,  device=device, dtype=torch.float32)
        # for i,pose in enumerate(allkeyframe_poses)}
                            
                           
                        #     # 预先算好每帧的 ΔR, Δt
                        #     deltaR = {}
                        #     deltat = {}
                        #     for i in orig:
                        #         ΔT = opt[i] @ torch.inverse(orig[i])   # 4×4
                        #         deltaR[i] = ΔT[:3, :3]                 # 3×3
                        #         deltat[i]= ΔT[:3,  3]                  # 3,
                        #         print("99999999999999999999999999999999999999orig_i99999999999999999999999999",i)
                        #     # 取出所有高斯点和它们的来源帧 ID
                        #     pts  = params['means3D']                   # [M×3]
                        #     fids = params['gaussian_frame_ids']        # [M]
                        #     print("////////////////////////////////////////////",pts.shape[0])
                        #     print("////////////////////////////////////////////",fids.shape[0])
                        #     # 对每个来源帧做一次校正
                        #     new_pts = []
                        #     for frame_id in torch.unique(fids):
                        #         mask = (fids == frame_id)
                        #         p_sub = pts[mask]                      # n×3
                        #         # p' = ΔR @ p + Δt
                        #         p_corr = (deltaR[int(frame_id)] @ p_sub.T).T + deltat[int(frame_id)]
                        #         print("99999999999999999999999999999999999999frameid99999999999999999999999999",frame_id)
                                
                        #         if not isinstance(p_corr, torch.Tensor):
                        #              p_corr = torch.nn.Parameter(torch.tensor(p_corr).to(device).float().contiguous().requires_grad_(True))
                        #         else:
                        #              p_corr = torch.nn.Parameter(p_corr.to(device).float().contiguous().requires_grad_(True))
                        #         new_pts.append(p_corr)     
                        #      # 拼回去
                        #     new_pts = torch.cat(new_pts, dim=0)
                        #     params['means3D'] = torch.nn.Parameter(new_pts.detach().clone().requires_grad_(True))
                          
                        # torch.cuda.empty_cache() 
        
        #   # Step 2: Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                    print("7777777777777777777777777777densify11777777777777777777777777777777")             
                else:
                    densify_curr_data = curr_data
                    print("7777777777777777777777777777densify2777777777777777777777777777777") 
                # dyn_ids = torch.tensor(dynamic_class_ids, device=device)
                # # 找到动态物体的像素
                # dynamic_mask = torch.isin(densify_curr_data['semantic_id'].long(), dyn_ids)
                # # 静态像素设为 True，动态设为 False
                # densify_curr_data['mask'] = ~dynamic_mask
                # Add new Gaussians to the scene based on the Silhouette
                print("==================================================================================load_semantics=",load_semantics,"================================================================")   
                params1, variables = add_new_gaussians(params1, params_opt_exclude, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx, config['mean_sq_dist_method'],
                                                      device, load_semantics=load_semantics)
                print("==================================================================================",params_opt_exclude,"================================================================")
                post_num_pts = params1['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
                
            # Update keyframes for gaussian mapping
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params1['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params1['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1],
                                                                num_keyframes, device=device)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                timestamp_keyframes.append(selected_time_idx)

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params1, params_opt_exclude, config['mapping']['lrs'], tracking=False) 
            
            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Add semantic id and colors
                if load_semantics:
                    if selected_rand_keyframe_idx == -1:
                        iter_data['semantic_id'] = semantic_id
                        iter_data['semantic_color'] = semantic_color
                    else:
                        iter_data['semantic_id'] = keyframe_list[selected_rand_keyframe_idx]['semantic_id']
                        iter_data['semantic_color'] = keyframe_list[selected_rand_keyframe_idx]['semantic_color']
                    # dyn_ids = torch.tensor(dynamic_class_ids, device=device)
                    # dynamic_mask = torch.isin(iter_data['semantic_id'].long(), dyn_ids)  # True 表示动态物体
                    # keep_mask = ~dynamic_mask  # True 表示静态像素
                    # keep_mask = keep_mask.reshape(-1)
                    # iter_data['mask']=keep_mask    
                # Loss for current frame
                loss, variables, losses = get_loss(params1, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                mapping=True, device=device, load_semantics=load_semantics)
                torch.cuda.empty_cache()
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True, load_semantics=load_semantics)
                # Backprop
                loss.backward()
                with torch.no_grad():   
                    # # Prune Gaussians
                    # if config['mapping']['prune_gaussians']:
                    #     params1, variables = prune_gaussians1(params1, params_opt_exclude, variables, optimizer, iter, config['mapping']['pruning_dict'])
                    #     if config['use_wandb']:
                    #         wandb_run.log({"Mapping/Number of Gaussians - Pruning": params1['means3D'].shape[0],
                    #                        "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params1, variables = densify(params1, variables, optimizer, iter, config['mapping']['densify_dict'], params_opt_exclude, device=device)
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params1['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                     # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params1, variables = prune_gaussians(params1, params_opt_exclude, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params1['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})        
                    # Optimizer Update
                    # 在 mapping 循环里，紧跟着 loss.backward():
                    # print("rot grad norm:", params1['cam_unnorm_rots'].grad.norm().item())
                    # print("trans grad norm:", params1['cam_trans'].grad.norm().item())

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params1, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                        else:
                            report_progress(params1, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
                torch.cuda.empty_cache()
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params1, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params1, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params1, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        if len(keyframe_list)>0:
            for i in range(0,len(keyframe_list)):
                with torch.no_grad():
                # Get the current estimated rotation & translation
                   curr_cam_rot = F.normalize(params1['cam_unnorm_rots'][..., keyframe_list[i]["id"]].detach())
                   curr_cam_tran = params1['cam_trans'][..., keyframe_list[i]["id"]].detach()
                   curr_w2c = torch.eye(4).to(device).float()
                   curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                   curr_w2c[:3, 3] = curr_cam_tran
                   keyframe_list[i]["est_w2c"]=curr_w2c            
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params1['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params1['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                if load_semantics:
                    curr_keyframe['semantic_id'] = semantic_id
                    curr_keyframe['semantic_color'] = semantic_color

                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
                # 回环检测与优化
                gray = cv2.cvtColor((curr_keyframe['color'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                _, des = extract_orb_features(gray)
                    # Prune Gaussians

                des_list = [des[j] for j in range(des.shape[0])]
                     
                          
                kf_bow = BowVector()
                vocab.transform(des_list, kf_bow)
                kf_bows_list.append(kf_bow)
                print("!!!!111111111111111111111111",len(kf_bows_list))
                if len(keyframe_list) > 150:  # Ensure enough keyframes for loop detectio
                    # Convert current color image to grayscale for DBoW2
                    
                 
                    # Detect loop candidates using DBoW2
                    current_gray = cv2.cvtColor((curr_keyframe['color'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    loop_candidates = loop_detection(keyframe_list, vocab, current_gray,kf_bows_list)
                    torch.cuda.empty_cache() 
                    if loop_candidates:
                        # ICP Refinement for Loop Candidates
                        
                        
                        loop_candidates.sort(key=lambda x: x[2], reverse=True)
                        top_k = loop_candidates[:min(8, len(loop_candidates))]
                        print("================top_k有==================",len(top_k))
                        for candidate in top_k:
                            # Extract source and target point clouds
                            source_idx, target_idx, score = candidate
                            print("================source_idx 有==================",source_idx)
                            print("================target_idx 有==================",target_idx)
                            source_frame = keyframe_list[source_idx]
                            target_frame = keyframe_list[target_idx]
                            # print("================source_frame 有==================",source_frame)
                            # print("================target_frame 有==================",target_frame)
                            # Compute point clouds for source and target
                            print("222222222222222222222222222")
                            source_pc_full = get_pointcloud(
                source_frame['color'].to(device),
                source_frame['depth'].to(device),
                intrinsics.to(device),
                source_frame['est_w2c'].to(device),
                transform_pts=True,
                load_semantics=load_semantics,
                semantic_id=source_frame.get('semantic_id', None).to(device) if load_semantics else None,
                semantic_color=source_frame.get('semantic_color', None).to(device) if load_semantics else None,
            ).cpu().numpy()
                            print("3333333333333333333333333333")
                            target_pc_full = get_pointcloud(
                target_frame['color'].to(device),#      optimizer = initialize_optimizer(params1, params_opt_exclude, config['mapping']['lrs'], tracking=False)
        #      with torch.no_grad():
        #             # Prune Gaussians
        #             if config['mapping']['prune_gaussians']:
        #                 params1, variables = prune_gaussians1(params1, params_opt_exclude, variables, optimizer, 20, config['mapping']['pruning_dict']) 
                target_frame['depth'].to(device),
                intrinsics.to(device),
                target_frame['est_w2c'].to(device),
                transform_pts=True,
                load_semantics=load_semantics,
                semantic_id=target_frame.get('semantic_id', None).to(device) if load_semantics else None,
                semantic_color=target_frame.get('semantic_color', None).to(device) if load_semantics else None,
            ).cpu().numpy()
                            # print("================source_pc_full有==================",source_pc_full)
                            # print("================target_pc_full有==================",target_pc_full)
                            # Initial transform between source and target
                            initial_transform = np.linalg.inv(source_frame['est_w2c'].cpu().numpy()) @ target_frame['est_w2c'].cpu().numpy()
                            print("4444444444444444444444444444")
                            # print("================initial_tranform有==================",initial_transform)
                            # 提取 xyz 部分用于 Open3D
                            source_pc_xyz = source_pc_full[:, :3]  # 只取前三列
                            target_pc_xyz = target_pc_full[:, :3]
                            # print("================source_pc_xyz有==================",source_pc_xyz)
                            # print("================target_pc_xyz有==================",target_pc_xyz)
                            print("============================Source points:", source_pc_xyz.shape[0], "Target points:", target_pc_xyz.shape[0])
                            assert not np.isnan(source_pc_xyz).any(), "NaN in source point cloud!"
                            assert not np.isnan(target_pc_xyz).any(), "NaN in target point cloud!"
                            assert not np.isinf(source_pc_xyz).any(), "Inf in source point cloud!"
                            assert not np.isinf(target_pc_xyz).any(), "Inf in target point cloud!"

                            # Refine using ICP
                            refined_transform, info = icp_optimization(
                                source_pc_xyz, target_pc_xyz, initial_transform
                                                                        )
                            torch.cuda.empty_cache() 
                            print("55555555555555555555555555555555")
                            
                            if info['fitness'] > 0.43 and info['rmse']<0.04:  # 或其他合理阈值
    
                                refined_loop_closures.append((source_idx, target_idx, refined_transform))
                            
                            #del initial_transform,target_pc_full,source_pc_full,target_pc_xyz,source_pc_xyz,source_frame,target_frame    
                        print("++++++++++++=refined_loop_closures的长度是++++++++++++++=",len(refined_loop_closures))    
        my_dist[time_idx]=len(keyframe_list)
        # Pose Graph Optimization (PGO) with loop closures
        if refined_loop_closures  and flag1==True and len(refined_loop_closures)>0:
                            # Extract keyframe poses
                            print("++++++++++++++++++++++++++++进PGO啦+++++++++++++++++++++++++++++++++++++++++")
                            flag2=True
                            with torch.no_grad():
                               for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
                                    candidate_cam_unnorm_rot1 = params1['cam_unnorm_rots'][..., time_idx].detach().clone()
                                    candidate_cam_tran1 = params1['cam_trans'][..., time_idx].detach().clone()
                                    params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot1
                                    params['cam_trans'][..., time_idx] = candidate_cam_tran1

                            if len(refined_loop_closures) > 120:
                                refined_loop_closures = sorted(refined_loop_closures, key=lambda x: -np.linalg.norm(x[2][:3, 3]))[:120]

                            keyframe_poses = [kf['est_w2c'].cpu().numpy() for kf in keyframe_list]
                            print("6666666666666666666num_frame66666666666666666",num_frames,"time_idx",time_idx)
                            # Perform PGO
                            
                            # renderer = make_renderer(535.4, 539.2, 320.1, 247.6, 640, 480,params,near=0.1, far=5.0)
                            optimized_poses = pgo_optimization(keyframe_poses, refined_loop_closures)
                            # # refined_w2c   = photometric_refine(keyframe_list, optimized_poses, renderer)
                            # print("7777777777777777time_idx7777777777777777777",time_idx)
                            # # Update keyframe poses with optimized results
                            for i, opt_pose in enumerate(optimized_poses):
                                # allkeyframe_list[keyframe_list[i]['id']]['est_w2c']=torch.tensor(opt_pose).to(device).float()
                                keyframe_list[i]['est_w2c'] = torch.tensor(opt_pose).to(device).float()
                                #R = opt_pose[:3, :3]
                                t = opt_pose[:3, 3]
                                old_quat = params['cam_unnorm_rots'][0, :, keyframe_list[i]['id']] 
                                R = torch.tensor(opt_pose[:3, :3]).to(device).float()
                                new_unnorm_quat = inverse_build_rotation(R, old_quat)
                                
                                print("-------------params-----------------",params)
                                print("-------------i------------------------",i)
                                print("-------------R------------------------",R)
                                with torch.no_grad():
                                     params['cam_trans'][..., keyframe_list[i]['id']] = torch.tensor(t).to(device).float()
                                     params['cam_unnorm_rots'][0, :, keyframe_list[i]['id']] = new_unnorm_quat

                                    #  #以下等下要删除
                                    #  params1['cam_trans'][..., keyframe_list[i]['id']] = torch.tensor(t).to(device).float()
                                    #  params1['cam_unnorm_rots'][0, :, keyframe_list[i]['id']] = new_unnorm_quat 
                                    

                                print("---------------keyframe_list[i]['id']---------------",keyframe_list[i]['id'])
                                print("0000000000000000000keyframe_list[i]['id']与time_idx000000000000000000000000",keyframe_list[i]['id'],time_idx)
                #             with torch.no_grad():
                #                   eval(dataset, params1, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 
                #  mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                #  device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True)       
                            
                #             print("Eval finished, program exit.")
                #             sys.exit(0)      
                            for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
                                # Load RGBD frames incrementally instead of all frames
                                print("888888888888888888888888888888888888888888888888888888888888888888888888")
                                if load_semantics:
                                   color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
                                else:
                                   color, depth, _, gt_pose = dataset[time_idx]
                                # Process poses
                                gt_w2c = torch.linalg.inv(gt_pose)
                                # Process RGB-D Data
                                color = color.permute(2, 0, 1) / 255
                                depth = depth.permute(2, 0, 1)
                                gt_w2c_all_frames1.append(gt_w2c)
                                curr_gt_w2c = gt_w2c_all_frames1
                                # Optimize only current time step for tracking
                                iter_time_idx = time_idx
                                # Initialize Mapping Data for selected frame
                                curr_data = {'cam': cam1, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics1,
                     'w2c': first_frame_w2c1, 'iter_gt_w2c_list': curr_gt_w2c}
        
                                if load_semantics:
                                   semantic_id = semantic_id.permute(2, 0, 1)
                                   semantic_color = semantic_color.permute(2, 0, 1) / 255
                                   curr_data['semantic_id'] = semantic_id
                                   curr_data['semantic_color'] = semantic_color
        
                                 # Initialize Data for Tracking
                                if seperate_tracking_res:
                                   tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
                                   tracking_color = tracking_color.permute(2, 0, 1) / 255
                                   tracking_depth = tracking_depth.permute(2, 0, 1)
                                   tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth,
                                  'id': iter_time_idx, 'intrinsics': tracking_intrinsics,
                                  'w2c': first_frame_w2c1,'iter_gt_w2c_list': curr_gt_w2c}
                                else:
                                   tracking_curr_data = curr_data

                                # Optimization Iterations
                                num_iters_mapping = config['mapping']['num_iters']
                                 #   # Step 2: Densification & KeyFrame-based Mapping
                                if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
                                    # Densification
                                    if config['mapping']['add_new_gaussians'] and time_idx > 0:
                                    # Setup Data for Densification
                                      if seperate_densification_res:
                                        # Load RGBD frames incrementally instead of all frames
                                        densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                                        densify_color = densify_color.permute(2, 0, 1) / 255
                                        densify_depth = densify_depth.permute(2, 0, 1)
                                        densify_curr_data = {'cam': densify_cam1, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics1, 'w2c': first_frame_w2c1, 'iter_gt_w2c_list': curr_gt_w2c}
                                      else:
                                        densify_curr_data = curr_data
               
                                      # Add new Gaussians to the scene based on the Silhouette
                                      params, variables1 = add_new_gaussians(params, params_opt_exclude1, variables1, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx, config['mean_sq_dist_method'],
                                                      device, load_semantics=load_semantics)
                                      print("==================================================================================",params_opt_exclude1,"================================================================")
                                      post_num_pts = params['means3D'].shape[0]
                                      if config['use_wandb']:
                                         wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
                
                                    # Update keyframes for gaussian mapping
                                    with torch.no_grad():
                                        # Get the current estimated rotation & translation
                                        curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                                        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                                        curr_w2c = torch.eye(4).to(device).float()
                                        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                                        curr_w2c[:3, 3] = curr_cam_tran
                                        # Select Keyframes for Mapping
                                        num_keyframes = config['mapping_window_size']-2
                                        if config['keyframe_every']==1:
                                            aa=(time_idx+1)//config['keyframe_every']
                                        elif  (time_idx == num_frames-2):
                                            if (time_idx+1)%config['keyframe_every']==0:
                                               aa=(time_idx+1)//config['keyframe_every']+1
                                            else:
                                               aa=(time_idx+1)//config['keyframe_every']+2       
                                        
                                        else:       
                                            aa=(time_idx+1)//config['keyframe_every']+1
                                        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",aa,time_idx,"time_idx==========================")    
                                        selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics1, keyframe_list[:my_dist.get(time_idx)-1],
                                                                num_keyframes, device=device)
                                        selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                                        if len(keyframe_list) > 0:
                                              # Add last keyframe to the selected keyframes
                                            selected_time_idx.append(keyframe_list[my_dist.get(time_idx)-1]['id'])
                                            selected_keyframes.append(my_dist.get(time_idx)-1)
                                        # Add current frame to the selected keyframes
                                        selected_time_idx.append(time_idx)
                                        selected_keyframes.append(-1)
                                        # Print the selected keyframes
                                        print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                                        timestamp_keyframes.append(selected_time_idx)

                                    # Reset Optimizer & Learning Rates for Full Map Optimization
                                    optimizer = initialize_optimizer(params, params_opt_exclude1, config['mapping']['lrs'], tracking=False) 

                                    # Mapping
                                    mapping_start_time = time.time()
                                    if num_iters_mapping > 0:
                                       progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                                    for iter in range(num_iters_mapping):
                                        iter_start_time = time.time()
                                        # Randomly select a frame until current time step amongst keyframes
                                        rand_idx = np.random.randint(0, len(selected_keyframes))
                                        selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                                        if selected_rand_keyframe_idx == -1:
                                          # Use Current Frame Data
                                           iter_time_idx = time_idx
                                           iter_color = color
                                           iter_depth = depth
                                        else:
                                           # Use Keyframe Data
                                           iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                                           iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                                           iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                                        iter_gt_w2c = gt_w2c_all_frames1[:iter_time_idx+1]
                                        iter_data = {'cam': cam1, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics1, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                                        # Add semantic id and colors
                                        if load_semantics:
                                           if selected_rand_keyframe_idx == -1:
                                              iter_data['semantic_id'] = semantic_id
                                              iter_data['semantic_color'] = semantic_color
                                           else:
                                              iter_data['semantic_id'] = keyframe_list[selected_rand_keyframe_idx]['semantic_id']
                                              iter_data['semantic_color'] = keyframe_list[selected_rand_keyframe_idx]['semantic_color']
                                        #    dyn_ids = torch.tensor(dynamic_class_ids, device=device)
                                        #    dynamic_mask = torch.isin(iter_data['semantic_id'].long(), dyn_ids)  # True 表示动态物体
                                        #    keep_mask = ~dynamic_mask  # True 表示静态像素
                                        #    keep_mask = keep_mask.reshape(-1)
                                        #    iter_data['mask']=keep_mask   
                                        # Loss for current frame
                                        loss, variables1, losses = get_loss(params, iter_data, variables1, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                mapping=True, device=device, load_semantics=load_semantics)
                                        torch.cuda.empty_cache()
                                        if config['use_wandb']:
                                            # Report Loss
                                           wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True, load_semantics=load_semantics)
                                        # Backprop
                                        loss.backward()
                                        with torch.no_grad():   
                                              # Prune Gaussians
                                            if config['mapping']['prune_gaussians']:
                                                params, variables1 = prune_gaussians(params, params_opt_exclude1, variables1, optimizer, iter, config['mapping']['pruning_dict'])
                                                if config['use_wandb']:
                                                    wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                                            # Gaussian-Splatting's Gradient-based Densification
                                            if config['mapping']['use_gaussian_splatting_densification']:
                                               params, variables1 = densify(params, variables1, optimizer, iter, config['mapping']['densify_dict'], params_opt_exclude1, device=device)
                                               if config['use_wandb']:
                                                  wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                                           # Optimizer Update
                                            print("-3---------------------------params['semantic_ids'].shape[0]-----------------------",params['semantic_ids'].shape[0])  
                                            print("-3、---------------------------params['means3d'].shape[0]-----------------------",params['means3D'].shape[0])
                                            print("-3、---------------------------params['rgb_colors'].shape[0]-----------------------",params['rgb_colors'].shape[0])  
                                            optimizer.step()
                                            optimizer.zero_grad(set_to_none=True)
                                            # print(params['cam_unnorm_rots'].grad.norm(), params['cam_trans'].grad.norm())

                                           # Report Progress
                                            if config['report_iter_progress']:
                                               if config['use_wandb']:
                                                  report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                                               else:
                                                  report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                                            else:
                                               progress_bar.update(1)
                                        # Update the runtime numbers
                                        iter_end_time = time.time()
                                        mapping_iter_time_sum1 += iter_end_time - iter_start_time
                                        mapping_iter_time_count1 += 1
                                        torch.cuda.empty_cache()
                                    if num_iters_mapping > 0:
                                       progress_bar.close()
                                    # Update the runtime numbers
                                    mapping_end_time = time.time()
                                    mapping_frame_time_sum += mapping_end_time - mapping_start_time
                                    mapping_frame_time_count += 1

                                    if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                                        try:
                                           # Report Mapping Progress
                                           progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                                           with torch.no_grad():
                                                if config['use_wandb']:
                                                   report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx, global_logging=True)
                                                else:
                                                   report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                                           progress_bar.close()
                                        except:
                                           ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                                           save_params_ckpt(params, ckpt_output_dir, time_idx)
                                           print('Failed to evaluate trajectory.')        
        #                         optimizer2 = initialize_optimizer(params, params_opt_exclude1, config['mapping']['lrs'], tracking=False)
        #                         with torch.no_grad():
        #                               # Prune Gaussians
        #                               if config['mapping']['prune_gaussians']:
        #                                  params, variables1 = prune_gaussians1(params, params_opt_exclude1, variables1, optimizer2, 20, config['mapping']['pruning_dict'])              
        #                         # … 你的 PGO 代码段 …
      
        # # Checkpoint every iteration
        # if flag2==False:
        #      optimizer1 = initialize_optimizer(params1, params_opt_exclude, config['mapping']['lrs'], tracking=False)
        #      with torch.no_grad():
        #             # Prune Gaussians
        #             if config['mapping']['prune_gaussians']:
        #                 params1, variables = prune_gaussians1(params1, params_opt_exclude, variables, optimizer1, 20, config['mapping']['pruning_dict']) 
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            if flag2==True:
              save_params_ckpt(params, ckpt_output_dir, time_idx)
              np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
            else:
              save_params_ckpt(params1, ckpt_output_dir, time_idx)
              np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))  
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1
        gc.collect()
        torch.cuda.empty_cache()
        if num_frames==time_idx+1 and flag2==True:
             optimizer = initialize_optimizer(params, params_opt_exclude1, config['mapping']['lrs'], tracking=False)
             with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables1 = prune_gaussians(params, params_opt_exclude1, variables1, optimizer, 20, config['mapping']['pruning_dict']) 
        if num_frames==time_idx+1 and flag2==False:
             optimizer = initialize_optimizer(params1, params_opt_exclude, config['mapping']['lrs'], tracking=False)
             with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params1, variables = prune_gaussians(params1, params_opt_exclude, variables, optimizer, 20, config['mapping']['pruning_dict']) 
    if config['save_timestamp_keyframes']:
        # Save keyframes selected at each timestamp
        max_length = max(len(inner) for inner in timestamp_keyframes)
        # Insert -1 for placeholder
        timestamp_keyframes_df = pd.DataFrame([inner + [-1 for _ in range(max_length - len(inner))] \
                                               for inner in timestamp_keyframes])
        timestamp_keyframes_df.to_csv(os.path.join(eval_dir, f"timestamp_keyframes.csv"), \
                                      index=False, header=False, na_rep='-1')

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            if flag2==True:
                eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True,dynamic_class_ids=[0])
            else:
                eval(dataset, params1, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True,dynamic_class_ids=[0])     
        else:
            
            if flag2==True:
                eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
        
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True,dynamic_class_ids=[0])
            else:
                eval(dataset, params1, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True,dynamic_class_ids=[0])  
    # Add Camera Parameters to Save them
    print("-4---------------------------params['semantic_ids'].shape[0]-----------------------",params['semantic_ids'].shape[0])  
    print("-4、---------------------------params['means3d'].shape[0]-----------------------",params['means3D'].shape[0])
    print("-4、---------------------------params['rgb_colors'].shape[0]-----------------------",params['rgb_colors'].shape[0])  
    params['timestep'] = variables1['timestep']
    params['intrinsics'] = intrinsics1.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c1.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    

    params1['timestep'] = variables['timestep']
    params1['intrinsics'] = intrinsics.detach().cpu().numpy()
    params1['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params1['org_width'] = dataset_config["desired_image_width"]
    params1['org_height'] = dataset_config["desired_image_height"]
    params1['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params1['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params1['gt_w2c_all_frames'] = np.stack(params1['gt_w2c_all_frames'], axis=0)
    params1['keyframe_time_indices'] = np.array(keyframe_time_indices)
    # print("2--------------params1['cam_unnorm_rots']------------",params1['cam_unnorm_rots'])
    # print("===========grounttruth+++++++++++++++",gt_w2c_all_frames)
    if flag2==True:
      # Save Parameters
      save_params_to_kitti_txt(params, output_dir1)
    else:
      save_params_to_kitti_txt(params1, output_dir1)  

    if load_semantics:
        params['semantic_ids'] = params['semantic_ids'].type(torch.uint8)
        params1['semantic_ids'] = params1['semantic_ids'].type(torch.uint8)
    if flag2==True:
      # Save Parameters
      save_params(params, output_dir)
      save_params(params1, output_dir2) 
    else:
      save_params(params1, output_dir)  
    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)