import numpy as np
from src.common.math_utils import apply_transform

def project_pinhole(points_world, T_cam_world, K):
    pts_cam = apply_transform(T_cam_world, points_world)
    X, Y, Z = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]

    eps = 1e-6
    valid = Z > eps

    u = K.fx * X / (Z + eps) + K.cx
    v = K.fy * Y / (Z + eps) + K.cy

    in_img = (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
    mask = valid & in_img

    uv = np.stack([u[mask], v[mask]], axis=1)
    return uv, mask
