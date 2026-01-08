import numpy as np
from src.common.math_utils import rot_z, rot_x, rot_y, euler_zyx

def make_T_cam_world():
    """
    Define camera frame explicitly:
    - Camera +Z : forward
    - Camera +X : right
    - Camera +Y : down
    """

    # Camera position in world (x forward, y left, z up)
    t = np.array([1.5, 0.0, 1.3])

    # Optional: pitch down
    pitch = np.deg2rad(10.0)
    yaw = np.deg2rad(0)
    roll = np.deg2rad(0)
    R_euler = euler_zyx(roll,yaw,pitch)
    
    # R_pitch = rot_z(pitch)
    # World -> Camera rotation
    # World axes: X forward, Y left, Z up
    # Camera axes: Z forward, X right, Y down
    R_world_cam = np.array([
        [ 0, -1,  0],   # world Y(left) -> camera X(right)
        [ 0,  0, -1],   # world Z(up)   -> camera Y(down)
        [ 1,  0,  0],   # world X(fwd)  -> camera Z(fwd)
    ], dtype=float)

    # R = R_pitch @ R_world_cam
    # R = R_euler @ R_world_cam

    T = np.eye(4)
    T[:3,:3] = R_euler @ R_world_cam
    T[:3, 3] = t
    return T
