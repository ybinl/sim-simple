import numpy as np
import matplotlib.pyplot as plt

from src.common.types import CameraIntrinsics
from src.common.math_utils import rot_z
from src.world.world import make_demo_world_points
from src.sensors.camera_pinhole import project_pinhole
from src.sensors.bev import world_xy_to_bev, clip_bev
from src.sensors.viz_sensor import plot_image, plot_bev

def make_T_cam_world():
    """
    Define camera frame explicitly:
    - Camera +Z : forward
    - Camera +X : right
    - Camera +Y : down
    """

    # Camera position in world (x forward, y left, z up)
    t = np.array([1.5, 0.0, 1.3])

    # World -> Camera rotation
    # World axes: X forward, Y left, Z up
    # Camera axes: Z forward, X right, Y down
    R_world_cam = np.array([
        [ 0, -1,  0],   # world Y(left) -> camera X(right)
        [ 0,  0, -1],   # world Z(up)   -> camera Y(down)
        [ 1,  0,  0],   # world X(fwd)  -> camera Z(fwd)
    ], dtype=float)

    # Optional: pitch down
    pitch = np.deg2rad(-10.0)
    R_pitch = rot_z(pitch)

    R = R_pitch @ R_world_cam

    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

# def make_T_cam_world():
#     t = np.array([1.5, 0.0, 1.3])
#     R = euler_zyx(
#         yaw=0.0,
#         pitch=np.deg2rad(-10.0),
#         roll=0.0
#     )

#     T = np.eye(4)
#     T[:3,:3] = R
#     T[:3, 3] = t
#     return T

def main():
    world_pts = make_demo_world_points()

    K = CameraIntrinsics(
        fx=900, fy=900,
        cx=640, cy=360,
        width=1280, height=720
    )

    T_cam_world = make_T_cam_world()

    uv_dict = {}
    for name, pts in world_pts.items():
        uv, _ = project_pinhole(pts, T_cam_world, K)
        uv_dict[name] = uv

    bev_size = (600,600)
    bev_dict = {}
    for name, pts in world_pts.items():
        bev_uv = world_xy_to_bev(
            pts,
            xlim=(0,60),
            ylim=(-10,10),
            bev_size=bev_size
        )
        bev_dict[name] = clip_bev(bev_uv, bev_size)

    plot_image(uv_dict, K.width, K.height)
    plot_bev(
        bev_dict,
        bev_size=bev_size,
        xlim=(0, 60),
        ylim=(-10, 10)
    )
    plt.show()

if __name__ == "__main__":
    main()
