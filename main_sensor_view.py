import matplotlib.pyplot as plt

from src.common.types import CameraIntrinsics
from src.world.world import make_demo_world_points
from src.sensors.camera_pinhole import project_pinhole
from src.sensors.camera_extrinsic import make_T_cam_world
from src.sensors.bev import world_xy_to_bev, clip_bev
from src.sensors.viz_sensor import plot_image, plot_bev

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
