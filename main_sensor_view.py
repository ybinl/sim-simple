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
        width=1280, height=720,
        # distortion=(k0,k1,k2,k3,k4),
        # undistortion=(u0,u1,u2,u3,u4),
    )

    T_cam_world = make_T_cam_world()

    uv_dict = {}
    for name, pts in world_pts.items():
        if len(pts) != 1: continue
        
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

# camera model pinhole with no distortion
# -> distortion, undistortion
# -> fisheye

# camera extrinsic
# -> x,y,z,roll,pitch,yaw

# map (world)
# -> lane
# -> traffic (car -> 선행차량/ 아닌차량, human -> 도로 위에 있는 사람, sidewalk 에 있는 사람)
# perception model -> tracking trajectory 인지 모듈, ..... (다양)
# -> crosswalk, traffic sign, stopline ( -> path planning에서 중요하게 보는 인지 입력)
# perception model -> 신호등이랑 정지선같이 법규적으로 인지가능한 signal들을 인지하는 모듈,....

# reference path 생성 -> planning 할 때 perception 출력을 사용함.
# v2x 로 신호등 신호 받아오기도 함. (서울시청 신호등 정보 실시간으로 통신해서 signal 주세요.)