from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# from camera
from src.sensors.camera_pinhole import project_pinhole
from src.sensors.camera_extrinsic import make_T_cam_world
from src.common.types import CameraIntrinsics

# from kinetic bicycle model
from src.dynamics.vehicle import VehicleState, VehicleParams, step_kinematic_bicycle

COLORS = {
    "lane_left": "gold",
    "lane_center": "black",
    "lane_right": "deepskyblue",

    "car_box": "lime",

    "stop_line": "red",
    "crosswalk": "orange",
    "crosswalk_stripes": "orange",   # stripes도 동일 계열
}
DEFAULT_COLOR = "gray"

def make_T_vehicle_world(state: VehicleState):
    """
    World(x fwd, y left, z up) -> Vehicle(x fwd, y left, z up)
    """
    c, s = np.cos(state.yaw), np.sin(state.yaw)
    R = np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1],
    ], dtype=float)
    t = np.array([state.x, state.y, 0.0], dtype=float)

    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = -R @ t
    return T


def _meter_to_bev_pixel(x, y, xlim, ylim, bev_size):
    W, H = bev_size
    u = (x - xlim[0]) / (xlim[1] - xlim[0]) * (W - 1)
    v = (1 - (y - ylim[0]) / (ylim[1] - ylim[0])) * (H - 1)
    return float(u), float(v)

def show_bev_viewer(
    bev_dict,
    bev_size,
    xlim,
    ylim,
    ref_path_xy,
    states,          # list of VehicleState
    traj_xy_list,    # list of (x,y) tuples, same length as states
    world_pts,
    title="Day2 Viewer",
):
    
    # --- helpers ---
    def _order_polygon_ccw(pts_uv: np.ndarray) -> np.ndarray:
        """Order 2D points counter-clockwise around centroid."""
        c = pts_uv.mean(axis=0)
        ang = np.arctan2(pts_uv[:,1] - c[1], pts_uv[:,0] - c[0])
        return pts_uv[np.argsort(ang)]  
    
    W, H = bev_size

    fig = plt.figure(figsize=(16, 8), dpi=80)

    # bev panel
    ax = fig.add_axes([0.06, 0.14, 0.52, 0.82])
    ax.set_title(title)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    # meter ticks
    xticks = np.linspace(0, W, 6)
    yticks = np.linspace(0, H, 6)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.linspace(xlim[0], xlim[1], 6).astype(int))
    ax.set_yticklabels(np.linspace(ylim[1], ylim[0], 6).astype(int))
    ax.set_xlabel("x [m] (forward)")
    ax.set_ylabel("y [m] (left)")
    ax.grid(True)


    for k, uv_list in bev_dict.items():
        if uv_list is None:
            continue

        col = COLORS.get(k, DEFAULT_COLOR)

        for uv in uv_list:
            if k in ("lane_left", "lane_center", "lane_right"):
                ax.plot(uv[:, 0], uv[:, 1], linewidth=2.5, color=col)

            elif k == "car_box":
                car_uv = uv
                if car_uv.shape[0] >= 4:
                    bottom4 = car_uv[:4].copy()
                    bottom4 = _order_polygon_ccw(bottom4)
                    poly = np.vstack([bottom4, bottom4[0]])
                    ax.plot(poly[:, 0], poly[:, 1], linewidth=2.5, color=col)
                else:
                    ax.scatter(car_uv[:, 0], car_uv[:, 1], s=30, color=col)

            elif k in ("crosswalk", "crosswalk_stripes", "stop_line"):
                ax.plot(uv[:, 0], uv[:, 1], linewidth=2.0, color=col)

            else:
                ax.scatter(uv[:, 0], uv[:, 1], s=6, color=col)


    # draw static reference path (once)
    ref_uv = []
    for (x, y) in ref_path_xy:
        u, v = _meter_to_bev_pixel(x, y, xlim, ylim, bev_size)
        ref_uv.append((u, v))
    ref_uv = np.array(ref_uv, dtype=float)
    ax.plot(ref_uv[:, 0], ref_uv[:, 1], linewidth=2, label="reference path")

    # dynamic artists (will update)

    traj_scatter = ax.scatter([], [], s=20, facecolors='none', edgecolors='red', label="trajectory")
    ego_pt = ax.scatter([], [], s=90, marker="+")
    heading_line, = ax.plot([], [], linewidth=2)
    ax.legend(loc="upper right")

    # cam panel
    ax_cam = fig.add_axes([0.62, 0.14, 0.36, 0.82])

    K = CameraIntrinsics(
        fx=900.0, fy=900.0,
        cx=640.0, cy=360.0,
        width=1280, height=720
    )

    ax_cam.set_title("Pinhole Camera Sensor Viewer")
    ax_cam.set_xlim(0, K.width)
    ax_cam.set_ylim(K.height, 0)
    ax_cam.set_xlabel("u [px]")
    ax_cam.set_ylabel("v [px]")
    ax_cam.grid(True)

    cam_artists = []  # frame마다 지우고 다시 그림


    # state for viewer
    idx = {"i": 0}
    playing = {"on": False}
    timer = fig.canvas.new_timer(interval=50)  # ms

    def draw_frame(i):
        i = int(np.clip(i, 0, len(states) - 1))
        idx["i"] = i

        # trajectory up to i
        traj_xy = np.array(traj_xy_list[: i + 1], dtype=float)
        traj_uv = np.array([_meter_to_bev_pixel(x, y, xlim, ylim, bev_size) for x, y in traj_xy])
        traj_scatter.set_offsets(traj_uv)
        
        # ego pose
        s = states[i]
        eu, ev = _meter_to_bev_pixel(s.x, s.y, xlim, ylim, bev_size)
        ego_pt.set_offsets(np.array([[eu, ev]]))

        # heading (yaw)
        Lh = 5.0
        hx = s.x + Lh * np.cos(s.yaw)
        hy = s.y + Lh * np.sin(s.yaw)
        hu, hv = _meter_to_bev_pixel(hx, hy, xlim, ylim, bev_size)
        heading_line.set_data([eu, hu], [ev, hv])

        for a in cam_artists:
            try:
                a.remove()
            except Exception:
                pass
        cam_artists.clear()

        T_cam_world = make_T_cam_world() @ make_T_vehicle_world(s)

        for name, pts_list in world_pts.items():
            if pts_list is None:
                continue

            col = COLORS.get(name, DEFAULT_COLOR)

            for pts3 in pts_list:
                uv, _ = project_pinhole(pts3, T_cam_world, K)
                if uv.shape[0] == 0:
                    continue

                if name in ("crosswalk", "crosswalk_stripes", "stop_line"):
                    ln, = ax_cam.plot(uv[:, 0], uv[:, 1], linewidth=2.0, color=col)
                    cam_artists.append(ln)
                else:
                    sc = ax_cam.scatter(uv[:, 0], uv[:, 1], s=6, color=col)
                    cam_artists.append(sc)


        # legend 중복 방지: 매 프레임 새로 갱신
        if len(cam_artists) > 0:
            ax_cam.legend(loc="upper right")


        ax.set_title(f"{title}  |  frame {i+1}/{len(states)}  |  v={s.v:.2f} m/s  yaw={np.rad2deg(s.yaw):.1f}°")
        fig.canvas.draw_idle()




    def on_prev(event):
        playing["on"] = False
        draw_frame(idx["i"] - 1)

    def on_next(event):
        playing["on"] = False
        draw_frame(idx["i"] + 1)

    def on_play_pause(event):
        playing["on"] = not playing["on"]

    def _tick():
        if not playing["on"]:
            return
        if idx["i"] >= len(states) - 1:
            playing["on"] = False
            return
        draw_frame(idx["i"] + 1)

    timer.add_callback(_tick)
    timer.start()

    # Buttons
    ax_prev = fig.add_axes([0.20, 0.03, 0.10, 0.07])
    ax_next = fig.add_axes([0.32, 0.03, 0.10, 0.07])
    ax_play = fig.add_axes([0.44, 0.03, 0.12, 0.07])

    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")
    b_play = Button(ax_play, "Play/Pause")

    b_prev.on_clicked(on_prev)
    b_next.on_clicked(on_next)
    b_play.on_clicked(on_play_pause)

    # initial
    draw_frame(0)
    plt.show()

