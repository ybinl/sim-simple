from __future__ import annotations

import numpy as np

# from bev
from src.sensors.bev import world_xy_to_bev, clip_bev

# from map
from src.world.world import make_demo_world_points, make_pangyo_world_pts

# from planning
from src.planning.planner import Planner, DriveState

# from kinetic bicycle model
from src.dynamics.vehicle import VehicleState, VehicleParams, step_kinematic_bicycle

from main_viewer import show_bev_viewer

# from control
from src.dynamics.control import (
    ControlParams,
    nearest_point_on_path,
    signed_lateral_error,
    heading_error,
    pure_pursuit_steer,
    speed_control
)
from src.planning.path import make_straight_lane_path, make_lane_change_path

def normalize_items_to_polylines(items):
    """
    items can be:
      - List[np.ndarray] where each is (Ni,2/3)
      - np.ndarray (N,2/3)  -> treated as one polyline
      - np.ndarray (3,)     -> treated as one point polyline (1,3)
    returns: List[np.ndarray]
    """
    if items is None:
        return []

    # List: already multiple polylines (or points)
    if isinstance(items, list):
        out = []
        for a in items:
            if a is None:
                continue
            arr = np.asarray(a)
            if arr.ndim == 1 and arr.shape[0] >= 2:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) > 0:
                out.append(arr)
        return out

    # ndarray
    arr = np.asarray(items)
    if arr.ndim == 1 and arr.shape[0] >= 2:
        return [arr.reshape(1, -1)]
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return [arr]

    return []
    
def ideal_perception(state, object_box_result, step):
    
    # object_box_result는 카메라 센서 로직에서 검출된 장애물의 위치
    # object_box 는 (x,y,0) center point가 검출되서 perception으로 들어옴
        
    return {     "object_box_result": object_box_result,
        "need_lane_change": step == 80,
        "lane_change_done": step == 160, 
        "approaching_intersection": step ==240,
        "turn_done": step == 320,
    }
    
def main():
    # --- BEV window bounds (meters) ---
    xlim = (-50.0, 100.0)
    ylim = (-20.0, 20.0)
    bev_size = (900, 600)

    # --- World ---
    world_pts = make_pangyo_world_pts()
    # world_pts = make_osm_world_pts()

    world_pts_norm = {}
    for name, items in world_pts.items():
        polylines = normalize_items_to_polylines(items)
        if len(polylines) > 0:
            world_pts_norm[name] = polylines

    bev_dict = {}
    for name, polylines in world_pts_norm.items():
        bev_dict[name] = []
        for pts3 in polylines:
            bev_uv = world_xy_to_bev(pts3, xlim=xlim, ylim=ylim, bev_size=bev_size)
            bev_uv = clip_bev(bev_uv, bev_size)
            if len(bev_uv) > 0:
                bev_dict[name].append(bev_uv)

    # --- planner ---
    planner = Planner()
    
    # --- Reference path ---
    ref_path = make_straight_lane_path(x_start=0.0, x_end=80.0, y=0.0)
    # ref_path = make_lane_change_path(x_start=0.0, x_end=80.0, y0=0.0, y1=3.5, x_change_start=20.0, x_change_end=40.0)

    # --- Vehicle and controllers ---
    vparams = VehicleParams(wheelbase=2.8)
    cparams = ControlParams(lookahead_base=4.0, lookahead_gain=0.25, k_speed=0.8)

    state = VehicleState(x=0.0, y=-1.0, yaw=np.deg2rad(0.0), v=0.0)
    v_ref = 10.0  # m/s

    # --- Simulation loop ---
    dt = 0.05
    T = 20.0
    steps = int(T / dt)

    states=[state]
    traj = [(state.x, state.y)]
    
    # map (world) 에 존재하는 장애물을 그대로 입력 받았다고 가정.
    obj_detection = world_pts.get("car_box", None)
    if obj_detection is not None:
        obj_2d = np.mean(obj_detection, axis=0)[:2]
    else:
        obj_2d = None
        
    for k in range(steps):
        perception = ideal_perception(state, obj_2d, k)
        drive_state, ref_path = planner.update(state, perception)
        
        # ref_path 를 제어 모듈이 받아서 사용함. planner 가 ref_path를 만들어줌
        ref_path = np.array(ref_path)
        
        # Errors (Pass/Fail #2)
        idx_near, _, _ = nearest_point_on_path(ref_path, state.x, state.y)
        e_lat = signed_lateral_error(state, ref_path, idx_near)
        e_head = heading_error(state, ref_path, idx_near)

        # Steering control (Pass/Fail #3)
        delta, _ = pure_pursuit_steer(state, ref_path, vparams, cparams)

        # Speed control (Pass/Fail #4)
        a = speed_control(state.v, v_ref, vparams, cparams)

        # Vehicle update (Pass/Fail #1)
        state = step_kinematic_bicycle(state, delta, a, dt, vparams)
        states.append(state)
        traj.append((state.x, state.y))
    
    show_bev_viewer(
        bev_dict=bev_dict,
        bev_size=bev_size,
        xlim=xlim,
        ylim=ylim,
        ref_path_xy=ref_path,
        states=states,
        traj_xy_list=traj,
        world_pts=world_pts_norm,
        title="Dynamics & Control BEV Viewer"
    )
if __name__ == "__main__":
    main()
