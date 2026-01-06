from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from src.dynamics.vehicle import VehicleState, VehicleParams, clamp

@dataclass(frozen=True)
class ControlParams:
    lookahead_base: float = 3.0    # m
    lookahead_gain: float = 0.2    # lookahead = base + gain*v
    k_speed: float = 0.8           # P gain for speed control

def nearest_point_on_path(path_xy: np.ndarray, x: float, y: float) -> tuple[int, float, np.ndarray]:
    """
    path_xy: (N,2)
    returns:
      idx: nearest index
      dist: Euclidean distance
      p: nearest point
    """
    dx = path_xy[:,0] - x
    dy = path_xy[:,1] - y
    d2 = dx*dx + dy*dy
    idx = int(np.argmin(d2))
    return idx, float(np.sqrt(d2[idx])), path_xy[idx]

def path_heading_at(path_xy: np.ndarray, idx: int) -> float:
    """
    Approx path tangent heading at index idx.
    """
    N = path_xy.shape[0]
    if N < 2:
        return 0.0
    i0 = max(0, min(N-2, idx))
    p0 = path_xy[i0]
    p1 = path_xy[i0+1]
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    return float(np.arctan2(dy, dx))

def signed_lateral_error(state: VehicleState, path_xy: np.ndarray, idx_near: int) -> float:
    """
    Signed cross-track error:
      sign is positive if vehicle is left of path direction.
    """
    p = path_xy[idx_near]
    psi_path = path_heading_at(path_xy, idx_near)

    # vector from path point to vehicle
    ex = state.x - p[0]
    ey = state.y - p[1]

    # left normal of path heading
    nx = -np.sin(psi_path)
    ny =  np.cos(psi_path)
    return float(ex*nx + ey*ny)

def heading_error(state: VehicleState, path_xy: np.ndarray, idx_near: int) -> float:
    psi_path = path_heading_at(path_xy, idx_near)
    e = psi_path - state.yaw
    # wrap
    e = (e + np.pi) % (2*np.pi) - np.pi
    return float(e)

def pure_pursuit_steer(
    state: VehicleState,
    path_xy: np.ndarray,
    params_vehicle: VehicleParams,
    params_ctrl: ControlParams
) -> tuple[float, dict]:
    """
    Pure pursuit:
      choose lookahead point ahead on path, compute curvature, steer.
    returns:
      delta (rad), debug info
    """
    # find nearest
    idx_near, _, _ = nearest_point_on_path(path_xy, state.x, state.y)

    # lookahead distance
    Ld = params_ctrl.lookahead_base + params_ctrl.lookahead_gain * state.v
    Ld = max(1.0, Ld)

    # find target index by accumulating along path
    target_idx = idx_near
    acc = 0.0
    for i in range(idx_near, path_xy.shape[0]-1):
        seg = path_xy[i+1] - path_xy[i]
        acc += float(np.linalg.norm(seg))
        target_idx = i+1
        if acc >= Ld:
            break

    tx, ty = path_xy[target_idx]
    # transform target point to vehicle frame
    dx = tx - state.x
    dy = ty - state.y
    # vehicle frame: x forward, y left
    x_v =  np.cos(state.yaw)*dx + np.sin(state.yaw)*dy
    y_v = -np.sin(state.yaw)*dx + np.cos(state.yaw)*dy

    # avoid numerical issues
    if x_v <= 1e-6:
        delta = 0.0
    else:
        curvature = 2.0 * y_v / (Ld*Ld)
        delta = np.arctan(params_vehicle.wheelbase * curvature)

    delta = clamp(delta, -params_vehicle.max_steer, params_vehicle.max_steer)

    dbg = {
        "idx_near": idx_near,
        "target_idx": target_idx,
        "Ld": Ld,
        "target_xy": (tx, ty),
        "target_in_vehicle": (x_v, y_v),
    }
    return float(delta), dbg

def speed_control(
    v: float,
    v_ref: float,
    params_vehicle: VehicleParams,
    params_ctrl: ControlParams
) -> float:
    """
    Simple P controller for speed.
    a = k*(v_ref - v)
    """
    a = params_ctrl.k_speed * (v_ref - v)
    a = clamp(a, params_vehicle.min_accel, params_vehicle.max_accel)
    return float(a)
