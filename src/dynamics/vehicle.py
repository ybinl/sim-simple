from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float  # rad
    v: float    # m/s

@dataclass(frozen=True)
class VehicleParams:
    wheelbase: float = 2.8   # m
    max_steer: float = np.deg2rad(35.0)  # rad
    max_accel: float = 3.0   # m/s^2
    min_accel: float = -6.0  # m/s^2 (brake)
    max_speed: float = 25.0  # m/s
    min_speed: float = 0.0   # m/s

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def step_kinematic_bicycle(
    state: VehicleState,
    delta: float,  # steering angle (front wheel), rad
    a: float,      # longitudinal accel, m/s^2
    dt: float,
    params: VehicleParams
) -> VehicleState:
    """
    Kinematic bicycle (no slip):
      x_dot = v * cos(yaw)
      y_dot = v * sin(yaw)
      yaw_dot = v/L * tan(delta)
      v_dot = a
    """
    delta = clamp(delta, -params.max_steer, params.max_steer)
    a = clamp(a, params.min_accel, params.max_accel)

    x = state.x + state.v * np.cos(state.yaw) * dt
    y = state.y + state.v * np.sin(state.yaw) * dt
    yaw = state.yaw + (state.v / params.wheelbase) * np.tan(delta) * dt
    v = state.v + a * dt

    v = clamp(v, params.min_speed, params.max_speed)

    # normalize yaw to [-pi, pi]
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi

    return VehicleState(x=x, y=y, yaw=yaw, v=v)
