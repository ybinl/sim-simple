from __future__ import annotations
import numpy as np

def make_straight_lane_path(x_start: float = 0.0, x_end: float = 80.0, y: float = 0.0, num: int = 400) -> np.ndarray:
    xs = np.linspace(x_start, x_end, num)
    ys = np.full_like(xs, y, dtype=float)
    return np.stack([xs, ys], axis=1)

def make_lane_change_path(x_start: float = 0.0, x_end: float = 80.0,
                          y0: float = 0.0, y1: float = 3.5,
                          x_change_start: float = 20.0, x_change_end: float = 40.0,
                          num: int = 500) -> np.ndarray:
    """
    Simple smooth lane change using cubic interpolation on y(x).
    """
    xs = np.linspace(x_start, x_end, num)
    ys = np.zeros_like(xs, dtype=float)

    for i, x in enumerate(xs):
        if x < x_change_start:
            ys[i] = y0
        elif x > x_change_end:
            ys[i] = y1
        else:
            # normalize s in [0,1]
            s = (x - x_change_start) / (x_change_end - x_change_start)
            # smoothstep cubic: 3s^2 - 2s^3
            w = 3*s*s - 2*s*s*s
            ys[i] = (1-w)*y0 + w*y1

    return np.stack([xs, ys], axis=1)

def make_turn_path(center, radius, angle_start, angle_end, num=100):
    angles = np.linspace(angle_start, angle_end, num)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return list(zip(xs, ys))
