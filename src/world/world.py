import numpy as np

def make_demo_world_points():
    xs = np.linspace(0, 60, 200)
    z = np.zeros_like(xs)

    lane_center = np.stack([xs, np.zeros_like(xs), z], axis=1)
    lane_left   = np.stack([xs,  1.75*np.ones_like(xs), z], axis=1)
    lane_right  = np.stack([xs, -1.75*np.ones_like(xs), z], axis=1)

    car_center = np.array([25.0, 1.0, 0.0])
    L, W, H = 4.5, 2.0, 1.5
    box = []
    for dz in [0, H]:
        for dx in [-L/2, L/2]:
            for dy in [-W/2, W/2]:
                box.append(car_center + np.array([dx, dy, dz]))
    car_box = np.array(box)

    return {
        "lane_center": lane_center,
        "lane_left": lane_left,
        "lane_right": lane_right,
        "car_box": car_box
    }
