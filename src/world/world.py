import numpy as np

def make_demo_world_points():
    lane_width = 3.5
    z0 = 0.0

    # =========================================================
    # 1) Main road (east-west, x direction)
    # =========================================================
    xs = np.linspace(-40, 40, 400)
    z_main = z0 * np.ones_like(xs)

    main_center = np.stack([xs, np.zeros_like(xs), z_main], axis=1)
    main_left   = np.stack([xs,  (lane_width/2)*np.ones_like(xs), z_main], axis=1)
    main_right  = np.stack([xs, -(lane_width/2)*np.ones_like(xs), z_main], axis=1)

    lane_center = np.vstack([main_center])
    lane_left   = np.vstack([main_left])
    lane_right  = np.vstack([main_right])
    
    # car_center = np.array([ -20.0, -1.0, 0.0 ])  # starting before intersection
    car_center = np.array([25.0, 1.0, 0.0])

    L, W, H = 4.5, 2.0, 1.5

    box = []
    for dz in [0.0, H]:
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


# def make_demo_world_points():
#     xs = np.linspace(0, 60, 200)
#     z = np.zeros_like(xs)

#     lane_center = np.stack([xs, np.zeros_like(xs), z], axis=1)
#     lane_left   = np.stack([xs,  1.75*np.ones_like(xs), z], axis=1)
#     lane_right  = np.stack([xs, -1.75*np.ones_like(xs), z], axis=1)

#     car_center = np.array([25.0, 1.0, 0.0])
#     L, W, H = 4.5, 2.0, 1.5
#     box = []
#     for dz in [0, H]:
#         for dx in [-L/2, L/2]:
#             for dy in [-W/2, W/2]:
#                 box.append(car_center + np.array([dx, dy, dz]))
#     car_box = np.array(box)

#     return {
#         "lane_center": lane_center,
#         "lane_left": lane_left,
#         "lane_right": lane_right,
#         "car_box": car_box
#     }
