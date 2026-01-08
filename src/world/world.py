import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

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

def shift_pts(items, dx, dy):
    if items is None:
        return None
    if isinstance(items, list):
        out=[]
        for a in items:
            if a is None: 
                continue
            b = np.asarray(a).copy()
            if b.ndim == 1:
                b = b.reshape(1,-1)
            b[:,0] += dx
            b[:,1] += dy
            out.append(b)
        return out
    b = np.asarray(items).copy()
    if b.ndim == 1:
        b = b.reshape(1,-1)
    b[:,0] += dx
    b[:,1] += dy
    return b

def make_pangyo_world_pts(
    area_half: float = 200.0,          # 전체 400m x 400m
    intersection_half: float = 25.0,   # 교차로 half-size (I)
    lane_width: float = 3.5,
    lanes_each_dir: int = 2,
    sample_step: float = 0.5,
    z0: float = 0.0,
    add_straight_connectors: bool = True,
    add_right_lane_centers: bool = True,
    dx: float = 0.0,
    dy: float = 0.0,
) -> Dict[str, object]:

    I = float(intersection_half)

    def _linspace_pts_1d(a: float, b: float, step: float) -> np.ndarray:
        n = max(2, int(np.ceil(abs(b - a) / step)) + 1)
        return np.linspace(a, b, n)

    def _make_straight_polyline(axis: str, s0: float, s1: float, offset: float) -> np.ndarray:
        ss = _linspace_pts_1d(s0, s1, sample_step)
        if axis == "x":
            x = ss
            y = np.full_like(ss, offset)
        elif axis == "y":
            y = ss
            x = np.full_like(ss, offset)
        else:
            raise ValueError("axis must be 'x' or 'y'")
        z = np.full_like(ss, z0)
        return np.stack([x, y, z], axis=1)

    def _offset_boundary(poly: np.ndarray, normal_xy: np.ndarray, dist: float) -> np.ndarray:
        n = normal_xy / (np.linalg.norm(normal_xy) + 1e-12)
        out = poly.copy()
        out[:, 0] += n[0] * dist
        out[:, 1] += n[1] * dist
        return out

    def _make_arc(center: np.ndarray, r: float, th0: float, th1: float, step_m: float) -> np.ndarray:
        arc_len = abs(th1 - th0) * r
        n = max(10, int(np.ceil(arc_len / step_m)) + 1)
        th = np.linspace(th0, th1, n)
        x = center[0] + r * np.cos(th)
        y = center[1] + r * np.sin(th)
        z = np.full_like(x, z0)
        return np.stack([x, y, z], axis=1)

    # ---------------------------------------
    # containers
    # ---------------------------------------
    lane_center: List[np.ndarray] = []
    lane_left:   List[np.ndarray] = []
    lane_right:  List[np.ndarray] = []

    # optional extra layers
    crosswalk: List[np.ndarray] = []
    stop_line: List[np.ndarray] = []

    # ---------------------------------------
    # straight approaches (교차로 안은 비우고 접근로만)
    # ---------------------------------------
    x_w0, x_w1 = -area_half, -I
    x_e0, x_e1 =  I,  area_half
    y_s0, y_s1 = -area_half, -I
    y_n0, y_n1 =  I,  area_half

    offsets_one_side = (np.arange(lanes_each_dir) + 0.5) * lane_width  # 0.5W, 1.5W ...

    # E-W (axis='x'): normal is +y
    for off in offsets_one_side:
        for c in (_make_straight_polyline("x", x_w0, x_w1, +off),
                  _make_straight_polyline("x", x_e0, x_e1, +off),
                  _make_straight_polyline("x", x_w0, x_w1, -off),
                  _make_straight_polyline("x", x_e0, x_e1, -off)):
            lane_center.append(c)
            lane_left.append(_offset_boundary(c, np.array([0.0, 1.0]), +lane_width/2))
            lane_right.append(_offset_boundary(c, np.array([0.0, 1.0]), -lane_width/2))

    # N-S (axis='y'): normal is +x
    for off in offsets_one_side:
        for c in (_make_straight_polyline("y", y_s0, y_s1, +off),
                  _make_straight_polyline("y", y_n0, y_n1, +off),
                  _make_straight_polyline("y", y_s0, y_s1, -off),
                  _make_straight_polyline("y", y_n0, y_n1, -off)):
            lane_center.append(c)
            lane_left.append(_offset_boundary(c, np.array([1.0, 0.0]), +lane_width/2))
            lane_right.append(_offset_boundary(c, np.array([1.0, 0.0]), -lane_width/2))

    # ---------------------------------------
    # (A) straight connectors inside intersection (gap 제거)
    # ---------------------------------------
    if add_straight_connectors:
        for off in offsets_one_side:
            # Eastbound connector: y = -off, x from -I to +I
            c = _make_straight_polyline("x", -I, +I, -off)
            lane_center.append(c)

            # Westbound connector: y = +off, x from +I to -I (방향은 상관없지만 보기 좋게)
            c = _make_straight_polyline("x", +I, -I, +off)
            lane_center.append(c)

            # Northbound connector: x = +off, y from -I to +I
            c = _make_straight_polyline("y", -I, +I, +off)
            lane_center.append(c)

            # Southbound connector: x = -off, y from +I to -I
            c = _make_straight_polyline("y", +I, -I, -off)
            lane_center.append(c)

    # ---------------------------------------
    # (B) right-turn guides (Korea: right-hand traffic)
    #  - 우회전은 "자기 차선 offset 유지"가 자연스럽게 이어짐
    #  - 코너 중심: (±I, ±I), 반경 r = I - off  (off < I 이어야 함)
    # ---------------------------------------
    if add_right_lane_centers:
        for off in offsets_one_side:
            r = I - off
            if r <= 1.0:  # 너무 큰 off면(차로가 교차로보다 넓어짐) skip
                continue

            # 1) Eastbound -> Southbound (SW corner), center (-I, -I)
            # start (-I, -off) -> end (-off, -I)
            c = _make_arc(center=np.array([-I, -I], float), r=r,
                          th0=np.pi/2, th1=0.0, step_m=sample_step)
            lane_center.append(c)

            # 2) Northbound -> Eastbound (SE corner), center (+I, -I)
            # start (+off, -I) -> end (+I, -off)
            c = _make_arc(center=np.array([+I, -I], float), r=r,
                          th0=np.pi, th1=np.pi/2, step_m=sample_step)
            lane_center.append(c)

            # 3) Westbound -> Northbound (NE corner), center (+I, +I)
            # start (+I, +off) -> end (+off, +I)
            c = _make_arc(center=np.array([+I, +I], float), r=r,
                          th0=-np.pi/2, th1=-np.pi, step_m=sample_step)
            lane_center.append(c)

            # 4) Southbound -> Westbound (NW corner), center (-I, +I)
            # start (-off, +I) -> end (-I, +off)
            c = _make_arc(center=np.array([-I, +I], float), r=r,
                          th0=0.0, th1=-np.pi/2, step_m=sample_step)
            lane_center.append(c)

    # ---------------------------------------
    # stop lines (간단)
    # ---------------------------------------
    stop_len = 2.0 * (lanes_each_dir * lane_width + 1.0)
    stop_line.append(np.array([[-I-1.5, -stop_len/2, z0], [-I-1.5, +stop_len/2, z0]]))
    stop_line.append(np.array([[+I+1.5, -stop_len/2, z0], [+I+1.5, +stop_len/2, z0]]))
    stop_line.append(np.array([[-stop_len/2, -I-1.5, z0], [+stop_len/2, -I-1.5, z0]]))
    stop_line.append(np.array([[-stop_len/2, +I+1.5, z0], [+stop_len/2, +I+1.5, z0]]))

    # ---------------------------------------
    # lead car (옵션)
    # ---------------------------------------
    car_center = np.array([+60.0, -(0.5 * lane_width), z0], dtype=float)
    L, W, H = 4.5, 2.0, 1.5
    box = []
    for dz in [0.0, H]:
        for dx_ in [-L/2, L/2]:
            for dy_ in [-W/2, W/2]:
                box.append(car_center + np.array([dx_, dy_, dz], dtype=float))
    car_box = np.array(box, dtype=float)

    world = {
        "lane_center": lane_center,
        "lane_left": lane_left,
        "lane_right": lane_right,
        "stop_line": stop_line,
        "car_box": car_box,
    }

    # shift
    dx=40.0
    dy=0.0
    for k in list(world.keys()):
        world[k] = shift_pts(world[k], dx, dy)

    return world

if __name__ == "__main__":
    world_pts = make_pangyo_world_pts()
    
    plt.figure()
    for k,pts in world_pts.items():
        if "lane_center" in k and pts is not None:
            for line_ in pts:
                plt.plot(line_[:,0], line_[:,1])
                
                
    plt.show()