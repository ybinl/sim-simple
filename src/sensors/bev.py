import numpy as np

def _as_points_array(points_world):
    """
    Accept:
      - np.ndarray (N,2) or (N,3)
      - List[np.ndarray] where each is (Ni,2)/(Ni,3)
    Return:
      - np.ndarray (N,2 or 3)
    """
    if isinstance(points_world, list):
        if len(points_world) == 0:
            return np.zeros((0, 3), dtype=float)
        # filter None/empty
        chunks = [np.asarray(p) for p in points_world if p is not None and len(p) > 0]
        if len(chunks) == 0:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(chunks)

    arr = np.asarray(points_world)
    # handle object array coming from np.array(list_of_arrays)
    if arr.ndim == 1 and arr.dtype == object:
        chunks = [np.asarray(p) for p in arr if p is not None and len(p) > 0]
        if len(chunks) == 0:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(chunks)

    return arr


def world_xy_to_bev(points_world, xlim, ylim, bev_size):
    W, H = bev_size
    xmin, xmax = xlim
    ymin, ymax = ylim

    pts = _as_points_array(points_world)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points_world must be (N,2+) array or List[(Ni,2+)], got shape={pts.shape}")

    x = pts[:, 0]
    y = pts[:, 1]

    u = (x - xmin) / (xmax - xmin) * (W - 1)
    v = (1.0 - (y - ymin) / (ymax - ymin)) * (H - 1)

    return np.stack([u, v], axis=1)

def clip_bev(bev_uv, bev_size):
    W, H = bev_size
    bev_uv = np.asarray(bev_uv, dtype=float)

    u = bev_uv[:, 0]
    v = bev_uv[:, 1]

    nan_mask = np.isnan(u) | np.isnan(v)
    in_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # NaN은 무조건 살리고, 정상 점은 화면 안에 있는 것만 살림
    keep = nan_mask | in_mask
    return bev_uv[keep]