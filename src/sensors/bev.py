import numpy as np

def world_xy_to_bev(points_world, xlim, ylim, bev_size):
    W, H = bev_size
    xmin, xmax = xlim
    ymin, ymax = ylim

    x = points_world[:,0]
    y = points_world[:,1]

    u = (x - xmin) / (xmax - xmin) * (W - 1)
    v = (1.0 - (y - ymin) / (ymax - ymin)) * (H - 1)

    return np.stack([u, v], axis=1)

def clip_bev(bev_uv, bev_size):
    W, H = bev_size
    u, v = bev_uv[:,0], bev_uv[:,1]
    mask = (u>=0)&(u<W)&(v>=0)&(v<H)
    return bev_uv[mask]
