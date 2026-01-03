import matplotlib.pyplot as plt
import numpy as np

def plot_image(uv_dict, width, height):
    plt.figure(figsize=(7,4))
    plt.title("Pinhole Camera Projection")
    plt.xlim(0, width)
    plt.ylim(height, 0)

    for k, uv in uv_dict.items():
        plt.scatter(uv[:,0], uv[:,1], s=4, label=k)

    plt.legend()
    plt.xlabel("u [px]")
    plt.ylabel("v [px]")

def plot_bev(bev_dict, bev_size, xlim, ylim, ego_yaw=0.0):
    W, H = bev_size
    plt.figure(figsize=(6,6))
    plt.title("BEV (Top-Down)")

    plt.xlim(0, W)
    plt.ylim(H, 0)

    # --- tick을 meter 기준으로 표시 ---
    xticks = np.linspace(0, W, 5)
    yticks = np.linspace(0, H, 5)

    xticklabels = np.linspace(xlim[0], xlim[1], 5).astype(int)
    yticklabels = np.linspace(ylim[1], ylim[0], 5).astype(int)

    plt.xticks(xticks, xticklabels)
    plt.yticks(yticks, yticklabels)

    plt.xlabel("x [m] (forward)")
    plt.ylabel("y [m] (left)")

    for k, uv in bev_dict.items():
        plt.scatter(uv[:,0], uv[:,1], s=4, label=k)

    # ego 원점 표시 (x=0, y=0)
    ego_u = (0 - xlim[0]) / (xlim[1] - xlim[0]) * (W - 1)
    ego_v = (1 - (0 - ylim[0]) / (ylim[1] - ylim[0])) * (H - 1)
    plt.scatter([ego_u], [ego_v], c="red", s=60, marker="+", label="ego (0,0)")

    # --- ego heading 표시 (yaw) ---
    heading_length_m = 5.0  # 5m 길이 화살표

    # ego heading endpoint in meter
    hx = heading_length_m * np.cos(ego_yaw)
    hy = heading_length_m * np.sin(ego_yaw)

    # meter -> BEV pixel
    head_u = (hx - xlim[0]) / (xlim[1] - xlim[0]) * (W - 1)
    head_v = (1 - (hy - ylim[0]) / (ylim[1] - ylim[0])) * (H - 1)

    plt.plot(
        [ego_u, head_u],
        [ego_v, head_v],
        color="red",
        linewidth=2,
        label="ego heading"
    )


    plt.legend()
    plt.grid(True)
