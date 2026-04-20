import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_pfm(file_path):
    """
    读取 PFM 文件
    返回:
        data: numpy.ndarray, float32
        scale: scale factor
    """
    file_path = str(file_path)
    with open(file_path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError("不是有效的 PFM 文件")

        dim_line = f.readline().decode("utf-8")
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("utf-8")

        match = re.match(r"^(\d+)\s(\d+)\s$", dim_line)
        if not match:
            raise ValueError("PFM 尺寸信息错误")

        width, height = map(int, match.groups())

        scale = float(f.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"   # 小端/大端
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")

        expected_channels = 3 if color else 1
        expected_size = width * height * expected_channels
        if data.size != expected_size:
            raise ValueError(
                f"数据大小不匹配，期望 {expected_size}，实际 {data.size}"
            )

        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)

        # PFM 默认从左下角开始存储，需要上下翻转
        data = np.flipud(data)

        return data.astype(np.float32), scale


def visualize_depth(depth, title="PFM Depth", cmap="plasma", save_path=None):
    """
    可视化深度图
    """
    depth = np.array(depth, dtype=np.float32)

    # 处理非法值
    valid_mask = np.isfinite(depth)
    if not np.any(valid_mask):
        raise ValueError("深度图中没有有效值")

    valid_values = depth[valid_mask]

    # 用分位数让显示更稳定，避免极端值影响
    vmin = np.percentile(valid_values, 2)
    vmax = np.percentile(valid_values, 98)

    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Depth")
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"可视化结果已保存到: {save_path}")

    plt.show()


def visualize_pfm():
    pfm_path = "./depths/frame_000001-dpt_large_384.pfm"   # 改成你的文件路径
    depth, scale = read_pfm(pfm_path)

    print("读取成功")
    print("shape:", depth.shape)
    print("dtype:", depth.dtype)
    print("scale:", scale)
    print("min:", np.nanmin(depth))
    print("max:", np.nanmax(depth))

    visualize_depth(
        depth,
        title=Path(pfm_path).name,
        cmap="plasma",
        save_path="depth_visualization.png"
    )


def visualize_ply():
    import open3d as o3d

    # 读取 ply 点云
    pcd = o3d.io.read_point_cloud("pointclouds/frame_000146.ply")
    pcd.transform([
    [1, 0, 0, 0],
    [0,-1, 0, 0],
    [0, 0,-1, 0],
    [0, 0, 0, 1]
    ])
    # 打印基本信息
    print(pcd)
    print("点数：", len(pcd.points))
    # 可视化
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    o3d.visualization.draw_geometries([pcd, axis], window_name="PLY Point Cloud")

from PIL import Image, ImageDraw

def create_checkerboard(
    cols=10,             # 棋盘格列数（格子数，不是内角点）
    rows=7,              # 棋盘格行数（格子数，不是内角点）
    square_size_px=100,  # 每个格子的像素大小
    save_path="checkerboard.png"
):
    width = cols * square_size_px
    height = rows * square_size_px

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0 = c * square_size_px
                y0 = r * square_size_px
                x1 = x0 + square_size_px
                y1 = y0 + square_size_px
                draw.rectangle([x0, y0, x1, y1], fill="black")

    img.save(save_path)
    print(f"棋盘格已保存到: {save_path}")


if __name__ == "__main__":
    create_checkerboard(
        cols=10,
        rows=7,
        square_size_px=120,
        save_path="checkerboard_10x7.png"
    )