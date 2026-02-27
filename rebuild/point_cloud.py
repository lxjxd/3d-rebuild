import os
import re
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# ------------------ 配置部分 ------------------
RGB_DIR = "rgb"
DEPTH_DIR = "depths"
OUTPUT_DIR = "output_pointclouds"

# 深度图缩放因子（非常重要！）
#   Kinect/Azure → 通常 depth_scale=1000.0（mm → m）
#   RealSense     → 通常 depth_scale=1000.0
#   TUM RGB-D     → 通常 depth_scale=5000.0 或 1000.0，看数据集说明
#   自采集数据    → 请确认深度值的单位
DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 5.0  # 截断深度（米），防止远处的噪声点

# 是否保存每一帧的点云
SAVE_EACH_PCD = True

# 是否尝试做帧间 RGB-D odometry
DO_ODOMETRY = True
# -------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_pfm(file_path):
    with open(file_path, "rb") as f:
        # Read header
        header = f.readline().decode("utf-8").rstrip()
        if header == "PF":
            channels = 3
        elif header == "Pf":
            channels = 1
        else:
            raise ValueError("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s+(\d+)\s*$", f.readline().decode("utf-8"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise ValueError("Malformed PFM header.")

        scale = float(f.readline().decode("utf-8").rstrip())
        if scale < 0:
            endian = "<"  # little endian
            scale = -scale
        else:
            endian = ">"  # big endian

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if channels == 3 else (height, width)

        # PFM stores data bottom-up
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale


def load_depth(depth_path: str) -> np.ndarray:
    """支持 .png (16bit) 和 .pfm 格式的深度图"""
    ext = Path(depth_path).suffix.lower()

    if ext == ".pfm":
        # 读取 pfm 格式（常见于 Blender、TUM部分序列、一些合成数据集）
        depth = read_pfm(depth_path)[0]  # 返回 (depth_array, scale)
        # pfm 通常已经是浮点米单位，但有时需要检查正负
        if np.any(depth < 0):
            depth = np.abs(depth)  # 有些数据集用负值表示有效深度
        return depth.astype(np.float32)
    elif ext in [".png", ".jpg"]:
        # 假设 16bit PNG
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Cannot read depth image: {depth_path}")
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0  # 假设 mm → m
        elif depth.dtype == np.float32:
            return depth
        else:
            raise ValueError(f"Unexpected depth dtype {depth.dtype} in {depth_path}")

    else:
        raise ValueError(f"Unsupported depth format: {ext}")


def pcd_pipeline():

    # 相机内参（请根据你的实际相机修改！！！）
    width = 1920
    height = 1440
    fx = 1400
    fy = 1400
    cx = width / 2
    cy = height / 2

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
    )

    # 获取文件列表（假设文件名对应）
    rgb_paths = sorted(Path(RGB_DIR).glob("*.png"))  # png / jpg / jpeg
    depth_paths = sorted(Path(DEPTH_DIR).glob("*.pfm"))  # png / pfm

    if len(rgb_paths) != len(depth_paths):
        print(
            f"警告：RGB 和 depth 文件数量不一致！ rgb:{len(rgb_paths)} depth:{len(depth_paths)}"
        )
        min_len = min(len(rgb_paths), len(depth_paths))
        rgb_paths = rgb_paths[:min_len]
        depth_paths = depth_paths[:min_len]

    print(f"共找到 {len(rgb_paths)} 对图像")

    # 初始化
    poses = [np.eye(4)]
    prev_rgbd = None

    for i, (rgb_p, depth_p) in enumerate(zip(rgb_paths, depth_paths)):
        print(f"处理 {i:04d} : {rgb_p.name} ↔ {depth_p.name}")

        # 读 RGB
        color_raw = o3d.io.read_image(str(rgb_p))

        # 读深度 → 转为 float32 米单位
        depth_np = load_depth(str(depth_p))
        depth_raw = o3d.geometry.Image((depth_np / DEPTH_SCALE).astype(np.float32))

        # 创建 RGBD
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw,
            depth_raw,
            depth_scale=1.0,  # 我们已经在前面除过了
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False,
        )
        """
        source_gpu = o3dc.pybind.geometry.RGBDImage.from_legacy(rgbd).to(
            o3dc.pybind.core.Device("CUDA:0")
        )
        target_gpu = o3dc.pybind.geometry.RGBDImage.from_legacy(prev_rgbd).to(
            o3dc.pybind.core.Device("CUDA:0")
        )

        intrinsic_gpu = o3dc.pybind.camera.PinholeCameraIntrinsic.from_legacy(intrinsic)
        """
        # 生成当前帧点云（相机坐标系）
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        if SAVE_EACH_PCD:
            save_path = os.path.join(OUTPUT_DIR, f"frame_{i:06d}.ply")
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"  保存单帧点云 → {save_path}")

        # ------------------- 位姿估计（简单前后帧 ICP / RGBD odometry） -------------------
        if DO_ODOMETRY and i > 0 and prev_rgbd is not None:
            [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd,
                prev_rgbd,
                intrinsic,
                odo_init=np.eye(4),
                option=o3d.pipelines.odometry.OdometryOption(),
                jacobian=o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            )

            if success:
                current_pose = poses[-1] @ trans
                poses.append(current_pose)
                print(f"  odometry 成功   info={info}")
            else:
                print("  odometry 失败，保持上一帧位姿")
                poses.append(poses[-1])

        prev_rgbd = rgbd

    print("\n处理完成。")
    print("正在合并所有点云到全局坐标系（可视化用）...")
    merged = o3d.geometry.PointCloud()

    for i, pose in enumerate(poses):
        pcd = o3d.io.read_point_cloud(os.path.join(OUTPUT_DIR, f"frame_{i:06d}.ply"))
        pcd.transform(pose)
        merged += pcd

    # 降采样（可选，防止内存爆炸）
    merged = merged.voxel_down_sample(voxel_size=0.02)

    o3d.visualization.draw_geometries(
        [merged],
        window_name="Merged Point Cloud (with estimated poses)",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    pcd_pipeline()
