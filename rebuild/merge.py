import open3d as o3d

def merge_and_visualize(poses=None):
    """合并所有点云并可视化

    Args:
        poses: 可选的位姿列表。如果为 None，则从文件加载。
    """
    # 加载 poses
    if poses is None:
        poses_path = os.path.join(OUTPUT_DIR, "poses.npy")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"poses 文件不存在: {poses_path}")
        poses = np.load(poses_path)
        print(f"从文件加载 poses: {poses_path}")

    print(f"共加载 {len(poses)} 个位姿")

    print("正在合并所有点云到全局坐标系...")
    merged = o3d.geometry.PointCloud()

    for i, pose in enumerate(poses):
        pcd_path = os.path.join(OUTPUT_DIR, f"frame_{i:06d}.ply")
        if os.path.exists(pcd_path):
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd.transform(pose)
            merged += pcd
        else:
            print(f"警告：找不到点云文件 {pcd_path}，跳过")

    # 降采样（可选，防止内存爆炸）
    merged = merged.voxel_down_sample(voxel_size=0.002)

    o3d.visualization.draw_plotly(
        [merged],
        window_name="Merged Point Cloud (with estimated poses)",
    )

