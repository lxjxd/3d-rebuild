import open3d as o3d
import numpy as np

from pathlib import Path
def check_ply():
    pcd = o3d.io.read_point_cloud("pointclouds/frame_000146.ply")
    points = np.asarray(pcd.points)
    
    print("最小值:", points.min(axis=0))
    print("最大值:", points.max(axis=0))
    print("范围:", points.max(axis=0) - points.min(axis=0))

def rename():


    folder = Path(r"calib_imgs")
    files = sorted([f for f in folder.iterdir() if f.is_file()])

    width = len(str(len(files)))  # 自动决定补零位数

    for i, file in enumerate(files, start=1):
        new_name = f"{i:0{width}d}{file.suffix}"
        new_path = folder / new_name
        file.rename(new_path)
        print(f"{file.name} -> {new_name}")