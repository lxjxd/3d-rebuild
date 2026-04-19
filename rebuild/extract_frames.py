import os

import cv2


def extract_frames(video_path, output_folder="frames"):
    """
    从视频中提取每一帧并保存为PNG图片
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return

    frame_count = 0
    saved_count = 0

    print("开始提取帧...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 保存每一帧（你可以在这里加条件，比如每5帧保存一次）
        output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
        cv2.imwrite(output_path, frame)
        saved_count += 1

        if frame_count % 300 == 0:
            print(f"已处理 {frame_count} 帧，已保存 {saved_count} 张图片")

    cap.release()
    print(f"完成！共读取 {frame_count} 帧，保存 {saved_count} 张PNG图片")
    print(f"图片保存在：{os.path.abspath(output_folder)}")


# 使用示例
if __name__ == "__main__":
    video_file = "input.mp4"  # ← 修改为你的视频路径
    output_dir = "frames"  # 输出文件夹名称

    extract_frames(video_file, output_dir)
