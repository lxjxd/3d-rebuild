import os
import cv2
from tqdm import tqdm

def resize_to_vertical_1080p(frame, target_width=1080, target_height=1920):
    """
    将任意分辨率帧转换为竖屏 1080x1920
    方式：等比缩放后居中裁剪，不拉伸
    """
    h, w = frame.shape[:2]

    # 原图和目标图宽高比
    src_ratio = w / h
    target_ratio = target_width / target_height

    if src_ratio > target_ratio:
        # 原图更宽：按高度缩放，再裁宽度
        new_h = target_height
        new_w = round(w * (target_height / h))
    else:
        # 原图更高或更窄：按宽度缩放，再裁高度
        new_w = target_width
        new_h = round(h * (target_width / w))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 居中裁剪
    x_start = 0 if (new_w - target_width) // 2 <0 else (new_w - target_width) // 2 
    y_start = 0 if (new_h - target_height) // 2 <0 else (new_h - target_height) // 2

    cropped = resized[y_start:y_start + target_height, x_start:x_start + target_width]

    return cropped


def extract_frames(video_path, output_folder="frames"):
    """
    检查视频分辨率，将视频帧统一转为竖屏1080x1920后保存为PNG
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return
    
    # 重新回到第一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return

    # 读取视频信息
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("视频信息：")
    print(f"分辨率：{orig_width} x {orig_height}")
    print(f"FPS：{fps:.2f}")
    print(f"总帧数：{total_frames}")
    print("目标分辨率：1080 x 1920（竖屏）")
    print("开始提取并转换帧...")

    pbar = tqdm(total = total_frames)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # 转为竖屏1080p
        output_frame = resize_to_vertical_1080p(frame, 1080, 1920)
        # 保存
        output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
        cv2.imwrite(output_path, output_frame)
        saved_count += 1
        pbar.update(1)
        if frame_count % 300 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧，已保存 {saved_count} 张图片")
    pbar.close()
    cap.release()

    print(f"完成！共读取 {frame_count} 帧，保存 {saved_count} 张PNG图片")
    print(f"图片保存在：{os.path.abspath(output_folder)}")


if __name__ == "__main__":
    video_file = "input.mov"   # 修改为你的视频路径
    output_dir = "frames"      # 输出文件夹名称

    extract_frames(video_file, output_dir)