import cv2
import numpy as np
import glob

# ===== 参数自己改 =====
pattern_size = (9, 6)      # 棋盘格内角点数: (列, 行)
square_size = 0.038         # 每个小格边长，单位米；20mm 就写 0.02
image_pattern = "calib_imgs/*.jpg"

# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# 世界坐标中的棋盘格点
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []   # 3D 点
imgpoints = []   # 2D 点

images = glob.glob(image_pattern)
if not images:
    raise RuntimeError("没有找到标定图片")

img_size = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    if not ret:
        print(f"未检测到角点: {fname}")
        continue

    corners2 = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    objpoints.append(objp)
    imgpoints.append(corners2)
    print(f"检测成功: {fname}")

if len(objpoints) < 10:
    raise RuntimeError(f"有效图片太少：{len(objpoints)} 张，建议至少 10 张")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

print("\n=== 标定结果 ===")
print("RMS reprojection error:", ret)
print("K =\n", K)
print("dist =\n", dist.ravel())

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

print(f"\nfx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
print(f"image size = {img_size}")