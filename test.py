import cv2
import numpy as np

# 定义棋盘格的大小
chessboard_size = (9, 6)

# 准备棋盘格角点的世界坐标
object_points = []  # 存储世界坐标系中的特征点坐标
image_points_left = []  # 存储左相机图像中的特征点坐标
image_points_right = []  # 存储右相机图像中的特征点坐标

# 生成棋盘格角点的世界坐标
for i in range(chessboard_size[1]):
    for j in range(chessboard_size[0]):
        object_points.append([j, i, 0])

# 生成左右相机的图像坐标
for i in range(chessboard_size[1]):
    for j in range(chessboard_size[0]):
        image_points_left.append([j, i])
        image_points_right.append([j + 2, i])

# 将特征点的坐标转换为数组
object_points = np.array([object_points], dtype=np.float32)
image_points_left = np.array([image_points_left], dtype=np.float32)
image_points_right = np.array([image_points_right], dtype=np.float32)

# 定义相机矩阵和畸变系数的初始值
camera_matrix_left = np.eye(3)
camera_matrix_right = np.eye(3)
dist_coeffs_left = np.zeros(5)
dist_coeffs_right = np.zeros(5)

# 定义图像大小
image_size = (640, 480)

# 进行立体校准
retval, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
    object_points, image_points_left, image_points_right, camera_matrix_left, dist_coeffs_left, camera_matrix_right,
    dist_coeffs_right, image_size, criteria=None, flags=cv2.CALIB_FIX_INTRINSIC
)

# 打印校准结果
print("左相机的相机矩阵：")
print(camera_matrix_left)
print("\n左相机的畸变系数：")
print(dist_coeffs_left)
print("\n右相机的相机矩阵：")
print(camera_matrix_right)
print("\n右相机的畸变系数：")
print(dist_coeffs_right)
print("\n旋转矩阵：")
print(R)
print("\n平移向量：")
print(T)
print("\n本征矩阵：")
print(E)
print("\n基础矩阵：")
print(F)