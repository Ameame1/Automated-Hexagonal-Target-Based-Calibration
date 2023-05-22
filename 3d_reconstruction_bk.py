import numpy as np
import cv2
import json
from target_localization import get_target_points
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_camera_parameters(json_path):

    with open(json_path, 'r') as f:
        camera_params = json.load(f)

    # extract camera parameters
    f = camera_params["f"]["val"]
    cx = camera_params["cx"]["val"]
    cy = camera_params["cy"]["val"]
    k1 = camera_params["k1"]["val"]
    k2 = camera_params["k2"]["val"]
    k3 = camera_params["k3"]["val"]
    p1 = camera_params["p1"]["val"]
    p2 = camera_params["p2"]["val"]

    # Construct the camera intrinsic parameter matrixs
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    # Construct distortion coefficient vector
    dist_coeffs = np.array([k1, k2, k3, p1, p2])

    return K, dist_coeffs

if __name__ == "__main__":

    # [1] Loaded all the images and store in a dict
    hyperparameter_dict ={
        "camera_11_left": {
            "camera_params": "camera parameters/zedLeft720p.json",
            "img_path": "camera 11/2022_12_15_15_51_19_927_rgb_left.png",
            "min_thresh": 100,
            "diff_thresh": 100,
            "min_area": 20,
            "max_area": 450,
            "axis_ratio_threshold": 0.25,
            "ellips_threshold": 0.1
            },
        "camera_11_right": {
            "camera_params": "camera parameters/zedRight720p.json",
            "img_path": "camera 11/2022_12_15_15_51_19_927_rgb_right.png",
            "min_thresh": 100,
            "diff_thresh": 100,
            "min_area": 20,
            "max_area": 450,
            "axis_ratio_threshold": 0.25,
            "ellips_threshold": 0.1
            },
        "camera_71": {
            "camera_params": "camera parameters/realsense71RGB.json",
            "img_path": "camera 71/2022_12_15_15_51_19_944_rgb.png",
            "min_thresh": 100,
            "diff_thresh": 20,
            "min_area": 10,
            "max_area": 450,
            "axis_ratio_threshold": 0.25,
            "ellips_threshold": 0.2
            },
        "camera_72": {
            "img_path": "camera 72/2022_12_15_15_51_19_956_rgb.png",
            "camera_params": "camera parameters/realsense72RGB.json",
            "min_thresh": 100,
            "diff_thresh": 15,
            "min_area": 7,
            "max_area": 450,
            "axis_ratio_threshold": 0.25,
            "ellips_threshold": 0.2
        },
        "camera_73": {
            "camera_params": "camera parameters/realsense73RGB.json",
            "img_path": "camera 73/2022_12_15_15_51_19_934_rgb.png",
            "min_thresh": 100,
            "diff_thresh": 50,
            "min_area": 10,
            "max_area": 450,
            "axis_ratio_threshold": 0.25,
            "ellips_threshold": 0.1
            },
        "camera_74": {
            "camera_params": "camera parameters/realsense74RGB.json",
            "img_path": "camera 74/2022_12_15_15_51_19_951_rgb.png",
            "min_thresh": 50,
            "diff_thresh": 50,
            "min_area": 10,
            "max_area": 450,
            "axis_ratio_threshold": 0.25,
            "ellips_threshold": 0.1
            },

    }
    # image_path_dict = {
    #     "camera_11_left": "camera 11/2022_12_15_15_51_19_927_rgb_left.png",
    #     "camera_11_right": "camera 11/2022_12_15_15_51_19_927_rgb_right.png",
    #     "camera_71": "camera 71/2022_12_15_15_51_19_944_rgb.png",
    #     "camera_72": "camera 72/2022_12_15_15_51_19_956_rgb.png",
    #     "camera_73": "camera 73/2022_12_15_15_51_19_934_rgb.png",
    #     "camera_74": "camera 74/2022_12_15_15_51_19_951_rgb.png",
    # }
    #
    # camera_path_dict = {
    #     "camera_11_left": "camera parameters/zedLeft720p.json",
    #     "camera_11_right": "camera parameters/zedRight720p.json",
    #     "camera_71": "camera parameters/realsense71RGB.json",
    #     "camera_72": "camera parameters/realsense72RGB.json",
    #     "camera_73": "camera parameters/realsense73RGB.json",
    #     "camera_74": "camera parameters/realsense74RGB.json",
    # }

    # image_dict = {}
    # for key in image_path_dict.keys():
    #     img_path = image_path_dict[key]
    #     img = cv2.imread(img_path)
    #     image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     image_dict[key] = image_rgb


    # [2] Get the target point in each image taken by different cameras
    #  here we could use thresold search to guarantee the number of targets we acquired from each image

    # [3] We first use the stereo camera: camera 11 to get the relative world coordinate of targets
    # [3.1] get the target list of two images
    left_img, left_target_list = get_target_points(image_path_dict["camera_11_left"])
    right_img, right_target_list = get_target_points(image_path_dict["camera_11_right"])

    # [3.2] construct target label based dict to map corresponding points
    left_target_dict = {}
    for targets_info in left_target_list:
        left_target_dict[targets_info["key"]] = targets_info["weighted_centroids"]

    right_target_dict = {}
    for targets_info in right_target_list:
        right_target_dict[targets_info["key"]] = targets_info["weighted_centroids"]

    # [3.3] get the coordinates of the same target point in left and right images
    image_points_left = []
    image_points_right = []
    for key in left_target_dict.keys():
        if right_target_dict.get(key, None) is not None:
            image_points_left.append(left_target_dict.get(key))
            image_points_right.append(right_target_dict.get(key))

    image_points_left = np.vstack(image_points_left)
    image_points_right = np.vstack(image_points_right)
    # [4] Get the camera intrinsic parameter matrix and distortion coefficients
    camera_matrix_left, distortion_coeffs_left = get_camera_parameters(camera_path_dict["camera_11_left"])
    camera_matrix_right, distortion_coeffs_right = get_camera_parameters(camera_path_dict["camera_11_right"])

    # TODO: Whether should we use these undistorted points, need to check
    # imagePoints_left_norm = cv2.undistortPoints(image_points_left, camera_matrix_left, distortion_coeffs_left, None,
    #                                             camera_matrix_left)
    # imagePoints_right_norm = cv2.undistortPoints(image_points_right, camera_matrix_right, distortion_coeffs_right, None,
    #                                              camera_matrix_right)
    # F, _ = cv2.findFundamentalMat(imagePoints_left_norm, imagePoints_left_norm, cv2.FM_RANSAC, 0.1, 0.99)

    # [5] Get the fundamental matrix from the left and right corresponding points
    F, _ = cv2.findFundamentalMat(image_points_left, image_points_right, cv2.FM_RANSAC, 0.1, 0.99)

    # [6] Get the Essential matrix from the Fundamental matrix
    E = camera_matrix_right.T @ F @ camera_matrix_left

    # [7] Calculate the Rotation matrix and translation vector
    retval, R, T, _ = cv2.recoverPose(E, image_points_left, image_points_right, camera_matrix_left, distortion_coeffs_left)

    # [8] Calculate the world coordinates of the target points

    # normalize the image points
    # image_point_left_norm = np.linalg.inv(camera_matrix_left) @ np.array([image_points_left[0], image_points_left[1], 1])
    # image_point_right_norm = np.linalg.inv(camera_matrix_right) @ np.array([image_points_right[0], image_points_right[1], 1])
    image_point_left_norm = cv2.undistortPoints(image_points_left, camera_matrix_left, distortion_coeffs_left, None, camera_matrix_left)
    image_point_right_norm = cv2.undistortPoints(image_points_right, camera_matrix_right, distortion_coeffs_right, None, camera_matrix_right)
    # triangular measurement
    rotation_translation_matrix = np.hstack((R, T))
    points_4d_homogeneous = cv2.triangulatePoints(np.eye(3, 4), rotation_translation_matrix,
                                                  image_point_left_norm, image_point_right_norm)

    # tranfer the coordinates to homogenous coordinates
    points_3d_homogeneous = points_4d_homogeneous / points_4d_homogeneous[3]

    # World coordinates of the target points
    target_point_world = points_3d_homogeneous[:3]

    # World coordinates of the right camera
    right_camera_world = -np.dot(R.T, T)

    # Print the world coordinates
    print("World coordinates of the target points:")
    print(target_point_world)
    print("\nWorld coordinates of the right camera:")
    print(right_camera_world)




    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the target points
    # ax.scatter(target_point_world[0], target_point_world[1], target_point_world[2], c='blue', label='Target Points')
    # # ax.text(target_point_world[0], target_point_world[1], target_point_world[2], 'Target Points',
    # #         color='blue', ha='center', va='center')
    #
    # # Plot the right camera position
    # ax.scatter(right_camera_world[0], right_camera_world[1], right_camera_world[2], c='red', label='Right Camera')
    # # ax.text(right_camera_world[0], right_camera_world[1], right_camera_world[2], 'Right Camera',
    # #         color='red', ha='center', va='center')
    #
    # ax.scatter(0.0, 0.0, 0.0, c='yellow', label='Left Camera')
    # # ax.text(0, 0, 0, 'Left Camera',
    # #         color='red', ha='center', va='center')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # # Add a legend
    # ax.legend()
    #
    # # Show the plot
    # plt.show()







