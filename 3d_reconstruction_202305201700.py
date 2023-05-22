import os.path

import numpy as np
import cv2
import json
from target_localization import get_target_points
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_relative_position(holographic_info_dict, left_camera, right_camera):
    # [4] Selected the left stereo camera as the reference camera
    left_camera_info = holographic_info_dict[left_camera]
    right_camera_info = holographic_info_dict[right_camera]

    # [4.1] Get the corresponding targets points
    left_targets_sets = set(left_camera_info["target_label_set"])
    right_targets_sets = set(right_camera_info["target_label_set"])

    corresponding_targets_set = left_targets_sets.intersection(right_targets_sets)
    print("Corresponding targets set: ", corresponding_targets_set)
    corresponding_targets_list = list(corresponding_targets_set)

    # [4.2] get the coordinates of the same target point in left and right images
    image_points_left = []
    image_points_right = []
    for key in corresponding_targets_list:
        image_points_left.append(np.array(left_camera_info.get(key)))
        image_points_right.append(np.array(right_camera_info.get(key)))

    image_points_left = np.vstack(image_points_left)
    image_points_right = np.vstack(image_points_right)

    # [4.3] Get the camera intrinsic parameter matrix and distortion coefficients
    camera_matrix_left = np.array(left_camera_info.get("camera_matrix"))
    distortion_coeffs_left = np.array(left_camera_info.get("distortion_coeffs"))
    camera_matrix_right = np.array(right_camera_info.get("camera_matrix"))
    distortion_coeffs_right = np.array(right_camera_info.get("distortion_coeffs"))

    F, _ = cv2.findFundamentalMat(image_points_left, image_points_right, cv2.FM_RANSAC)

    # [6] Get the Essential matrix from the Fundamental matrix
    E = camera_matrix_right.T @ F @ camera_matrix_left

    # [7] Calculate the Rotation matrix and translation vector
    retval, R, T, _ = cv2.recoverPose(E, image_points_left, image_points_right, camera_matrix_left, distortion_coeffs_left)

    # [8] Calculate the world coordinates of the target points

    # Projection matrix for left camera
    P1 = np.dot(camera_matrix_left, np.hstack((np.eye(3), np.zeros((3, 1)))))

    # Projection matrix for right camera
    P2 = np.dot(camera_matrix_right, np.hstack((R, T)))

    # 3D positions of the points
    point_4d_hom = cv2.triangulatePoints(P1, P2, image_points_left.T, image_points_right.T)
    point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    target_point_world = point_3d[:3, :].T

    return R, T, corresponding_targets_list, target_point_world



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
    # Construct Hyperparameter dict for future use
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
    camera_file = "camera_info.json"
    if os.path.exists(camera_file):
        with open(camera_file, "r") as file:
            holographic_info_dict = json.load(file)
    else:
        # [1] Get all the camera and corresponding images targets information
        holographic_info_dict = {}
        for key in hyperparameter_dict.keys():
            # [2] Get the target information of each camera image
            image, target_list = get_target_points(**hyperparameter_dict[key])

            # store all required information in a dict
            info_dict = {}
            info_dict["img"] = image.tolist()

            target_label_set = []
            for target in target_list:
                tareget_label = target["key"]
                target_label_set.append(tareget_label)
                info_dict[tareget_label] = target["weighted_centroids"].tolist()

            info_dict["target_label_set"] = target_label_set

            # [3] Get the camera intrinsic parameter matrix and distortion coefficients
            camera_matrix, distortion_coeffs = get_camera_parameters(hyperparameter_dict["camera_11_left"]["camera_params"])
            info_dict["camera_matrix"] = camera_matrix.tolist()
            info_dict["distortion_coeffs"] = distortion_coeffs.tolist()

            # Add all the camera info to the dict
            holographic_info_dict[key] = info_dict

        # save all the camera information instead of computing each time
        with open(camera_file, 'w') as file:
            json.dump(holographic_info_dict, file)

    # [4] Selected the left stereo camera as the reference camera
    left_camera_info = holographic_info_dict["camera_11_left"]
    right_camera_info = holographic_info_dict["camera_11_right"]

    # [4.1] Get the corresponding targets points
    left_targets_sets = set(left_camera_info["target_label_set"])
    right_targets_sets = set(right_camera_info["target_label_set"])

    corresponding_targets_set = left_targets_sets.intersection(right_targets_sets)
    # print("Corresponding targets set: ", corresponding_targets_set)
    corresponding_targets_list = list(corresponding_targets_set)
    corresponding_targets_list.sort()

    print("Corresponding targets list: ", corresponding_targets_list)
    # [4.2] get the coordinates of the same target point in left and right images
    image_points_left = []
    image_points_right = []
    for key in corresponding_targets_list:
        image_points_left.append(np.array(left_camera_info.get(key)))
        image_points_right.append(np.array(right_camera_info.get(key)))

    image_points_left = np.vstack(image_points_left)
    image_points_right = np.vstack(image_points_right)

    # [4.3] Get the camera intrinsic parameter matrix and distortion coefficients
    camera_matrix_left = np.array(left_camera_info.get("camera_matrix"))
    distortion_coeffs_left = np.array(left_camera_info.get("distortion_coeffs"))
    camera_matrix_right = np.array(right_camera_info.get("camera_matrix"))
    distortion_coeffs_right = np.array(right_camera_info.get("distortion_coeffs"))

    F, _ = cv2.findFundamentalMat(image_points_left, image_points_right, cv2.FM_RANSAC)

    # [6] Get the Essential matrix from the Fundamental matrix
    E = camera_matrix_right.T @ F @ camera_matrix_left

    # [7] Calculate the Rotation matrix and translation vector
    retval, R, T, _ = cv2.recoverPose(E, image_points_left, image_points_right, camera_matrix_left, distortion_coeffs_left)

    # [8] Calculate the world coordinates of the target points

    # Projection matrix for left camera
    P1 = np.dot(camera_matrix_left, np.hstack((np.eye(3), np.zeros((3, 1)))))

    # Projection matrix for right camera
    P2 = np.dot(camera_matrix_right, np.hstack((R, T)))

    # 3D positions of the points
    point_4d_hom = cv2.triangulatePoints(P1, P2, image_points_left.T, image_points_right.T)
    point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    target_point_world = point_3d[:3, :].T


    # World coordinates of the right camera
    right_camera_world = -np.dot(R.T, T)

    # Print the world coordinates
    print("World coordinates of the target points:")
    print(target_point_world)
    print("\nWorld coordinates of the right camera:")
    print(right_camera_world)


    # [9] Store all the target_point_world in a world_points_dict
    word_points_dict = {}
    for i, key in enumerate(corresponding_targets_list):
        print("Key: ", key)
        word_points_dict[key] = target_point_world[(i*6):(i+1)*6, :]

    # [10] calculate the R & T of all cameras
    targets_world_label_set = set(corresponding_targets_list)
    for camera_key in holographic_info_dict.keys():
        camera_info = holographic_info_dict[camera_key]

        # store corresponding  points in world coordinates and image cooridiantes
        curr_target_point_world = []
        curr_target_point_image = []

        curr_target_label_set = set(camera_info["target_label_set"])
        curr_correspond_labels = curr_target_label_set.intersection(targets_world_label_set)
        print("Current correspond targets: ", curr_correspond_labels)
        if len(curr_correspond_labels) == 0:
            print(camera_key + " does not get corresponding world position")
            continue
        for label in curr_correspond_labels:
            curr_target_point_world.append(word_points_dict[label])
            curr_target_point_image.append(np.array(camera_info[label]))

        # construct the object points and image points for cv2.solvePnP
        curr_target_point_world = np.vstack(curr_target_point_world)
        curr_target_point_image = np.vstack(curr_target_point_image)

        curr_camera_matrix = np.array(camera_info.get("camera_matrix"))
        curr_distortion_coeffs = np.array(camera_info.get("distortion_coeffs"))

        # using solvePnP to get the rotation matrix and translation vector
        _, rvec, tvec = cv2.solvePnP(curr_target_point_world, curr_target_point_image, curr_camera_matrix, curr_distortion_coeffs)
        R_camera, _ = cv2.Rodrigues(rvec)

        # store the R and T into the camera info
        camera_info["R"] = R_camera
        camera_info["T"] = tvec

        # from image coordinate to the world 3D position



    print("test")



    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the target points
    # ax.scatter(target_point_world[:, 0], target_point_world[:, 1], target_point_world[:, 2], c='blue', label='Target Points')
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
    #
    # # print("Hello")







