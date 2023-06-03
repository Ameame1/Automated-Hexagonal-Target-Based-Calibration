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


    # Projection matrix for left camera
    R1 = left_camera_info["R"]
    T1 = left_camera_info["T"]
    P1 = np.dot(camera_matrix_left, np.hstack((R1, T1)))
    # P1 = np.dot(camera_matrix_left, np.hstack((np.eye(3), np.zeros((3, 1)))))


    # Projection matrix for right camera
    R2 = right_camera_info["R"]
    T2 = right_camera_info["T"]
    P2 = np.dot(camera_matrix_right, np.hstack((R2, T2)))

    # 3D positions of the points
    point_4d_hom = cv2.triangulatePoints(P1, P2, image_points_left.T, image_points_right.T)
    point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    target_point_world = point_3d[:3, :].T

    return corresponding_targets_list, target_point_world



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


def compute_transformation_matrix(points_A, points_B):
    # 构建增广矩阵
    augmented_matrix = np.ones((points_A.shape[0], 4))
    augmented_matrix[:, :3] = points_A

    # 使用最小二乘法估计转换矩阵
    transformation_matrix, _ = np.linalg.lstsq(augmented_matrix, points_B, rcond=None)[:2]

    return transformation_matrix

def cooridinate_transfer_1(refer_R, tvec, target_local_coordinate):
    refer_obj_coord = refer_R @ target_local_coordinate.T + tvec
    return refer_obj_coord

def cooridinate_transfer_2(refer_R, tvec, target_local_coordinate):
    refer_obj_coord = refer_R.T @ (target_local_coordinate.T - tvec)
    return refer_obj_coord


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


    # [4] select camera 11 as the reference camera

    target_local_coordinate = np.array([
        [0, 90.01599884033203, 0],
        [77.95613861083984, 45.007999420166016, 0],
        [77.95613861083984, -45.007999420166016, 0],
        [0, -90.01599884033203, 0],
        [-77.95613861083984, -45.007999420166016, 0],
        [-77.95613861083984, 45.007999420166016, 0]
    ])
    homogeneous_object_points = np.hstack((target_local_coordinate, np.ones((target_local_coordinate.shape[0], 1))))

    refer_camera_info = holographic_info_dict["camera_11_left"]
    refer_target_label_set = list(refer_camera_info["target_label_set"])
    refer_target_label_set.sort()
    refer_camera_matrix = np.array(refer_camera_info["camera_matrix"])
    refer_distortion_coeffs = np.array(refer_camera_info["distortion_coeffs"])
    refer_camera_info["R"] = np.eye(3)
    refer_camera_info["T"] = np.zeros((3, 1))

    word_points_dict = {}
    # align all the targets to reference coordinates
    for target_label in refer_target_label_set:
        target_imag_points = np.array(refer_camera_info[target_label])

        _, rvec, tvec = cv2.solvePnP(target_local_coordinate, target_imag_points, refer_camera_matrix,
                                           refer_distortion_coeffs)
        refer_R, _ = cv2.Rodrigues(rvec)

        # way 1: directly transpose
        refer_obj_coord = cooridinate_transfer_2(refer_R, tvec, target_local_coordinate)
        # refer_obj_coord = refer_R @ target_local_coordinate.T + tvec

        # way 2: inverse transfer
        # refer_obj_coord = cooridinate_transfer_2(refer_R, tvec, target_local_coordinate)




        # refer_camera_points_world = np.linalg.inv(refer_camera_matrix) @ refer_obj_coord

        word_points_dict[target_label] = refer_obj_coord.T



    camera_key_list = list(holographic_info_dict.keys())

    # camera_key_list.remove("camera_11_left")
    for camera_key in camera_key_list:

        # [10] calculate the R & T of all cameras
        targets_world_label_set = set(word_points_dict.keys())
        camera_info = holographic_info_dict[camera_key]

        # store corresponding  points in world coordinates and image cooridiantes
        curr_target_point_world = []
        curr_target_point_image = []

        curr_target_label_set = set(camera_info["target_label_set"])
        curr_correspond_labels = curr_target_label_set.intersection(targets_world_label_set)
        curr_correspond_labels = list(curr_correspond_labels)
        curr_correspond_labels.sort()
        # print("Current correspond targets: ", curr_correspond_labels)
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
        _, curr_rvec, curr_tvec = cv2.solvePnP(curr_target_point_world[:12, :], curr_target_point_image[:12, :], curr_camera_matrix, curr_distortion_coeffs)
        R_camera, _ = cv2.Rodrigues(curr_rvec)

        # store the R and T into the camera info
        camera_info["R"] = R_camera
        camera_info["T"] = curr_tvec

        # from image coordinate to the world 3D position
        # homogeneous_imgpoints_ref = np.hstack((curr_target_point_image, np.ones((curr_target_point_image.shape[0], 1))))
        homogeneous_wordpoints_ref = np.hstack((curr_target_point_world, np.ones((curr_target_point_world.shape[0], 1))))
        # Projection matrix for the camera
        P = np.dot(curr_camera_matrix, np.hstack((R_camera, curr_tvec)))

        imgpoints_projected_hom = P @ homogeneous_wordpoints_ref.T
        imgpoints_projected = imgpoints_projected_hom / np.tile(imgpoints_projected_hom[-1, :], (3, 1))
        imgpoints_projected = imgpoints_projected[:2, :].T
        error = cv2.norm(curr_target_point_image, imgpoints_projected.reshape(-1, 2), cv2.NORM_L2) / len(imgpoints_projected)
        print("Projection error: ", error)

        for local_target_label in curr_target_label_set:
            if local_target_label in curr_correspond_labels:
                continue
            local_target_imag_points = np.array(camera_info[local_target_label])

            _, local_rvec, local_tvec = cv2.solvePnP(target_local_coordinate, local_target_imag_points, curr_camera_matrix,
                                         curr_distortion_coeffs)

            local_R, _ = cv2.Rodrigues(local_rvec)

            # local_obj_coord = local_R @ target_local_coordinate.T + local_tvec
            local_obj_coord = cooridinate_transfer_2(local_R, local_tvec, target_local_coordinate)
            local_obj_coord = local_obj_coord.T
            # refer_points = np.linalg.inv(R_camera) @ (local_obj_coord - curr_tvec)
            refer_points = cooridinate_transfer_1(R_camera, curr_tvec, local_obj_coord)

            word_points_dict[local_target_label] = refer_points.T

            # print("label: ", target_label)
            # if target_label in curr_correspond_labels:
            #     original_refer_points = word_points_dict[target_label]
            #     refer_points = R_camera @ original_refer_points.T + curr_tvec
            #     print("coordinate diff1: ", refer_points.T - local_obj_coord.T)



            # if target_label in curr_correspond_labels:
            #     original_refer_points = word_points_dict[target_label]
            #     print("coordinate diff1: ", refer_points - original_refer_points)
            #     # print("coordinate diff2: ", refer_points_2 - original_refer_points)
            # print("coordinate: ", refer_points)

    # print("test")

    # gather the targets points
    target_point_world = []
    for target_key in word_points_dict.keys():
        target_point_world.append(word_points_dict[target_key])
    target_point_world = np.vstack(target_point_world)
    # gather the camera points
    camera_points = []
    camera_name_list = []
    for camera_key in holographic_info_dict.keys():
        camera_coord = holographic_info_dict[camera_key]["T"].reshape(1,3)
        camera_points.append(camera_coord)
        camera_name_list.append(camera_key)

    camera_points = np.vstack(camera_points)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the target points

    ax.scatter(target_point_world[:, 0], target_point_world[:, 1], target_point_world[:, 2], c='blue', s=5, label='Target Points')
    # ax.text(target_point_world[0], target_point_world[1], target_point_world[2], 'Target Points',
    #         color='blue', ha='center', va='center')

    # Plot the right camera position
    camera_color_list = ["red", "green", "yellow", "black", "purple", "gray"]
    for camera_idx in range(len(camera_name_list)):
        camera_name = camera_name_list[camera_idx]
        camera_coord = camera_points[camera_idx]
        ax.scatter(camera_coord[0], camera_coord[1], camera_coord[2], c=camera_color_list[camera_idx], s=20, label= camera_name)
        # ax.text(right_camera_world[0], right_camera_world[1], right_camera_world[2], 'Right Camera',
        #         color='red', ha='center', va='center')

    # ax.scatter(0.0, 0.0, 0.0, c='yellow', label='Left Camera')
    # # ax.text(0, 0, 0, 'Left Camera',
    # #         color='red', ha='center', va='center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()





