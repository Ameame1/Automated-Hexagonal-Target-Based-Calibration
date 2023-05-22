import numpy as np
import cv2
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def rough_target_detection(img, min_thresh=50, diff_thresh=50):
    """
    Return mask of the image that meets the conditions for each pixel:
        condition 1: min(r,g,b) < min_thresh
        condition 2: max(r,g,b) - min(r,g,b) > diff_thresh
    :param image:
    :param min_thresh: the threshold of minimum value of RGB channels in an image
    :param diff_thresh: the threshold of difference between the values in RGB channels
    :return: a mask that meets condition 1 and condition 2
    """
    img_c_min = img.min(axis=2)
    img_c_max = img.max(axis=2)

    # the mask that meet condition 1
    M1 = (img_c_min < min_thresh)
    # the mask that meet condition 2
    M2 = (img_c_max - img_c_min) > diff_thresh

    # the mask that meet condition 1 or condition 2
    M = np.logical_or(M1, M2)
    # change the data type from bool to unsigned integer
    M = M.astype(np.uint8)

    return M

def area_threshold_filter(mask, min_area=20, max_area=450):
    """
    This function will filter out the connected components with too small and too big area
    :param mask: the image mask that need to be filtered
    :param min_area: the minimum area threshold
    :param max_area: the maximum area threshold
    :return: a new mask
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # filter the connection components
    filtered_labels = np.zeros_like(labels)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            filtered_labels[labels == label] = 255

    # change the image to binary image
    filtered_M = np.where(filtered_labels == 255, 255, 0).astype(np.uint8)

    return filtered_M

def axis_ratio_filter(mask, axis_ratio_threshold=0.3):
    """
    This function will filter the components that are not obviously not round
    :param mask: the image mask that need to be filtered
    :param axis_ratio_threshold: the axis ratio of a component should greater than this threshold
    :return: a new mask
    """

    # Using the axis ratio to remove clusters that are obviously not round
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # filter the connection components
    filtered_labels = np.zeros_like(labels)

    for label in range(1, num_labels):


        # get current connected area mask
        connected_area_mask =  (labels == label).astype(np.uint8)
        # extract pixel coordinates of the connected area
        indices = np.where(connected_area_mask != 0)
        x_coords = indices[0]
        y_coords = indices[1]

        # Calculate covariance matrix， note that if connected_area_mask has only on elements, we should ignore it
        if x_coords.shape[0] > 1:
            pixel_coordinates = np.vstack((x_coords, y_coords))
            # Get the convariance matrix of current connected clusters' coordinates
            covariance_matrix = np.cov(pixel_coordinates)
            # Calculate the eigenvalues of covariance matrix
            eigenvalues = np.linalg.eigvals(covariance_matrix)
            # Take the square root of the eigenvalues
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            # Get delta_min and delat_max
            delta_min = sqrt_eigenvalues.min()
            delta_max = sqrt_eigenvalues.max()
            axis_ratio = delta_min / delta_max
        else:
            axis_ratio = 0.0
        if axis_ratio > axis_ratio_threshold:
            filtered_labels[labels == label] = 255

    # change the image to binary image
    filtered_M = np.where(filtered_labels == 255, 255, 0).astype(np.uint8)

    return filtered_M

# define the function to get the matrix A
def get_derivative_matrix(x, y):
    return np.array([x * x, x * y, y * y])

def get_largest_residual_error(matrix_A):
    """
    This function is to calculate the largest residual errors of matrix A
    :param matrix_A:
    :return:
    """
    one_vector = np.ones(6)
    ellips_params = (np.linalg.inv(matrix_A.T @ matrix_A)) @ matrix_A.T @ one_vector

    residual_errors = matrix_A @ ellips_params - one_vector

    residual_errors = np.abs(residual_errors)

    return residual_errors.max()
    """
    This function will figure out these components that can form the six dots targets
    :param mask: the mask that include hexagon targets
    :param ellips_threshold: threshold to filter out those components that cannot form obvious hexagon
    :return: a new mask
    """
def target_detection(mask, ellips_threshold=0.1):
    """
    This function will figure out these components that can form the six dots targets
    :param mask: the mask that include hexagon targets
    :param ellips_threshold: threshold to filter out those components that cannot form obvious hexagon
    :return:
        filtered_M: a new mask,
        labels: cluster labels matrix,
        centroids: cluster centroids,
        target_hexagen_cluster_indices: the targets that meet the residual error condition
    """

    # Step 1: get the 5 nearest clusters of each selected cluster
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # Build a KDTree from the centroids points
    kdtree = KDTree(centroids)
    # Find the 5 nearest neighbors for each point
    distances, indices = kdtree.query(centroids, k=6)  # k+1 to include the point itself

    # Calculate each group of clusters coordinate & the matrix A
    derivate_matrix_set = np.zeros((centroids.shape[0], 6, 3))
    for idx in range(centroids.shape[0]):
        mean_value = centroids[indices[idx]].mean(axis=0)
        matrix_A = np.zeros((6, 3))
        for cluster_idx in range(6):
            x, y = centroids[indices[idx, cluster_idx]] - mean_value
            matrix_A[cluster_idx] = get_derivative_matrix(x, y)

        derivate_matrix_set[idx] = matrix_A


    selected_cluster_sets = set()

    target_num_list = [] # store all the target label that detected
    # filter these clusters that can form a target
    for idx in range(centroids.shape[0]):
        # if the cluster has been selected, ignore it
        if idx in selected_cluster_sets:
            continue
        matrix_A = derivate_matrix_set[idx]
        residual_errors = get_largest_residual_error(matrix_A)
        # print("residual_errors", residual_errors)

        if residual_errors < ellips_threshold:
            selected_cluster_sets.update(indices[idx])
            target_num_list.append(idx)


    # filter the connection components
    filtered_labels = np.zeros_like(labels)

    for label_idx in selected_cluster_sets:
        filtered_labels[labels == label_idx] = 255

    # change the image to binary image
    filtered_M = np.where(filtered_labels == 255, 255, 0).astype(np.uint8)

    target_hexagen_cluster_indices = indices[target_num_list]

    return filtered_M, labels, centroids, target_hexagen_cluster_indices

def target_label_align(image_rgb, labels, centroids, indices):

    # color channel map dict
    color_dict = {
        0: "R",
        1: "G",
        2: "B"
    }

    target_list = []

    # process each selected target in a loop
    for target_idx in range(indices.shape[0]):
        clusters_idxes = indices[target_idx]

        # initial weighted centroids of each dot in a target
        weighted_centroids = np.zeros((6, 2))
        # initial color value matrix of each dot in a target
        color_value_matrix = np.zeros((6, 3))

        # total_mask = np.zeros_like(labels)

        # process each cluster (dots in hexagon) in a target
        for idx in range(clusters_idxes.shape[0]):
            cluster_idx = clusters_idxes[idx]

            # get the mask of current cluster
            connected_area_mask = (labels == cluster_idx).astype(np.uint8)

            # Dilate the current cluster mask, here we use a square kernels
            kernel = np.ones((3, 3), np.uint8)  # square kernel
            dilated_mask = cv2.dilate(connected_area_mask, kernel, iterations=1)
            #         print("original sum: ", connected_area_mask.sum(), " dilated sum: ", dilated_mask.sum())

            # get dilated image of the current cluster
            dilated_img = image_rgb * dilated_mask[:, :, np.newaxis]
            # compute the average value of RGB channels
            avg_rgb_value = np.sum(dilated_img, axis=(0, 1)) / dilated_mask.sum()
            # compute the L2 norm of each pixel position
            avg_dilated_img = (dilated_img - avg_rgb_value) * dilated_mask[:, :, np.newaxis]
            l2_norm_avg_img = np.linalg.norm(avg_dilated_img, axis=2)
            # get the weight of each pixel in the map
            weighted_p = 1 - l2_norm_avg_img / l2_norm_avg_img.max()
            weighted_p = weighted_p * dilated_mask

            # compute the weighted centroid of this cluster
            dilated_indices = np.where(dilated_mask != 0)
            # Get the weighted dilated evaluation
            weighted_x, weighted_y = 0, 0
            for x, y in zip(dilated_indices[0], dilated_indices[1]):
                weighted_x += x * weighted_p[x, y]
                weighted_y += y * weighted_p[x, y]
            weighted_x = weighted_x / weighted_p.sum()
            weighted_y = weighted_y / weighted_p.sum()
            weighted_centroids[idx] = np.array([weighted_y, weighted_x])

            # for visualization
            # total_mask = np.logical_or(total_mask, connected_area_mask)
            # Get masked image of current cluster (one dot in
            masked_img = image_rgb * connected_area_mask[:, :, np.newaxis]
            #
            color_value_matrix[idx] = np.sum(masked_img, axis=(0,1))

            # color_value_matrix[idx] = np.array([red_sum, green_sum, blue_sum])
        #         print(red_sum, green_sum, blue_sum)

        color_key_list = list(np.argmax(color_value_matrix, axis=1))
        color_list = [color_dict.get(key) for key in color_key_list]

        # if the number of  "Blue Dot" is not one, ignore this target
        if color_list.count("B") != 1:
            print("This targets has some problem, ignore it:", color_list)
            continue

        # clockwise arrangement calculation
        blue_idx = color_list.index("B")
        blue_cluster_index = clusters_idxes[blue_idx]
        blue_centroid = centroids[blue_cluster_index]
        target_centroids = centroids[clusters_idxes]

        # calculate the polar coordinates angle of each point relative to the starting point
        angles = np.arctan2(target_centroids[:, 1] - blue_centroid[1], target_centroids[:, 0] - blue_centroid[0])

        # use argsort to sort angles，return the indices of sorted value
        sorted_indices = np.argsort(angles)

        # rearrange the color sequence in closewise order
        rearraged_color_list = [color_list[idx] for idx in sorted_indices]
        #     print(rearraged_color_list)

        # generate the color label string
        target_label = "".join(rearraged_color_list[1:])

        sorted_clusters_idxes = clusters_idxes[sorted_indices]
        #     print("original cluster indices: ", clusters_idxes, " sorted cluster indices: ", sorted_clusters_idxes)
        sorted_weighted_centroids = weighted_centroids[sorted_indices]
        #     print("target_label is ", target_label)
        #     print(centroids[sorted_clusters_idxes])
        #     print(sorted_weighted_centroids)

        # construct the final results of each target, including: key=color label string, targets coordinates
        target_dict = {}
        target_dict["key"] = target_label
        # target_dict["cluster_idxes"] = sorted_clusters_idxes
        target_dict["original_centroids"] = centroids[sorted_clusters_idxes]
        target_dict["weighted_centroids"] = sorted_weighted_centroids

        target_list.append(target_dict)

        # compute the weighted centroids
    #     target_image = image_rgb * total_mask[:, :, np.newaxis]
    #     fig = plt.figure(figsize=(10,10))
    #     plt.imshow(target_image)
    #     plt.axis("off")
    #     plt.show()
    return target_list


def get_target_points(img_path,
                      min_thresh=50,
                      diff_thresh=50,
                      min_area=20,
                      max_area=450,
                      axis_ratio_threshold=0.3,
                      ellips_threshold=0.1,
                      **kwargs,
                      ):
    img = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get the M
    mask = rough_target_detection(image_rgb, min_thresh, diff_thresh)
    # filter unwanted components that are too small or too big
    area_filtered_mask = area_threshold_filter(mask, min_area, max_area)
    # filter unwanted components that are obviously not round
    axis_ratio_mask = axis_ratio_filter(area_filtered_mask, axis_ratio_threshold)
    # selected targets clusters from targets mask
    targets_mask, cluster_labels, centroids, targets_list = target_detection(axis_ratio_mask, ellips_threshold)
    # target label and align
    target_info_list = target_label_align(image_rgb, cluster_labels, centroids, targets_list)

    return image_rgb, target_info_list


if __name__ == "__main__":
    # get target position of all images

    image_path_dict = {
        "camera_11_left": "camera 11/2022_12_15_15_51_19_927_rgb_left.png",
        "camera_11_right": "camera 11/2022_12_15_15_51_19_927_rgb_right.png",
        "camera_71": "camera 71/2022_12_15_15_51_19_944_rgb.png",
        "camera_72": "camera 72/2022_12_15_15_51_19_956_rgb.png",
        "camera_73": "camera 73/2022_12_15_15_51_19_934_rgb.png",
        "camera_74": "camera 74/2022_12_15_15_51_19_951_rgb.png",
    }

    image_path_dict = {
        "camera_11_left": "camera 11/2022_12_15_15_51_19_927_rgb_left.png",
        "camera_11_right": "camera 11/2022_12_15_15_51_19_927_rgb_right.png",
        "camera_71": "camera 71/2022_12_15_15_51_19_944_rgb.png",
        "camera_72": "camera 72/2022_12_15_15_51_19_956_rgb.png",
        "camera_73": "camera 73/2022_12_15_15_51_19_934_rgb.png",
        "camera_74": "camera 74/2022_12_15_15_51_19_951_rgb.png",
    }

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

    image_dict = {}


    for key in image_path_dict.keys():
        img_path = image_path_dict[key]
        img = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_dict[key] = image_rgb


    test_img = image_dict.get("camera_11_left")
    test_img_copy = test_img.copy()
    # get the M
    mask = rough_target_detection(test_img)
    # filter unwanted components that are too small or too big
    area_filtered_mask = area_threshold_filter(mask)
    # filter unwanted components that are obviously not round
    axis_ratio_mask = axis_ratio_filter(area_filtered_mask)
    # selected targets clusters from targets mask
    targets_mask, cluster_labels, centroids, targets_list = target_detection(axis_ratio_mask)
    # target label and align
    target_info_list = target_label_align(test_img,cluster_labels, centroids, targets_list)
    # test_img_copy, target_info_list = get_target_points(image_path_dict["camera_74"])
    # draw the target position of on the image
    draw_width = 5
    for target in target_info_list:
        label = target.get("key")
        label_list = list(label)
        label_list.insert(0, "B")
        text_label = "Hexa Target " + label
        centroids = target.get("weighted_centroids")
        for i, coordinate in enumerate(centroids):
            x = int(coordinate[0] + 0.5)
            y = int(coordinate[1] + 0.5)
            # draw the rectangle
            cv2.rectangle(test_img_copy, (x-draw_width, y-draw_width), (x + draw_width, y + draw_width), (0, 255,0), 1)

            # add label
            # text_label = text_label + "_" + str(i + 1)
            text_label = label_list[i] + str(i+1)
            cv2.putText(test_img_copy, text_label, (x-draw_width, y - draw_width), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

    print(target_info_list)

    figure = plt.figure(figsize=(20, 20))
    ax1 = figure.add_subplot(321)
    ax2 = figure.add_subplot(322)
    ax3 = figure.add_subplot(323)
    ax4 = figure.add_subplot(324)
    ax5 = figure.add_subplot(325)
    ax6 = figure.add_subplot(326)

    ax1.imshow(test_img)
    ax1.set_title("Original Image")

    ax2.imshow(mask)
    ax2.set_title("Raw mask")

    ax3.imshow(area_filtered_mask)
    ax3.set_title("Mask after filtering the small and large area")

    ax4.imshow(axis_ratio_mask)
    ax4.set_title("Mask after filtering out components obviously not round")

    ax5.imshow(targets_mask)
    ax5.set_title("Targets Mask")

    ax6.imshow(test_img_copy)
    ax6.set_title("Detected Targets")

    # plt.imshow(test_img_copy)

    plt.show()
