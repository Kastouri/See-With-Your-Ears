import numpy as np
XIAOMI_INTRINSIC_CAM = np.array([[3.09066638e+03, 0.00000000e+00, 2.01042027e+03], 
                                 [0.00000000e+00, 3.08805119e+03, 1.50205810e+03], 
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

XIAOMI_INTRINSIC_CAM_06 = np.array([[2.01527037e+03, 0.00000000e+00, 2.15621798e+03],
                                    [0.00000000e+00, 2.11383807e+03, 1.56036525e+03],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


def im_point_to_angles(im_point, object_distance, camera_intrinsic_matrix=XIAOMI_INTRINSIC_CAM_06):
    u, v =  im_point
    z_cam = -object_distance
    cx = camera_intrinsic_matrix[0,2]
    cy = camera_intrinsic_matrix[1,2]
    fu = camera_intrinsic_matrix[0,0]
    fv = camera_intrinsic_matrix[1,1]
    x_cam = (u - cx)/fv * z_cam
    y_cam = (v - cy)/fu * z_cam

    
def im_point_to_angles_inter(im_point, img_size, fov, camera_intrinsic_matrix=XIAOMI_INTRINSIC_CAM_06):
    # determine camera fov
    width, height = img_size
    fov_u, fov_v = fov
    u, v = im_point

    # determine angles
    angle_u = (u - width/2) / width * fov_u
    angle_v = (height/2 - v) / height * fov_v

    return angle_u, angle_v


def determine_cam_fov(img_size, camera_intrinsic_matrix=XIAOMI_INTRINSIC_CAM_06):
    # determine camera fov
    width, height = img_size
    fu = camera_intrinsic_matrix[0, 0]
    fv = camera_intrinsic_matrix[1, 1]
    # determine fov
    fov_u = 2 * np.rad2deg(2 * np.arctan2(width, 2 * fu))
    fov_v = 2 * np.rad2deg(2 * np.arctan2(height, 2 * fv))

    return fov_u, fov_v
