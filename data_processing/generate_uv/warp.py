import numpy as np
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import cv2.aruco as aruco
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract sample plane.')
    parser.add_argument('data_root', type=str)
    parser.add_argument('save_root', type=str)
    parser.add_argument('--main_cam_id', type=int, default=0)
    parser.add_argument('--texture_resolution', type=int, default=1024)
    parser.add_argument('--down_size',type=int, default=2)
    
    args = parser.parse_args()

    data_root = args.data_root + "sfm/"
    output_root = args.data_root + args.save_root + f"texture_{args.texture_resolution}/"

    # Load the region of interest (ROI) from the file
    roi = np.fromfile(output_root+"roi_{}.bin".format(args.down_size), np.int32).reshape([6,])

    # Initialize the ArUco dictionary for marker detection
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

    
    frame = []
    gray = []
    keys = []
    descriptors = []

    # Create a SIFT detector object
    sift = cv2.SIFT_create()
    
    cameras = [args.main_cam_id, 1-args.main_cam_id] 
    for which_cam in cameras:
        # Read, blur, and resize the undistorted image
        udt_img = cv2.imread(data_root+f"{which_cam}_0_udt.png")
        udt_img = cv2.GaussianBlur(udt_img, (roi[4], 1), 0)
        udt_img = cv2.GaussianBlur(udt_img, (1, roi[5]), 0)
        udt_img = cv2.resize(udt_img,(udt_img.shape[1]//args.down_size,udt_img.shape[0]//args.down_size), cv2.INTER_LINEAR)
        cv2.imwrite(data_root+f"cam_{which_cam}.png", udt_img)
        
        frame.append(udt_img)

        # Detect keypoints and compute descriptors using SIFT
        kp, des = sift.detectAndCompute(udt_img, None)
        keys.append(kp)
        descriptors.append(des)

        # Draw keypoints on the image and save it
        tmp_frame_keys = cv2.drawKeypoints(udt_img, kp, None)

        cv2.imwrite(data_root+f"cam_{which_cam}_keys.png", tmp_frame_keys)
    
    ratio = 0.50
    matcher = cv2.BFMatcher()

    # Perform KNN matching of descriptors between the two images
    raw_matches = matcher.knnMatch(descriptors[0], descriptors[1], k = 2)
    
    # Apply ratio test to select good matches
    good_matches = []
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])
    
    matches = cv2.drawMatchesKnn(frame[0], keys[0], frame[1], keys[1], good_matches, None, flags = 2)
    cv2.imwrite(data_root+"matches.png", matches)
    
    if len(good_matches) > 4:
        # Extract point coordinates from the good matches
        ptsA = np.float32([keys[0][m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ptsB = np.float32([keys[1][m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute the homography matrix using RANSAC
        ransacReprojThreshold = 4

        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        
        # Warp the second image to align with the first image
        img_warp = cv2.warpPerspective(frame[1], H, (frame[0].shape[1],frame[0].shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        cv2.imwrite(data_root+f"cam_{1-args.main_cam_id}_warp.png", img_warp)
        H.astype(np.float64).tofile(output_root+"H.bin")

        # Extract and save the cropped regions from the aligned images
        w_start, h_start, w_end, h_end = roi[:4]
        img_0 = frame[0][h_start:h_end,w_start:w_end,:]
        img_1 = img_warp[h_start:h_end,w_start:w_end,:]

        cv2.imwrite(data_root+"image0_crop.png", img_0)
        cv2.imwrite(data_root+"image1_crop.png", img_1)

