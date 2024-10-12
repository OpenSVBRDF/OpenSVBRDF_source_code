import numpy as np
import cv2

class Camera():
    """
    Camera encapsulates operations related to camera calibration and undistortion.

    Functions: 
        project:            Project the spatial coordinates onto the image plane.
        undistort_2steps:   Undistort the source image using cv2.initUndistortRectifyMap() and cv2.remap().
        get_cam_pos:        Retrieve the camera position.
        get_trans_matrix:   Retrieve the rotation and translation matrices.
        get_camera_matrix:  Retrieve the camera matrix.
        get_distortion_matrix: Retrieve the distortion matrix.
        get_height:         Retrieve the image height.
        get_width:          Retrieve the image width.
        
    """
    def __init__(self,intrinsic_file,extrinsic_file):
        # Load intrinsic parameters (camera matrix and distortion coefficients)
        cv_file = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        self.A = cv_file.getNode("camera_matrix").mat()
        self.D = cv_file.getNode("distortion_coefficients").mat()
        cv_file.release()

        # Load extrinsic parameters (rotation vector and translation vector)
        cv_file = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
        self.rvec = cv_file.getNode("rvec").mat()
        self.tvec = cv_file.getNode("tvec").mat()
        self.rvec = np.stack(self.rvec,axis=0)
        self.tvec = np.stack(self.tvec,axis=0)
        cv_file.release()
        
        # Convert rotation vector to rotation matrix
        self.R = cv2.Rodrigues(self.rvec)[0]
        
        self.T = self.tvec

        # Compute camera position in the world coordinate system
        self.cam_pos = np.matmul(-np.linalg.inv(self.R),self.T)
        
        # Prepare projection matrix
        self.matP = np.zeros([4,4],np.float32)
        self.matP[:3,:3] = self.R
        self.matP[:3,[3]] = self.T
        
        # Image dimensions
        self.width = 5328
        self.height = 4608
        
        self.ncm, _ = cv2.getOptimalNewCameraMatrix(self.A, self.D, (self.width, self.height), 1, (self.width, self.height), 0)

        # Initialize undistortion and rectification map
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.A, self.D, None, self.ncm,
                (self.width, self.height), cv2.CV_16SC2)
        
    def get_cam_pos(self):
        """Retrieve the camera position."""
        return self.cam_pos
        
    def get_trans_matrix(self):
        """Retrieve the rotation and translation matrices."""
        return self.R, self.T

    def get_camera_matrix(self):
        """Retrieve the camera matrix."""
        return self.A

    def get_distortion_matrix(self):
        """Retrieve the distortion matrix."""
        return self.D

    def project(self, pos):
        """
        Project spatial coordinates onto the image plane.
        
        Parameters:
        pos (np.ndarray): Spatial coordinates.
        
        Returns:
        np.ndarray: Image coordinates.
        """
        cam_coord_pos = np.matmul(self.R, pos)+self.T
        cam_coord_pos /= cam_coord_pos[[2],:]
        
        cam_coord = np.matmul(self.A, cam_coord_pos).T
        
        return cam_coord[:,:2]
    
    def undistort_2steps(self,src):
        """
        Undistort the source image using precomputed maps.
        
        Parameters:
        src (np.ndarray): Source image.
        
        Returns:
        np.ndarray: Undistorted image.
        """
        dst = cv2.remap(src, self.map1, self.map2, cv2.INTER_LINEAR)
        return dst
    
    def get_height(self):
        """Retrieve the image height."""
        return self.height
    
    def get_width(self):
        """Retrieve the image width."""
        return self.width
