import cv2
import numpy as np
import glob

class CameraCalibrator:
    def __init__(self, pattern_size=(9, 6)):
        self.pattern_size = pattern_size
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix = None
        self.distortion_coefficients = None
    
    def calibrate(self, images_folder_path):
        # 객체 점 생성
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)

        # 파일 목록 읽기
        images = glob.glob(images_folder_path)

        # 객체점과 이미지점 준비
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

        # 카메라 보정
        if len(self.objpoints) > 0 and len(self.imgpoints) > 0:
            ret, self.camera_matrix, self.distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        else:
            print("No chessboard corners found in the provided images")
        
    def __call__(self, img):
        if self.camera_matrix is not None and self.distortion_coefficients is not None:
            return cv2.undistort(img, self.camera_matrix, self.distortion_coefficients)
        else:
            print("Camera not calibrated yet")
            return img