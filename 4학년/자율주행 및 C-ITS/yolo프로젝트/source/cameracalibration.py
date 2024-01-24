import cv2
import numpy as np
import glob



class CameraCalibrator:
    def __init__(self, nx, ny):
        self.nx = nx  # 체스보드 코너의 x축(열) 갯수
        self.ny = ny  # 체스보드 코너의 y축(행) 갯수
        self.calibration_images_success = []  # 보정 성공 이미지 리스트
        self.calibration_images_error = []  # 보정 실패 이미지 리스트

    def calibrate(self, images_folder_path):
        # 체스보드 코너의 3D 좌표 생성
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        # 모든 이미지에서 3D 좌표와 2D 좌표(체스보드 코너) 저장할 배열 생성
        objpoints = []  # 실제 3D 공간에서의 좌표
        imgpoints = []  # 이미지 평면에서의 좌표

        # 보정용 이미지들을 불러옴
        images = glob.glob(images_folder_path)

        # 모든 이미지를 하나씩 돌며 체스보드 코너 찾기
        for fname in images:
            # 이미지 로드 후 그레이스케일 변환
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 체스보드 코너 찾기
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # 코너를 찾은 경우, 3D 좌표와 2D 좌표를 배열에 추가
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # 코너를 표시한 이미지 저장
                img = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                self.calibration_images_success.append(img)
            # 코너를 찾지 못한 경우, 실패 이미지 리스트에 저장
            else:
                self.calibration_images_error.append(img)

        # 카메라 보정
        ret, self.camera_matrix, self.distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def __call__(self, img):
        # 왜곡 보정 후 이미지 반환
        return cv2.undistort(img, self.camera_matrix, self)