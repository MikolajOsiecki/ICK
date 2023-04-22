# program to capture single image from webcam in python

# importing OpenCV library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import glob
import time
import random

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
# for i in range(10):
#     cam = VideoCapture(i)
#     if cam.isOpened():
#         cam_port = i
#         print(i)

class Calibration:
    def __init__(self, take_pictures = True, cam_port=0, picture_count=20, save_location='calibration/'):
        self.cam_port = cam_port
        self.create_calib_photos = take_pictures
        self.number_of_pictures = picture_count
        self.save_location = save_location
     

    def create_calibration_images(self):
        """Create calibration images   

        Args:
            cam_port (int): port of camera to use
            number_of_pictures (int): number of calib photos to take
        """
        print("Welcome to calibration mode. Please! place the checkerboard in front of the camera and press enter to start capturing images")
        input()
        cam = cv2.VideoCapture(self.cam_port)
        if not (cam.isOpened()):
            print("Could not open video device")
        else:
            for i in range(self.number_of_pictures):
                result, frame = cam.read()
                cv2.imwrite(str(self.save_location) +'calibration' + str(i) + '.jpg', frame)
                time.sleep(0.5)
                if result:
                    print(f"Image {i} captured successfully")
                else:
                    print("No image detected. Please! try again")
        cam.release()

    def calibrate_camera(self):
        """Calibrate camera using checkerboard images. And return camera matrix, distortion coefficients, rotation and translation vectors.

        Returns:
            list: objpoints list of object points
            list: imgpoints list of image points
            numpy.ndarray: mtx camera matrix
            numpy.ndarray: dist distortion coefficients
            list: rvecs list of rotation vectors
            list: tvecs list of translation vectors
            numpy.ndarray: newcameramtx optimal camera matrix
            tuple: roi region of interest
        """
        if self.create_calib_photos:
            self.create_calibration_images()
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*4,3), np.float32)
        objp[:,:2] = np.mgrid[0:6,0:4].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        img_names = []
        images = glob.glob(str(self.save_location) +'*.jpg')
        # print(images)
        calibrated = False
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (6,4), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                self.imgpoints.append(corners2)
                print(f'Found checkerboard for {fname}')
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (6,4), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
                calibrated = True
                img_names.append(fname)  
        cv2.destroyAllWindows()
        if not calibrated:
            print(f'No checkerboard found! Calibration failed!! \nPlease! check the checkerboard and try again')
        else:
            # print(objpoints)
            # print(imgpoints)  
            #get random image with checkerboard
            checkerboard = random.choice(img_names)
            print(f'Using {checkerboard} for undistortion')
            img = cv2.imread(checkerboard)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            h,  w = img.shape[:2]
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            # print(type(self.roi))
            return self.objpoints, self.imgpoints, self.mtx, self.dist, self.rvecs, self.tvecs, self.newcameramtx, self.roi

def calulate_error(self):
    """Calculate the error of the calibration"""
    self.mean_error = 0
    for i in range(len(self.objpoints)):
        imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
        error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        self.mean_error += error
    print( "total error: {}".format(self.mean_error/len(self.objpoints)) )
    return self.mean_error

