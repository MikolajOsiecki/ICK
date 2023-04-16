import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import glob
import time
import random

class Obstacle_Photo:
    """
    Class for taking photos and finding contours of obstacles
    
    Args:
        save_path: path to save photos
        cam_port: camera port
        mtx: camera matrix
        dist: distortion coefficients
        newcameramtx: new camera matrix
        roi: region of interest
    """
    def __init__(self, save_path = '/obstacle', cam_port=0, mtx = [], dist = [], newcameramtx = [], roi = (tuple) ):
        self.cam_port = cam_port
        self.save_path = save_path
        self.mtx = mtx
        self.dist = dist
        self.newcameramtx = newcameramtx
        self.roi = roi


    def take_photo(self):
        """_summary_: Take a single photo using camera and save it
        """
        cam = cv2.VideoCapture(self.cam_port)
        if not (cam.isOpened()):
            print("Could not open video device")
        else:
            result, frame = cam.read()
            if result:
                # print(f"Image captured successfully")
                cv2.imwrite(str(self.save_path)+'target.png', frame)
            else:
                print("No image detected. Please! try again")
        cam.release()


    def undistort_image(self):
        """
        Undistort image using calibration parameters"""
        target = cv2.imread(str(self.save_path)+'target.png')
        dst = cv2.undistort(target, self.mtx, self.dist, None, self.newcameramtx)
        # crop the image
        x, y, w, h = self.roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(str(self.save_path)+'/calibresult.png', dst)

    def find_contours(self):
        """
        Find and draw contours of the image
        """
        im = cv2.imread(str(self.save_path)+'/calibresult.png')
        assert im is not None, "file could not be read, check if calibration was succesfull"
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont = cv2.drawContours(im, contours, -1, (0,255,0), 3)
        # print(contours)
        cv2.imshow('image',cont)
        cv2.imwrite(str(self.save_path)+'output.png', cont)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def find_obstacle(self):
        """
        Take a photo and undistort it
        """
        self.take_photo()
        self.undistort_image()

    def find_obstacle_contours(self):
        """
        Take a photo and print contours
        """
        self.find_obstacle()
        self.find_contours()
