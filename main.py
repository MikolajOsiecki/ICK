import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import glob
import time
import threading
from threading import Event
from Calibration import Calibration
from Obsatcle_Photo import Obstacle_Photo
from matcher import Matcher

PhotoEvent = Event()
MatchCounts = [0,0,0]
Scores = [0,0,0]
FoundObstacle = ["Kubek", "Butelka", "Baton", "None"]   #list of names, can be changed
threads = []

cam_port = 0 #  laptop camera
# cam_port = "/dev/video2" # usb camera
calib_retry_count = 2
calib_path = '/home/nosfreat/AGH/ICK/calibration/'
obstacle_path = '/home/nosfreat/AGH/ICK/obstacle/'
path_for_matcher = '/home/nosfreat/AGH/ICK/obstacle/calibresult.png'
kubek_path = '/home/nosfreat/AGH/ICK/Kubek/'
butelka_path = '/home/nosfreat/AGH/ICK/Butelka/'
baton_path = '/home/nosfreat/AGH/ICK/Baton/'



def select_obstacle():
    max_score_id = np.argmax(Scores)
    if MatchCounts[max_score_id] > 1:
        return FoundObstacle[max_score_id]
    else:
        return FoundObstacle[-1]

class ObstaclePhotoThread (threading.Thread):
    """
    Thread for periodic photo taking

    Args:
        threadID: thread ID
        name: thread name
        Obstacle: Obstacle_Photo class instance
    """
    def __init__(self, threadID, name, Obstacle = Obstacle_Photo):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.Obstacle = Obstacle
        self._stop_event = threading.Event()

    def run(self):
        print( "Starting " + self.name)
        while True:
            if self._stop_event.is_set():
                break
            print(select_obstacle())
            self.get_photo()
            PhotoEvent.set()
            if self._stop_event.is_set():
                break
            time.sleep(1)
            PhotoEvent.clear()
            time.sleep(14)

    def stop(self):
        self._stop_event.set()

    def get_photo(self):
        self.Obstacle.find_obstacle()

class MatchingThread (threading.Thread):
    """
    Thread for matching photo with template

    Args:
        threadID: thread ID
        name: thread name
        matcher: Matcher class instance
    """
    def __init__(self, threadID, name, matcher = Matcher):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.matcher = matcher
        self._stop_event = threading.Event()

    def run(self):
        while True:
            if self._stop_event.is_set():
                break
            PhotoEvent.wait()
            if self._stop_event.is_set():
                break
            print( "Matching " + self.name)
            self.match_target()

    def stop(self):
        PhotoEvent.set()
        self._stop_event.set()

    def match_target(self):
        score, match_count = self.matcher.match_phots()
        Scores[self.threadID] = score
        MatchCounts[self.threadID] = match_count

####################################################

def main():
    Calib = Calibration(take_pictures=False, cam_port= cam_port, picture_count= 20,save_location=calib_path)
    for calib_retry in range(calib_retry_count):
        try:
            objpoints, imgpoints, mtx, dist, rvecs, tvecs, newcameramtx, roi = Calib.calibrate_camera()
            print("Calibration succesfull")
            break
        except:
            print("Calibration failed")
        print(f"Calibration retry {calib_retry+1} of {calib_retry_count}")
        calib_retry = calib_retry + 1

    PhotoThread = ObstaclePhotoThread(99, "ObstaclePhoto", Obstacle = Obstacle_Photo(cam_port=cam_port, save_path=obstacle_path, mtx = mtx, dist = dist, newcameramtx = newcameramtx, roi = roi))
    threads.append(PhotoThread)
    MatchThreadKubek = MatchingThread(0, "MatchKubek", matcher=Matcher(template_path=kubek_path, img_path=path_for_matcher, min_match_count=70))
    threads.append(MatchThreadKubek)
    MatchThreadButelka = MatchingThread(1, "MatchButelka", matcher=Matcher(template_path=butelka_path, img_path=path_for_matcher, min_match_count=70))
    threads.append(MatchThreadButelka)
    MatchThreadBaton = MatchingThread(2, "MatchBaton", matcher=Matcher(template_path=baton_path, img_path=path_for_matcher, min_match_count=70))
    threads.append(MatchThreadBaton)
    for thread in threads:
        thread.start()
    
    while True:
        input("Press q to quit")
        if input :
            print("Quitting")
            for thread in threads:
                thread.stop()
            for thread in threads:
                thread.join()
            break


if __name__ == "__main__":
    main()