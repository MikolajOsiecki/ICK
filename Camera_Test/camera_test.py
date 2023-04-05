import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture("/dev/video2")

# Set smaller resolution
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 640
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 480

while cam.isOpened():
    ret, frame = cam.read()
    if ret:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break