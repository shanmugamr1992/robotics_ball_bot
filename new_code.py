from cortano import RemoteInterface
from ultralytics import YOLO
import numpy as np
import cv2
import cv2
import numpy as np
import time
import pygame
import sys
from lan import lan

model = YOLO("yolov8m.pt")

lan.start("10.0.0.119", 9999, frame_shape=(360, 640))

ball_should_be_within = (240, 115, 70, 70)

motor_vals = [0] * 10
sensor_vals = np.zeros((20,), np.int32)
pygame.init()
                                  # w, h
screen = pygame.display.set_mode((640, 360))
clock = pygame.time.Clock()
screen.fill((63, 63, 63))

def decode_depth_frame(f):
    R = np.left_shift(frame[:, :, 0].astype(np.uint16), 5)
    G = frame[:, :, 1].astype(np.uint16)
    B = np.left_shift(frame[:, :, 2].astype(np.uint16), 5)
    I = np.bitwise_or(R, G, B)
    return I
    
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
    
def get_midpoint(boxA):
    width = abs(boxA[0] - boxA[2])
    height = abs(boxA[1] - boxA[3])
    mid_point = (int(boxA[0]+(width/2)), int(boxA[1]+(height/2)))
    return mid_point

def ball_position(ball_should_be_within,bbox):
    mp_target = midpoint(ball_should_be_within)
    mp_current = midpoint(bbox)
    if bb_intersection_over_union(ball_should_be_within,bbox) <= 0.95:
        if mp_current[0] > mp_target[0]:
            keys_letters["LEFT"] = 1
        elif mp_current[0] < mp_target[0]:
            keys_letters["RIGHT"] = 1
        elif abs(mp_current[0] - mp_target[0]) <= 3:
            if mp_current[1] > mp_target[1]:
                keys_letters["UP"] = 1
            elif mp_current[1] < mp_target[1]:
                keys_letters["DOWN"] = 1
    else:
        keys_letters["LEFT"] = 0
        keys_letters["RIGHT"] = 0
        keys_letters["UP"] = 0
        keys_letters["DOWN"] = 0
        
keys_letters={'a':0, 'w':0, 's':0, 'd':0, 'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0, 'q':0}
key_number_mappings={
    97: 'a',
    119: 'w',
    115: 's',
    100: 'd',
    1073741906: 'UP',
    1073741905: 'DOWN',
    1073741904: 'LEFT',
    1073741903: 'RIGHT',
    113: 'q'
}
