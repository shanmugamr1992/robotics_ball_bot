from ultralytics import YOLO
import numpy as np
from .ball_bbox import Ballbbox


class ObjectDetectionYOLO:
    def __init__(self, model):
        self.model = YOLO(model)

    def get_ball_detections(self, color_image, depth_image):
        result = self.model(color_image, verbose=False)[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        balls_detected = []
        for cls, bbox in zip(classes, bboxes):
            if cls == 32:
                (x1, y1, x2, y2) = bbox
                midpoint = int((x1+x2)/2), int((y1+y2)/2)
                depth_in_cm = None
                if depth_image is not None:
                    depth_in_cm = depth_image[midpoint[1], midpoint[0]]/10
                balls_detected.append(Ballbbox(*bbox, depth_in_cm))

        return balls_detected
