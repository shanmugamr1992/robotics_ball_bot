from ultralytics import YOLO
import numpy as np
import cv2
import time
import sys
from . import lan
from .ball_bbox import Ballbbox

class Robot:
    def __init__(
        self, ip = "10.0.0.119", ball_should_be_within = (341, 255, 422, 338), 
        model="yolov8s.pt", port=9999, distance_threshold=200,
        lever_down_potentiometer_reading = 3300, lever_up_potentiometer_reading = 3680,
        forward_speed = 35, turning_speed = 40, tolerance = 4, compute_reference_box = False):
        lan.stop()
        lan.start(ip, port, frame_shape=(360, 640))
        self.clear_motors()
        self.reference_bbox = Ballbbox(*ball_should_be_within)      
        self.model = YOLO(model)
        self.distance_threshold = distance_threshold
        self.motor_vals = [0]*10
        self.lever_down_potentiometer_reading = lever_down_potentiometer_reading
        self.lever_up_potentiometer_reading = lever_up_potentiometer_reading
        self.forward_speed = forward_speed
        self.turning_speed = turning_speed
        self.tolerance = tolerance
        if compute_reference_box:
            self.reference_bbox = self.compute_reference_box()
            reference_bbox=self.reference_bbox
            print(f'Ref coordinates : {(reference_bbox.x1, reference_bbox.y1, reference_bbox.x2, reference_bbox.y2)}')

    def display(self, image, tgt_bbox, iou):
        reference_bbox = self.reference_bbox
        if tgt_bbox is not None:
            image = cv2.rectangle(image, (tgt_bbox.x1, tgt_bbox.y1), (tgt_bbox.x2, tgt_bbox.y2), (0,0, 255), 1)
            image = cv2.circle(image, (tgt_bbox.midx, tgt_bbox.midy), 4, (0,0,255), -1)
        else:
            tgt_bbox = reference_bbox
        image = cv2.rectangle(image, (reference_bbox.x1, reference_bbox.y1), (reference_bbox.x2, reference_bbox.y2), (0,255,0), 1)
        image = cv2.circle(image, (reference_bbox.midx, reference_bbox.midy), 4, (0,255, 0), -1)
        image = cv2.putText(image, f'IOU: {round(iou,2)}%, D: {tgt_bbox.depth_in_cm} cm', (tgt_bbox.x1, tgt_bbox.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0, 255), 1)
        cv2.imshow('POV', image)
        cv2.waitKey(1)
        
    def compute_reference_box(self):
        time.sleep(3)
        color_image,depth_image = self.get_camera_pic()
        if np.sum(color_image) == 0:
            print('Image is empty')
            sys.exit(0)
        balls_detected = self.get_ball_detections(color_image, depth_image)
        return balls_detected[0]        
        
    def get_potentiometer_reading(self):
        return lan.read()[1]

    def get_camera_pic(self):
        return lan.get_frame() 

    def send_motor_vals(self):
        lan.write(self.motor_vals)
        
    def clear_motors(self, claw_has_ball = False):
        self.motor_vals = [0]*10
        if claw_has_ball : 
            self.motor_vals[7] = -20 
        self.send_motor_vals()
        
    def move_forward(self, speed = None):
        if speed is None:
            speed = self.forward_speed
        self.motor_vals[0] = -speed # Left motor (Negative -> Forward)
        self.motor_vals[9] = speed # Right motor (Positive -> Forward)
        self.send_motor_vals()

    def move_backward(self, speed = None):
        if speed is None:
            speed = self.forward_speed
        self.motor_vals[0] = speed
        self.motor_vals[9] = -speed
        self.send_motor_vals()

    def turn_right(self, speed = None):
        if speed is None:
            speed = self.turning_speed 
        self.motor_vals[0] = -speed 
        self.motor_vals[9] = 0 
        self.send_motor_vals()

    def turn_left(self, speed = None):
        if speed is None:
            speed = self.turning_speed
        self.motor_vals[0] = 0 
        self.motor_vals[9] = speed
        self.send_motor_vals()
        
    def close_claw(self):
        self.clear_motors()
        time.sleep(1)
        self.motor_vals[7] = -50
        print(self.motor_vals)
        self.send_motor_vals()
        time.sleep(1.5)
        # TODO: Add check to see if ball is picked up (potentiometer ?)

    def open_claw(self):
        self.motor_vals[7] = 50
        self.send_motor_vals()
        time.sleep(1)      
        self.clear_motors()

    def lever_down(self, claw_has_ball = False):
        # Need to give mild voltage to hold the ball intact
        self.clear_motors(claw_has_ball)
        # Beyond this gravity will bring it down
        while self.get_potentiometer_reading() > self.lever_down_potentiometer_reading:
            self.motor_vals[6] = -40
            self.send_motor_vals()
               
    def lever_up(self, claw_has_ball = False):
        self.clear_motors(claw_has_ball)    
        while self.get_potentiometer_reading() < self.lever_up_potentiometer_reading:
            self.motor_vals[6] = 40
            self.send_motor_vals()      
        self.clear_motors(claw_has_ball)
        
    def get_ball_detections(self, color_image, depth_image):       
        result = self.model(color_image, verbose=False)[0]        
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        balls_detected = []
        for cls, bbox in zip(classes, bboxes):
            if cls == 32:
                (x1, y1, x2, y2) = bbox
                midpoint = int((x1+x2)/2), int((y1+y2)/2)
                depth_in_cm = depth_image[midpoint[1], midpoint[0]]/10
                balls_detected.append(Ballbbox(*bbox, depth_in_cm))
        return balls_detected

    def get_target_ball_bbox(self, balls_detected):
    # Finding based on euclidean distance between center of the bbox and ref_box
    # (Maybe we should not go for boxes in xtreme left or right ? (Turning takes longer).
    # TODO: So maybe should give more weight to balls close  in the x axis than y axis
        target_bbox = None
        min_dist = None
        for ball_bbox in balls_detected:
            dist = ball_bbox.get_distance_to(self.reference_bbox)
            if min_dist is None or dist < min_dist:
                target_bbox = ball_bbox
                min_dist = dist               
        return target_bbox
    
    def move_to_target_ball_bbox(self, current_target_bbox, color_image):
        self.clear_motors()
        reference_bbox = self.reference_bbox 
        # TODO : Maybe have some time constraint as well (1 min) if not sw
        iou = current_target_bbox.get_iou_with(reference_bbox) 
        while iou < 0.80:
            # TODO: Should calculate this based on checking how far the ball can be within the reference frame 
            # i.e how much tolerance on x and y plane
            # TODO : Can also add some speed based numbers
            self.display(color_image, current_target_bbox, iou)
            tolerance = max(abs(current_target_bbox.midy - reference_bbox.midy)*.1, self.tolerance)
            fspeed = None
            tspeed = None
            dist = current_target_bbox.depth_in_cm
            if current_target_bbox.depth_in_cm > 55 :
                fspeed = 40
                tspeed = 45
            print(f'Iou {iou} , distance: {dist}')
            
            if abs(current_target_bbox.midx - reference_bbox.midx) < tolerance:                   
                self.move_backward(fspeed) if current_target_bbox.midy > reference_bbox.midy else self.move_forward(fspeed)
            else: 
                self.turn_left(tspeed) if current_target_bbox.midx < reference_bbox.midx else self.turn_right(tspeed)                     
            
            color_image,depth_image = self.get_camera_pic()
            balls_detected = self.get_ball_detections(color_image, depth_image)
            if len(balls_detected) == 0:
                return False

            # TODO : Or we should find the ball closest to the current target box (ie.) This will be like tracking
            # Reason : I am afraid if there are two balls right next to each other it might just be oscilatting 
            # i.e when it turns right the right ball will become closer, then when it turns left the left ball might be closer
            current_target_bbox = self.get_target_ball_bbox(balls_detected) 
            iou = current_target_bbox.get_iou_with(reference_bbox) 
            
        self.clear_motors()
        print(f'Came out of the loop for :{(current_target_bbox.midx, current_target_bbox.midy)}')
        print(f'Reference coordinates    :{(reference_bbox.midx, reference_bbox.midy)}')
        print(f'IOU                      :{iou}')
        
        return True
   
    def pickup_ball(self):    
        self.lever_down(claw_has_ball = False)
        self.close_claw()
        self.lever_up(claw_has_ball = True)
        return True # TODO: Should return true or false based on pickup

    def move_to_opp_wall(self):
        self.move_forward()
        time.sleep(2)
        # TODO : Should implement this

    def drop_ball_and_reset(self):
        self.lever_down(claw_has_ball = True)
        self.open_claw()
        self.lever_up()

    def random_walk(self):
        #Just going to turn around here
        self.turn_right()
    
    def run(self):
        
        while True:           
            self.random_walk()
            color_image,depth_image = self.get_camera_pic()
            self.display(color_image, None, 0)
            balls_detected = self.get_ball_detections(color_image, depth_image)
            
            # No balls are detected
            if len(balls_detected) == 0:
                continue
            
            target_bbox = self.get_target_ball_bbox(balls_detected)

            # If this target ball is too far out, we could do some more exploration
            if self.distance_threshold and target_bbox.depth_in_cm > self.distance_threshold:
                continue

            if not self.move_to_target_ball_bbox(target_bbox, color_image):
                continue
            
            if not self.pickup_ball():
                continue

            self.move_to_opp_wall()

            self.drop_ball_and_reset()