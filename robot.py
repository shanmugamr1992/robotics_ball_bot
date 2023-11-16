
import numpy as np
import cv2
import time
import sys
from object_detection.object_detection import ObjectDetection
from communication.communication_module import CommunicationModule
from object_detection.ball_bbox import Ballbbox
from april_tags.april_tag import AprilTag


class Robot:
    def __init__(
            self, ip="10.0.0.122", ball_should_be_within=(341, 255, 422, 338),
            model="yolov8s.pt", port=9999, distance_threshold=200,
            lever_down_potentiometer_reading=3300, lever_up_potentiometer_reading=3680,
            long_dist_increase_speed_delta=5, dist_threshold=50, forward_speed=35, turning_speed=40, tolerance=4, compute_reference_box=False, angle_threshold=10, run_on_nano=False):

        self.cam = cv2.VideoCapture(3)
        self.communication_module = CommunicationModule.get_module(
            run_on_nano, ip, port)
        self.clear_motors()
        self.reference_bbox = Ballbbox(*ball_should_be_within)
        self.object_detection_module = ObjectDetection.get_object_detection_module(
            run_on_nano, model)
        self.distance_threshold = distance_threshold
        self.motor_vals = [0]*10
        self.lever_down_potentiometer_reading = lever_down_potentiometer_reading
        self.lever_up_potentiometer_reading = lever_up_potentiometer_reading
        self.forward_speed = forward_speed
        self.turning_speed = turning_speed
        self.tolerance = tolerance
        self.run_on_nano = run_on_nano
        self.dist_threshold = dist_threshold
        self.long_dist_increase_speed_delta = long_dist_increase_speed_delta
        self.april_tag_module = AprilTag(run_on_nano)
        self.angle_threshold = angle_threshold

        if compute_reference_box:
            self.reference_bbox = self.compute_reference_box()
            reference_bbox = self.reference_bbox
            print(
                f'Ref coordinates : {(reference_bbox.x1, reference_bbox.y1, reference_bbox.x2, reference_bbox.y2)}')

    def display(self, image, tgt_bbox, iou):
        reference_bbox = self.reference_bbox
        if tgt_bbox is not None:
            image = cv2.rectangle(
                image, (tgt_bbox.x1, tgt_bbox.y1), (tgt_bbox.x2, tgt_bbox.y2), (0, 0, 255), 1)
            image = cv2.circle(
                image, (tgt_bbox.midx, tgt_bbox.midy), 4, (0, 0, 255), -1)
        else:
            tgt_bbox = reference_bbox
        image = cv2.rectangle(image, (reference_bbox.x1, reference_bbox.y1),
                              (reference_bbox.x2, reference_bbox.y2), (0, 255, 0), 1)
        image = cv2.circle(image, (reference_bbox.midx,
                           reference_bbox.midy), 4, (0, 255, 0), -1)
        image = cv2.putText(image, f'IOU: {round(iou,2)}%, D: {tgt_bbox.depth_in_cm} cm', (
            tgt_bbox.x1, tgt_bbox.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow('POV', image)
        cv2.waitKey(1)

    def compute_reference_box(self):
        time.sleep(3)
        color_image, depth_image = self.get_camera_pic(usb_cam=True)
        if np.sum(color_image) == 0:
            print('Image is empty')
            sys.exit(0)
        balls_detected = self.object_detection_module.get_ball_detections(
            color_image, depth_image)
        return balls_detected[0]

    def get_lever_potentiometer_reading(self):
        sensor_readings = self.communication_module.get_sensor_readings()
        return sensor_readings[0]

    def get_claw_potentiometer_reading(self):
        sensor_readings = self.communication_module.get_sensor_readings()
        return sensor_readings[1]

    def get_camera_pic(self, usb_cam=False):
        if usb_cam:
            depth_image = None
            color_image = None
            ret, frame = self.cam.read()
            if ret is True:
                color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            color_image, depth_image = self.communication_module.get_camera_pic()
        return color_image, depth_image

    def send_motor_vals(self):
        self.communication_module.send_motor_vals(self.motor_vals)

    def clear_motors(self, claw_has_ball=False):
        self.motor_vals = [0]*10
        if claw_has_ball:
            self.motor_vals[7] = -20
        self.send_motor_vals()

    def move_forward(self, speed=None):
        if speed is None:
            speed = self.forward_speed
        self.motor_vals[0] = -speed  # Left motor (Negative -> Forward)
        self.motor_vals[9] = speed  # Right motor (Positive -> Forward)
        self.send_motor_vals()

    def move_backward(self, speed=None):
        if speed is None:
            speed = self.forward_speed
        self.motor_vals[0] = speed
        self.motor_vals[9] = -speed
        self.send_motor_vals()

    def turn_right(self, speed=None):
        if speed is None:
            speed = self.turning_speed
        self.motor_vals[0] = -speed
        self.motor_vals[9] = 0
        self.send_motor_vals()

    def turn_left(self, speed=None):
        if speed is None:
            speed = self.turning_speed
        self.motor_vals[0] = 0
        self.motor_vals[9] = speed
        self.send_motor_vals()

    def close_claw(self):
        self.clear_motors()
        time.sleep(1)
        self.motor_vals[7] = -50
        self.send_motor_vals()
        time.sleep(1.5)
        # TODO: Add check to see if ball is picked up (potentiometer ?)

    def open_claw(self):
        self.motor_vals[7] = 50
        self.send_motor_vals()
        time.sleep(1)
        self.clear_motors()

    def lever_down(self, claw_has_ball=False):
        # Need to give mild voltage to hold the ball intact
        self.clear_motors(claw_has_ball)
        # Beyond this gravity will bring it down
        while self.get_lever_potentiometer_reading() > self.lever_down_potentiometer_reading:
            self.motor_vals[6] = -40
            self.send_motor_vals()

    def lever_up(self, claw_has_ball=False):
        self.clear_motors(claw_has_ball)
        while self.get_lever_potentiometer_reading() < self.lever_up_potentiometer_reading:
            self.motor_vals[6] = 40
            self.send_motor_vals()
        self.clear_motors(claw_has_ball)

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

    def move_to_target_ball_bbox(self, current_target_bbox):
        color_image, _ = self.get_camera_pic(usb_cam=True)
        self.clear_motors()
        reference_bbox = self.reference_bbox
        # TODO : Maybe have some time constraint as well (1 min) if not sw
        iou = current_target_bbox.get_iou_with(reference_bbox)
        while iou < 0.80:
            # TODO: Should calculate this based on checking how far the ball can be within the reference frame
            # i.e how much tolerance on x and y plane
            # TODO : Can also add some speed based numbers
            if not self.run_on_nano:
                self.display(color_image, current_target_bbox, iou)
            tolerance = max(abs(current_target_bbox.midy -
                            reference_bbox.midy)*.1, self.tolerance)

            fspeed = None
            tspeed = None
            if current_target_bbox.depth_in_cm and current_target_bbox.depth_in_cm > self.distance_threshold:
                fspeed = self.forward_speed + self.long_dist_increase_speed_delta
                tspeed = self.turning_speed + self.long_dist_increase_speed_delta

            if abs(current_target_bbox.midx - reference_bbox.midx) < tolerance:
                self.move_backward(
                    fspeed) if current_target_bbox.midy > reference_bbox.midy else self.move_forward(fspeed)
            else:
                self.turn_left(
                    tspeed) if current_target_bbox.midx < reference_bbox.midx else self.turn_right(tspeed)

            color_image, depth_image = self.get_camera_pic(usb_cam=True)
            balls_detected = self.object_detection_module.get_ball_detections(
                color_image, depth_image)
            if len(balls_detected) == 0:
                return False

            # TODO : Or we should find the ball closest to the current target box (ie.) This will be like tracking
            # Reason : I am afraid if there are two balls right next to each other it might just be oscilatting
            # i.e when it turns right the right ball will become closer, then when it turns left the left ball might be closer
            current_target_bbox = self.get_target_ball_bbox(balls_detected)
            iou = current_target_bbox.get_iou_with(reference_bbox)

        self.clear_motors()
        print(
            f'Came out of the loop for :{(current_target_bbox.midx, current_target_bbox.midy)}')
        print(
            f'Reference coordinates    :{(reference_bbox.midx, reference_bbox.midy)}')
        print(f'IOU                      :{iou}')

        return True

    def pickup_ball(self):
        self.lever_down(claw_has_ball=False)
        self.close_claw()
        self.lever_up(claw_has_ball=True)
        return True  # TODO: Should return true or false based on pickup

    def almost_reached_wall(self, x, y, target_angle, current_angle):
        diff_in_angles = min((target_angle - current_angle) %
                             360, (current_angle - target_angle) % 360)
        target_close_to_wall = False
        if target_angle == 0 and x > 134:
            target_close_to_wall = True
        elif target_angle == 90 and y > 62:
            target_close_to_wall = True
        elif target_angle == 180 and x < 10:
            target_close_to_wall = True
        elif target_angle == 270 and y < 10:
            target_close_to_wall = True
        else:
            return False
        if target_close_to_wall and diff_in_angles < self.angle_threshold:
            return True
        else:
            return False

    def move_to_target_wall(self, target_angle):
        color_image, _ = self.get_camera_pic()
        x, y, current_angle = self.april_tag_module.get_position_and_rotation_of_camera(
            color_image)
        almost_reached_wall = self.almost_reached_wall(
            x, y, target_angle, current_angle)
        while not almost_reached_wall:
            # TODO Might have to make angle threshold small initally and increase it we get close tot he wall, doesn't matter.
            diff_in_angles = min((target_angle - current_angle) %
                                 360, (current_angle - target_angle) % 360)
            if diff_in_angles < self.angle_threshold:
                self.move_forward()
            elif target_angle in [0, 90, 180]:
                if current_angle > target_angle and current_angle < 180 + target_angle:
                    self.turn_right()
                else:
                    self.turn_left()
            else:
                if current_angle > 90 and current_angle < 270:
                    self.turn_left()
                else:
                    self.turn_right()

            color_image, _ = self.get_camera_pic()
            location = self.april_tag_module.get_position_and_rotation_of_camera(
                color_image)

            if location is not None:  # If its none, it will just use the previous values of x and y
                x, y, current_angle = location

            almost_reached_wall = self.almost_reached_wall(
                x, y, target_angle, current_angle)

        self.move_forward()
        time.sleep(2)  # Its okay if the bot hits the wall. Go little extra
        self.clear_motors

    def find_target_angle(self, x, y):
        """
            Angle : +x axis => 0 , +y axis => 90
           (0,72)                                       (144,72)
            -----------------------W2---------------------
            |                                            |
            |                                            |
            W1                                           W4
            |                                            |
            |                                            |
            -----------------------W0---------------------
           (0,0)                                        (144,0)
        """
        # TODO Can add weighting to give more importance to W0
        # w0, w1,   w2,    w3
        dist_to_walls = [y, x, 72 - y, 144 - x]
        min_idx = dist_to_walls.index(min(dist_to_walls))
        if min_idx == 0:
            return 270
        elif min_idx == 1:
            return 180
        elif min_idx == 2:
            return 90
        else:
            return 0

    def move_to_drop_off_location(self):
        color_image, _ = self.get_camera_pic()
        location = self.april_tag_module.get_position_and_rotation_of_camera(
            color_image)
        while location is None:
            self.random_walk_till_detection(
                detect_object="april_tag")
        x, y, _ = location
        angle = self.find_target_angle(x, y)
        self.move_to_target_wall(angle)

    def drop_ball_and_reset(self):
        self.lever_down(claw_has_ball=True)
        self.open_claw()
        self.lever_up()
        forward_time = time.time() + 2
        while time.time() < forward_time:
            self.move_backward()
        self.clear_motors()

    def random_walk_till_detection(self, detect_object="ball"):
        # Turn right for 5 seconds, then left for 5 seconds then move back for 2 seconds and repeat
        curr_time = time.time()
        while True:
            if detect_object == "ball":
                color_image, depth_image = self.get_camera_pic(usb_cam=True)
                if not self.run_on_nano:
                    self.display(color_image, None, 0)
                balls_detected = self.object_detection_module.get_ball_detections(
                    color_image, depth_image)
                if len(balls_detected) > 0:
                    return balls_detected
            else:
                color_image, _ = self.get_camera_pic()
                location = self.april_tag_module.get_position_and_rotation_of_camera(
                    color_image)
                if location is not None:
                    return location

            if time.time() < curr_time + 5:
                self.turn_right()
            elif time.time() < curr_time + 10:
                self.turn_left()
            else:
                self.move_backward()
                time.sleep(2)
                self.clear_motors()
                curr_time = time.time()

    def run(self):

        while True:
            balls_detected = self.random_walk_till_detection(
                detect_object="ball")

            target_bbox = self.get_target_ball_bbox(balls_detected)

            # If this target ball is too far out, we could do some more exploration
            if self.distance_threshold and target_bbox.depth_in_cm and target_bbox.depth_in_cm > self.distance_threshold:
                continue

            if not self.move_to_target_ball_bbox(target_bbox):
                continue

            if not self.pickup_ball():
                continue

            self.move_to_drop_off_location()

            self.drop_ball_and_reset()
