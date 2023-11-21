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
            self, ip="10.0.0.122", ball_should_be_within=(258, 367, 361, 472),
            model="yolov8s.pt", port=9999, distance_threshold=200,
            lever_down_potentiometer_reading=3550, lever_up_potentiometer_reading=3680,
            long_dist_increase_speed_delta=7, dist_threshold=50, forward_speed=43, turning_speed=37, tolerance=5, compute_reference_box=False, angle_threshold=10, run_on_nano=True):

        self.cam = None
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
        self.switch_cam(type="usb")
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
        image = cv2.putText(image, f'IOU: {round(0,2)}%, D: {tgt_bbox.depth_in_cm} cm', (
            tgt_bbox.x1, tgt_bbox.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow('POV', image)
        cv2.waitKey(1)

    def compute_reference_box(self):
        time.sleep(1)
        color_image, depth_image = self.get_camera_pic(usb_cam=True)
        from PIL import Image
        Image.fromarray(color_image).save('check.jpeg')
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

    def switch_cam(self, type="usb"):
        if type == "usb":
            if self.communication_module.cam.pipeline:
                self.communication_module.cam.pipeline.stop()
                self.communication_module.cam.pipeline = None
            self.cam = cv2.VideoCapture(0)
        else:
            if self.cam:
                self.cam.release()
                self.cam = None
            self.communication_module.cam.open()

    def get_camera_pic(self, usb_cam=False):
        if usb_cam:
            if self.cam is None:
                self.switch_cam(type="usb")
            depth_image = None
            color_image = None
            ret, frame = self.cam.read()
            if ret is True:
                color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            if self.communication_module.cam.pipeline is None:
                self.switch_cam("realsense")
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

    def turn_right(self, speed=None, multiplication_factor=0):
        if speed is None:
            speed = self.turning_speed
        speed = speed + 5  # Right side turning is slow so we add this
        self.motor_vals[0] = -speed
        self.motor_vals[9] = -int(multiplication_factor * speed)
        self.send_motor_vals()

    def turn_left(self, speed=None, multiplication_factor=0):
        if speed is None:
            speed = self.turning_speed
        self.motor_vals[0] = int(multiplication_factor * speed)
        self.motor_vals[9] = speed
        self.send_motor_vals()

    def close_claw(self):
        self.clear_motors()
        self.motor_vals[7] = -50
        self.send_motor_vals()
        time.sleep(1.2)

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
            self.motor_vals[6] = -25
            self.send_motor_vals()

    def lever_up(self, claw_has_ball=False):
        self.clear_motors(claw_has_ball)
        while self.get_lever_potentiometer_reading() < self.lever_up_potentiometer_reading:
            self.motor_vals[6] = 75
            self.send_motor_vals()
        self.clear_motors(claw_has_ball)

    def get_target_ball_bbox(self, balls_detected):
        target_bbox = None
        min_dist = None
        for ball_bbox in balls_detected:
            dist = ball_bbox.get_distance_to(self.reference_bbox)
            if min_dist is None or dist < min_dist:
                target_bbox = ball_bbox
                min_dist = dist
        return target_bbox

    def move_to_target_ball_bbox(self, current_target_bbox: Ballbbox):
        color_image, _ = self.get_camera_pic(usb_cam=True)
        self.clear_motors()
        reference_bbox = self.reference_bbox
        in_pickup_range = current_target_bbox.get_proximity_to(reference_bbox)
        start_time = time.time()
        while not in_pickup_range:
            if time.time() > start_time + 45:  # Avoid that ball after 45 seconds and do random stuff
                print(
                    'AFTER 45 SECONDS COULD NOT PICKUP BALL, SO MOVING BACK AND SPINNING LEFT AND STARTING OVER')
                self.move_backward()
                time.sleep(1)
                self.clear_motors()
                self.turn_left(speed=80, multiplication_factor=1)
                time.sleep(1)
                self.clear_motors()
                return False

            if not self.run_on_nano:
                self.display(color_image, current_target_bbox, 0)

            tolerance = max(abs(current_target_bbox.midy -
                            reference_bbox.midy)*.1, self.tolerance)

            if abs(current_target_bbox.midx - reference_bbox.midx) < tolerance:
                fspeed = None
                if abs(current_target_bbox.midy - reference_bbox.midy) > 20:
                    fspeed = self.forward_speed + self.long_dist_increase_speed_delta
                self.move_backward(
                    fspeed) if current_target_bbox.midy > reference_bbox.midy else self.move_forward(fspeed)
            else:
                self.turn_left() if current_target_bbox.midx < reference_bbox.midx else self.turn_right()

            color_image, depth_image = self.get_camera_pic(usb_cam=True)
            balls_detected = self.object_detection_module.get_ball_detections(
                color_image, depth_image)

            if len(balls_detected) == 0:
                return False

            current_target_bbox = self.get_target_ball_bbox(balls_detected)
            in_pickup_range = current_target_bbox.get_proximity_to(
                reference_bbox)

        print(
            f'Came out of the loop for :{(current_target_bbox.midx, current_target_bbox.midy)}')
        print(
            f'Reference coordinates    :{(reference_bbox.midx, reference_bbox.midy)}')

        self.move_forward(speed=40)
        time.sleep(0.5)
        self.clear_motors()
        return True

    def is_bot_at_edge(self):
        location = self.get_current_location()
        print(f'BOT LOCATION AFTER PICKUP OF BALL {location}')
        x, y, angle = location
        if x < 25 or x > 120 or y < 25 or y > 48:
            return True, location
        else:
            return False, location

    # TODO : Write this logic properly
    def handle_bot_at_edge(self, location):
        x, y, current_angle = location
        if (
            x < 20 and (current_angle > 340 or current_angle < 20) or
            x > 124 and (current_angle > 160 and current_angle < 200) or
            y < 20 and (current_angle > 70 and current_angle < 110) or
            y > 52 and (current_angle > 250 and current_angle < 290)
        ):
            self.move_backward()
            time.sleep(0.7)

    def pickup_ball(self):
        self.close_claw()
        if self.get_claw_potentiometer_reading() > 2450:
            print('WARNING : BALL WAS NOT GRABBED')
            self.move_backward()
            time.sleep(0.5)
            self.clear_motors()
            self.reset_to_initial_state()
            return False

        bot_at_edge, location = self.is_bot_at_edge()

        if bot_at_edge:
            print('BOT IS AT THE EDGE')
            self.handle_bot_at_edge(location)

        self.lever_up(claw_has_ball=True)
        if self.get_claw_potentiometer_reading() > 2450:  # Ball was dropped in some way when lever was up
            print('BALL WAS DROPPED AFTER LEVER UP. RESETTING')
            self.reset_to_initial_state()
            return False

        return True

    def almost_reached_wall(self, y, current_angle, max_time):
        if time.time() > max_time:
            print('MAX TIME REACHED FOR DROP OFF BALL. RESETTING')
            return True
        # TODO This was 14
        if y < 18 and (current_angle > 70 and current_angle < 110):
            print(f'REACHED TARGET WALL {y},{current_angle}')
            return True
        else:
            return False

    def move_to_target_wall(self):
        location = self.get_current_location()
        x, y, current_angle = location
        max_time = time.time() + 15

        almost_reached_wall = self.almost_reached_wall(
            y, current_angle, max_time)

        while not almost_reached_wall:
            if current_angle > 80 and current_angle < 100:
                self.move_forward(speed=70)
            elif current_angle > 270 or current_angle < 90:
                # TODO : Increased the speed. Check if this is okay or old one was okay.
                self.turn_left(multiplication_factor=0.5,
                               speed=self.turning_speed+5)
            else:
                self.turn_right(multiplication_factor=0.5,
                                speed=self.turning_speed+5)

            color_image, _ = self.get_camera_pic()
            location = self.april_tag_module.get_position_and_rotation_of_camera(
                color_image)

            if location is not None:  # If its none, it will just use the previous values of x and y
                x, y, current_angle = location

            almost_reached_wall = self.almost_reached_wall(
                y, current_angle, max_time)

        self.move_forward()
        time.sleep(1)  # Its okay if the bot hits the wall. Go little extra
        self.clear_motors()

    def drop_ball_and_reset(self):
        self.lever_down(claw_has_ball=True)
        self.open_claw()
        self.lever_up()
        backward_time = time.time() + 1.5
        while time.time() < backward_time:
            self.move_backward(speed=80)
        self.lever_down()
        self.clear_motors()
        # TODO : Can get location here and do something if needed

    def get_current_location(self):
        location = None
        color_image, _ = self.get_camera_pic(usb_cam=False)
        start_time = time.time()
        while location is None:
            if time.time() < start_time + 4:
                # TODO : Should check if this speed detection works
                self.turn_left(multiplication_factor=1)
            elif time.time() < start_time + 8:
                self.turn_right(multiplication_factor=1)
            else:
                self.move_backward()
                start_time = time.time()
            location = self.april_tag_module.get_position_and_rotation_of_camera(
                color_image)

        return location

    # TODO : Code this properly
    def random_walk_till_ball_detection(self):
        start_time = time.time()
        # This will make it look for april tags so might be a problem. Check if its needed always
        # x, y, angle = self.get_current_location()
        while True:
            color_image, depth_image = self.get_camera_pic(usb_cam=True)
            if not self.run_on_nano:
                self.display(color_image, None, 0)
            balls_detected = self.object_detection_module.get_ball_detections(
                color_image, depth_image)
            if len(balls_detected) > 0:
                return balls_detected

            if time.time() < start_time + 2:
                self.turn_left(multiplication_factor=1, speed=80)
            elif time.time() < start_time + 4:
                self.turn_right(multiplication_factor=1, speed=80)
            else:
                self.move_forward()
                time.sleep(2)
                self.clear_motors()
                curr_time = time.time()

    def reset_to_initial_state(self):
        self.clear_motors()
        if self.get_lever_potentiometer_reading() > 3000:
            self.lever_down()
        if self.get_claw_potentiometer_reading() > 10 and self.get_claw_potentiometer_reading() < 3800:
            self.open_claw()

    def run(self):
        print('RESTTING TO INITIAL STATE')
        self.reset_to_initial_state()

        while True:
            print('RANDOM WALK TILL BALL DETECTION')
            balls_detected = self.random_walk_till_ball_detection()

            print('MOVING TO  PICKUP BALL')
            target_bbox = self.get_target_ball_bbox(balls_detected)

            if not self.move_to_target_ball_bbox(target_bbox):
                print('COULD NOT MOVE TO BALL, RESETTING')
                continue

            print('PICKING UP BALL')
            if not self.pickup_ball():
                print('COULD NOT PICK UP BALL, RESETTING')
                continue

            print('MOVING TO TARGET WALL')
            self.move_to_target_wall()

            print('DROPPING BALL AND RESETTING')
            self.drop_ball_and_reset()
