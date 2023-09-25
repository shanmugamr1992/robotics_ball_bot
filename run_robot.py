import cv2
from robot import Robot
import time

if __name__ == "__main__":
  r = Robot(run_on_nano=False)
  time.sleep(4)
  r.run()
