import cv2
from robot import Robot
import time
import sys

if __name__ == "__main__":
  r = Robot(run_on_nano=sys.argv[1])
  time.sleep(2)
  r.run()
