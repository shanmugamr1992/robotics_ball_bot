import cv2
from robot import Robot
import time
import sys

if __name__ == "__main__":
  if sys.argv[1] == "True":
    run_on_nano=True
  else:
    run_on_nano=False

  r = Robot(run_on_nano=run_on_nano,ip="10.0.0.119")
  time.sleep(2)
  r.run()
