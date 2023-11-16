from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy
from .ball_bbox import Ballbbox


class ObjectDetectionMobilenet:
    def __init__(self):
        self.model = detectNet("ssd-mobilenet-v2", threshold=0.1)

    def get_ball_detections(self, color_image, depth_image):
        cuda_image = cudaFromNumpy(color_image)
        results = self.model.Detect(cuda_image)
        balls_detected = []
        for result in results:
            if result.ClassID == 37:
                (x1, y1, x2, y2) = int(result.Left), int(
                    result.Top), int(result.Right), int(result.Bottom)
                midx, midy = result.Center
                depth_in_cm = None
                if depth_image is not None:
                    depth_in_cm = depth_image[int(midy), int(midx)]/10
                balls_detected.append(Ballbbox(x1, y1, x2, y2, depth_in_cm))
        return balls_detected
