
class ObjectDetection:

    @classmethod
    def get_object_detection_module(cls, run_on_nano = False, model = None):
        if run_on_nano:
            from .object_detection_mobilenet import ObjectDetectionMobilenet
            return ObjectDetectionMobilenet()
        else :
            from .object_detection_yolo import ObjectDetectionYOLO
            return ObjectDetectionYOLO(model)