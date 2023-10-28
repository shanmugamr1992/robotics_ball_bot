from .tags import Tags
import cv2
import numpy as np

class AprilTag:
    def __init__(self, run_on_nano=False):
        if run_on_nano:
            from dt_apriltags import Detector
        else:
            from pyapriltags import Detector
        self.detector = Detector(families='tag16h5',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)
        fx = 460.92495728   # FOV(x) -> depth2xyz -> focal length (x)
        fy = 460.85058594   # FOV(y) -> depth2xyz -> focal length (y)
        cx = 315.10949707   # 640 (width) 
        cy = 176.72598267   # 320 (height)
        
        self.camera_params = ( fx, fy, cx, cy )
        self.tag_size = 3.0
        self.tag_poses = Tags.tags

    def detect_tags(self, color_image):
        tags = self.detector.detect(
            img=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY),
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size)
        
        tags = [tag for tag in tags if tag.decision_margin > 30 and tag.tag_id in self.tag_poses.keys()]
        return tags  

    def Rt2T(self, R, t):
      T = np.eye(4, dtype=np.float32)
      T[:3, :3] = R
      T[:3, 3:] = t
      return T    


    def localize(self, tags):
      # use the formula present in the video, just with the first tag
      if len(tags) == 0: return None
      #TODO Can also sort and get the one with highest margin or a weighted average
      tag = tags[0] #
      T_camera_apriltag = self.Rt2T(tag.pose_R, tag.pose_t)
      T_apriltag_camera = np.linalg.inv(T_camera_apriltag)
        
      T_map_apriltag = self.tag_poses[tag.tag_id]
      
      T_map_camera = T_map_apriltag @ T_apriltag_camera
      return T_map_camera

    def get_position_and_rotation_of_camera(self, color_image):
        tags = self.detect_tags(color_image)
        T_matrix = self.localize(tags)
        if T_matrix is not None: 
            x, y = T_matrix[0, 3], T_matrix[1, 3]
            Tinv = np.linalg.inv(T_matrix)
            yaw = np.degrees(np.arctan2(Tinv[2,1],Tinv[2,0]))
            #yaw = Rotation.from_matrix(T_matrix[:3, :3]).as_euler("zyx", degrees=True)[0]
            return (x,y,yaw)
        else:
            return None