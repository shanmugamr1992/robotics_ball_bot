import math


class Ballbbox:
    def __init__(self, x1, y1, x2, y2, depth_in_cm=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.midx = int((x1+x2)/2)
        self.midy = int((y1+y2)/2)
        self.depth_in_cm = depth_in_cm

    def get_distance_to(self, bbox):
        return math.sqrt((self.midx - bbox.midx)**2 + (self.midy - bbox.midy)**2)

    def get_proximity_to(self, bbox):
        diffy = abs(self.midy - bbox.midy)
        return diffy < 10
