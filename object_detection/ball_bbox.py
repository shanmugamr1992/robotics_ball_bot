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

    def get_iou_with(self, bbox):
    	xA = max(self.x1, bbox.x1)
    	yA = max(self.y1, bbox.y1)
    	xB = min(self.x2, bbox.x2)
    	yB = min(self.y2, bbox.y2)
    	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    	boxAArea = (self.x2 - self.x1 + 1) * (self.y2 - self.y1 + 1)
    	boxBArea = (bbox.x2 - bbox.x1 + 1) * (bbox.y2 - bbox.y1 + 1)        
    	iou = interArea / float(boxAArea + boxBArea - interArea)
    	return iou
        