import numpy as np

class faceQuant():
    def __init__(self, frame):
        self.frame = frame

    def _dist(self, p1, p2):
        x1, y1 = self.frame[(p1, "x")], self.frame[(p1, "y")]
        x2, y2 = self.frame[(p2, "x")], self.frame[(p2, "y")]
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def get_opening(self, prefix):
        length = self._dist(f"{prefix} back", f"{prefix} front")
        width = self._dist(f"{prefix} top", f"{prefix} bottom")
        return width/length

    def get_eye_opening(self):
        return self.get_opening("eye")

    def get_ear_opening(self):
        return self.get_opening("ear")
    
    def get_angle(self, bp1, bp2, bp3):
        A = np.array([self.frame[(bp1, "x")],
                      self.frame[(bp1, "y")]])

        B = np.array([self.frame[(bp2, "x")],
                      self.frame[(bp2, "y")]])

        C = np.array([self.frame[(bp3, "x")],
                      self.frame[(bp3, "y")]])

        BA = A - B
        BC = C - B

        cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.degrees(np.arccos(cos_angle))
    
    def get_ear_position(self):
        return self.get_angle("ear top", "ear bottom", "eye front")

    def get_snout_position(self):
        return self.get_angle("nose bottom", "nose top", "eye front")
    
    def get_mouth_position(self):
        return self.get_angle("mouth", "eye front", "nose bottom")
    
    def get_face_inclination(self):
        return self.get_angle("ear bottom", "eye front", "nose bottom")-90
    
    def get_face_quant(self):
        return [self.get_eye_opening(), self.get_ear_opening(), self.get_ear_position(), self.get_snout_position(), self.get_mouth_position(), self.get_face_inclination()]
    