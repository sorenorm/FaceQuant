import numpy as np
import pandas as pd

class faceQuant():
    def __init__(self, frame = None):
        if frame is not None:
            self.frame = frame

    def set_frame(self, frame):
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
    
    def get_face_quant(self, frame = None):
        if frame is not None:
            self.set_frame(frame)
        return [self.get_eye_opening(), self.get_ear_opening(), self.get_ear_position(), self.get_snout_position(), self.get_mouth_position(), self.get_face_inclination()]

class FaceQuantVec:
    def __init__(self, df):
        self.df = df

    def _dist(self, p1, p2):
        x1 = self.df[(p1, "x")]
        y1 = self.df[(p1, "y")]
        x2 = self.df[(p2, "x")]
        y2 = self.df[(p2, "y")]
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def get_opening(self, prefix):
        length = self._dist(f"{prefix} back", f"{prefix} front")
        width  = self._dist(f"{prefix} top",  f"{prefix} bottom")
        return width / length

    def get_angle(self, bp1, bp2, bp3):
        A = np.stack([self.df[(bp1, "x")], self.df[(bp1, "y")]], axis=1)
        B = np.stack([self.df[(bp2, "x")], self.df[(bp2, "y")]], axis=1)
        C = np.stack([self.df[(bp3, "x")], self.df[(bp3, "y")]], axis=1)

        BA = A - B
        BC = C - B

        cos_angle = np.sum(BA * BC, axis=1) / (
            np.linalg.norm(BA, axis=1) * np.linalg.norm(BC, axis=1)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.degrees(np.arccos(cos_angle))

    def compute(self):
        return pd.DataFrame({
            "eye_opening":  self.get_opening("eye"),
            "ear_opening":  self.get_opening("ear"),
            "ear_position": self.get_angle("ear top", "ear bottom", "eye front"),
            "snout_position": self.get_angle("nose bottom", "nose top", "eye front"),
            "mouth_position": self.get_angle("mouth", "eye front", "nose bottom"),
            "face_inclination": self.get_angle("ear bottom", "eye front", "nose bottom") - 90
        })
