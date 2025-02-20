from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Tuple
from PIL import Image, ImageDraw
import math
from enum import Enum

# 0: Nose
# 1: Left Eye ,2: Right Eye
# 3: Left Ear ,4: Right Ear
# 5: Left Shoulder ,6: Right Shoulder
# 7: Left Elbow ,8: Right Elbow
# 9: Left Wrist ,10: Right Wrist
# 11: Left Hip ,12: Right Hip
# 13: Left Knee ,14: Right Knee
# 15: Left Ankle ,16: Right Ankle

class KeypointNames(str, Enum):
    NOSE = "Burun"
    L_EYE = "Sol Göz"
    R_EYE = "Sağ Göz"
    L_EAR = "Sol Kulak"
    R_EAR = "Sağ Kulak"
    L_SHOULDER = "Sol Omuz"
    R_SHOULDER = "Sağ Omuz"
    L_ELBOW = "Sol Dirsek"
    R_ELBOW = "Sağ Dirsek"
    L_WRIST = "Sol Bilek"
    R_WRIST = "Sağ Bilek"
    L_HIP = "Sol Kalça"
    R_HIP = "Sağ Kalça"
    L_KNEE = "Sol Diz"
    R_KNEE = "Sağ Diz"
    L_ANKLE = "Sol Ayak Bileği"
    R_ANKLE = "Sağ Ayak Bileği"
    NECK = "Kafa"


class PostureAnalyzer:
    def __init__(self):
        self.model = YOLO("yolov8x-pose.pt")

        self.perspectives = {
            "front": [(0, 1), (1, 3), (0, 2), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)],
            "back": [(3, 3), (4, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)],
            "right": [(0, 2), (2, 4), (6, 8), (8, 10), (12, 14), (14, 16)],
            "left": [(0, 1), (1, 3), (5, 7), (7, 9), (11, 13), (13, 15)]
        }

        self.angle_dict = {
            "front": [],
            "back": [],
            "right": [],
            "left": []
        }

        self.keypoints = None

    @staticmethod
    def draw_keypoints(perspective: List[Tuple], image: np.ndarray, keypoints: List[Tuple]) -> np.ndarray:
        if perspective not in ["right",  "left"]:
            for kp1, kp2 in perspective:
                x1, y1, conf1 = int(keypoints[kp1][0]), int(keypoints[kp1][1]), keypoints[kp1][2]
                x2, y2, conf2 = int(keypoints[kp2][0]), int(keypoints[kp2][1]), keypoints[kp2][2]
                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(image, (x1, y1), (x2, y2), (230, 230, 230), 8)

        keypoints_ = list(set(x for pair in perspective for x in pair))
        for i in keypoints_:
            x, y, conf = int(keypoints[i][0]), int(keypoints[i][1]), keypoints[i][2]
            if conf > 0.5:
                cv2.circle(image, (x, y), 20, (50, 201, 255), -1)

        return image

    def analyze_sides(self, image: Image, direction: str, height: int, img_np: np.ndarray):
        ankle_x = self.keypoints[16][0] if direction == "right" else self.keypoints[15][0]
        ankle_y = self.keypoints[16][1] if direction == "right" else self.keypoints[15][1]
        ear_x = self.keypoints[4][0] if direction == "right" else self.keypoints[3][0]
        ear_y = self.keypoints[4][1] if direction == "right" else self.keypoints[3][1]
        shoulder_x = self.keypoints[6][0] if direction == "right" else self.keypoints[5][0]
        shoulder_y = self.keypoints[6][1] if direction == "right" else self.keypoints[5][1]
        knee_x = self.keypoints[14][0] if direction == "right" else self.keypoints[13][0]
        knee_y = self.keypoints[14][1] if direction == "right" else self.keypoints[13][1]
        hip_x = self.keypoints[12][0] if direction == "right" else self.keypoints[11][0]
        hip_y = self.keypoints[12][1] if direction == "right" else self.keypoints[11][1]

        # PELVISTEN AYAK BILEGINE VE KULAK MEMESINE CIZGI
        self.drawline((hip_x, hip_y), (ankle_x, ankle_y), "b")
        self.drawline((hip_x, hip_y), (ear_x, ear_y), "b")

        # KALCADAN REFERANS CIZGISI
        self.drawline((hip_x, 0), (hip_x, height), "y")

        # AYAKTAN REFERANS CIZGISI
        self.drawline((ankle_x, image.size[1]), (ankle_x, 0), "p")

        # NECK
        neck_reference = (hip_x, ear_y)
        neck_angle = self.calculate_angles((ear_x, ear_y), neck_reference, (hip_x, hip_y))
        self.angle_dict[direction].append({
            "angle": neck_angle,
            "coord": [float(ear_x), float(ear_y)],
            "name": KeypointNames.NECK
        })
        self.drawline(neck_reference, (ear_x, ear_y), "o")

        # SHOULDERS
        shoulder_reference = (hip_x, shoulder_y)
        shoulder_angle = self.calculate_angles((shoulder_x, shoulder_y), shoulder_reference, (hip_x, hip_y))
        self.angle_dict[direction].append({
            "angle": shoulder_angle,
            "coord": [float(shoulder_x), float(shoulder_y)],
            "name": KeypointNames.R_SHOULDER if direction == "right" else KeypointNames.L_SHOULDER
        })
        self.drawline(shoulder_reference, (shoulder_x, shoulder_y), "g")
        self.drawline((hip_x, hip_y), (shoulder_x, shoulder_y), "g")


        # KNEES
        knee_reference = (hip_x, knee_y)
        knee_angle = self.calculate_angles((knee_x, knee_y), knee_reference, (hip_x, hip_y))
        self.angle_dict[direction].append({
            "angle": knee_angle,
            "coord": [float(knee_x), float(knee_y)],
            "name": KeypointNames.L_KNEE if direction == "left" else KeypointNames.R_KNEE
        })

        # ANKLES
        foot_reference = (hip_x, ankle_y)
        foot_degree = self.calculate_angles((ankle_x, ankle_y), foot_reference, (hip_x, hip_y))
        self.angle_dict[direction].append({
            "angle": foot_degree,
            "coord": [float(ankle_x), float(ankle_y)],
            "name": KeypointNames.L_ANKLE if direction == "left" else KeypointNames.R_ANKLE
        })

        return image

    def analyze_front(self, image: Image):
        nose_x = self.keypoints[0][0]
        nose_y = self.keypoints[0][1]
        left_ear_x = self.keypoints[3][0]
        left_ear_y = self.keypoints[3][1]
        right_ear_x = self.keypoints[4][0]
        right_ear_y = self.keypoints[4][1]
        left_shoulder_x = self.keypoints[5][0]
        left_shoulder_y = self.keypoints[5][1]
        right_shoulder_x = self.keypoints[6][0]
        right_shoulder_y = self.keypoints[6][1]
        left_elbow_x = self.keypoints[7][0]
        left_elbow_y = self.keypoints[7][1]
        right_elbow_x = self.keypoints[8][0]
        right_elbow_y = self.keypoints[8][1]
        left_hip_x = self.keypoints[11][0]
        left_hip_y = self.keypoints[11][1]
        right_hip_x = self.keypoints[12][0]
        right_hip_y = self.keypoints[12][1]
        left_knee_x = self.keypoints[13][0]
        left_knee_y = self.keypoints[13][1]
        right_knee_x = self.keypoints[14][0]
        right_knee_y = self.keypoints[14][1]
        left_ankle_x = self.keypoints[15][0]
        left_ankle_y = self.keypoints[15][1]
        right_ankle_x = self.keypoints[16][0]
        right_ankle_y = self.keypoints[16][1]

        middle_head_x = right_ear_x - (right_ear_x - left_ear_x) / 2
        middle_head_y = right_ear_y + (left_ear_y - right_ear_y) / 2
        middle_foot_x = left_ankle_x - (left_ankle_x - right_ankle_x) / 2
        middle_foot_y = left_ankle_y + (left_ankle_y - right_ankle_y) / 2

        y_diff = middle_foot_y - middle_head_y
        x_diff = middle_foot_x - middle_head_x
        opposite = middle_head_y

        slope = math.atan2(y_diff, x_diff)
        adjacent = opposite / math.tan(slope)

        self.drawline((middle_head_x-adjacent, 0), (middle_foot_x+adjacent, image.size[1]), "g") # Vucudu ortalayan cizgi
        self.drawline((middle_foot_x, image.size[1]), (middle_foot_x, 0), 'b') # Resmi ortalayan referans cizgisi

        # SHOULDERS
        self.drawline((nose_x, nose_y), (left_shoulder_x, left_shoulder_y), "r")
        self.drawline((nose_x, nose_y), (right_shoulder_x, right_shoulder_y), "r")
        self.drawline((left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), "r")
        shoulder_angle = self.cosinus_theorem((nose_x, nose_y),
                                              (left_shoulder_x, left_shoulder_y),
                                              (right_shoulder_x, right_shoulder_y))

        self.angle_dict["front"].append({
            "angle": shoulder_angle,
            "coord": [float(left_shoulder_x), float(left_shoulder_y)],
            "name": KeypointNames.L_SHOULDER})

        # HIPS
        self.drawline((middle_foot_x, middle_foot_y), (left_hip_x, left_hip_y), "r")
        self.drawline((middle_foot_x, middle_foot_y), (right_hip_x, right_hip_y), "r")
        self.drawline((left_hip_x, left_hip_y), (right_hip_x, right_hip_y), "r")
        hip_angle = self.cosinus_theorem((middle_foot_x, middle_foot_y),
                                         (left_hip_x, left_hip_y),
                                         (right_hip_x, right_hip_y))
        self.angle_dict["front"].append({
            "angle": hip_angle,
            "coord": [float(left_hip_x), float(left_hip_y)],
            "name": KeypointNames.L_HIP})

        # FOOTS
        self.drawline((middle_foot_x, middle_foot_y), (left_ankle_x, left_ankle_y), "r")
        self.drawline((middle_foot_x, middle_foot_y), (right_ankle_x, right_ankle_y), "r")
        foot_angle = self.cosinus_theorem((middle_foot_x, middle_foot_y),
                                          (left_ankle_x, left_ankle_y),
                                          (right_ankle_x, right_ankle_y))
        self.angle_dict["front"].append({
            "angle": foot_angle,
            "coord": [float(left_ankle_x), float(left_ankle_y)],
            "name": KeypointNames.L_ANKLE
        })

        # ELBOWS
        elbow_angle = self.cosinus_theorem((middle_head_x, middle_head_y),
                                           (left_elbow_x, left_elbow_y),
                                           (right_elbow_x, right_elbow_y))
        self.angle_dict["front"].append({
            "angle": elbow_angle,
            "coord": [float(right_elbow_x), float(right_elbow_y)],
            "name": KeypointNames.R_ELBOW
        })

        # KNEES
        knee_angle = self.cosinus_theorem((middle_foot_x, middle_foot_y),
                                          (left_knee_x, left_knee_y),
                                          (right_knee_x, right_knee_y))
        self.angle_dict["front"].append({
            "angle": knee_angle,
            "coord": [float(right_knee_x), float(right_knee_y)],
            "name": KeypointNames.R_KNEE
        })
        return image

    def analyze_back(self, image: Image):
        left_ear_x = self.keypoints[3][0]
        left_ear_y = self.keypoints[3][1]
        right_ear_x = self.keypoints[4][0]
        right_ear_y = self.keypoints[4][1]
        left_shoulder_x = self.keypoints[5][0]
        left_shoulder_y = self.keypoints[5][1]
        right_shoulder_x = self.keypoints[6][0]
        right_shoulder_y = self.keypoints[6][1]
        left_elbow_x = self.keypoints[7][0]
        left_elbow_y = self.keypoints[7][1]
        right_elbow_x = self.keypoints[8][0]
        right_elbow_y = self.keypoints[8][1]
        left_hip_x = self.keypoints[11][0]
        left_hip_y = self.keypoints[11][1]
        right_hip_x = self.keypoints[12][0]
        right_hip_y = self.keypoints[12][1]
        left_knee_x = self.keypoints[13][0]
        left_knee_y = self.keypoints[13][1]
        right_knee_x = self.keypoints[14][0]
        right_knee_y = self.keypoints[14][1]
        left_ankle_x = self.keypoints[15][0]
        left_ankle_y = self.keypoints[15][1]
        right_ankle_x = self.keypoints[16][0]
        right_ankle_y = self.keypoints[16][1]

        middle_head_x = right_ear_x - (right_ear_x - left_ear_x) / 2
        middle_head_y = right_ear_y + (left_ear_y - right_ear_y) / 2
        middle_foot_x = left_ankle_x - (left_ankle_x - right_ankle_x) / 2
        middle_foot_y = left_ankle_y + (left_ankle_y - right_ankle_y) / 2

        y_diff = middle_foot_y - middle_head_y
        x_diff = middle_foot_x - middle_head_x
        opposite = middle_head_y

        slope = math.atan2(y_diff, x_diff)
        adjacent = opposite / math.tan(slope)

        self.drawline((middle_head_x-adjacent, 0), (middle_foot_x+adjacent, image.size[1]), "g") # Vucudu ortalayan cizgi
        self.drawline((middle_foot_x, image.size[1]), (middle_foot_x, 0), 'b') # Resmi ortalayn referans cizgisi

        # OMUZLAR
        self.drawline((left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), "r") # Omuzlar arasi cizgi
        self.drawline((middle_head_x, middle_head_y), (left_shoulder_x, left_shoulder_y), "r")
        self.drawline((middle_head_x, middle_head_y), (right_shoulder_x, right_shoulder_y), "r")
        shoulder_angle = self.cosinus_theorem((middle_head_x, middle_head_y),
                                               (left_shoulder_x, left_shoulder_y),
                                               (right_shoulder_x, right_shoulder_y))
        self.angle_dict["back"].append({
            "angle": shoulder_angle,
            "coord": [float(right_shoulder_x), float(right_shoulder_y)],
            "name": KeypointNames.R_SHOULDER
        })

        # HIPS
        self.drawline((middle_head_x, middle_head_y), (left_hip_x, left_hip_y), "r")
        self.drawline((middle_head_x, middle_head_y), (right_hip_x, right_hip_y), "r")
        self.drawline((left_hip_x, left_hip_y), (right_hip_x, right_hip_y), "r")
        hip_angle = self.cosinus_theorem((middle_head_x, middle_head_y),
                                         (left_hip_x, left_hip_y),
                                         (right_hip_x, right_hip_y))
        self.angle_dict["back"].append({
            "angle": hip_angle,
            "coord": [float(right_hip_x), float(right_hip_y)],
            "name": KeypointNames.R_HIP
        })

        # KNEES
        self.drawline((middle_foot_x, middle_foot_y), (left_knee_x, left_knee_y), "r")
        self.drawline((middle_foot_x, middle_foot_y), (right_knee_x, right_knee_y), "r")
        self.drawline((left_knee_x, left_knee_y), (right_knee_x, right_knee_y), "r")
        knee_angle = self.cosinus_theorem((middle_foot_x, middle_foot_y),
                                          (left_knee_x, left_knee_y),
                                          (right_knee_x, right_knee_y))
        self.angle_dict["back"].append({
            "angle": knee_angle,
            "coord": [float(left_knee_x), float(left_knee_y)],
            "name": KeypointNames.L_KNEE
        })

        # FOOTS
        self.drawline((middle_foot_x, middle_foot_y), (left_ankle_x, left_ankle_y), "r")
        self.drawline((middle_foot_x, middle_foot_y), (right_ankle_x, right_ankle_y), "r")
        foot_angle = self.cosinus_theorem((middle_foot_x, middle_foot_y),
                                          (left_ankle_x, left_ankle_y),
                                          (right_ankle_x, right_ankle_y))
        self.angle_dict["back"].append({
            "angle": foot_angle,
            "coord": [float(right_ankle_x), float(right_ankle_y)],
            "name": KeypointNames.R_ANKLE
        })

        # ELBOWS
        elbow_angle = self.cosinus_theorem((middle_head_x, middle_head_y),
                                           (left_elbow_x, left_elbow_y),
                                           (right_elbow_x, right_elbow_y))
        self.angle_dict["back"].append({
            "angle": elbow_angle,
            "coord": [float(left_elbow_x), float(left_elbow_y)],
            "name": KeypointNames.L_ELBOW
        })

        return image

    def drawline(self, point_a, point_b, color: str, width=10):
        if color == "r":
            color = (0, 0, 255)
        if color == "g":
            color = (0, 255, 0)
        if color == "b":
            color = (255, 0, 0)
        if color == "y":
            color = (0, 255, 255)
        if color == "o":
            color = (0, 180, 255)
        if color == "p":
            color = (255, 0, 127)

        point_a = (int(point_a[0]), int(point_a[1]))
        point_b = (int(point_b[0]), int(point_b[1]))

        self.draw.line([point_a, point_b], fill=color, width=width)

    @staticmethod
    def calculate_angles(point_ac: Tuple, point_bc: Tuple, point_ab: Tuple) -> float:
        """
        point_ac: hesaplamasi yapilacak bolgenin koordinatlari
        point_bc: hesaplama icin referans alinan noktanin koordinatlari
        point_ab: hesaplamasi yapilacak bolgenin duz postur cizgisindeki noktanin koordinatlari
        """
        leg_a = abs(point_ac[0] - point_ab[0])
        leg_b = abs(point_bc[1] - point_ab[1])

        angle = math.degrees(math.atan(leg_a / leg_b))
        angle = float(angle)

        # kenarlar cok kisaysa aci da 90a yakin olur
        if angle > 85:
            return 0
        return round(angle, 2)

    @staticmethod
    def calculate_edge_length(point_a: Tuple, point_b: Tuple) -> float:
        """
        kenar uzunluk hesaplama
        """
        distance = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
        return distance

    def cosinus_theorem(self, point_a: Tuple, point_b: Tuple, point_c: Tuple) -> float:
        """
        A ustteki B soldaki C sagdaki nokta
        """
        leg_ab = self.calculate_edge_length(point_a, point_b)
        leg_ac = self.calculate_edge_length(point_a, point_c)
        leg_bc = self.calculate_edge_length(point_b, point_c)

        radian_b = math.acos((leg_ab**2 + leg_bc**2 - leg_ac**2) / (2 * leg_ab * leg_bc))
        radian_c = math.acos((leg_ac**2 + leg_bc**2 - leg_ab**2) / (2 * leg_ac * leg_bc))

        angle_b = math.degrees(radian_b)
        angle_c = math.degrees(radian_c)

        diff = abs(angle_b - angle_c)

        return round(diff, 2)

    def analyze(self, image, image_np, perspective):
        result = self.model(image)
        self.keypoints = result[0].keypoints.data[0].cpu().numpy()

        image_np = self.draw_keypoints(self.perspectives[perspective], image_np, self.keypoints)

        height = image_np.shape[0]
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_np)
        self.draw = ImageDraw.Draw(image)

        if perspective in ["right", "left"]:
            image = self.analyze_sides(image, perspective, height, image_np)
        elif perspective == "front":
            image = self.analyze_front(image)
        elif perspective == "back":
            image = self.analyze_back(image)

        return image, self.angle_dict[perspective]
