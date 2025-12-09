# ----------------------------------------#
#               IMPORTS                   #
# ----------------------------------------#

import cv2
from ultralytics import YOLO
import random
import numpy as np

#----------------------------------------#
#           HELPER FUNCTIONS             #
#   1. def cross_orientation_angle(crop) #
#   2. def straighten_cross(crop, angle) #
#----------------------------------------#

def cross_orientation_angle_hough(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=20, maxLineGap=5)
    if lines is None:
        return None, mask

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # normalize to [0, 180)
        angle = angle % 180.0
        angles.append(angle)

    if not angles:
        return None, mask

    # reduce 0–180 symmetry by folding to [0,90] (cross has 2 main directions)
    angles_folded = [a if a <= 90 else a-90 for a in angles]

    # take median to be robust
    angle_main = np.median(angles_folded)

    return angle_main, mask

def straighten_cross(crop, angle):
    if angle is None:
        return crop, None

    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

    straight = cv2.warpAffine(
        crop, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0)
    )

    return straight, angle

#----------------------------------------#
#       MAIN INTEGRATION FUNCTION        #
#----------------------------------------#
def main_integration(img_path):
    # --- CHOOSE BEST MODEL --- #
    model = YOLO("runs/detect/train/weights/best.pt")

    # --- TEST ON RANDOM TEST SET IMAGE --- #
    results = model.predict(img_path, conf=0.25)

    for result in results:

        # --- DETECT --- #
        print("Detection:\n")
        print(result.boxes.xyxy)

        original = result.orig_img
        boxes = result.boxes

        if boxes is None or len(boxes)==0:
            print("X-X-X-   no detections found   -X-X-X")
            return None, None

        # --- CHOOSE BEST DETECTION --- #
        best_detection_index = boxes.conf.argmax().item()
        best_result = boxes[best_detection_index]

        x1, y1, x2, y2 = best_result.xyxy[0].int().tolist()

        # --- CROP ROI OF DETECTION --- #
        H, W = original.shape[:2]
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))

        crop = original[y1:y2, x1:x2].copy()

        cv2.imshow("Best Detection Crop W/" + str(str(best_result.conf[0])) + " CONFIDENCE", crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    template = cv2.imread("TEMPLATE.png")
    original_orientation, template_mask = cross_orientation_angle_hough(template)
    crop_orientation, crop_mask = cross_orientation_angle_hough(crop)

    print ("OG orientation = " + str(original_orientation))
    print("crop's orientation = " + str(crop_orientation-original_orientation))

    straight, angle = straighten_cross(crop, (crop_orientation-original_orientation+45))

    cv2.imshow("straighten", straight)
    cv2.imwrite("RESULTED_DETECTION.png", straight)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    CROSS_ROLL = angle
    print("CROSS ROLL = " + str(CROSS_ROLL))
    print("however, orientation = " + str(crop_orientation-original_orientation))

    return CROSS_ROLL, crop


