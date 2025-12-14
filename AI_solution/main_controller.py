import time
import math
import glob
from vision_functions import *


# ----- FUNCTIONS ----- #
#
#   1. scanner([yaw_min, yaw_step, yaw_max], [pitch_min, pitch_step, pitch_max], sleep_interval)
IMAGE_GLOB_PATTERN = "dataset/images/test/*.jpg"

image_paths = sorted(glob.glob(IMAGE_GLOB_PATTERN))
frame_index = 0  # over the image_paths list

def scanner(yaw_lims=[-45, 5, 45], pitch_lims=[-10, 5, 10], sleep_interval=0.02):
    MIN_PITCH = pitch_lims[0]
    MAX_PITCH = pitch_lims[2]
    PITCH_STEP = pitch_lims[1]

    # --- for each pitch angle, sweep yaw angles --- #
    direction = 1

    pitch = MIN_PITCH
    while (pitch <= MAX_PITCH):
        if direction == 1:
            START_YAW = yaw_lims[0]
            END_YAW = yaw_lims[2]
            YAW_STEP = yaw_lims[1]
        else:
            START_YAW = yaw_lims[2]
            END_YAW = yaw_lims[0]
            YAW_STEP = -1 * yaw_lims[1]

        yaw = START_YAW

        while not done_mini_sweep(yaw, END_YAW, YAW_STEP):

            # --- gets current frame --- #
            img_path = next_image_path()

            if img_path is None:
                print("No more images available. Stopping search.")
                return False, None, None, None, None

            print(f"  Using image: {img_path}")
            print("yaw step: " + str(YAW_STEP) + ", current yaw: " + str(yaw))
            print("--- detecting @ pitch = " + str(pitch) + ", yaw = " + str(yaw))

            roll, crop_img = main_integration(img_path)
            if roll is not None:
                print("\n>>> CROSS DETECTED <<<")
                print(f"Barrel yaw:   {yaw:.2f}°")
                print(f"Barrel pitch: {pitch:.2f}°")
                print(f"Cross roll:   {roll:.2f}°")

                return True, yaw, pitch, roll, crop_img

            print("  -> No cross detected in this image. Continuing scan.")

            time.sleep(sleep_interval)
            yaw = yaw + YAW_STEP


        direction *= -1
        pitch = pitch + PITCH_STEP

    # finished all yaw/pitch combinations, no detection
    print("\nNo cross found in the defined yaw/pitch search space.")
    return False, None, None, None, None

def next_image_path():
    global frame_index

    if frame_index >= len(image_paths):
        return None

    path = image_paths[frame_index]
    frame_index += 1
    return path

def done_mini_sweep(x, end, step):
    return x > end if step > 0 else x < end


def main():
    if len(image_paths) == 0:
        print(f"No images found with pattern: {IMAGE_GLOB_PATTERN}")
        return

    print(f"Loaded {len(image_paths)} images for test search.")

    found, yaw_deg, pitch_deg, roll_deg, crop_img = scanner()

    if not found:
        print("\nSearch finished: cross NOT found.")
    else:
        print("\nSearch finished: cross FOUND.")
        print(f"Final yaw:   {yaw_deg:.2f}°")
        print(f"Final pitch: {pitch_deg:.2f}°")
        print(f"Cross roll:  {roll_deg:.2f}°")

        # Optional visualization of the cropped cross
        if crop_img is not None:
            cv2.imshow("Detected cross crop", crop_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
