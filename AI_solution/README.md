# Cross Detection System & Synthetic Dataset Generator


##  Overview

his project simulates a camera attached to a tank's barrel. 
The tank can move its barrel in a roll-pitch-yaw coordinates. 
The code detects a cross using YOLO v8 DNN trained on a custom generated dataset.
After the detection, the roll angle of the cross is estimated using Hough lines and the cross is "fixed" into its straightened version. 
The result is the straight cross image, saved as 'RESULTED_DETECTION.png' and the printout of the roll-pitch-yaw angles inwhich the cross was found. 

---

## Project Structure

```
project/
│
├── train.py                  # trains YOLO v8 on the custom dataset created.
├── generate_dataset.py       # Custom dataset generator (online BG image + cross shape)
├── vision_functions.py       # Cross detection + tilt angle estimation
├── main_controller.py        # Yaw–pitch scan and computer vision integration
├── TEMPLATE.png              # Basic cross template
├── runs/                     # YOLO training results
├── dataset/                  # Generated training/validation/test sets
│   ├── images/
│   └── labels/
└── README.md                 # Project documentation
```

---

## Synthetic Dataset Generator

The dataset generator produces YOLO‑compatible data containing:

- **Positive samples** → a randomized cross appears in each image  
- **Negative samples** → clean background images with *no cross*  
- **Randomized transformations**
  - Rotation  
  - Tilt/shear  
  - Scaling  
  - Morphological variation  
  - Random placement  
  - Real backgrounds downloaded online

---

## vision_functions.py
Using your trained YOLO model (`best.pt`), the system:

1. Locates the cross inside the image  
2. Extracts its bounding‑box crop  
3. Computes the **roll angle** using Hough‑lines  
4. Returns:  
   ```
   roll_angle ('CROSS_ROLL'), cropped_cross_image ('crop')
   ```

---

## main_controller.py

A simplified simulation of a tank turret scanning for a target:

- Performs **yaw sweeps** (left→right)
- Performs **pitch sweeps** (down→up)
- At each step:
  1. Load a test set's frame  
  2. Run the cross detection and angle estimation
  3. If cross found → STOP  
  4. Return:
     - yaw
     - pitch
     - detected roll angle  
  5. Save the resulted crop image  

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics (YOLO)

Install:

```
pip install opencv-python numpy ultralytics
```

---

## Running the Dataset Generator

```
python generate_dataset.py
```

This creates a full YOLO‑ready dataset inside the `dataset/` directory.

---
## Training the NN 

```aiignore
python train.py
```
## Running the Algorithm

```
python main_controller.py
```





## Notes

- Background‑only images are majority in the test set to evaluate false‑positive robustness.  
- Cross geometry is planar, so orientation estimation is reliable without camera intrinsics.  
- The testings are dependent on the order of the images in the test set. In order for the user to test on another order the images need to be shuffled manually (at this moment).

---

## License

This project is open for personal, academic and research use.


