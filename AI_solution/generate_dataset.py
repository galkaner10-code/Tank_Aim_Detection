import numpy as np
import cv2
import requests
import random
import os


# ---------------------------------------------------------
#  SAME GENERATOR FUNCTIONS (unchanged)
# ---------------------------------------------------------

def generate_cross(size, thickness, arm_length):
    template = np.zeros((size, size, 4), np.uint8)
    # horizontal
    x1 = size//2 - int(arm_length * size)
    y1 = size//2 - int(thickness*size)//2
    x2 = size//2 + int(arm_length * size)
    y2 = size//2 + int(thickness*size)//2
    template[y1:y2, x1:x2] = (255,255,255,255)
    # vertical
    y1 = size//2 - int(arm_length * size)
    x1 = size//2 - int(thickness*size)//2
    y2 = size//2 + int(arm_length * size)
    x2 = size//2 + int(thickness*size)//2
    template[y1:y2, x1:x2] = (255,255,255,255)
    return template


def rotate_image(image, angle):
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def resize_image(image, scale):
    newW = max(1, int(image.shape[1] * scale))
    newH = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (newW,newH), interpolation=cv2.INTER_AREA)


def random_tilt_rgba(image, max_shear=0.35):
    h, w = image.shape[:2]
    shx = random.uniform(-max_shear, max_shear)
    shy = random.uniform(-max_shear, max_shear)
    cx, cy = w/2, h/2
    M = np.array([[1.0, shx, -shx*cy],
                  [shy, 1.0, -shy*cx]], dtype=np.float32)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0,0,0,0))


def import_backgrounds_from_web(w, h, retries=4):
    url = f"https://picsum.photos/{w}/{h}"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=10)
            arr = np.frombuffer(r.content, np.uint8)
            bg = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bg is not None:
                return bg
        except:
            pass
    # fallback synthetic
    return np.random.randint(0,255,(h,w,3),np.uint8)


def place_rgba(background, template, cx, cy):
    H, W = background.shape[:2]
    th, tw = template.shape[:2]
    x1 = cx - tw//2
    y1 = cy - th//2

    bg_x1 = max(0, x1)
    bg_y1 = max(0, y1)
    bg_x2 = min(W, x1 + tw)
    bg_y2 = min(H, y1 + th)

    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return background, None

    tx1 = bg_x1 - x1
    ty1 = bg_y1 - y1
    tx2 = tx1 + (bg_x2 - bg_x1)
    ty2 = ty1 + (bg_y2 - bg_y1)

    tmpl_crop = template[ty1:ty2, tx1:tx2]
    alpha = tmpl_crop[:,:,3:4] / 255.0
    region = background[bg_y1:bg_y2, bg_x1:bg_x2]

    blended = (1-alpha)*region + alpha*tmpl_crop[:,:,:3]
    background[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype(np.uint8)

    mask = template[:,:,3] > 20
    ys, xs = np.where(mask)
    if len(xs)==0:
        return background, None

    t_xmin, t_xmax = xs.min(), xs.max()
    t_ymin, t_ymax = ys.min(), ys.max()

    b_xmin = np.clip(x1 + t_xmin, 0, W-1)
    b_xmax = np.clip(x1 + t_xmax, 0, W-1)
    b_ymin = np.clip(y1 + t_ymin, 0, H-1)
    b_ymax = np.clip(y1 + t_ymax, 0, H-1)

    xc = (b_xmin + b_xmax) / (2*W)
    yc = (b_ymin + b_ymax) / (2*H)
    bw = (b_xmax - b_xmin) / W
    bh = (b_ymax - b_ymin) / H

    return background, (xc, yc, bw, bh)



# ---------------------------------------------------------
# NEW DATASET GENERATOR
# ---------------------------------------------------------

def generate_dataset(
    out="dataset",
    n_train=300, n_val=60,
    n_test_cross=20,        # FEW test images with cross
    n_test_bg=200,          # MANY background-only test images
    min_size=400, max_size=1200,
    base_cross_size=512
):

    # create dirs
    for d in ["train", "val", "test"]:
        os.makedirs(f"{out}/images/{d}", exist_ok=True)
        os.makedirs(f"{out}/labels/{d}", exist_ok=True)

    print("Creating base cross...")
    base_cross = generate_cross(base_cross_size, 0.15, 0.45)
    cv2.imwrite("TEMPLATE.png", base_cross)

    # ------------------------------
    # FUNCTION: cross image creator
    # ------------------------------
    def make_cross_image(split, img_id):
        W = random.randint(min_size, max_size)
        H = random.randint(min_size, max_size)
        bg = import_backgrounds_from_web(W, H)

        tmpl = base_cross.copy()
        if random.random() < 0.7:
            tmpl = generate_cross(
                base_cross_size,
                thickness=random.uniform(0.10,0.22),
                arm_length=random.uniform(0.35,0.50)
            )

        tmpl = rotate_image(tmpl, random.uniform(0,360))
        tmpl = random_tilt_rgba(tmpl)
        tmpl = resize_image(tmpl, random.uniform(0.2,0.8))

        cx = random.randint(0, W-1)
        cy = random.randint(0, H-1)

        final_img, bbox = place_rgba(bg, tmpl, cx, cy)
        if bbox is None:
            return False

        cv2.imwrite(f"{out}/images/{split}/{img_id}.jpg", final_img)

        xc, yc, w, h = bbox
        with open(f"{out}/labels/{split}/{img_id}.txt","w") as f:
            f.write(f"0 {xc} {yc} {w} {h}\n")

        return True

    # ------------------------------
    # FUNCTION: background-only creator
    # ------------------------------
    def make_background_image(split, img_id):
        W = random.randint(min_size, max_size)
        H = random.randint(min_size, max_size)
        bg = import_backgrounds_from_web(W, H)

        cv2.imwrite(f"{out}/images/{split}/{img_id}.jpg", bg)
        open(f"{out}/labels/{split}/{img_id}.txt","w").close()  # empty label file

    # ------------------------------
    # GENERATE TRAIN (cross only)
    # ------------------------------
    print("TRAIN SPLIT...")
    img_id = 0
    while img_id < n_train:
        if make_cross_image("train", img_id):
            img_id += 1

    # ------------------------------
    # GENERATE VAL (cross only)
    # ------------------------------
    print("VAL SPLIT...")
    img_id = 0
    while img_id < n_val:
        if make_cross_image("val", img_id):
            img_id += 1

    # ------------------------------
    # GENERATE TEST = cross few + many backgrounds
    # ------------------------------
    print("TEST SPLIT...")
    img_id = 0

    # few cross images
    made = 0
    while made < n_test_cross:
        if make_cross_image("test", img_id):
            img_id += 1
            made += 1

    # many backgrounds
    for _ in range(n_test_bg):
        make_background_image("test", img_id)
        img_id += 1

    print("\n✔ DONE – dataset ready!")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

generate_dataset(
    out="dataset",
    n_train=300,
    n_val=60,
    n_test_cross=20,   # FEW test cross images
    n_test_bg=200,     # MANY bg-only test images
)
