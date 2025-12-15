import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt


os.makedirs("dummydata", exist_ok=True)

H, W = 30, 90
ION_SIZE = 10

BG_MEAN = 400
BG_STD  = 20

BRIGHT_PEAK = 6000
BRIGHT_STD  = 100

POINT_PER_100 = 20
LINE_PER_100  = 10

FLICKER_PROB = 0.8
FLICKER_AMP_STD = 0.25


# =============================
# (A) Drift / motion model
# =============================

# global slow drift state (random walk)
def macromotion():
    dx = np.random.uniform(-2.0, 2.0)
    dy = np.random.uniform(-2.0, 2.0)
    return dx, dy


def micromotion():
    dx = np.random.normal(0, 0.25)
    dy = np.random.normal(0, 0.25)
    return dx, dy

# =============================
# (B) Image primitives
# =============================

def make_background():
    img = np.random.normal(BG_MEAN, BG_STD, (H, W)).astype(np.float32)

    lowf = np.random.normal(0, 5, (H, W)).astype(np.float32)
    lowf = cv2.GaussianBlur(lowf, (19,19), 7.0)
    return img + lowf


def add_prnu(img):
    prnu = np.random.normal(1.0, 0.002, img.shape).astype(np.float32)
    return img * prnu


def draw_bright_ion(img, cx, cy, scale=1.0):
    r = ION_SIZE // 2
    sigma = 2.0 + np.random.normal(0, 0.2)  # PSF blur jitter

    for dy in range(-r, r):
        for dx in range(-r, r):
            yy = int(round(cy + dy))
            xx = int(round(cx + dx))
            if 0 <= yy < H and 0 <= xx < W:
                dist = np.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
                peak = np.random.normal(BRIGHT_PEAK, BRIGHT_STD)
                grain = np.random.normal(0, 50 + 150 * dist)
                img[yy, xx] += scale * (peak * dist + grain)

    return img


def add_read_noise(img):
    return img + np.random.normal(0, 5, img.shape).astype(np.float32)


def add_point_noise(img):
    y = np.random.randint(0, H)
    x = np.random.randint(0, W)
    img[y, x] += 20000
    return img


def add_line_noise(img):
    if np.random.rand() < 0.5:
        y = np.random.randint(0, H)
        img[y, :] += 20000
    else:
        x = np.random.randint(0, W)
        img[:, x] += 20000
    return img


# =============================
# (C) 20-frame base generation
# =============================

def generate_20_images(base_index):

    # nominal ion positions
    cx1_nom, cy_nom = 20, 15
    cx2_nom = 70

    # quantum state
    state1 = np.random.randint(0, 2)
    state2 = np.random.randint(0, 2)

    with open(f"dummydata/state_{base_index:03d}.json", "w") as f:
        json.dump({"s1": state1, "s2": state2}, f)

    # base-level slow drift
    macro_dx, macro_dy = macromotion()
    # noise frame selection
    point_frames = set(np.random.choice(20, POINT_PER_100//5, replace=False))
    line_frames  = set(np.random.choice(20, LINE_PER_100//5, replace=False))

    for k in range(20):

        img = make_background()

        # frame-level micromotion jitter
        jdx, jdy = micromotion()

        cx1 = cx1_nom + macro_dx + jdx
        cy1 = cy_nom  + macro_dy + jdy
        cx2 = cx2_nom + macro_dx + jdx
        cy2 = cy_nom  + macro_dy + jdy


        if state1 == 1 and np.random.rand() < FLICKER_PROB:
            scale1 = max(np.random.normal(1.0, FLICKER_AMP_STD), 0.0)
            img = draw_bright_ion(img, cx1, cy1, scale1)

        if state2 == 1 and np.random.rand() < FLICKER_PROB:
            scale2 = max(np.random.normal(1.0, FLICKER_AMP_STD), 0.0)
            img = draw_bright_ion(img, cx2, cy2, scale2)

        img = add_prnu(img)
        img = add_read_noise(img)

        if k in point_frames:
            img = add_point_noise(img)
        if k in line_frames:
            img = add_line_noise(img)

        img = np.clip(img, 0, 65535).astype(np.uint16)
        np.save(f"dummydata/img_{base_index:03d}_{k:02d}.npy", img)

        # visualization preview
        if base_index < 5 and k == 0:
            vis = img.astype(np.float32) - BG_MEAN
            vis[vis < 0] = 0
            vis = vis ** 0.4
            vis = (vis / (vis.max() + 1e-6) * 255).astype(np.uint8)
            cv2.imwrite(f"dummydata/img_{base_index:03d}_{k:02d}.png", vis)

    return state1, state2


# =============================
# (D) Dataset generation
# =============================

N = 200
for i in range(N):
    st = generate_20_images(i)
    print(f"Base {i}: state={st}")

print("DONE generating dummy data.")
