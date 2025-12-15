import numpy as np
import cv2
import os
import json
import glob
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


###########################################
# (A) Drift 관련 함수 
###########################################

# 세트(base_index) 사이 slow drift (1~5 pixel 정도)
def slow_drift_offset():
    dx = np.random.uniform(-1, 1)     # realistic slow drift
    dy = np.random.uniform(-1, 1)
    return dx, dy

# 20 프레임 내부 fast jitter (0.3 pixel 이하)
def fast_jitter():
    dx = np.random.normal(0, 0.3)     # micromotion + readout jitter
    dy = np.random.normal(0, 0.3)
    return dx, dy

# subpixel shift
def shift_image(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )



###########################################
# (B) 기존 이미지 생성 함수들 
###########################################

def make_background():
    HF_STD = 10
    img = np.random.normal(BG_MEAN, HF_STD, (H, W)).astype(np.float32)

    LF_STD = 5
    lowf = np.random.normal(0, LF_STD, (H, W)).astype(np.float32)
    lowf = cv2.GaussianBlur(lowf, (19,19), 7.0)

    return img + lowf


def add_prnu(img):
    prnu = np.random.normal(1.0, 0.002, img.shape).astype(np.float32)
    return img * prnu


def draw_bright_ion(img, cx, cy, scale=1.0):
    r = ION_SIZE // 2
    sigma = 2.0 + np.random.uniform(-0.2, 0.2)

    for dy in range(-r, r):
        for dx in range(-r, r):
            yy = cy + dy
            xx = cx + dx
            if 0 <= yy < H and 0 <= xx < W:
                dist = np.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
                peak = np.random.normal(BRIGHT_PEAK, BRIGHT_STD)

                noise_strength = 20 + 200 * dist
                grain = np.random.normal(0, noise_strength)

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



###########################################
# (C) 20-프레임 생성 (drift 포함)
###########################################

def generate_20_images(base_index):

    cx1, cy = 20, 15
    cx2 = 70

    state1 = np.random.randint(0, 2)
    state2 = np.random.randint(0, 2)

    # JSON 저장
    with open(f"dummydata/state_{base_index:03d}.json", "w") as f:
        json.dump({"s1": state1, "s2": state2}, f)

    # 이 세트에 대한 slow drift (모든 20프레임에 동일하게 적용)
    slow_dx, slow_dy = slow_drift_offset()

    # 20프레임 중 노이즈 넣는 곳
    point_frames = set(np.random.choice(20, POINT_PER_100//5, replace=False))
    line_frames  = set(np.random.choice(20, LINE_PER_100//5, replace=False))

    for k in range(20):

        img = make_background()

        # 이온 그리기
        if state1 == 1 and np.random.rand() < FLICKER_PROB:
            scale1 = np.random.normal(1.0, FLICKER_AMP_STD)
            scale1 = max(scale1, 0.0)   # 음수 방지
            img = draw_bright_ion(img, cx1, cy, scale=scale1)

        if state2 == 1 and np.random.rand() < FLICKER_PROB:
            scale2 = np.random.normal(1.0, FLICKER_AMP_STD)
            scale2 = max(scale2, 0.0)
            img = draw_bright_ion(img, cx2, cy, scale=scale2)

        # frame-to-frame jitter
        jitter_dx, jitter_dy = fast_jitter()

        # slow drift + jitter 합산
        total_dx = slow_dx + jitter_dx
        total_dy = slow_dy + jitter_dy

        # drift 적용
        img = shift_image(img, total_dx, total_dy)

        img = add_prnu(img)
        img = add_read_noise(img)

        if k in point_frames:
            img = add_point_noise(img)
        if k in line_frames:
            img = add_line_noise(img)

        img = np.clip(img, 0, 65535).astype(np.uint16)
        np.save(f"dummydata/img_{base_index:03d}_{k:02d}.npy", img)

        # PNG 저장 유지
        if base_index < 10 and k == 0:
            vis = img.astype(np.float32) - BG_MEAN
            vis[vis < 0] = 0
            vis = vis ** 0.4
            vis = (vis / (vis.max() + 1e-6) * 255).astype(np.uint8)
            cv2.imwrite(f"dummydata/img_{base_index:03d}_{k:02d}.png", vis)

    return (state1, state2)



###########################################
# (D) 실제 데이터 생성 루프
###########################################

N = 200
for i in range(N):
    st = generate_20_images(i)
    print(f"Base {i}: state={st}")

print("Done generating drift-included data.")



###########################################
# (E) 시각화 함수들 (원본 그대로!)
###########################################

def show_npy_abs(img, vmin=300, vmax=1700):
    plt.figure(figsize=(6,3))
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Absolute-scale view")
    plt.show()


def preview_png(base_index):
    raw_path = f"dummydata/img_{base_index:03d}_00.npy"
    if not os.path.exists(raw_path):
        print("RAW not found:", raw_path)
        return

    img = np.load(raw_path)

    plt.figure(figsize=(6,3))
    plt.imshow(img, cmap='gray', vmin=300, vmax=1700)
    plt.colorbar()
    plt.title(f"Absolute-scale view (base {base_index})")
    plt.show()


def preview_mean_brightness(base_index):
    means = []
    for k in range(20):
        path = f"dummydata/img_{base_index:03d}_{k:02d}.npy"
        if os.path.exists(path):
            img = np.load(path)
            means.append(img.mean())

    plt.figure(figsize=(5,3))
    plt.hist(means, bins=10, color='blue')
    plt.xlabel("Mean brightness")
    plt.ylabel("Count")
    plt.title(f"Histogram of 20-frame mean brightness (base {base_index})")
    plt.show()
    


path = "dummydata/img_000_00.npy"

img = np.load(path)

print("Shape:", img.shape)
print("Dtype:", img.dtype)
# print("Contents:")
print(img)