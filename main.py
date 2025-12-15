################################################## IMPORT ##################################################
import numpy as np
import os
import json
from glob import glob

# sklearn module
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # 로지스틱
from sklearn.svm import SVC                         # SVM
from sklearn.neighbors import KNeighborsClassifier  # K-NN
from sklearn.metrics import f1_score                # 평가 방법
from sklearn.model_selection import GroupShuffleSplit

############################################################################################################








############################################# Params Setting ###############################################

# 이온 trap 위치 initial setting [ 90 * 30 image ]
ion1_x = 20
ion2_x = 70
ion_y = 15

# drift 고려한 crop에 쓰일 파라미터
ROI_SIZE = 16     # crop 사이즈
SEARCH_RADIUS = 2 # drift 탐색 범위 (2  pixel 오차 가능) - ??? micromotion 어느정도인지?
CENTER_PATCH = 10 # 밝기 평가용 패치 크기 (10x10, ion size랑 맞춤)
# patch 10 안 하면 1개의 노이즈에 취약해서 10 * 10 평균 밝기로 존재 추정

#########################################################################################################








############################################# Function Define #############################################


# 0 1 2를 이온 state로 변환, dictionary에서 key에 해당하는 값을 찾아서 출력
def class_to_str(key):
    mapping = {0: "00", 1: "10", 2: "11"}
    result = mapping[key]
    return result


# 편의를 위해 { 00 -> 0, 01 / 10 -> 1, 11 -> 2 } 로 mapping
def state_to_class(a, b):
    if a == 0 and b == 0:
        return 0
    if a == 1 and b == 1:
        return 2
    return 1   # 01 or 10


# 밝기 -> 숫자 (경계값으로 구분하는 방식)
def brightness_to_label(brightness_mean):
    if brightness_mean < 500:
        return 0
    elif brightness_mean < 1500:
        return 1
    else:           # 1500 이상
        return 2


# brightness_to_label에서 판단한 라벨로 state(00, 10, 11)를 str으로 저장
def combine_frame_labels(a, b):
    if a == 2 and b == 2:   # 둘 다 1500 이상이면 11(2)
        return 2
    if a >= 1 or b >= 1:    # 하나라도 조금 밝으면 10 (1) (1대신 2를 넣으면 threshold 정확도 0.8)
        return 1            
    return 0                # 해당되지 않으면 00 (0)


# drift 때문에 발생하는 center 이동 감지
def find_local_center(img, pred_x, pred_y, search_radius=SEARCH_RADIUS, patch_size=CENTER_PATCH):
    half_p = patch_size // 2    # 중심 기준 크롭 위해서 반 계산. 상하좌우 5pixel 정도

    best_x, best_y = pred_x, pred_y # 현재까지의 값 중 가장 그럴듯한 값 대입
    best_mean = -1.0                # 초기값 설정

    # x y -5 ~ 5만큼 주변 스캔하여 이온 중심 예측하기
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            # -2부터 예측하기
            mod_x = pred_x + dx
            mod_y = pred_y + dy

            x0 = mod_x - half_p
            x1 = mod_x + half_p
            y0 = mod_y - half_p
            y1 = mod_y + half_p

            # 위에서 얻은 좌표로 img slicing해서 밝기 평균 구하기, 가장 높은 지점이 best slicing
            patch = img[y0:y1, x0:x1]
            m = patch.mean()
            if m > best_mean:
                best_mean = m
                best_x, best_y = mod_x, mod_y
            # 연산 끝나면 다음 차례로 돌아가기
    return best_x, best_y


# find_local_center에서 찾은 좌표를 기준으로 이온 배열 데이터 저장
def crop_roi(img, x_pred, y_pred, size=ROI_SIZE):
    half_p = size // 2

    # 좌표 적용
    x_adjust, y_adjust = find_local_center(img, x_pred, y_pred)

    y0 = y_adjust - half_p
    y1 = y_adjust + half_p
    x0 = x_adjust - half_p
    x1 = x_adjust + half_p

    roi = img[y0:y1, x0:x1]
    if roi.shape != (size, size):
        return None
    return roi


# 20번 state predict한 값 hist에 넣고 max인 구간으로 판단 - 실제 실험에서는 약 100번
def final_state_20frame(labels):
    hist = np.bincount(labels, minlength=3)
    return np.argmax(hist)


# ML에 맞게 데이터 변환 - 정확도 향상의 핵심 (non-ML 방식은 이 과정 없이 모두 테스트 후 hist로 판단)
def load_ML_dataset():
    # ML 학습 위해 X, Y list 만들기
    feature = []
    label = []
    base_ids = []

    # 해당 패턴의 모든 파일들을 리스트 형식으로 저장
    states = glob("dummydata/state_*.json")

    # dummydata 폴더에서 true state 값이 담긴 json과 밝기 분포 배열 담긴 npy를 읽어서 
    # feature 벡터와 GT label을 생성
    for js in states:
        base_idx = int(os.path.basename(js)[6:9]) # index 6 ~ 8까지 불러옴 ex) 010, 011, 012, ...
        frame_features = []   # base 안의 20 frame 내용 저장
        
        with open(js, "r") as f:
            statefile = json.load(f )

        # state 00 01 10 11을 class 0 1 2로 저장
        groundtruth_class = state_to_class(statefile["s1"], statefile["s2"])

        
        # ChatGPT에게 ML model이 학습할 수 있도록 데이터를 변환해달라고 요청함.
        # 아래 for 문은 모두 ChatGPT의 도움을 받음
        for k in range(20):
            path = f"dummydata/img_{base_idx:03d}_{k:02d}.npy"
            if not os.path.exists(path):
                continue

            img = np.load(path)
            
            # 이온 중심 기준 크롭
            roi_ion1 = crop_roi(img, ion1_x, ion_y)
            roi_ion2 = crop_roi(img, ion2_x, ion_y)

            # ML model에 고정 길이 벡터를 입력해야 하므로, ROI(2D)를 1D로 flatten해서 ion1, ion2를 이어붙임 -> 두 이온의 패턴을 함께 보고 분류함
            feature_vec = np.concatenate([roi_ion1.flatten(), roi_ion2.flatten()])

            # 해당 frame의 ion1 + ion2 공간 패턴을 base 단위의 배열에 저장
            frame_features.append(feature_vec)


        # 20 frame이 정상적으로 모였을 때만 사용
        if len(frame_features) == 20:
            # frame 단위 noise와 flicker은 평균으로 남기고 ??????????? base level state 정보만 남김 
            base_feature = np.mean(frame_features, axis=0)
            feature.append(base_feature)    # ML X 정보
            label.append(groundtruth_class) # ML Y 정보
            base_ids.append(base_idx)       # 해당 feature가 어느 base에서 왔는지 (연산 시 같은 base에서 중복 연산 하지 않도록)

    return np.array(feature), np.array(label), np.array(base_ids)


# load_ML_dataset에서 변환한 데이터를 ML에 학습시킴
def train_ML_models():
    feature, label, base_ids = load_ML_dataset()

    print("Loaded ML dataset:", feature.shape, label.shape)

    if len(feature) == 0:
        print("[ERROR]: data가 로드되지 않음")
        return None

    X_train, _, y_train, _ = train_test_split(feature, label, test_size=0.25, random_state=0, stratify=label)

    models = {
        "LR": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="rbf", probability=True),
        "kNN": KNeighborsClassifier(3)
    }
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        print(f"{name} trained.")

    print("\nML training COMPLETE!\n")
    return models


# ChatGPT에게 모델 별 accuracy를 table로 출력하는 방법을 물어보고 해당 형식을 사용함.
# 모델 별 accuracy 및 f1-score 출력
def evaluate_final_accuracy(models):
    print("=========== FINAL 20-FRAME RESULTS (TABLE) ===========\n")
    # 결과값을 배열에 저장
    gt_all = []
    thr_all = []
    lr_all = []
    svm_all = []
    knn_all = []

    # 얼마나 correct 했는지 count
    correct_thr = 0
    correct_lr  = 0
    correct_svm = 0
    correct_knn = 0
    total = 200

    # 표 형식
    row_fmt = "{:<4} | {:<4} | {:<4} | {:<4} | {:<4} | {:<4} | {:<3} | {:<3} | {:<3} | {:<3}"

    # 헤더 출력
    print(row_fmt.format("Base", "GT", "Thr", "LR", "SVM", "kNN", "Thr", "LR", "SVM", "kNN"))
    print("-" * 70)

    # total 200번 동안 예측한 것, O인지 X인지 출력
    for base in range(total):
        with open(f"dummydata/state_{base:03d}.json", "r") as f:
            st = json.load(f)
        gt_state = state_to_class(st["s1"], st["s2"])

        thr_labels = []
        ml_labels_lr  = []
        ml_labels_svm = []
        ml_labels_knn = []

        for frame_index in range(20):
            img = np.load(f"dummydata/img_{base:03d}_{frame_index:02d}.npy")
            # 크기 조정
            roi_ion1 = crop_roi(img, ion1_x, ion_y)
            roi_ion2 = crop_roi(img, ion2_x, ion_y)
            
            # 밝기 평균 계산 - non-ML 방식
            ion1_centermean = roi_ion1[3:13, 3:13].mean()
            ion2_centermean = roi_ion2[3:13, 3:13].mean()
            fl1= brightness_to_label(ion1_centermean)
            fl2 = brightness_to_label(ion2_centermean)
            thr_labels.append(combine_frame_labels(fl1, fl2))

            # ML 방식
            feat = np.concatenate([roi_ion1.flatten(), roi_ion2.flatten()])
            ml_labels_lr.append(models["LR"].predict([feat])[0])
            ml_labels_svm.append(models["SVM"].predict([feat])[0])
            ml_labels_knn.append(models["kNN"].predict([feat])[0])

        # 예측값 저장
        thr_predict = final_state_20frame(thr_labels)
        lr_predict  = final_state_20frame(ml_labels_lr)
        svm_predict = final_state_20frame(ml_labels_svm)
        knn_predict = final_state_20frame(ml_labels_knn)

        # 맞았는지 (correct:1, wrong:0)
        thr_ok = (thr_predict == gt_state)
        lr_ok  = (lr_predict  == gt_state)
        svm_ok = (svm_predict == gt_state)
        knn_ok = (knn_predict == gt_state)

        # correct count
        correct_thr += thr_ok
        correct_lr  += lr_ok
        correct_svm += svm_ok
        correct_knn += knn_ok

        # O인지 X인지 표시
        ok = lambda b: "O" if b else "X"

        # 행마다 row_fmt (~~ | ~~ | ... | ~~) 사용하여 출력
        print(row_fmt.format(
            base,
            class_to_str(gt_state),
            class_to_str(thr_predict),
            class_to_str(lr_predict),
            class_to_str(svm_predict),
            class_to_str(knn_predict),
            ok(thr_ok),
            ok(lr_ok),
            ok(svm_ok),
            ok(knn_ok),
        ))

        gt_all.append(gt_state)
        thr_all.append(thr_predict)
        lr_all.append(lr_predict)
        svm_all.append(svm_predict)
        knn_all.append(knn_predict)

    print("-" * 65)
    print(f"Threshold accuracy = {correct_thr}/{total} = {correct_thr/total:.3f}")
    print(f"LR accuracy        = {correct_lr}/{total}  = {correct_lr/total:.3f}")
    print(f"SVM accuracy       = {correct_svm}/{total}  = {correct_svm/total:.3f}")
    print(f"kNN accuracy       = {correct_knn}/{total}  = {correct_knn/total:.3f}")
    print("F1-scores (macro):")
    print(f"Threshold = {f1_score(gt_all, thr_all, average='macro'):.4f}")
    print(f"LR        = {f1_score(gt_all, lr_all,  average='macro'):.4f}")
    print(f"SVM       = {f1_score(gt_all, svm_all, average='macro'):.4f}")
    print(f"kNN       = {f1_score(gt_all, knn_all, average='macro'):.4f}")

##############################################################################################################










########################### Main Loop ########################################################################

if __name__ == "__main__":
    print("Experiment START")
    models = train_ML_models()
    evaluate_final_accuracy(models)
    print("\nExperiment Done")


##############################################################################################################
