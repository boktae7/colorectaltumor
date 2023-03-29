import os
import numpy as np
import cv2
import torch

from data import load_npz

def arrage_result(box_list, score_list):
    sort_score_list = np.sort(score_list)[::-1]
    sort_box_list = []
    dup_count = 0
    for score in sort_score_list:
        dup_num = len(np.where(score_list==score)[0])
        if dup_num == 1:
            dup_count = 0
        sort_box_list.append(box_list[np.where(score_list==score)[0][dup_count]])
        if dup_num > (dup_count+1):
            dup_count += 1
    return sort_box_list, sort_score_list

def generate_hm_new(input_shape, height, width, landmarks, sigma=4): # sigma -> 랜드마크 기준 얼마나 퍼진 heatmap 인지
    hms = np.zeros(shape=(len(landmarks), height, width), dtype=np.float32)
    divide_factor = input_shape / height
    for i in range(len(landmarks)):
        if (landmarks[i][0] != -1.0) and (landmarks[i][1] != -1.0):
            x, y = round(landmarks[i][0]/divide_factor), round(landmarks[i][1]/divide_factor)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
            br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

            c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], width)
            aa, bb = max(0, ul[1]), min(br[1], height)
            gaussian = np.maximum(hms[i, aa:bb, cc:dd], _makeGaussian_new(sigma)[a:b, c:d])
            hms[i, aa:bb, cc:dd] = gaussian

    return hms

def generate_hm_new_2point(input_shape, height, width, landmarks, sigma=4): # sigma -> 랜드마크 기준 얼마나 퍼진 heatmap 인지
    hms = np.zeros(shape=(1, height, width), dtype=np.float32)
    divide_factor = input_shape / height
    for i in range(len(landmarks)):
        if (landmarks[i][0] != -1.0) and (landmarks[i][1] != -1.0):
            x, y = round(landmarks[i][0]/divide_factor), round(landmarks[i][1]/divide_factor)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
            br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

            c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], width)
            aa, bb = max(0, ul[1]), min(br[1], height)
            gaussian = np.maximum(hms[0, aa:bb, cc:dd], _makeGaussian_new(sigma)[a:b, c:d])
            hms[0, aa:bb, cc:dd] = gaussian

    return hms

def landmark_exception(landmark):
    fix_landmark = landmark.astype(np.int64)
    if len(landmark) < 3 :
        if fix_landmark[0][0] > fix_landmark[1][0]:
            tmp = fix_landmark[0][0]
            fix_landmark[0][0] = fix_landmark[1][0]
            fix_landmark[1][0] = tmp

        if fix_landmark[0][1] > fix_landmark[1][1]:
            tmp = fix_landmark[0][1]
            fix_landmark[0][1] = fix_landmark[1][1]
            fix_landmark[1][1] = tmp

        for i in range(2):
            for j in range(2):
                if fix_landmark[i][j] < 80:
                    fix_landmark[i][j] = 0
                    if i == 1:
                        fix_landmark[1][0] = 0
                        fix_landmark[1][1] = 0
    else:
        if fix_landmark[0] > fix_landmark[2]:
            tmp = fix_landmark[0]
            fix_landmark[0] = fix_landmark[2]
            fix_landmark[2] = tmp

        if fix_landmark[1] > fix_landmark[3]:
            tmp = fix_landmark[1]
            fix_landmark[1] = fix_landmark[3]
            fix_landmark[3] = tmp

        for i in range(4):
            if fix_landmark[i] < 80:
                fix_landmark[i] = 0
                if i == 2:
                    fix_landmark[2] = 0
                    fix_landmark[3] = 0

    return fix_landmark

def landmark_exception_cut(landmark):
    fix_landmark = landmark.astype(np.int64)
    if len(landmark) < 3 :
        if fix_landmark[0][0] > fix_landmark[1][0]:
            tmp = fix_landmark[0][0]
            fix_landmark[0][0] = fix_landmark[1][0]
            fix_landmark[1][0] = tmp

        if fix_landmark[0][1] > fix_landmark[1][1]:
            tmp = fix_landmark[0][1]
            fix_landmark[0][1] = fix_landmark[1][1]
            fix_landmark[1][1] = tmp

        for i in range(2):
            for j in range(2):
                if fix_landmark[i][j] < 5:
                    fix_landmark[i][j] = 0
                    if i == 1:
                        fix_landmark[1][0] = 0
                        fix_landmark[1][1] = 0
    else:
        if fix_landmark[0] > fix_landmark[2]:
            tmp = fix_landmark[0]
            fix_landmark[0] = fix_landmark[2]
            fix_landmark[2] = tmp

        if fix_landmark[1] > fix_landmark[3]:
            tmp = fix_landmark[1]
            fix_landmark[1] = fix_landmark[3]
            fix_landmark[3] = tmp

        for i in range(4):
            if fix_landmark[i] < 5:
                fix_landmark[i] = 0
                if i == 2:
                    fix_landmark[2] = 0
                    fix_landmark[3] = 0

    return fix_landmark

def _makeGaussian_new(sigma=3):
    size = 6 * sigma + 3

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    gaussian = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return gaussian

def convert_heatmap_to_landmark(heatmap):
    return np.flip(np.array(np.unravel_index(heatmap.argmax(), heatmap.shape)))

def convert_heatmap_to_landmark_2output(heatmap):
    # print("")
    # print(np.flip(np.array([np.unravel_index(heatmap.argmax(), heatmap.shape)])))
    # row,col = np.flip(np.array(np.unravel_index(np.argsort(heatmap.ravel()),heatmap.shape)))
    # row, col = row[:-1],col[:-1]
    # print(row[0], col[0])
    # print(heatmap[row[0], col[0]])
    # print(row[1], col[1])
    # print(heatmap[row[1], col[1]])   generate_hm_new(input_shape, height, width, landmarks, sigma=4)
    max_point = np.flip(np.array(np.unravel_index(heatmap.argmax(), heatmap.shape)))
    landmarks = [max_point,max_point]
    zero_heatamp = generate_hm_new(128, 128, 128, landmarks, sigma=15)
    
    max_v = np.max(heatmap)
    min_v = np.min(heatmap)
    p_nor = (heatmap - min_v) / (max_v - min_v)
    # new_img = pd_slices[idx] * (1-zero_heatamp[idx])
    new_img_2 = p_nor * (1-zero_heatamp[0])

    sec_point = convert_heatmap_to_landmark(new_img_2)
    
    pd_land = np.array([max_point, sec_point], dtype=np.int64)
    return pd_land

def make_multislice_input(dir_5slice, batch_size, number_list, pd_landmarks):
    multi_input = []
    multi_gt = []
    for i in range(batch_size):
        target_path = f"{dir_5slice}{number_list[i]}.npz"
        if os.path.isfile(target_path):
            ct, min_x, min_y, max_x, max_y, _, _, _ = load_npz(target_path)
        else:
            print("Target File Check")
            print(target_path)
            exit()
        cut = pd_landmarks[i]
        cts = []
        gts = []
        for j in range(ct.shape[0]): 
            cut_img = ct[j, cut[1]:cut[1]+(cut[3]-cut[1]), cut[0]:cut[0]+(cut[2]-cut[0])]
            cut_img = cv2.resize(cut_img, dsize=(128,128), interpolation=cv2.INTER_AREA)
            cts.append(cut_img)
            gts.append([[min_x, min_y], [max_x, max_y]])
        # for j in range(3): 
        #     cut_img = ct[j+1, cut[1]:cut[1]+(cut[3]-cut[1]), cut[0]:cut[0]+(cut[2]-cut[0])]
        #     cut_img = cv2.resize(cut_img, dsize=(128,128), interpolation=cv2.INTER_AREA)
        #     cts.append(cut_img)
        #     gts.append([[min_x, min_y], [max_x, max_y]])

        multi_input.append(cts)
        multi_gt.append(gts)
    multi_input_tensor = torch.from_numpy(np.array(multi_input)).type(torch.FloatTensor)
    return multi_input_tensor, multi_gt

def make_multislice_train(dir_5slice, batch_size, number_list):
    multi_input = []
    multi_gt = []
    for i in range(batch_size):
        target_path = f"{dir_5slice}{number_list[i]}.npz"
        if os.path.isfile(target_path):
            ct, min_x, min_y, max_x, max_y, _, _, _ = load_npz(target_path)
        else:
            print("Target File Check")
            print(target_path)
            exit()
        cts = []
        gts = []
        # for j in range(ct.shape[0]): 
        #     cut_img = ct[j]
        #     cts.append(cut_img)
        #     gts.append([[min_x, min_y], [max_x, max_y]])
        for j in range(3): 
            cut_img = ct[j+1]
            cts.append(cut_img)
            gts.append([[min_x, min_y], [max_x, max_y]])

        multi_input.append(cts)
        multi_gt.append(gts)
    multi_input_tensor = torch.from_numpy(np.array(multi_input)).type(torch.FloatTensor)
    return multi_input_tensor