import os
from re import L
from turtle import pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pandas as pd

import cv2
import openpyxl
import torch
from func import *

def cal_score(gt_landmark, pd_landmark, shape):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    tp_x1 = gt_landmark[0][0]
    if pd_landmark[0][0] > tp_x1:
        tp_x1 = pd_landmark[0][0]

    tp_x2 = gt_landmark[1][0]
    if pd_landmark[1][0] < tp_x2:
        tp_x2 = pd_landmark[1][0]

    tp_y1 = gt_landmark[0][1]
    if pd_landmark[0][1] > tp_y1:
        tp_y1 = pd_landmark[0][1]
        
    tp_y2 = gt_landmark[1][1]
    if pd_landmark[1][1] < tp_y2:
        tp_y2 = pd_landmark[1][1]

    ## check intersection
    is_inter = False
    if (tp_x2 - tp_x1) >= 0:
        if (tp_y2 - tp_y1) >= 0:
            is_inter = True
    
    if is_inter:
        TP = (tp_x2 - tp_x1 + 1) * (tp_y2 - tp_y1 + 1)
    else:
        TP = 0
    FP = (pd_landmark[1][0] - pd_landmark[0][0] + 1) * (pd_landmark[1][1] - pd_landmark[0][1] + 1) - TP
    FN = (gt_landmark[1][0] - gt_landmark[0][0] + 1) * (gt_landmark[1][1] - gt_landmark[0][1] + 1) - TP
    TN = int(shape[0] * shape[1]) - TP - FP - FN - TN
    return TP, FP, TN, FN

def Cal_Result(TP, FP, TN, FN):
    ## Return dice, specitificity, sensitivity, precision
    # TP, FP, TN, FN = perf_measure(gt_img, pred_img)

    dice_value = 2 * TP / ((TP + FP) + (TP + FN))
    sensitivity = TP / (TP + FN)
    specitificity = TN / (FP + TN)
    iou_value = TP / (TP + FP + FN)

    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else : 
        precision = 0

    return iou_value, dice_value, specitificity, sensitivity, precision

def patient_score(data_path):
    is_first = True
    patient_list = []
    for file_name in os.listdir(data_path):
        if "~" not in file_name and "BoxInfo" not in file_name and "Patient" not in file_name:
            print(f"{file_name} Patient analysis....")
            excel = pd.read_excel(f"{data_path}{file_name}")
            target_list = list(excel.columns[7:])
            target_num = len(target_list)
            target_list.insert(0, 'Patient')
    
            if is_first:
                is_first = False
                for num in range(len(excel["Patient"])):
                    patient_list.append(excel["Patient"][num].split("_")[1])
                patient_list = sorted(set(patient_list))
            
            name = file_name.split('_')

            path_excel_result = data_path + 'Patient_'
            for n_idx in range(1, len(name)):
                path_excel_result += name[n_idx] + '_'
            path_excel_result = path_excel_result[:-1]
            wb_result = openpyxl.Workbook()
            worksheet = wb_result.active
            worksheet.append(target_list)
            wb_result.save(path_excel_result)
             
            patient_result = []
            for patient in patient_list:
                score_list = [] 
                for num in range(len(excel["Patient"])):
                    if patient in excel["Patient"][num]:
                        for t_num in range(target_num):
                            score_list.append(excel[target_list[t_num+1]][num])
                avg_list = []
                for i in range(target_num):
                    score = []
                    for j in range(len(score_list)):
                        if (j % target_num) == i:
                            score.append(score_list[j])
                    avg_list.append(round(np.mean(score), 4))
                avg_list.insert(0, patient)
                patient_result.append(avg_list)

            wb_result = openpyxl.load_workbook(path_excel_result)
            ws = wb_result.active
            for p in range(len(patient_result)):
                ws.append(patient_result[p])
            wb_result.save(path_excel_result)                        
    print("Patient Analysis Finish")

def patient_score_boxinfo(data_path):
    is_first = True
    patient_list = []
    for file_name in os.listdir(data_path):
        if "~" not in file_name and "BoxInfo" in file_name and "Patient" not in file_name:
            print(f"{file_name} Patient analysis....")
            excel = pd.read_excel(f"{data_path}{file_name}")
            target_list = []
            for col in range(len(list(excel.columns))):
                if "_S" in excel.columns[col]:
                    target_list.append(excel.columns[col])
            target_num = len(target_list)
            target_list.insert(0, 'Patient')
    
            if is_first:
                is_first = False
                for num in range(len(excel["Patient"])):
                    patient_list.append(excel["Patient"][num].split("_")[1])
                patient_list = sorted(set(patient_list))
            
            name = file_name.split('_')

            path_excel_result = data_path + 'Patient_'
            for n_idx in range(1, len(name)):
                path_excel_result += name[n_idx] + '_'
            path_excel_result = path_excel_result[:-1]
            wb_result = openpyxl.Workbook()
            worksheet = wb_result.active
            worksheet.append(target_list)
            wb_result.save(path_excel_result)
             
            patient_result = []
            for patient in patient_list:
                score_list = [] 
                for num in range(len(excel["Patient"])):
                    if patient in excel["Patient"][num]:
                        for t_num in range(target_num):
                            score_list.append(excel[target_list[t_num+1]][num])
                avg_list = []
                for i in range(target_num):
                    score = []
                    for j in range(len(score_list)):
                        if (j % target_num) == i:
                            score.append(score_list[j])
                    avg_list.append(round(np.mean(score), 4))
                avg_list.insert(0, patient)
                patient_result.append(avg_list)

            wb_result = openpyxl.load_workbook(path_excel_result)
            ws = wb_result.active
            for p in range(len(patient_result)):
                ws.append(patient_result[p])
            wb_result.save(path_excel_result)                        
    print("Patient Analysis Finish")

def cal_cutgt(ori_gt, cut_land):
    ratio_x = 128 / (cut_land[2] - cut_land[0])
    ratio_y = 128 / (cut_land[3] - cut_land[1])
    cut_gt = []
    if int(ori_gt[0][0]) > 0:
        cut_x1 = ori_gt[0][0]-cut_land[0]
        cut_y1 = ori_gt[0][1]-cut_land[1]
        cut_x2 = ori_gt[1][0]-cut_land[0]
        cut_y2 = ori_gt[1][1]-cut_land[1]
        cut_gt = np.array([[cut_x1 * ratio_x, cut_y1 * ratio_y], [cut_x2 * ratio_x, cut_y2 * ratio_y]], dtype=np.int64)
    else:
        cut_gt = np.array([[0, 0], [1, 1]], dtype=np.int64)

    return cut_gt, [ratio_x, ratio_y]

def detradd_net(save_dir, name_dir, network, device, input_tensor_cpu, gt_landmarks_cpu, detr_out, select_boxes, box_score_choice, cut_landmarks, number_list, pred_shape, is_save, is_o2, is_multi):   
    batch_size = input_tensor_cpu.shape[0]
    
    ## Hourglass Network
    new_pd_landmarks = []
    pd_heatmap_l1 = []
    pd_heatmap_l2 = []
    ori_pd_landmarks = []
    gt_shape = 512
    in_shape = 128
    ratio_gt_pd = int(gt_shape / pred_shape)
    ratio_in_pd = int(in_shape / pred_shape)
    for num in range(batch_size):
        if not is_multi:
            ori_img = input_tensor_cpu[num][0]
        else:
            ori_img = input_tensor_cpu[num][1]
        cut = cut_landmarks[num] 
        cut_img = ori_img[cut[1]:cut[1]+(cut[3]-cut[1]), cut[0]:cut[0]+(cut[2]-cut[0])]
        cut_img = cv2.resize(cut_img, dsize=(in_shape,in_shape), interpolation=cv2.INTER_AREA)
        # new_land = np.reshape(cut, (2,2))
        if cut[2] > 50:
            _, ratio = cal_cutgt(gt_landmarks_cpu[num], cut)
            input_tensor = torch.tensor(cut_img, dtype=torch.float64, device=device)
            input_tensor = input_tensor.view(1, 1, input_tensor.shape[0], input_tensor.shape[1]).float()
            pred_tensor = network(input_tensor)
            pred_slices = pred_tensor.detach().cpu().numpy()[0]
            pd_heatmap_l1.append(pred_slices[0])
            if not is_o2:
                pd_heatmap_l2.append(pred_slices[1])
            pred_land = []
            
            if not is_o2:
                for i in range(2):
                    landmark = convert_heatmap_to_landmark(pred_slices[i, :, :]) * ratio_in_pd 
                    pred_land.append(landmark)
            else:
                pred_land = convert_heatmap_to_landmark_2output(pred_slices[0, :, :]) * ratio_in_pd 

            pred_land = np.array(pred_land)
            pred_land = landmark_exception_cut(pred_land)
            ori_pd_landmarks.append(pred_land)
            if pred_land[1][0] > 80 and pred_land[1][1] > 80:
                tmp_land = np.array([[pred_land[0][0] * (1/ratio[0]), pred_land[0][1] * (1/ratio[1])], [pred_land[1][0] * (1/ratio[0]), pred_land[1][1] * (1/ratio[1])]])
                new_land = np.array([[tmp_land[0][0]+cut[0], tmp_land[0][1]+cut[1]],[tmp_land[1][0]+cut[0], tmp_land[1][1]+cut[1]]], dtype=np.int64)
                new_land = landmark_exception(new_land)
            else:
                new_land = np.array([[0, 0],[0, 0]])
        else:
            new_land = np.array([[0, 0],[0, 0]])
            pd_heatmap_l1.append(np.zeros((pred_shape, pred_shape)))
            if not is_o2:
                pd_heatmap_l2.append(np.zeros((pred_shape, pred_shape)))
            ori_pd_landmarks.append(new_land)
        new_pd_landmarks.append(new_land)
    ## Save Img 
    if is_save:
        if batch_size > 8:
            batch_size = 8
        fig_col = int(batch_size/2)

        ## detection result
        fig = plt.figure() 
        gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
        for num in range(batch_size):
            ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
            if not is_multi:
                ax.imshow(input_tensor_cpu[num][0], cmap='gray')
            else:
                ax.imshow(input_tensor_cpu[num][1], cmap='gray')
            gt = gt_landmarks_cpu[num] 
            pds = detr_out[num]['boxes']
            for i in range(len(pds)):
                for j in range(4):
                    if pds[i][j] < 0:
                        pds[i][j] = 0
                ax.add_patch(patches.Rectangle((int(pds[i][0]), int(pds[i][1])), int(pds[i][2]) - int(pds[i][0]), int(pds[i][3]) - int(pds[i][1]), edgecolor = 'blue', fill=False))
            ax.add_patch(patches.Rectangle(gt[0], gt[1][0] - gt[0][0], gt[1][1] - gt[0][1], edgecolor = 'red', fill=False))
            score = 'BG'
            if len(pds) > 0:
                best_idx = torch.argmax(detr_out[num]['scores'])
                score = round(float(detr_out[num]['scores'][best_idx]), 3) ## best score 
            title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}_{score}'
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
        plt.margins(0,0)
        os.makedirs(f"{save_dir}DT/", exist_ok=True)
        plt.savefig(f"{save_dir}DT/DT_{name_dir}.jpg")
        plt.savefig(f"{save_dir}{name_dir}_0.jpg")
        plt.close()
        
        ## dt choice box 
        fig = plt.figure() 
        gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
        for num in range(batch_size):
            ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
            if not is_multi:
                ax.imshow(input_tensor_cpu[num][0], cmap='gray')
            else:
                ax.imshow(input_tensor_cpu[num][1], cmap='gray')
            gt = gt_landmarks_cpu[num] 
            pds = select_boxes[num]
            for i in range(len(pds)):
                pd = pds[i].reshape(2,2)
                ax.add_patch(patches.Rectangle(pd[0], pd[1][0] - pd[0][0], pd[1][1] - pd[0][1], edgecolor = 'blue', fill=False))
                # ax.add_patch(patches.Rectangle((int(pds[i][0]), int(pds[i][1])), int(pds[i][2]) - int(pds[i][0]), int(pds[i][3]) - int(pds[i][1]), edgecolor = 'blue', fill=False))
            ax.add_patch(patches.Rectangle(gt[0], gt[1][0] - gt[0][0], gt[1][1] - gt[0][1], edgecolor = 'red', fill=False))
            score = 'BG'
            if len(pds) > 0:
                best_idx = torch.argmax(detr_out[num]['scores'])
                score = round(float(detr_out[num]['scores'][best_idx]), 3) ## best score 
            title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}_{score}'
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
        plt.margins(0,0)
        os.makedirs(f"{save_dir}DT_Choice/", exist_ok=True)
        plt.savefig(f"{save_dir}DT_Choice/Choice_{name_dir}.jpg")
        plt.savefig(f"{save_dir}{name_dir}_1.jpg")
        plt.close()

        ## dt output box 
        fig = plt.figure() 
        gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
        for num in range(batch_size):
            ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
            if not is_multi:
                ax.imshow(input_tensor_cpu[num][0], cmap='gray')
            else:
                ax.imshow(input_tensor_cpu[num][1], cmap='gray')
            gt = gt_landmarks_cpu[num] 
            cut = cut_landmarks[num] 
            ax.add_patch(patches.Rectangle((cut[0], cut[1]), cut[2] - cut[0], cut[3] - cut[1], edgecolor = 'blue', fill=False))
            ax.add_patch(patches.Rectangle(gt[0], gt[1][0] - gt[0][0], gt[1][1] - gt[0][1], edgecolor = 'red', fill=False))
            score = box_score_choice[num]
            title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}_{score}'
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
        plt.margins(0,0)
        os.makedirs(f"{save_dir}DT_Result/", exist_ok=True)
        plt.savefig(f"{save_dir}DT_Result/Ori_{name_dir}.jpg")
        plt.savefig(f"{save_dir}{name_dir}_2.jpg")
        plt.close()

        ## Cut resize img
        fig = plt.figure() 
        gs = gridspec.GridSpec(nrows=2, ncols=fig_col)   
        for num in range(batch_size):
            if not is_multi:
                ori_img = input_tensor_cpu[num][0]
            else:
                ori_img = input_tensor_cpu[num][1]
            cut = cut_landmarks[num] 
            cut_img = ori_img[cut[1]:cut[1]+(cut[3]-cut[1]), cut[0]:cut[0]+(cut[2]-cut[0])]
            cut_img = cv2.resize(cut_img, dsize=(128,128), interpolation=cv2.INTER_AREA)
            ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
            ax.imshow(cut_img, cmap='gray')
            if cut[2] > 50:
                cut_gt, _ = cal_cutgt(gt_landmarks_cpu[num], cut)
                ax.add_patch(patches.Rectangle(cut_gt[0], cut_gt[1][0] - cut_gt[0][0], cut_gt[1][1] - cut_gt[0][1], edgecolor = 'red', fill=False))        
            title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
        plt.margins(0,0)
        os.makedirs(f"{save_dir}Cut_Img/", exist_ok=True)
        plt.savefig(f"{save_dir}Cut_Img/Cut_{name_dir}.jpg")
        plt.savefig(f"{save_dir}{name_dir}_3.jpg")
        plt.close()

        ## land 1 heat map
        fig = plt.figure() 
        gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
        for num in range(batch_size):
            ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
            ax.imshow(pd_heatmap_l1[num])
            cut = cut_landmarks[num] 
            if cut[2] > 50:
                gt, _ = cal_cutgt(gt_landmarks_cpu[num], cut)
            else:
                gt = gt_landmarks_cpu[num] / ratio_gt_pd
            pd = ori_pd_landmarks[num] / ratio_in_pd
            ax.scatter(int(gt[0][0]), int(gt[0][1]), c='r', s=10)
            ax.scatter(pd[0][0], pd[0][1], c='blue', s=10)
            if is_o2:
                ax.scatter(int(gt[1][0]), int(gt[1][1]), c='r', s=10)
                ax.scatter(pd[1][0], pd[1][1], c='blue', s=10)
            title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
        plt.margins(0,0)
        os.makedirs(f'{save_dir}/HMap1/', exist_ok=True)
        plt.savefig(f"{save_dir}/HMap1/HMap1_{name_dir}.jpg")
        plt.savefig(f"{save_dir}{name_dir}_4.jpg")
        plt.close()

        if not is_o2:
            ## land 2 heat map
            fig = plt.figure() 
            gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
            for num in range(batch_size):
                ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
                ax.imshow(pd_heatmap_l2[num])
                cut = cut_landmarks[num] 
                if cut[2] > 50:
                    gt, _ = cal_cutgt(gt_landmarks_cpu[num], cut)
                else:
                    gt = gt_landmarks_cpu[num] / ratio_gt_pd
                pd = ori_pd_landmarks[num] / ratio_in_pd
                ax.scatter(int(gt[1][0]), int(gt[1][1]), c='r', s=10)
                ax.scatter(pd[1][0], pd[1][1], c='blue', s=10)
                title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
                ax.set_title(title)
                ax.axis('off')
            plt.tight_layout()
            plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
            plt.margins(0,0)
            os.makedirs(f'{save_dir}/HMap2/', exist_ok=True)
            plt.savefig(f"{save_dir}/HMap2/HMap2_{name_dir}.jpg")
            plt.savefig(f"{save_dir}{name_dir}_5.jpg")
            plt.close()
        
        ## final output
        fig = plt.figure() 
        gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
        for num in range(batch_size):
            ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
            if not is_multi:
                ax.imshow(input_tensor_cpu[num][0], cmap='gray')
            else:
                ax.imshow(input_tensor_cpu[num][1], cmap='gray')
            gt = gt_landmarks_cpu[num]
            pd = new_pd_landmarks[num]
            ax.add_patch(patches.Rectangle(gt[0], gt[1][0] - gt[0][0], gt[1][1] - gt[0][1], edgecolor = 'red', fill=False))
            ax.add_patch(patches.Rectangle(pd[0], pd[1][0] - pd[0][0], pd[1][1] - pd[0][1], edgecolor = 'blue', fill=False))
            title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
        plt.margins(0,0)
        os.makedirs(f'{save_dir}/Result/', exist_ok=True)
        plt.savefig(f"{save_dir}/Result/Result_{name_dir}.jpg")
        plt.savefig(f"{save_dir}{name_dir}_6.jpg")
        plt.close()  

    return new_pd_landmarks

def save_result_image(save_dir, name_dir, input_tensor_cpu, gt_landmarks_cpu, pred_tensor_cpu, gt_tensor_cpu, number_list):   
    batch_size = input_tensor_cpu.shape[0]
    if batch_size > 8:
        batch_size = 8
    ratio_gt_pd = int(input_tensor_cpu.shape[2] / pred_tensor_cpu.shape[2])
    fig_col = int(batch_size/2)
    total_input = []
    total_gt_landmark = []
    total_pd_landmark = []
    total_pd_heatmap_l1 = []
    total_pd_heatmap_l2 = []
    total_gt_heatmap_l1 = []
    total_gt_heatmap_l2 = []
    for num in range(batch_size):
        input_slice = input_tensor_cpu[num][0]
        gt_landmark = gt_landmarks_cpu[num]
        pred_slices = pred_tensor_cpu[num]
        number_slice = number_list[num]
        gt_slices = gt_tensor_cpu[num]

        gt_landmarks = np.array(gt_landmark, dtype=np.int64)

        pred_landmarks = []
        for i in range(pred_slices.shape[0]): ## pred_slices.shape[0] == num landmark 
            landmark = convert_heatmap_to_landmark(pred_slices[i, :, :]) 
            # landmark = convert_heatmap_to_landmark_2output(pred_slices[i, :, :]) 
            pred_landmarks.append(landmark)
        pred_landmarks = np.array(pred_landmarks)

        total_input.append(input_slice)
        total_gt_landmark.append(gt_landmarks)
        total_pd_landmark.append(pred_landmarks)
        total_pd_heatmap_l1.append(pred_slices[0])
        total_pd_heatmap_l2.append(pred_slices[1])
        total_gt_heatmap_l1.append(gt_slices[0])
        total_gt_heatmap_l2.append(gt_slices[1])

    ## input + box
    fig = plt.figure() 
    gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
    for num in range(len(total_input)):
        ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
        ax.imshow(total_input[num], cmap='gray')
        gt = total_gt_landmark[num] 
        pd = total_pd_landmark[num] * ratio_gt_pd 
        # pd = total_pd_landmark[num][0] * ratio_gt_pd
        ax.add_patch(patches.Rectangle(gt[0], gt[1][0] - gt[0][0], gt[1][1] - gt[0][1], edgecolor = 'red', fill=False))
        ax.add_patch(patches.Rectangle(pd[0], pd[1][0] - pd[0][0], pd[1][1] - pd[0][1], edgecolor = 'blue', fill=False))
        title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
    plt.margins(0,0)
    os.makedirs(f'{save_dir}/Result/', exist_ok=True)
    plt.savefig(f"{save_dir}/Result/Result_{name_dir}.jpg")
    plt.close()

    ## land heat map
    fig = plt.figure() 
    gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
    for num in range(len(total_input)):
        ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
        ax.imshow(total_pd_heatmap_l1[num])
        gt = total_gt_landmark[num]
        pd = total_pd_landmark[num]
        # pd = total_pd_landmark[num][0]
        ax.scatter(int(gt[0][0]/ratio_gt_pd), int(gt[0][1]/ratio_gt_pd), c='r', s=10)
        ax.scatter(pd[0][0], pd[0][1], c='blue', s=10)

        # ax.scatter(int(gt[1][0]/ratio_gt_pd), int(gt[1][1]/ratio_gt_pd), c='r', s=10)
        # ax.scatter(pd[1][0], pd[1][1], c='blue', s=10)

        title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
    plt.margins(0,0)
    os.makedirs(f'{save_dir}/HMap1/', exist_ok=True)
    plt.savefig(f"{save_dir}/HMap1/HMap1_{name_dir}.jpg")
    plt.close()

    ## land 2 heat map
    fig = plt.figure() 
    gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
    for num in range(len(total_input)):
        ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
        ax.imshow(total_pd_heatmap_l2[num])
        gt = total_gt_landmark[num]
        pd = total_pd_landmark[num]
        ax.scatter(int(gt[1][0]/ratio_gt_pd), int(gt[1][1]/ratio_gt_pd), c='r', s=10)
        ax.scatter(pd[1][0], pd[1][1], c='blue', s=10)
        title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
    plt.margins(0,0)
    os.makedirs(f'{save_dir}/HMap2/', exist_ok=True)
    plt.savefig(f"{save_dir}/HMap2/HMap2_{name_dir}.jpg")
    plt.close()

    ## gt heat map 1
    fig = plt.figure() 
    gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
    for num in range(len(total_input)):
        ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
        ax.imshow(total_gt_heatmap_l1[num])
        # ax.scatter(int(gt[0][0]/ratio_gt_pd), int(gt[0][1]/ratio_gt_pd), c='r', s=10)
        title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
    plt.margins(0,0)
    os.makedirs(f'{save_dir}/HMap1_G/', exist_ok=True)
    plt.savefig(f"{save_dir}/HMap1_G/HMap1_G_{name_dir}.jpg")
    plt.close()

    ##  gt heat map 2
    fig = plt.figure() 
    gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
    for num in range(len(total_input)):
        ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
        ax.imshow(total_gt_heatmap_l2[num])
        # ax.scatter(int(gt[1][0]/ratio_gt_pd), int(gt[1][1]/ratio_gt_pd), c='r', s=10)
        title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
    plt.margins(0,0)
    os.makedirs(f'{save_dir}/HMap2_G/', exist_ok=True)
    plt.savefig(f"{save_dir}/HMap2_G/HMap2_G_{name_dir}.jpg")
    plt.close()

def save_result_image_cut(save_npz_dir, save_img_dir, name_dir, input_tensor_cpu, gt_landmarks_cpu, cut_landmarks, number_list, is_save, multislice, save_npz=True):   
    batch_size = input_tensor_cpu.shape[0]
    
    ## Get & Save Cut Data
    cut_imgs = []
    cut_nums = []
    cut_gts = []
    ori_pds = []
    if save_npz:
        for num in range(batch_size):
            if multislice:
                ori_img = input_tensor_cpu[num][1]
            else:
                ori_img = input_tensor_cpu[num][0]
            cut = cut_landmarks[num] 
            cut_img = ori_img[cut[1]:cut[1]+(cut[3]-cut[1]), cut[0]:cut[0]+(cut[2]-cut[0])]
            cut_img = cv2.resize(cut_img, dsize=(128,128), interpolation=cv2.INTER_AREA)
            if cut[2] > 50:
                cut_imgs.append(cut_img)
                cut_nums.append(number_list[num])
                cut_gt, _= cal_cutgt(gt_landmarks_cpu[num], cut)
                cut_gt = landmark_exception_cut(cut_gt)
                cut_gts.append(cut_gt)
                ori_pds.append(cut)
                
                if cut_gt[0][0] < 0:
                    print(number_list[num])
                    print(cut_gt)
                    exit()
                npz_name = f"{save_npz_dir}{number_list[num]}.npz"
                np.savez_compressed(npz_name, cut_img = cut_img, number = number_list[num], cut_gt=cut_gt, ori_img = ori_img, ori_gt = gt_landmarks_cpu[num], cut_mark = cut)
    ## IMG Save
    if is_save:
        if batch_size > 8:
            batch_size = 8
        fig_col = int(batch_size/2)

        ## input + box
        fig = plt.figure() 
        gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
        for num in range(batch_size):
            ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
            if save_npz:
                if multislice:
                    ax.imshow(input_tensor_cpu[num][1], cmap='gray')
                else:
                    ax.imshow(input_tensor_cpu[num][0], cmap='gray')
            else:
                ax.imshow(input_tensor_cpu[num], cmap='gray')
            gt = gt_landmarks_cpu[num] 
            cut = cut_landmarks[num] 
            ax.add_patch(patches.Rectangle((cut[0], cut[1]), cut[2] - cut[0], cut[3] - cut[1], edgecolor = 'blue', fill=False))
            ax.add_patch(patches.Rectangle(gt[0], gt[1][0] - gt[0][0], gt[1][1] - gt[0][1], edgecolor = 'red', fill=False))
            title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
        plt.margins(0,0)
        plt.savefig(f"{save_img_dir}Ori_{name_dir}.jpg")
        plt.close()

        ## Cut resize img
        fig = plt.figure() 
        gs = gridspec.GridSpec(nrows=2, ncols=fig_col)
        for num in range(batch_size):
            if save_npz:
                if multislice:
                    ax.imshow(input_tensor_cpu[num][1], cmap='gray')
                else:
                    ax.imshow(input_tensor_cpu[num][0], cmap='gray')
            else:
                ori_img = input_tensor_cpu[num]
            cut = cut_landmarks[num] 
            cut_img = ori_img[cut[1]:cut[1]+(cut[3]-cut[1]), cut[0]:cut[0]+(cut[2]-cut[0])]
            cut_img = cv2.resize(cut_img, dsize=(128,128), interpolation=cv2.INTER_AREA)
            ax = fig.add_subplot(gs[int(num/fig_col), num%fig_col])
            ax.imshow(cut_img, cmap='gray')
            if cut[2] > 50:
                cut_gt, _ = cal_cutgt(gt_landmarks_cpu[num], cut)
                ax.add_patch(patches.Rectangle(cut_gt[0], cut_gt[1][0] - cut_gt[0][0], cut_gt[1][1] - cut_gt[0][1], edgecolor = 'red', fill=False))
            title = f'{number_list[num].split("_")[1]}_{number_list[num].split("_")[2]}'
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0.1, wspace = 0.1)
        plt.margins(0,0)
        plt.savefig(f"{save_img_dir}Cut_{name_dir}.jpg")
        plt.close()

    return cut_imgs, cut_nums, cut_gts, ori_pds

def create_box(boxes_cpu, scores_cpu, th, box_usenum, shape_x, shape_y):
    min_x = shape_x
    min_y = shape_y
    max_x = 0
    max_y = 0
    select_boxes = []
    zero_img = np.zeros((512, 512))
    add_input_channel = len(boxes_cpu)
    if add_input_channel >= (box_usenum+1):
        add_input_channel = box_usenum
    zero_box_count = 0
    for bbnum in range(add_input_channel):
        if boxes_cpu[bbnum][2] < 80 or boxes_cpu[bbnum][2] < 80:
            zero_box_count += 1 
    for bbnum in range(add_input_channel):
        select_boxes.append(landmark_exception(boxes_cpu[bbnum]))
        for x in range(shape_x):
            if x >= boxes_cpu[bbnum][0] and x <= boxes_cpu[bbnum][2]:
                for y in range(shape_y):
                    if y >= boxes_cpu[bbnum][1] and y <= boxes_cpu[bbnum][3]:
                        if bbnum == 0:
                            zero_img[x][y] += scores_cpu[bbnum]
                        else:
                            if x >= 80 and y >= 80:
                                zero_img[x][y] += scores_cpu[bbnum]
                            else:
                                if zero_box_count > (add_input_channel/2):
                                    zero_img[x][y] += scores_cpu[bbnum]
        
    for x in range(shape_x):
        for y in range(shape_y):
            if zero_img[x][y] <= th:
                zero_img[x][y] = 0.0
            else:
                if x <= min_x:
                    min_x = x
                if y <= min_y:
                    min_y = y
                if x >= max_x:
                    max_x = x
                if y >= max_y:
                    max_y = y

    return zero_img, [min_x, min_y, max_x, max_y], select_boxes

def calculate_box_score(box_usenum, create_box, pred_boxes, pred_scores, number_slice=""):
    box_list = []
    score_list = []
    score_weight_list = []
    select_score_list = []
    select_score_weight_list = []
    min_x = create_box[0]
    min_y = create_box[1]
    max_x = create_box[2]
    max_y = create_box[3]
    c_width = max_x - min_x + 1
    c_height = max_y - min_y + 1
    for i in range(box_usenum):
        pred_box = pred_boxes[i]
        pred_score = pred_scores[i]
        if min_x < pred_box[0]:
            min_x = pred_box[0]
        if min_y < pred_box[1]:
            min_y = pred_box[1]
        if max_x > pred_box[2]:
            max_x = pred_box[2]
        if max_y > pred_box[3]:
            max_y = pred_box[3]

        ## check intersection
        is_inter = False
        if (max_x - min_x) >= 0:
            if (max_y - min_y) >= 0:
                is_inter = True
        if is_inter:
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            weight_score = ((width * height) / (c_width * c_height)) * pred_score 
            pred_box = pred_box.reshape(2, 2)
            box_list.append(pred_box)
            score_list.append(pred_score)
            score_weight_list.append(weight_score)
            select_score_list.append(pred_score)
            select_score_weight_list.append(weight_score)
        else:
            tmp_result = [" No ", " No "]
            box_list.append(tmp_result)
            score_list.append(pred_score)
            score_weight_list.append(0)
    return box_list, score_list, score_weight_list, select_score_list, select_score_weight_list
