from re import S
from tkinter.messagebox import NO
import openpyxl
# from DETR.calculate import patient_score, patient_score_boxinfo
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import cv2
import numpy as np
import time
import nrrd
from calculate import *
from img_process import *

def train_epoch(args, postprocessors, model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                postprocessors_2, model_2: torch.nn.Module, criterion_2: torch.nn.Module, optimizer_2: torch.optim.Optimizer, device: torch.device, epoch: int, max_norm: float = 0, ):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    model_2.train()
    criterion_2.train()
    metric_logger_2 = utils.MetricLogger(delimiter="  ")
    metric_logger_2.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger_2.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    DIR_Train_S1 = f"{args.output_dir}IMG_Result/Train_S1/"      
    os.makedirs(DIR_Train_S1, exist_ok=True)
    DIR_Train_S2 = f"{args.output_dir}IMG_Result/Train_S2/"      
    os.makedirs(DIR_Train_S2, exist_ok=True)

    print_freq = 10
    idx_data = 0
    idx_data_2 = 0
    img_save_count = 0
    img_max_num = 5
    if "Test" in args.output_dir:
        img_max_num = 500
    img_num_2 = 0
    img_save_name = []
    epoch_score_1 = 0
    epoch_score_2 = 0

    # multiconv = conv.Multi_Conv(5).to(device)
    CUT_Imgs = []
    Cut_Imgs_ori = []
    Cut_GT_marks = []
    Cut_GT_marks_ori = []
    Cut_PD_marks_ori = []
    Cut_marks = []
    Cut_Colon_marks_ori = []
    Cut_length = []
    Cut_ratio = []
    Cut_Number_list = []

    S1_GT_marks = []
    S1_PD_marks = []
    S2_GT_marks = []
    S2_PD_marks = []
    Tumor_Num_Total_s1 = 0
    Tumor_Num_Detect_s1 = 0
    Tumor_Num_Detect_Zero_s1 = 0
    Tumor_Num_Total_s2 = 0
    Tumor_Num_Detect_s2 = 0
    Tumor_Num_Detect_Zero_s2 = 0
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        input_tensor = batch['ct'].to(device)
        number_list = batch['number']
        gt_landmarks = batch['landmarks'].to(device) 

        cur_bath_size = input_tensor.shape[0]
        idx_data += cur_bath_size
        shape_x = input_tensor.shape[2]
        shape_y = input_tensor.shape[3]
            
        # input_slices = input_tensor.cpu().numpy()
        input_tensor_cpu = input_tensor.detach().cpu().numpy()
        gt_landmarks_cpu = gt_landmarks.cpu().numpy()
    
        input_slices_rgb = []
        for num in range(input_tensor.shape[0]):
            tmp_img = input_tensor_cpu[num]  ## (1, 512, 512)
            tmp_img = tmp_img.squeeze(0)
            # tmp_img = cv2.resize(tmp_img, (224, 224), interpolation=cv2.INTER_CUBIC)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
            input_slices_rgb.append(tmp_img)
        input_tensor_rgb = torch.tensor(input_slices_rgb).to(device= device)
        input_tensor_rgb = input_tensor_rgb.transpose(1,3)
        input_tensor_rgb = input_tensor_rgb.transpose(2,3)
        # input_tensor_rgb = input_tensor_rgb.to(device)
            
        targets_s1 = []
        d = {}
        for idx in range(cur_bath_size):
            if gt_landmarks_cpu[idx][0][0] == gt_landmarks_cpu[idx][1][0]:
                gt_landmarks_cpu[idx][1][0] += 1
            if gt_landmarks_cpu[idx][0][1] == gt_landmarks_cpu[idx][1][1]:
                gt_landmarks_cpu[idx][1][1] += 1
            center_x = ((gt_landmarks_cpu[idx][0][0] + gt_landmarks_cpu[idx][1][0]) / 2) / shape_x
            center_y = ((gt_landmarks_cpu[idx][0][1] + gt_landmarks_cpu[idx][1][1]) / 2) / shape_y
            bbox_w = (gt_landmarks_cpu[idx][1][0] - gt_landmarks_cpu[idx][0][0]) / shape_x
            bbox_y = (gt_landmarks_cpu[idx][1][1] - gt_landmarks_cpu[idx][0][1]) / shape_y

            bbox = [center_x, center_y, bbox_w, bbox_y]
            if "ZERO" not in number_list[idx]:
                label = [1]
                Tumor_Num_Total_s1 += 1
            else:
                label = [args.zero_label]
            gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
            d['boxes'] = gt_bbox
            d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
            targets_s1.append(d)  
            d = {}   

        outputs = model(input_tensor_rgb) ## return out, hs[-1].squeeze()  # return out, [100, 256]

        loss_dict = criterion(outputs[0], targets_s1) ## return losses, indices
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[0][k] * weight_dict[k] for k in loss_dict[0].keys() if k in weight_dict)
        loss_dict_reduced = utils.reduce_dict(loss_dict[0])
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("")
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        t_sizes = []
        for t_num in range(cur_bath_size):
            t_sizes.append([512, 512])
        orig_target_sizes = torch.tensor(t_sizes, dtype=torch.int64, device=device)
        results = postprocessors['bbox'](outputs[0], orig_target_sizes)
        is_s1_save = True
        for idx_all in range(len(results)):
            is_pred_zero = False
            best_idx = int(torch.argmax(results[idx_all]['scores']))
            # pred_landmarks = results[0]['boxes'][best_idx].cpu().numpy().reshape(2, 2)
            epoch_score_1 += float(results[idx_all]['scores'][best_idx])

            boxes = results[idx_all]['boxes']
            scores = results[idx_all]['scores']
            boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
            scores_cpu_ori = scores.cpu().numpy()
            boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

            if args.box_mode == "SB":
                if scores_cpu[0] >= args.box_th:
                    min_x = boxes_cpu[0][0]
                    if min_x < 0:
                        min_x = 0
                    min_y = boxes_cpu[0][1]
                    if min_y < 0:
                        min_y = 0
                    max_x = boxes_cpu[0][2]
                    max_y = boxes_cpu[0][3]
                    if min_x == max_x:
                        max_x += 1
                    if max_y == min_y:
                        max_y += 1
                else:
                    min_x = 0
                    min_y = 0
                    max_x = 1
                    max_y = 1
            else:
                zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                box_score = np.max(zero_img)
                min_x = 512
                min_y = 512
                max_x = 0
                max_y = 0

                if box_score > args.box_th:
                    for x in range(create_box_points[0], create_box_points[2] + 1):
                        for y in range(create_box_points[1], create_box_points[3] + 1):
                            if zero_img[x][y] != 0.0:
                                if x <= min_x:
                                    min_x = x
                                if y <= min_y:
                                    min_y = y
                                if x >= max_x:
                                    max_x = x
                                if y >= max_y:
                                    max_y = y

                ## if detr can't found colon
                if min_x == 512:
                    min_x = 0
                    min_y = 0
                    max_x = 1
                    max_y = 1
                if min_x == max_x:
                    max_x += 1
                if max_y == min_y:
                    max_y += 1
            if max_y < 80 and "ZERO" in number_list[idx_all]:                      ## zero detect
                Tumor_Num_Detect_Zero_s1 += 1
                
            elif max_y > 100 and "ZERO" not in number_list[idx_all]:
                Tumor_Num_Detect_s1 += 1

            pred_landmarks = [[min_x, min_y],[max_x, max_y]]
            S1_GT_marks.append(gt_landmarks_cpu[idx_all])
            S1_PD_marks.append(pred_landmarks)
            
            patient = f"{number_list[idx_all].split('_')[0]}_{number_list[idx_all].split('_')[1][4:]}"
            slice_num = int(number_list[idx_all].split('_')[2])
            if max_y > 80: #### pd tumor
                contor_list, contour_num = get_include_landmarks(args.data_seg_contour, patient, slice_num, pred_landmarks)
            else:          #### pd normal
                contor_list, contour_num = get_noraml_landmarks(args.data_seg_contour, patient, slice_num)
                is_pred_zero = True
            for c_idx in range(contour_num):
                try:                
                    min_x_over = contor_list[c_idx][0][0]
                    min_y_over = contor_list[c_idx][0][1]
                    max_x_over = contor_list[c_idx][1][0]
                    max_y_over = contor_list[c_idx][1][1]
                except:
                    print('')
                    print(f"{patient} - {slice_num}")
                    print(f"{c_idx} {contour_num} == {len(contor_list)} / min : {min_x} {min_y} / max : {max_x} {max_y}")
                    print(len(contor_list))
                    print(contor_list[0])
                    print(contor_list[1])
                    exit()
                gt_segmark = [[min_x_over, min_y_over],[max_x_over, max_y_over]]
                
                TP, _, _, _ = cal_score(gt_segmark, pred_landmarks, (shape_x, shape_y))
                if max_y_over > 50:
                    # correct_per = TP/((gt_segmarks[idx_all][1][0]-gt_segmarks[idx_all][0][0])*(gt_segmarks[idx_all][1][1]-gt_segmarks[idx_all][0][1]))
                    correct_per = TP/((gt_segmark[1][0]-gt_segmark[0][0])*(gt_segmark[1][1]-gt_segmark[0][1]))
                    idx = idx_all
                    # if (correct_per >= 0.01 and "ZERO" not in number_list[idx]) or is_pred_zero:
                    if (correct_per >= 0.01) or is_pred_zero:
                        cut_margin_min_x = 20
                        cut_margin_min_y = 20
                        cut_margin_max_x = 20
                        cut_margin_max_y = 20
                 
                        min_x_cut = gt_segmark[0][0]
                        min_y_cut = gt_segmark[0][1]
                        max_x_cut = gt_segmark[1][0]
                        max_y_cut = gt_segmark[1][1]
                        
                        ## square box
                        min_x_cut_ori = min_x_cut
                        min_y_cut_ori = min_y_cut
                        max_x_cut_ori = max_x_cut
                        max_y_cut_ori = max_y_cut

                        cut_length_x = max_x_cut - min_x_cut
                        cut_length_y = max_y_cut - min_y_cut

                        if cut_length_x < cut_length_y:
                            max_x_cut += int((cut_length_y - cut_length_x)/2)
                            if (cut_length_y - cut_length_x) % 2 == 1:
                                max_x_cut += 1
                            min_x_cut -= int((cut_length_y - cut_length_x)/2)
                            if max_x_cut > 510:
                                max_x_cut = max_x_cut_ori
                                min_x_cut = min_x_cut_ori - (cut_length_y - cut_length_x)
                                if min_x_cut < 1:
                                    cut_adj = 1 - min_x_cut 
                                    max_x_cut += cut_adj
                                    min_x_cut += cut_adj
                            if min_x_cut < 1:
                                max_x_cut = max_x_cut_ori + (cut_length_y - cut_length_x)
                                min_x_cut = min_x_cut_ori
                                if max_x_cut > 510:
                                    cut_adj = max_x_cut - 510 
                                    max_x_cut -= cut_adj
                                    min_x_cut += cut_adj
                        else:
                            max_y_cut += int((cut_length_x - cut_length_y)/2)
                            min_y_cut -= int((cut_length_x - cut_length_y)/2)
                            if (cut_length_x - cut_length_y) % 2 == 1:
                                max_y_cut += 1

                            if max_y_cut > 510:
                                max_y_cut = max_y_cut_ori
                                min_y_cut = min_y_cut_ori - (cut_length_x - cut_length_y)
                                if min_y_cut < 1:
                                    cut_adj = 1 - min_y_cut 
                                    max_y_cut += cut_adj
                                    min_y_cut += cut_adj
                            if min_y_cut < 1:
                                max_y_cut = max_y_cut_ori + (cut_length_x - cut_length_y)
                                min_y_cut = min_y_cut_ori
                                if max_y_cut > 510:
                                    cut_adj = max_y_cut - 510 
                                    max_y_cut -= cut_adj
                                    min_y_cut -= cut_adj

                        if min_x_cut < 20:
                            cut_margin_min_x = 0
                        if min_y_cut < 20:              
                            cut_margin_min_y = 0
                        if max_x_cut > 491:  
                            cut_margin_max_x = 0
                        if max_y_cut > 491:  
                            cut_margin_max_y = 0

                        cut_length_x = max_x_cut - min_x_cut
                        cut_length_y = max_y_cut - min_y_cut
                        if cut_length_x != cut_length_y:
                            print("x = ", cut_length_x)
                            print(max_x_cut, min_x_cut)
                            print("y = ", cut_length_y)
                            print(max_y_cut, min_y_cut)
                            exit()          

                        cut_img = input_tensor_cpu[idx][0][min_y_cut-cut_margin_min_y:max_y_cut+cut_margin_max_y,min_x_cut-cut_margin_min_x:max_x_cut+cut_margin_max_x]
                        
                        TP_contour, _, _, _ = cal_score(gt_landmarks_cpu[idx], [[min_x_cut, min_y_cut],[max_x_cut, max_y_cut]], (shape_x, shape_y))
                        if TP_contour > 0:
                            gt_cut_min_x = gt_landmarks_cpu[idx][0][0] - (min_x_cut - cut_margin_min_x)
                            if gt_landmarks_cpu[idx][0][0] < (min_x_cut-cut_margin_min_x):
                                gt_cut_min_x = 5              
                            gt_cut_min_y = gt_landmarks_cpu[idx][0][1] - (min_y_cut - cut_margin_min_y)
                            if gt_landmarks_cpu[idx][0][1] < (min_y_cut-cut_margin_min_y):
                                gt_cut_min_y = 5 
                            gt_cut_max_x = gt_landmarks_cpu[idx][1][0] - (gt_landmarks_cpu[idx][0][0] - (gt_landmarks_cpu[idx][0][0] - (min_x_cut - cut_margin_min_x)))
                            if gt_landmarks_cpu[idx][1][0] > (max_x_cut+cut_margin_max_x):
                                gt_cut_max_x = (max_x_cut+cut_margin_max_x) - (min_x_cut-cut_margin_min_x) - 5
                            gt_cut_max_y = int(gt_landmarks_cpu[idx][1][1]) - (gt_landmarks_cpu[idx][0][1] - (gt_landmarks_cpu[idx][0][1] - (min_y_cut - cut_margin_min_y)))
                            if gt_landmarks_cpu[idx][1][1] > (max_y_cut+cut_margin_max_y):
                                gt_cut_max_y = (max_y_cut+cut_margin_max_y) - (min_y_cut-cut_margin_min_y) - 5
                        else:
                            gt_cut_min_x = 0
                            gt_cut_min_y = 0
                            gt_cut_max_x = 1
                            gt_cut_max_y = 1

                        s1_cut_marks = [[min_x_cut-cut_margin_min_x, min_y_cut-cut_margin_min_y],[max_x_cut+cut_margin_max_x, max_y_cut+cut_margin_max_y]]
                        # SEG_Imgs_All.append(seg_imgs[idx])
                        if img_save_count <= img_max_num and is_s1_save:
                            img_save_count += 1
                            is_s1_save = False
                            if "Test" in DIR_Train_S1:
                                is_s1_save = True
                            
                            img_save_name.append(f"{number_list[idx]}")
                            seg_colon = f"{number_list[idx].split('_')[0]}_{number_list[idx].split('_')[1][4:]}"
                            slice_num = int(number_list[idx].split('_')[2])
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            try:
                                target_seg = seg_data[:,:,slice_num]
                                target_seg = np.flip(target_seg, axis=0)
                                target_seg = np.rot90(target_seg, 3)
                            except:
                                print("")
                                print(f"{number_list[idx]} colon data check")
                                print(seg_data.shape)
                                print("")
                                exit()

                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Train_S1}E{epoch}_{number_list[idx]}_{c_idx}.tiff"
                            gt_cut_marks = [[gt_cut_min_x, gt_cut_min_y], [gt_cut_max_x, gt_cut_max_y]]
                            
                            save_1st(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks, gt_cut_marks, s1_cut_marks, dir_save)
                        
                        shape_y_cut, shape_x_cut = cut_img.shape
                        cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                        gt_cut_min_x = int(gt_cut_min_x*(256 / shape_x_cut))
                        gt_cut_min_y = int(gt_cut_min_y*(256 / shape_y_cut))
                        gt_cut_max_x = int(gt_cut_max_x*(256 / shape_x_cut))
                        gt_cut_max_y = int(gt_cut_max_y*(256 / shape_y_cut))

                        if gt_cut_max_y < 10:
                            gt_cut_min_x = 0
                            gt_cut_min_y = 0
                            gt_cut_max_x = 1
                            gt_cut_max_y = 1

                        CUT_Imgs.append(cut_img)
                        Cut_GT_marks.append(np.array([[gt_cut_min_x,gt_cut_min_y],[gt_cut_max_x,gt_cut_max_y]]))         
                        Cut_Number_list.append(number_list[idx])       
                        # Cut_Colon_marks_ori.append(gt_segmarks[idx])
                        Cut_Colon_marks_ori.append(gt_segmark)
                        Cut_Imgs_ori.append(input_tensor_cpu[idx][0])
                        Cut_GT_marks_ori.append(gt_landmarks_cpu[idx])  
                        Cut_length.append([[(min_x_cut-cut_margin_min_x), (min_y_cut-cut_margin_min_y)],[(gt_landmarks_cpu[idx][0][0] - (gt_landmarks_cpu[idx][0][0] - (min_x_cut - cut_margin_min_x))), 
                                            (gt_landmarks_cpu[idx][0][1] - (gt_landmarks_cpu[idx][0][1] - (min_y_cut - cut_margin_min_y)))]])   
                        Cut_ratio.append([(256 / shape_x_cut), (256 / shape_y_cut)])
                        Cut_PD_marks_ori.append(pred_landmarks)
                        Cut_marks.append(s1_cut_marks)
                                                
                        ## 2nd stage
                        # if len(CUT_Imgs) < 0:
                        
                        if len(CUT_Imgs) == args.batch_size or cur_bath_size < args.batch_size:  
                            idx_data_2 += int(len(CUT_Imgs))
                            
                            targets = []
                            d = {} 
                            t_sizes_2 = []
                            for t_num in range(len(CUT_Imgs)):
                                t_sizes_2.append([256, 256])
                            for s_num in range(len(CUT_Imgs)): 
                                bbox = []   
                                if Cut_GT_marks[s_num][0][0] >= Cut_GT_marks[s_num][1][0]:
                                    Cut_GT_marks[s_num][1][0] = Cut_GT_marks[s_num][0][0] + 1
                                if Cut_GT_marks[s_num][0][1] >= Cut_GT_marks[s_num][1][1]:
                                    Cut_GT_marks[s_num][1][1] = Cut_GT_marks[s_num][0][1] + 1
                                center_x = ((Cut_GT_marks[s_num][0][0] + Cut_GT_marks[s_num][1][0]) / 2) / 256
                                center_y = ((Cut_GT_marks[s_num][0][1] + Cut_GT_marks[s_num][1][1]) / 2) / 256
                                bbox_w = (Cut_GT_marks[s_num][1][0] - Cut_GT_marks[s_num][0][0]) / 256
                                bbox_y = (Cut_GT_marks[s_num][1][1] - Cut_GT_marks[s_num][0][1]) / 256
                                bbox = [center_x, center_y, bbox_w, bbox_y]                            

                                if "ZERO" not in Cut_Number_list[s_num]:
                                    label = [1]
                                    Tumor_Num_Total_s2 += 1
                                else:
                                    label = [args.zero_label]
                                    
                                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                                d['boxes'] = gt_bbox
                                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                                targets.append(d)     
                                d = {}   
                            
                            input_slices_rgb = []
                            for num in range(len(CUT_Imgs)):
                                tmp_img = CUT_Imgs[num] 
                                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
                                input_slices_rgb.append(tmp_img)
                            input_tensor_rgb_2 = torch.tensor(input_slices_rgb).to(device= device)
                            input_tensor_rgb_2 = input_tensor_rgb_2.transpose(1,3)
                            input_tensor_rgb_2 = input_tensor_rgb_2.transpose(2,3)
                            outputs_2 = model_2(input_tensor_rgb_2)
                
                            loss_dict_2 = criterion_2(outputs_2[0], targets) ## return losses, indices
                            weight_dict = criterion_2.weight_dict
                            losses_2 = sum(loss_dict_2[0][k] * weight_dict[k] for k in loss_dict_2[0].keys() if k in weight_dict)
                            loss_dict_reduced = utils.reduce_dict(loss_dict_2[0])
                            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
                            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
                            loss_value_2 = losses_reduced_scaled.item()
                            print(f"Train {idx_data} / {len(data_loader.dataset)} Bbox_Loss = {round(loss_dict[0]['loss_bbox'].item(),4)} CIoU_Loss = {round(loss_dict[0]['loss_giou'].item(),4)} \
                                    2nd - Bbox_Loss = {round(loss_dict_2[0]['loss_bbox'].item(),4)} CIoU_Loss = {round(loss_dict_2[0]['loss_giou'].item(),4)}", end="\r")
                            
                            if not math.isfinite(loss_value_2):
                                print("")
                                print("Loss 2nd stageis {}, stopping training".format(loss_value_2))
                                print(loss_dict_reduced)
                                sys.exit(1)

                            optimizer_2.zero_grad()
                            losses_2.backward()
                            
                            if max_norm > 0:
                                torch.nn.utils.clip_grad_norm_(model_2.parameters(), max_norm)
                            optimizer_2.step()

                            metric_logger_2.update(loss=loss_value_2, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
                            metric_logger_2.update(class_error=loss_dict_reduced['class_error'])
                            metric_logger_2.update(lr=optimizer_2.param_groups[0]["lr"])

                            orig_target_sizes = torch.tensor(t_sizes_2, dtype=torch.int64, device=device)
            
                            results_2 = postprocessors_2['bbox'](outputs_2[0], orig_target_sizes)

                            for idx in range(len(results_2)):
                                best_idx = int(torch.argmax(results_2[idx]['scores']))
                                epoch_score_2 += float(results_2[idx]['scores'][best_idx])
                                # pred_landmarks = results_2[idx]['boxes'][best_idx].cpu().numpy().reshape(2, 2)

                                boxes = results_2[idx]['boxes']
                                scores = results_2[idx]['scores']
                                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                                scores_cpu_ori = scores.cpu().numpy()
                                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                                if args.box_mode == "SB":
                                    if scores_cpu[0] >= args.box_th:
                                        min_x = boxes_cpu[0][0]
                                        if min_x < 0:
                                            min_x = 0
                                        min_y = boxes_cpu[0][1]
                                        if min_y < 0:
                                            min_y = 0
                                        max_x = boxes_cpu[0][2]
                                        max_y = boxes_cpu[0][3]
                                        if min_x == max_x:
                                            max_x += 1
                                        if max_y == min_y:
                                            max_y += 1
                                    else:
                                        min_x = 0
                                        min_y = 0
                                        max_x = 1
                                        max_y = 1
                                else:
                                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                                    box_score = np.max(zero_img)
                                    min_x = 512
                                    min_y = 512
                                    max_x = 0
                                    max_y = 0

                                    if box_score > args.box_th:
                                        for x in range(create_box_points[0], create_box_points[2] + 1):
                                            for y in range(create_box_points[1], create_box_points[3] + 1):
                                                if zero_img[x][y] != 0.0:
                                                    if x <= min_x:
                                                        min_x = x
                                                    if y <= min_y:
                                                        min_y = y
                                                    if x >= max_x:
                                                        max_x = x
                                                    if y >= max_y:
                                                        max_y = y

                                    ## if detr can't found colon
                                    if min_x == 512:
                                        min_x = 0
                                        min_y = 0
                                        max_x = 1
                                        max_y = 1
                                    if min_x == max_x:
                                        max_x += 1
                                    if max_y == min_y:
                                        max_y += 1
                                if (max_y-min_y) == 1 and "ZERO" in Cut_Number_list[idx]:                      ## zero detect
                                    Tumor_Num_Detect_Zero_s2 += 1
                                elif max_y > 10 and "ZERO" not in Cut_Number_list[idx]:
                                    Tumor_Num_Detect_s2 += 1

                                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                                S2_GT_marks.append(Cut_GT_marks[idx])
                                S2_PD_marks.append(pred_landmarks)

                                if Cut_Number_list[idx] in img_save_name:
                                    img_num_2 += 1
                                    img_list = [CUT_Imgs[idx], Cut_Imgs_ori[idx]]
                                    dir_save = f"{DIR_Train_S2}E{epoch}_{Cut_Number_list[idx]}_{img_num_2}.tiff"
                                    save_2nd(img_list, Cut_GT_marks[idx], pred_landmarks, Cut_Colon_marks_ori[idx], Cut_GT_marks_ori[idx], Cut_PD_marks_ori[idx], Cut_marks[idx],
                                                    Cut_length[idx], Cut_ratio[idx], dir_save)


                            CUT_Imgs = []
                            Cut_Imgs_ori = []
                            Cut_GT_marks = []
                            Cut_Number_list = []
                            Cut_Colon_marks_ori = []
                            Cut_GT_marks_ori = []
                            Cut_length = []
                            Cut_ratio = []
                            Cut_PD_marks_ori = []
                            Cut_marks = []

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger_2.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    Iou_s1 = 0
    Dice_s1 = 0
    Speciticity_s1 = 0
    Sensitivity_s1 = 0
    Precision_s1 = 0
    for idx in range(len(S1_GT_marks)):
        TP, FP, TN, FN = cal_score(S1_GT_marks[idx], S1_PD_marks[idx], (512, 512))
        iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
        Iou_s1 += iou
        Dice_s1 += dice
        Speciticity_s1 += specitificity
        Sensitivity_s1 += sensitivity
        Precision_s1 += precision
    Iou_s1 /= float(len(S1_GT_marks))
    Dice_s1 /= float(len(S1_GT_marks))
    Speciticity_s1 /= float(len(S1_GT_marks))
    Sensitivity_s1 /= float(len(S1_GT_marks))
    Precision_s1 /= float(len(S1_GT_marks))

    Iou_s2 = 0
    Dice_s2 = 0
    Speciticity_s2 = 0
    Sensitivity_s2 = 0
    Precision_s2 = 0
    train_stats_s2 = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    Tumor_Num_Total_Zero_s2 = 0
    if len(S2_GT_marks) > 0:
        for idx in range(len(S2_GT_marks)):
            TP, FP, TN, FN = cal_score(S2_GT_marks[idx], S2_PD_marks[idx], (256, 256))
            iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
            Iou_s2 += iou
            Dice_s2 += dice
            Speciticity_s2 += specitificity
            Sensitivity_s2 += sensitivity
            Precision_s2 += precision
        Iou_s2 /= float(len(S2_GT_marks))
        Dice_s2 /= float(len(S2_GT_marks))
        Speciticity_s2 /= float(len(S2_GT_marks))
        Sensitivity_s2 /= float(len(S2_GT_marks))
        Precision_s2 /= float(len(S2_GT_marks))
        epoch_score_2 /= idx_data_2
        train_stats_s2 = {k: meter.global_avg for k, meter in metric_logger_2.meters.items()}

        Tumor_Num_Total_Zero_s2 = len(S2_GT_marks) - Tumor_Num_Total_s2
        
    epoch_score_1 /= idx_data
    
    train_stats_s1 = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    Tumor_Num_Total_Zero_s1 = len(data_loader.dataset) - Tumor_Num_Total_s1
    train_results_s1 = [Iou_s1, Dice_s1, Sensitivity_s1, Precision_s1, Speciticity_s1, Tumor_Num_Total_s1, Tumor_Num_Detect_s1, Tumor_Num_Total_Zero_s1, Tumor_Num_Detect_Zero_s1]
    train_results_s2 = [Iou_s2, Dice_s2, Sensitivity_s2, Precision_s2, Speciticity_s2, Tumor_Num_Total_s2, Tumor_Num_Detect_s2, Tumor_Num_Total_Zero_s2, Tumor_Num_Detect_Zero_s2]
    print("")
    return train_stats_s1, epoch_score_1, train_results_s1, train_stats_s2, epoch_score_2, train_results_s2, len(S2_GT_marks)

@torch.no_grad()
def evaluate(args, model, postprocessors, criterion, data_loader, model_2, postprocessors_2, criterion_2, epoch, device):
    # data_loader에는 coco val 이미지 5000장에 대한 정보가 들어가 있다. type( util.misc.NestedTensor, dict ) 
    model.eval()
    criterion.eval()
    model_2.eval()
    criterion_2.eval()
    """
    category_query = torch.load('category_query.pth', map_location=torch.device('cpu'))
    mask = torch.isnan(category_query)
    category_query[mask] = 0
    category_query = category_query.cuda(device)

    weight = torch.nn.Parameter( category_query )
    model.query_embed.weight = weight
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger_2 = utils.MetricLogger(delimiter="  ")
    metric_logger_2.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    DIR_Val_S1 = f"{args.output_dir}IMG_Result/Val_S1/"      
    os.makedirs(DIR_Val_S1, exist_ok=True)
    DIR_Val_S2 = f"{args.output_dir}IMG_Result/Val_S2/"      
    os.makedirs(DIR_Val_S2, exist_ok=True)

    print_freq = 10
    idx_data = 0
    idx_data_2 = 0
    img_save_count = 0
    img_max_num = 20
    img_save_name = []

    epoch_score_1 = 0
    epoch_score_2 = 0

    # multiconv = conv.Multi_Conv(5).to(device)
    CUT_Imgs = []
    Cut_Imgs_ori = []
    Cut_GT_marks = []
    Cut_GT_marks_ori = []
    Cut_PD_marks_ori = []
    Cut_marks = []
    Cut_Colon_marks_ori = []
    Cut_length = []
    Cut_ratio = []
    Cut_Number_list = []

    S1_GT_marks = []
    S1_PD_marks = []
    S2_GT_marks = []
    S2_PD_marks = []
    Tumor_Num_Total_s1 = 0
    Tumor_Num_Detect_s1 = 0
    Tumor_Num_Detect_Zero_s1 = 0
    Tumor_Num_Total_s2 = 0
    Tumor_Num_Detect_s2 = 0
    Tumor_Num_Detect_Zero_s2 = 0
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            input_tensor = batch['ct'].to(device)
            number_list = batch['number']
            gt_landmarks = batch['landmarks'].to(device) 

            cur_bath_size = input_tensor.shape[0]
            idx_data += cur_bath_size
            shape_x = input_tensor.shape[2]
            shape_y = input_tensor.shape[3]
                
            input_tensor_cpu = input_tensor.detach().cpu().numpy()
            gt_landmarks_cpu = gt_landmarks.cpu().numpy()

            input_slices_rgb = []
            for num in range(input_tensor.shape[0]):
                tmp_img = input_tensor_cpu[num]  ## (1, 512, 512)
                tmp_img = tmp_img.squeeze(0)
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
                input_slices_rgb.append(tmp_img)
            input_tensor_rgb = torch.tensor(input_slices_rgb).to(device= device)
            input_tensor_rgb = input_tensor_rgb.transpose(1,3)
            input_tensor_rgb = input_tensor_rgb.transpose(2,3)
            
            targets_s1 = []
            d = {}
            for idx in range(cur_bath_size):    
                if gt_landmarks_cpu[idx][0][0] == gt_landmarks_cpu[idx][1][0]:
                    gt_landmarks_cpu[idx][1][0] += 1
                if gt_landmarks_cpu[idx][0][1] == gt_landmarks_cpu[idx][1][1]:
                    gt_landmarks_cpu[idx][1][1] += 1
                center_x = ((gt_landmarks_cpu[idx][0][0] + gt_landmarks_cpu[idx][1][0]) / 2) / shape_x
                center_y = ((gt_landmarks_cpu[idx][0][1] + gt_landmarks_cpu[idx][1][1]) / 2) / shape_y
                bbox_w = (gt_landmarks_cpu[idx][1][0] - gt_landmarks_cpu[idx][0][0]) / shape_x
                bbox_y = (gt_landmarks_cpu[idx][1][1] - gt_landmarks_cpu[idx][0][1]) / shape_y
                bbox = [center_x, center_y, bbox_w, bbox_y]
               
                if "ZERO" not in number_list[idx]:
                    label = [1]
                    Tumor_Num_Total_s1 += 1
                else:
                    label = [args.zero_label]
                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                d['boxes'] = gt_bbox
                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                targets_s1.append(d)  
                d = {}   

            outputs, decoder_out = model(input_tensor_rgb)
            loss_dict, indices = criterion(outputs, targets_s1) ## return losses, indices
            weight_dict = criterion.weight_dict
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                **loss_dict_reduced_scaled,
                                **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            
            t_sizes = []
            for t_num in range(cur_bath_size):
                t_sizes.append([shape_y, shape_x])
            orig_target_sizes = torch.tensor(t_sizes, dtype=torch.int64, device=device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            is_s1_save = True
            for idx_all in range(len(results)):
                is_pred_zero = False
                best_idx = int(torch.argmax(results[idx_all]['scores']))
                # pred_landmarks = results[0]['boxes'][best_idx].cpu().numpy().reshape(2, 2)
                epoch_score_1 += float(results[idx_all]['scores'][best_idx])

                boxes = results[idx_all]['boxes']
                scores = results[idx_all]['scores']
                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                scores_cpu_ori = scores.cpu().numpy()
                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                if args.box_mode == "SB":
                    if scores_cpu[0] >= args.box_th:
                        min_x = boxes_cpu[0][0]
                        if min_x < 0:
                            min_x = 0
                        min_y = boxes_cpu[0][1]
                        if min_y < 0:
                            min_y = 0
                        max_x = boxes_cpu[0][2]
                        max_y = boxes_cpu[0][3]
                        if min_x == max_x:
                            max_x += 1
                        if max_y == min_y:
                            max_y += 1
                    else:
                        min_x = 0
                        min_y = 0
                        max_x = 1
                        max_y = 1
                else:
                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                    box_score = np.max(zero_img)
                    min_x = 512
                    min_y = 512
                    max_x = 0
                    max_y = 0

                    if box_score > args.box_th:
                        for x in range(create_box_points[0], create_box_points[2] + 1):
                            for y in range(create_box_points[1], create_box_points[3] + 1):
                                if zero_img[x][y] != 0.0:
                                    if x <= min_x:
                                        min_x = x
                                    if y <= min_y:
                                        min_y = y
                                    if x >= max_x:
                                        max_x = x
                                    if y >= max_y:
                                        max_y = y

                    ## if detr can't found colon
                    if min_x == 512:
                        min_x = 0
                        min_y = 0
                        max_x = 1
                        max_y = 1
                    if min_x == max_x:
                        max_x += 1
                    if max_y == min_y:
                        max_y += 1
                if max_y < 80 and "ZERO" in number_list[idx_all]:                      ## zero detect
                    Tumor_Num_Detect_Zero_s1 += 1
                    
                elif max_y > 100 and "ZERO" not in number_list[idx_all]:
                    Tumor_Num_Detect_s1 += 1

                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                S1_GT_marks.append(gt_landmarks_cpu[idx_all])
                S1_PD_marks.append(pred_landmarks)

                patient = f"{number_list[idx_all].split('_')[0]}_{number_list[idx_all].split('_')[1][4:]}"
                slice_num = int(number_list[idx_all].split('_')[2])
                if max_y > 80: #### pd tumor
                    contor_list, contour_num = get_include_landmarks(args.data_seg_contour, patient, slice_num, pred_landmarks)
                else:          #### pd normal
                    contor_list, contour_num = get_noraml_landmarks(args.data_seg_contour, patient, slice_num)
                    is_pred_zero = True
                for c_idx in range(contour_num):
                    try:                
                        min_x_over = contor_list[c_idx][0][0]
                        min_y_over = contor_list[c_idx][0][1]
                        max_x_over = contor_list[c_idx][1][0]
                        max_y_over = contor_list[c_idx][1][1]
                    except:
                        print('')
                        print(f"{patient} - {slice_num}")
                        print(f"{c_idx} {contour_num} == {len(contor_list)} / min : {min_x} {min_y} / max : {max_x} {max_y}")
                        print(len(contor_list))
                        print(contor_list[0])
                        print(contor_list[1])
                        exit()
                gt_segmark = [[min_x_over, min_y_over],[max_x_over, max_y_over]]
                
                TP, _, _, _ = cal_score(gt_segmark, pred_landmarks, (shape_x, shape_y))
                if max_y_over > 50:
                    correct_per = TP/((gt_segmark[1][0]-gt_segmark[0][0])*(gt_segmark[1][1]-gt_segmark[0][1]))
                    idx = idx_all
                    
                    # if (correct_per >= 0.01 and "ZERO" not in number_list[idx]) or is_pred_zero:
                    if (correct_per >= 0.01) or is_pred_zero:
                        cut_margin_min_x = 20
                        cut_margin_min_y = 20
                        cut_margin_max_x = 20
                        cut_margin_max_y = 20
                       
                        min_x_cut = gt_segmark[0][0]
                        min_y_cut = gt_segmark[0][1]
                        max_x_cut = gt_segmark[1][0]
                        max_y_cut = gt_segmark[1][1]
                        
                        ## square box
                        min_x_cut_ori = min_x_cut
                        min_y_cut_ori = min_y_cut
                        max_x_cut_ori = max_x_cut
                        max_y_cut_ori = max_y_cut
                        cut_length_x = max_x_cut - min_x_cut
                        cut_length_y = max_y_cut - min_y_cut
                        if cut_length_x < cut_length_y:
                            max_x_cut += int((cut_length_y - cut_length_x)/2)
                            if (cut_length_y - cut_length_x) % 2 == 1:
                                max_x_cut += 1
                            min_x_cut -= int((cut_length_y - cut_length_x)/2)
                            if max_x_cut > 510:
                                max_x_cut = max_x_cut_ori
                                min_x_cut = min_x_cut_ori - (cut_length_y - cut_length_x)
                                if min_x_cut < 1:
                                    cut_adj = 1 - min_x_cut 
                                    max_x_cut += cut_adj
                                    min_x_cut += cut_adj
                            if min_x_cut < 1:
                                max_x_cut = max_x_cut_ori + (cut_length_y - cut_length_x)
                                min_x_cut = min_x_cut_ori
                                if max_x_cut > 510:
                                    cut_adj = max_x_cut - 510 
                                    max_x_cut -= cut_adj
                                    min_x_cut += cut_adj
                        else:
                            max_y_cut += int((cut_length_x - cut_length_y)/2)
                            min_y_cut -= int((cut_length_x - cut_length_y)/2)
                            if (cut_length_x - cut_length_y) % 2 == 1:
                                max_y_cut += 1

                            if max_y_cut > 510:
                                max_y_cut = max_y_cut_ori
                                min_y_cut = min_y_cut_ori - (cut_length_x - cut_length_y)
                                if min_y_cut < 1:
                                    cut_adj = 1 - min_y_cut 
                                    max_y_cut += cut_adj
                                    min_y_cut += cut_adj
                            if min_y_cut < 1:
                                max_y_cut = max_y_cut_ori + (cut_length_x - cut_length_y)
                                min_y_cut = min_y_cut_ori
                                if max_y_cut > 510:
                                    cut_adj = max_y_cut - 510 
                                    max_y_cut -= cut_adj
                                    min_y_cut -= cut_adj

                        if min_x_cut < 20:
                            cut_margin_min_x = 0
                        if min_y_cut < 20:              
                            cut_margin_min_y = 0
                        if max_x_cut > 491:  
                            cut_margin_max_x = 0
                        if max_y_cut > 491:  
                            cut_margin_max_y = 0

                        cut_length_x = max_x_cut - min_x_cut
                        cut_length_y = max_y_cut - min_y_cut
                        if cut_length_x != cut_length_y:
                            print("")
                            print(number_list[idx_all])
                            print("x = ", cut_length_x)
                            print(max_x_cut, min_x_cut)
                            print("y = ", cut_length_y)
                            print(max_y_cut, min_y_cut)
                            exit()

                        cut_img = input_tensor_cpu[idx][0][min_y_cut-cut_margin_min_y:max_y_cut+cut_margin_max_y,min_x_cut-cut_margin_min_x:max_x_cut+cut_margin_max_x]
                        TP_contour, _, _, _ = cal_score(gt_landmarks_cpu[idx], [[min_x_cut, min_y_cut],[max_x_cut, max_y_cut]], (shape_x, shape_y))
                        if TP_contour > 0:
                            gt_cut_min_x = gt_landmarks_cpu[idx][0][0] - (min_x_cut - cut_margin_min_x)
                            if gt_landmarks_cpu[idx][0][0] < (min_x_cut-cut_margin_min_x):
                                gt_cut_min_x = 5              
                            gt_cut_min_y = gt_landmarks_cpu[idx][0][1] - (min_y_cut - cut_margin_min_y)
                            if gt_landmarks_cpu[idx][0][1] < (min_y_cut-cut_margin_min_y):
                                gt_cut_min_y = 5 
                            gt_cut_max_x = gt_landmarks_cpu[idx][1][0] - (gt_landmarks_cpu[idx][0][0] - (gt_landmarks_cpu[idx][0][0] - (min_x_cut - cut_margin_min_x)))
                            if gt_landmarks_cpu[idx][1][0] > (max_x_cut+cut_margin_max_x):
                                gt_cut_max_x = (max_x_cut+cut_margin_max_x) - (min_x_cut-cut_margin_min_x) - 5
                            gt_cut_max_y = int(gt_landmarks_cpu[idx][1][1]) - (gt_landmarks_cpu[idx][0][1] - (gt_landmarks_cpu[idx][0][1] - (min_y_cut - cut_margin_min_y)))
                            if gt_landmarks_cpu[idx][1][1] > (max_y_cut+cut_margin_max_y):
                                gt_cut_max_y = (max_y_cut+cut_margin_max_y) - (min_y_cut-cut_margin_min_y) - 5
                        else:
                            gt_cut_min_x = 0
                            gt_cut_min_y = 0
                            gt_cut_max_x = 1
                            gt_cut_max_y = 1
                        s1_cut_marks = [[min_x_cut-cut_margin_min_x, min_y_cut-cut_margin_min_y],[max_x_cut+cut_margin_max_x, max_y_cut+cut_margin_max_y]]
                        # SEG_Imgs_All.append(seg_imgs[idx])
                        if img_save_count <= img_max_num and is_s1_save:
                            img_save_count += 1
                            is_s1_save = False
                            img_save_name.append(number_list[idx])

                            seg_colon = f"{number_list[idx].split('_')[0]}_{number_list[idx].split('_')[1][4:]}"
                            slice_num = int(number_list[idx].split('_')[2])
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                
                            try:
                                target_seg = seg_data[:,:,slice_num]
                                target_seg = np.flip(target_seg, axis=0)
                                target_seg = np.rot90(target_seg, 3)
                            except:
                                print("")
                                print(f"{number_list[idx]} colon data check")
                                print(seg_data.shape)
                                print("")
                                exit()

                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_S1}E{epoch}_{number_list[idx]}_{c_idx}.tiff"
                            gt_cut_marks = [[gt_cut_min_x, gt_cut_min_y], [gt_cut_max_x, gt_cut_max_y]]
                            
                            save_1st(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks, gt_cut_marks, s1_cut_marks, dir_save)
                        
                        shape_y_cut, shape_x_cut = cut_img.shape
                        cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                        gt_cut_min_x = int(gt_cut_min_x*(256 / shape_x_cut))
                        gt_cut_min_y = int(gt_cut_min_y*(256 / shape_y_cut))
                        gt_cut_max_x = int(gt_cut_max_x*(256 / shape_x_cut))
                        gt_cut_max_y = int(gt_cut_max_y*(256 / shape_y_cut))

                        if gt_cut_max_y < 10:
                            gt_cut_min_x = 0
                            gt_cut_min_y = 0
                            gt_cut_max_x = 1
                            gt_cut_max_y = 1

                        CUT_Imgs.append(cut_img)
                        Cut_GT_marks.append(np.array([[gt_cut_min_x,gt_cut_min_y],[gt_cut_max_x,gt_cut_max_y]]))         
                        Cut_Number_list.append(number_list[idx])       
                        Cut_Colon_marks_ori.append(gt_segmark)
                        Cut_Imgs_ori.append(input_tensor_cpu[idx][0])
                        Cut_GT_marks_ori.append(gt_landmarks_cpu[idx])  
                        Cut_length.append([[(min_x_cut-cut_margin_min_x), (min_y_cut-cut_margin_min_y)],[(gt_landmarks_cpu[idx][0][0] - (gt_landmarks_cpu[idx][0][0] - (min_x_cut - cut_margin_min_x))), 
                                            (gt_landmarks_cpu[idx][0][1] - (gt_landmarks_cpu[idx][0][1] - (min_y_cut - cut_margin_min_y)))]])   
                        Cut_ratio.append([(256 / shape_x_cut), (256 / shape_y_cut)])
                        Cut_PD_marks_ori.append(pred_landmarks)
                        Cut_marks.append(s1_cut_marks)
                                                
                        ## 2nd stage
                        # if len(CUT_Imgs) < 0:
                        if len(CUT_Imgs) == args.batch_size or cur_bath_size < args.batch_size:  
                            idx_data_2 += int(len(CUT_Imgs))
        
                            targets = []
                            d = {} 
                            t_sizes_2 = []
                            for t_num in range(len(CUT_Imgs)):
                                t_sizes_2.append([256, 256])
                            for s_num in range(len(CUT_Imgs)): 
                                bbox = []  
                                if Cut_GT_marks[s_num][0][0] >= Cut_GT_marks[s_num][1][0]:
                                    Cut_GT_marks[s_num][1][0] = Cut_GT_marks[s_num][0][0] + 1
                                if Cut_GT_marks[s_num][0][1] >= Cut_GT_marks[s_num][1][1]:
                                    Cut_GT_marks[s_num][1][1] = Cut_GT_marks[s_num][0][1] + 1 
                                center_x = ((Cut_GT_marks[s_num][0][0] + Cut_GT_marks[s_num][1][0]) / 2) / 256
                                center_y = ((Cut_GT_marks[s_num][0][1] + Cut_GT_marks[s_num][1][1]) / 2) / 256
                                bbox_w = (Cut_GT_marks[s_num][1][0] - Cut_GT_marks[s_num][0][0]) / 256
                                bbox_y = (Cut_GT_marks[s_num][1][1] - Cut_GT_marks[s_num][0][1]) / 256
                                bbox = [center_x, center_y, bbox_w, bbox_y]
                                if "ZERO" not in Cut_Number_list[s_num]:
                                    label = [1]
                                    Tumor_Num_Total_s2 += 1
                                else:
                                    label = [args.zero_label]
                                    
                                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                                d['boxes'] = gt_bbox
                                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                                targets.append(d)     
                                d = {}   
                            input_slices_rgb = []
                            for num in range(len(CUT_Imgs)):
                                tmp_img = CUT_Imgs[num] 
                                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
                                input_slices_rgb.append(tmp_img)
                            input_tensor_rgb_2 = torch.tensor(input_slices_rgb).to(device= device)
                            input_tensor_rgb_2 = input_tensor_rgb_2.transpose(1,3)
                            input_tensor_rgb_2 = input_tensor_rgb_2.transpose(2,3)
                            outputs_2, decoder_out = model_2(input_tensor_rgb_2)
                
                            loss_dict_2, indices = criterion_2(outputs_2, targets) ## return losses, indices
                            weight_dict = criterion.weight_dict
                            loss_dict_reduced = utils.reduce_dict(loss_dict)
                            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
                            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                                        for k, v in loss_dict_reduced.items()}
                            metric_logger_2.update(loss=sum(loss_dict_reduced_scaled.values()),
                                                **loss_dict_reduced_scaled,
                                                **loss_dict_reduced_unscaled)
                            metric_logger_2.update(class_error=loss_dict_reduced['class_error'])
                            print(f"Validation {idx_data} / {len(data_loader.dataset)} Bbox_Loss = {round(loss_dict['loss_bbox'].item(),4)} CIoU_Loss = {round(loss_dict['loss_giou'].item(),4)} \
                                    2nd - Bbox_Loss = {round(loss_dict_2['loss_bbox'].item(),4)} CIoU_Loss = {round(loss_dict_2['loss_giou'].item(),4)}", end="\r")                 

                            orig_target_sizes = torch.tensor(t_sizes_2, dtype=torch.int64, device=device)
            
                            results_2 = postprocessors_2['bbox'](outputs_2, orig_target_sizes)

                            for idx in range(len(results_2)):
                                best_idx = int(torch.argmax(results_2[idx]['scores']))
                                epoch_score_2 += float(results_2[idx]['scores'][best_idx])
                                # pred_landmarks = results_2[idx]['boxes'][best_idx].cpu().numpy().reshape(2, 2)

                                boxes = results_2[idx]['boxes']
                                scores = results_2[idx]['scores']
                                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                                scores_cpu_ori = scores.cpu().numpy()
                                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                                if args.box_mode == "SB":
                                    if scores_cpu[0] >= args.box_th:
                                        min_x = boxes_cpu[0][0]
                                        if min_x < 0:
                                            min_x = 0
                                        min_y = boxes_cpu[0][1]
                                        if min_y < 0:
                                            min_y = 0
                                        max_x = boxes_cpu[0][2]
                                        max_y = boxes_cpu[0][3]
                                        if min_x == max_x:
                                            max_x += 1
                                        if max_y == min_y:
                                            max_y += 1
                                    else:
                                        min_x = 0
                                        min_y = 0
                                        max_x = 1
                                        max_y = 1
                                else:
                                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                                    box_score = np.max(zero_img)
                                    min_x = 512
                                    min_y = 512
                                    max_x = 0
                                    max_y = 0

                                    if box_score > args.box_th:
                                        for x in range(create_box_points[0], create_box_points[2] + 1):
                                            for y in range(create_box_points[1], create_box_points[3] + 1):
                                                if zero_img[x][y] != 0.0:
                                                    if x <= min_x:
                                                        min_x = x
                                                    if y <= min_y:
                                                        min_y = y
                                                    if x >= max_x:
                                                        max_x = x
                                                    if y >= max_y:
                                                        max_y = y

                                    ## if detr can't found colon
                                    if min_x == 512:
                                        min_x = 0
                                        min_y = 0
                                        max_x = 1
                                        max_y = 1
                                    if min_x == max_x:
                                        max_x += 1
                                    if max_y == min_y:
                                        max_y += 1
                                if max_y < 10 and "ZERO" in Cut_Number_list[idx]:                      ## zero detect
                                    Tumor_Num_Detect_Zero_s2 += 1
                                elif max_y > 10 and "ZERO" not in Cut_Number_list[idx]:
                                    Tumor_Num_Detect_s2 += 1

                                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                                S2_GT_marks.append(Cut_GT_marks[idx])
                                S2_PD_marks.append(pred_landmarks)

                                if Cut_Number_list[idx] in img_save_name:
                                    img_list = [CUT_Imgs[idx], Cut_Imgs_ori[idx]]
                                    dir_save = f"{DIR_Val_S2}E{epoch}_{Cut_Number_list[idx]}.tiff"
                                    # gt_cut_marks = [[gt_cut_min_x, gt_cut_min_y], [gt_cut_max_x, gt_cut_max_y]]
                                    # s1_cut_marks = [[min_x_cut-cut_margin_min_x, min_y_cut-cut_margin_min_y],[max_x_cut+cut_margin_max_x, max_y_cut+cut_margin_max_y]]
                                    save_2nd(img_list, Cut_GT_marks[idx], pred_landmarks, Cut_Colon_marks_ori[idx], Cut_GT_marks_ori[idx], Cut_PD_marks_ori[idx], Cut_marks[idx],
                                                Cut_length[idx], Cut_ratio[idx], dir_save)

                            CUT_Imgs = []
                            Cut_Imgs_ori = []
                            Cut_GT_marks = []
                            Cut_Number_list = []
                            Cut_Colon_marks_ori = []
                            Cut_GT_marks_ori = []
                            Cut_length = []
                            Cut_ratio = []
                            Cut_PD_marks_ori = []
                            Cut_marks = []

    Iou_s1 = 0
    Dice_s1 = 0
    Speciticity_s1 = 0
    Sensitivity_s1 = 0
    Precision_s1 = 0
    for idx in range(len(S1_GT_marks)):
        TP, FP, TN, FN = cal_score(S1_GT_marks[idx], S1_PD_marks[idx], (512, 512))
        iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
        Iou_s1 += iou
        Dice_s1 += dice
        Speciticity_s1 += specitificity
        Sensitivity_s1 += sensitivity
        Precision_s1 += precision
    Iou_s1 /= float(len(S1_GT_marks))
    Dice_s1 /= float(len(S1_GT_marks))
    Speciticity_s1 /= float(len(S1_GT_marks))
    Sensitivity_s1 /= float(len(S1_GT_marks))
    Precision_s1 /= float(len(S1_GT_marks))

    Iou_s2 = 0
    Dice_s2 = 0
    Speciticity_s2 = 0
    Sensitivity_s2 = 0
    Precision_s2 = 0
    val_stats_s2 = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    Tumor_Num_Total_Zero_s2 = 0
    if len(S2_GT_marks) > 0:
        for idx in range(len(S2_GT_marks)):
            TP, FP, TN, FN = cal_score(S2_GT_marks[idx], S2_PD_marks[idx], (256, 256))
            iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
            Iou_s2 += iou
            Dice_s2 += dice
            Speciticity_s2 += specitificity
            Sensitivity_s2 += sensitivity
            Precision_s2 += precision
        Iou_s2 /= float(len(S2_GT_marks))
        Dice_s2 /= float(len(S2_GT_marks))
        Speciticity_s2 /= float(len(S2_GT_marks))
        Sensitivity_s2 /= float(len(S2_GT_marks))
        Precision_s2 /= float(len(S2_GT_marks))
        epoch_score_2 /= idx_data_2
        val_stats_s2 = {k: meter.global_avg for k, meter in metric_logger_2.meters.items()}
        Tumor_Num_Total_Zero_s2 = len(S2_GT_marks) - Tumor_Num_Total_s2

    epoch_score_1 /= idx_data
    
    val_stats_s1 = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    Tumor_Num_Total_Zero_s1 = len(data_loader.dataset) - Tumor_Num_Total_s1    
    val_results_s1 = [Iou_s1, Dice_s1, Sensitivity_s1, Precision_s1, Speciticity_s1, Tumor_Num_Total_s1, Tumor_Num_Detect_s1, Tumor_Num_Total_Zero_s1, Tumor_Num_Detect_Zero_s1]
    val_results_s2 = [Iou_s2, Dice_s2, Sensitivity_s2, Precision_s2, Speciticity_s2, Tumor_Num_Total_s2, Tumor_Num_Detect_s2, Tumor_Num_Total_Zero_s2, Tumor_Num_Detect_Zero_s2]
    print("")
    return val_stats_s1, epoch_score_1, val_results_s1, val_stats_s2, epoch_score_2, val_results_s2, len(S2_GT_marks)


def test_img():
    fig = plt.figure(figsize=(8,8)) 
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(input_tensor_cpu[idx][0], cmap='gray')
    ax.set_title("Ori")
    ax.axis('off')
    ax = fig.add_subplot(gs[0,1])
    ax.imshow(seg_imgs[idx], cmap='gray')
    ax.set_title("gt")
    ax.add_patch(patches.Rectangle((gt_segmarks[idx][0][0]-2, gt_segmarks[idx][0][1]-2), gt_segmarks[idx][1][0] - gt_segmarks[idx][0][0], gt_segmarks[idx][1][1] - gt_segmarks[idx][0][1], edgecolor = 'b', fill=False))
    ax.axis('off')
    ax = fig.add_subplot(gs[1,0])
    ax.imshow(input_tensor_cpu[idx][0], cmap='gray')
    ax.add_patch(patches.Rectangle((gt_segmarks[idx][0][0]-2, gt_segmarks[idx][0][1]-2), gt_segmarks[idx][1][0] - gt_segmarks[idx][0][0], gt_segmarks[idx][1][1] - gt_segmarks[idx][0][1], edgecolor = 'b', fill=False))
    ax.add_patch(patches.Rectangle((gt_landmarks_cpu[idx][0][0], gt_landmarks_cpu[idx][0][1]), gt_landmarks_cpu[idx][1][0] - gt_landmarks_cpu[idx][0][0], gt_landmarks_cpu[idx][1][1] - gt_landmarks_cpu[idx][0][1], edgecolor = 'g', fill=False))
    ax.add_patch(patches.Rectangle((pred_landmarks[0][0], pred_landmarks[0][1]), pred_landmarks[1][0] - pred_landmarks[0][0], pred_landmarks[1][1] - pred_landmarks[0][1], edgecolor = 'r', fill=False))
    ax.set_title("Pred")
    ax.axis('off')
    ax = fig.add_subplot(gs[1,1])
    ax.imshow(input_tensor_cpu[idx][0][min_y_cut-cut_margin_min_y:max_y_cut+cut_margin_max_y,min_x_cut-cut_margin_min_x:max_x_cut+cut_margin_max_x], cmap='gray')
    ax.add_patch(patches.Rectangle((gt_cut_min_x, gt_cut_min_y), gt_cut_max_x - gt_cut_min_x, gt_cut_max_y - gt_cut_min_y, edgecolor = 'g', fill=False))
    ax.set_title("CUT")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"Y:/yskim/BoundingBox/result/seg_{number_list[idx]}.tiff")
    plt.axis('off')
    plt.close()