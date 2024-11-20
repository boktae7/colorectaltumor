import os
from turtle import Turtle
import util.misc as utils
import cv2
import numpy as np
import time
import nrrd
import torch

from calculate import *
from img_process import *

@torch.no_grad()
def val(args, model, postprocessors, criterion, data_loader, model_2, postprocessors_2, criterion_2, epoch, device, colon_info):
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
    header = 'Val:'

    output_dir = f"{args.output_dir}Epoch_{epoch}/"
    DIR_Val_S1 = f"{output_dir}IMG_Result/Val_S1/"      
    os.makedirs(DIR_Val_S1, exist_ok=True)
    DIR_Val_S2 = f"{output_dir}IMG_Result/Val_S2/"      
    os.makedirs(DIR_Val_S2, exist_ok=True)
    DIR_Val_NoC = f"{output_dir}IMG_Result/Val_Noc/"      
    os.makedirs(DIR_Val_NoC, exist_ok=True)

    print_freq = 10
    idx_data = 0
    idx_data_2 = 0
    img_save_count = 0
    img_save_count_noc = 0
    img_max_num = 100
    img_max_num_noc = 15
    img_save_name = []

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

    Score_1st_s1 = []
    Score_2nd_s1 = []
    Score_1st_s2 = []
    Score_2nd_s2 = []

    S1_GT_marks = []
    S1_PD_marks = []
    S2_GT_marks = []
    S2_PD_marks = []

    T_Num_Total_s1 = 0
    T_Num_Detect_s1 = 0
    N_Num_Detect_s1 = 0
    T_Num_Total_s2 = 0
    T_Num_Detect_s2 = 0
    N_Num_Detect_s2 = 0

    Tumor_Results_S1 = []
    Zero_Results_S1 = []
    Tumor_Results_S2 = []
    Zero_Results_S2 = []

    Patient_S1 = []
    Slice_S1 = []
    Iszero_S1 = []
    IsNext = []
    Patient_S2 = []
    Slice_S2 = []
    Iszero_S2 = []

    Iou_s1 = []
    Dice_s1 = []
    Speciticity_s1 = []
    Sensitivity_s1 = []
    Precision_s1 = []
    Iou_s2 = []
    Dice_s2 = []
    Speciticity_s2 = []
    Sensitivity_s2 = []
    Precision_s2 = []

    Remain_boxes_s1 = []
    Remain_scores_s1 = []
    Remain_score_weights_s1 = []
    Select_scores_tumor_s1 = []
    Select_scores_weight_tumor_s1 = []
    Select_scores_zero_s1 = []
    Select_scores_weight_zero_s1 = []
    Select_use_scores_s1 = []
    Select_use_score_weights_s1 = []
    Remain_boxes_s2 = []
    Remain_scores_s2 = []
    Remain_score_weights_s2 = []
    Select_scores_tumor_s2 = []
    Select_scores_weight_tumor_s2 = []
    Select_scores_zero_s2 = []
    Select_scores_weight_zero_s2 = []
    Select_use_scores_s2 = []
    Select_use_score_weights_s2 = []

    Iou_s2_ori = []
    Dice_s2_ori = []
    Speciticity_s2_ori = []
    Sensitivity_s2_ori = []
    Precision_s2_ori = []
    S2_GT_marks_ori = []
    S2_PD_marks_ori = []

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

                is_zero = 'X'
                if "ZERO" not in number_list[idx]:
                    label = [1]
                    T_Num_Total_s1 += 1
                else:
                    label = [args.zero_label]
                    is_zero = 'O'
                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                d['boxes'] = gt_bbox
                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                targets_s1.append(d)  
                d = {}   

                number_slice = number_list[idx]   
                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                slice_name = number_slice.split("_")[2]
                
                Patient_S1.append(patient)
                Slice_S1.append(slice_name)
                Iszero_S1.append(is_zero)

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
                t_sizes.append([512, 512])
            orig_target_sizes = torch.tensor(t_sizes, dtype=torch.int64, device=device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            is_s1_save = True
            is_noc_save = True
            for idx_all in range(len(results)):
                is_pred_zero = False
                best_idx = int(torch.argmax(results[idx_all]['scores']))
                boxes = results[idx_all]['boxes']
                scores = results[idx_all]['scores']
                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                scores_cpu_ori = scores.cpu().numpy()
                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                score = round(float(scores_cpu[0]), 4)
                score_2 = round(float(scores_cpu[1]), 4)
                Score_1st_s1.append(score)
                Score_2nd_s1.append(score_2)

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

                    box_list = []
                    for i in range(args.box_usenum):
                        box_list.append(boxes_cpu[i].reshape(2,2))
      
                    score_list = scores_cpu[0:4]
                    score_weight_list = [0, 0, 0, 0]
                    select_score_list = score_list
                    select_score_weihgt_list = score_list

                else:
                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                    box_list, score_list, score_weight_list, select_score_list, select_score_weihgt_list = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)

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
                    
                Remain_boxes_s1.append(box_list)
                Remain_scores_s1.append(score_list)
                Remain_score_weights_s1.append(score_weight_list)
                if len(select_score_list) > 0:
                    Select_use_scores_s1.extend(select_score_list)
                    Select_use_score_weights_s1.extend(select_score_weihgt_list)

                if max_y < 80 and "ZERO" in number_list[idx_all]:                      ## zero detect
                    N_Num_Detect_s1 += 1
                elif max_y > 80 and "ZERO" not in number_list[idx_all]:
                    T_Num_Detect_s1 += 1
                
                if max_y < 80:
                    is_pred_zero = True

                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                S1_GT_marks.append(gt_landmarks_cpu[idx_all])
                S1_PD_marks.append(pred_landmarks)

                TP, FP, TN, FN = cal_score(gt_landmarks_cpu[idx_all], pred_landmarks, (512, 512))
                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
                Iou_s1.append(iou)
                Dice_s1.append(dice)
                Speciticity_s1.append(specitificity)
                Sensitivity_s1.append(sensitivity)
                Precision_s1.append(precision)

                if "ZERO" in number_list[idx_all]:
                    Zero_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_zero_s1.extend(select_score_list)
                    Select_scores_weight_zero_s1.extend(select_score_weihgt_list)
                else:
                    Tumor_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_tumor_s1.extend(select_score_list)
                    Select_scores_weight_tumor_s1.extend(select_score_weihgt_list)
                    
                patient = f"{number_list[idx_all].split('_')[0]}_{number_list[idx_all].split('_')[1]}"
                slice_num = int(number_list[idx_all].split('_')[2])
                try:
                    min_x = int(colon_info[patient][str(slice_num)][0])
                    min_y = int(colon_info[patient][str(slice_num)][1])
                    max_x = int(colon_info[patient][str(slice_num)][2])
                    max_y = int(colon_info[patient][str(slice_num)][3])
                except Exception as e:
                    print("")
                    print(f"{patient} _ {slice_num} Check")
                    print(e)
                    exit()
                gt_segmark = [[min_x, min_y],[max_x, max_y]]
                
                TP, _, _, _ = cal_score(gt_segmark, pred_landmarks, (shape_x, shape_y))
                is_next = False
                
                if gt_segmark[0][1] > 50: ## miny >50 이 colon seg 있는 slice 라고 가정 
                    correct_per = TP/((gt_segmark[1][0]-gt_segmark[0][0])*(gt_segmark[1][1]-gt_segmark[0][1]))

                    idx = idx_all

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
                    
                    if min_x_cut < 20:
                        cut_margin_min_x = 0
                    if min_y_cut < 20:              
                        cut_margin_min_y = 0
                    if max_x_cut > 491:  
                        cut_margin_max_x = 0
                    if max_y_cut > 491:  
                        cut_margin_max_y = 0
                    
                    cut_img = input_tensor_cpu[idx][0][min_y_cut-cut_margin_min_y:max_y_cut+cut_margin_max_y,min_x_cut-cut_margin_min_x:max_x_cut+cut_margin_max_x]
                    
                    if (correct_per >= 0.01 and "ZERO" not in number_list[idx]) or is_pred_zero:
                        is_next = True
                        
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

                        s1_cut_marks = [[min_x_cut-cut_margin_min_x, min_y_cut-cut_margin_min_y],[max_x_cut+cut_margin_max_x, max_y_cut+cut_margin_max_y]]
                        
                        if img_save_count < img_max_num and is_s1_save:
                            img_save_count += 1
                            is_s1_save = False
                            img_save_name.append(number_list[idx])
                            
                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            target_seg = seg_data[:,:,slice_num]
                            target_seg = np.flip(target_seg, axis=0)
                            target_seg = np.rot90(target_seg, 3)
                        
                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_S1}E{epoch}_{number_list[idx]}.tiff"
                            gt_cut_marks = [[gt_cut_min_x, gt_cut_min_y], [gt_cut_max_x, gt_cut_max_y]]
                            
                            save_1st(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks, gt_cut_marks, s1_cut_marks, dir_save)
                        
                        shape_y_cut, shape_x_cut = cut_img.shape
                        cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                        gt_cut_min_x *= (256 / shape_x_cut)
                        gt_cut_min_y *= (256 / shape_y_cut)
                        gt_cut_max_x *= (256 / shape_x_cut)
                        gt_cut_max_y *= (256 / shape_y_cut)

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
                                
                                is_zero = 'X'
                                if "ZERO" not in Cut_Number_list[s_num]:
                                    label = [1]
                                    T_Num_Total_s2 += 1
                                else:
                                    label = [args.zero_label]
                                    is_zero = 'O'
                                    
                                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                                d['boxes'] = gt_bbox
                                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                                targets.append(d)     
                                d = {}   

                                number_slice = Cut_Number_list[s_num]
                                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                                slice_name = number_slice.split("_")[2]
                                
                                Patient_S2.append(patient)
                                Slice_S2.append(slice_name)
                                Iszero_S2.append(is_zero)
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
                                boxes = results_2[idx]['boxes']
                                scores = results_2[idx]['scores']
                                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                                scores_cpu_ori = scores.cpu().numpy()
                                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                                score = round(float(scores_cpu[0]), 4)
                                score_2 = round(float(scores_cpu[1]), 4)
                                Score_1st_s2.append(score)
                                Score_2nd_s2.append(score_2)

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

                                    score_list = scores_cpu[0:4]
                                    score_weight_list = [0, 0, 0, 0]
                                    select_scores = score_list
                                    select_score_weihgts = score_list
                                else:
                                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                                    box_list, score_list, score_weight_list, select_scores, select_score_weihgts = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)
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
                                
                                Remain_boxes_s2.append(box_list)
                                Remain_scores_s2.append(score_list)
                                Remain_score_weights_s2.append(score_weight_list)
                                if len(select_score_list) > 0:
                                    Select_use_scores_s2.extend(select_scores)
                                    Select_use_score_weights_s2.extend(select_score_weihgts)
                                    
                                if max_y < 10 and "ZERO" in Cut_Number_list[idx]:                      ## zero detect
                                    N_Num_Detect_s2 += 1
                                elif max_y > 10 and "ZERO" not in Cut_Number_list[idx]:
                                    T_Num_Detect_s2 += 1

                                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                                S2_GT_marks.append(Cut_GT_marks[idx])
                                S2_PD_marks.append(pred_landmarks)

                                TP, FP, TN, FN = cal_score(Cut_GT_marks[idx], pred_landmarks, (256, 256))
                                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)

                                Iou_s2.append(iou)
                                Dice_s2.append(dice)
                                Speciticity_s2.append(specitificity)
                                Sensitivity_s2.append(sensitivity)
                                Precision_s2.append(precision)

                                if "ZERO" in Cut_Number_list[idx]:
                                    Zero_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                    Select_scores_zero_s2.extend(select_score_list)
                                    Select_scores_weight_zero_s2.extend(select_score_weihgt_list)   
                                else:
                                    Tumor_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                    Select_scores_tumor_s2.extend(select_score_list)
                                    Select_scores_weight_tumor_s2.extend(select_score_weihgt_list)

                                if Cut_Number_list[idx] in img_save_name:
                                    img_list = [CUT_Imgs[idx], Cut_Imgs_ori[idx]]
                                    dir_save = f"{DIR_Val_S2}E{epoch}_{Cut_Number_list[idx]}.tiff"
                                    
                                    save_2nd(img_list, Cut_GT_marks[idx], pred_landmarks, Cut_Colon_marks_ori[idx], Cut_GT_marks_ori[idx], Cut_PD_marks_ori[idx], Cut_marks[idx],
                                                Cut_length[idx], Cut_ratio[idx], dir_save)
                                
                                if pred_landmarks[1][1] > 10: 
                                    pd_min_x = (pred_landmarks[0][0] / Cut_ratio[idx][0]) + Cut_length[idx][0][0]
                                    pd_min_y = (pred_landmarks[0][1] / Cut_ratio[idx][1]) + Cut_length[idx][0][1]
                                    pd_max_x = (pred_landmarks[1][0] / Cut_ratio[idx][0]) + Cut_length[idx][1][0]
                                    pd_max_y = (pred_landmarks[1][1] / Cut_ratio[idx][1]) + Cut_length[idx][1][1]
                                else:
                                    pd_min_x = 0
                                    pd_min_y = 0
                                    pd_max_x = 0
                                    pd_max_y = 0

                                TP_ori, FP_ori, TN_ori, FN_ori = cal_score(Cut_GT_marks_ori[idx], [[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]], (512, 512))
                                iou_ori, dice_ori, specitificity_ori, sensitivity_ori, precision_ori = Cal_Result(TP_ori, FP_ori, TN_ori, FN_ori)

                                Iou_s2_ori.append(iou_ori)
                                Dice_s2_ori.append(dice_ori)
                                Speciticity_s2_ori.append(specitificity_ori)
                                Sensitivity_s2_ori.append(sensitivity_ori)
                                Precision_s2_ori.append(precision_ori)
                                S2_GT_marks_ori.append(Cut_GT_marks_ori[idx])
                                S2_PD_marks_ori.append([[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]])

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
                    else:
                        if img_save_count_noc < img_max_num_noc and is_noc_save:
                            img_save_count_noc += 1
                            is_noc_save = False
                            
                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            target_seg = seg_data[:,:,slice_num]
                            target_seg = np.flip(target_seg, axis=0)
                            target_seg = np.rot90(target_seg, 3)
                            
                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_NoC}E{epoch}_{number_list[idx]}.tiff"
                            save_NoC(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks,  dir_save)

                IsNext.append(is_next)
   
    N_Num_Total_s1 = len(data_loader.dataset) - T_Num_Total_s1  
    N_Num_Total_s2 = len(S2_GT_marks) - T_Num_Total_s2
    results_static_s1 = [Iou_s1, Dice_s1, Sensitivity_s1, Precision_s1, Speciticity_s1, T_Num_Total_s1, T_Num_Detect_s1, N_Num_Total_s1, N_Num_Detect_s1]
    results_static_s2 = [Iou_s2, Dice_s2, Sensitivity_s2, Precision_s2, Speciticity_s2, T_Num_Total_s2, T_Num_Detect_s2, N_Num_Total_s2, N_Num_Detect_s2]
    results_box_s1 = [Remain_boxes_s1, Remain_scores_s1, Remain_score_weights_s1, Select_scores_tumor_s1, Select_scores_weight_tumor_s1, Select_scores_zero_s1, 
                        Select_scores_weight_zero_s1, Select_use_scores_s1, Select_use_score_weights_s1, Score_1st_s1, Score_2nd_s1]
    results_box_s2 = [Remain_boxes_s2, Remain_scores_s2, Remain_score_weights_s2, Select_scores_tumor_s2, Select_scores_weight_tumor_s2, Select_scores_zero_s2, 
                        Select_scores_weight_zero_s2, Select_use_scores_s2, Select_use_score_weights_s2, Score_1st_s2, Score_2nd_s2]

    results_s1 = [results_static_s1, S1_GT_marks, S1_PD_marks, results_box_s1, Tumor_Results_S1, Zero_Results_S1]
    results_s2 = [results_static_s2, S2_GT_marks, S2_PD_marks, results_box_s2, Tumor_Results_S2, Zero_Results_S2]
    results_slice = [Patient_S1, Slice_S1, Iszero_S1, IsNext, Patient_S2, Slice_S2, Iszero_S2]
    results_s2_ori = [Patient_S2, Slice_S2, Iszero_S2, S2_GT_marks_ori, S2_PD_marks_ori, Iou_s2_ori, Dice_s2_ori, Speciticity_s2_ori, Sensitivity_s2_ori, Precision_s2_ori]
    print("")
    print(f"{epoch} Epoch Validation Fnish, Excel Write Start")
    return  results_s1, len(S2_GT_marks), results_s2, results_slice, results_s2_ori

@torch.no_grad()
def val_max_colon(args, model, postprocessors, criterion, data_loader, model_2, postprocessors_2, criterion_2, epoch, device, max_segmark, colon_info):
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
    header = 'Val:'

    output_dir = f"{args.output_dir}Epoch_{epoch}/"
    DIR_Val_S1 = f"{output_dir}IMG_Result/Val_S1/"      
    os.makedirs(DIR_Val_S1, exist_ok=True)
    DIR_Val_S2 = f"{output_dir}IMG_Result/Val_S2/"      
    os.makedirs(DIR_Val_S2, exist_ok=True)
    DIR_Val_NoC = f"{output_dir}IMG_Result/Val_Noc/"      
    os.makedirs(DIR_Val_NoC, exist_ok=True)

    print_freq = 10
    idx_data = 0
    idx_data_2 = 0
    img_save_count = 0
    img_save_count_noc = 0
    img_max_num = 100
    img_max_num_noc = 15
    img_save_name = []

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

    Score_1st_s1 = []
    Score_2nd_s1 = []
    Score_1st_s2 = []
    Score_2nd_s2 = []

    S1_GT_marks = []
    S1_PD_marks = []
    S2_GT_marks = []
    S2_PD_marks = []

    T_Num_Total_s1 = 0
    T_Num_Detect_s1 = 0
    N_Num_Detect_s1 = 0
    T_Num_Total_s2 = 0
    T_Num_Detect_s2 = 0
    N_Num_Detect_s2 = 0

    Tumor_Results_S1 = []
    Zero_Results_S1 = []
    Tumor_Results_S2 = []
    Zero_Results_S2 = []

    Patient_S1 = []
    Slice_S1 = []
    Iszero_S1 = []
    IsNext = []
    Patient_S2 = []
    Slice_S2 = []
    Iszero_S2 = []

    Iou_s1 = []
    Dice_s1 = []
    Speciticity_s1 = []
    Sensitivity_s1 = []
    Precision_s1 = []
    Iou_s2 = []
    Dice_s2 = []
    Speciticity_s2 = []
    Sensitivity_s2 = []
    Precision_s2 = []

    Remain_boxes_s1 = []
    Remain_scores_s1 = []
    Remain_score_weights_s1 = []
    Select_scores_tumor_s1 = []
    Select_scores_weight_tumor_s1 = []
    Select_scores_zero_s1 = []
    Select_scores_weight_zero_s1 = []
    Select_use_scores_s1 = []
    Select_use_score_weights_s1 = []
    Remain_boxes_s2 = []
    Remain_scores_s2 = []
    Remain_score_weights_s2 = []
    Select_scores_tumor_s2 = []
    Select_scores_weight_tumor_s2 = []
    Select_scores_zero_s2 = []
    Select_scores_weight_zero_s2 = []
    Select_use_scores_s2 = []
    Select_use_score_weights_s2 = []

    Iou_s2_ori = []
    Dice_s2_ori = []
    Speciticity_s2_ori = []
    Sensitivity_s2_ori = []
    Precision_s2_ori = []
    S2_GT_marks_ori = []
    S2_PD_marks_ori = []

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

                is_zero = 'X'
                if "ZERO" not in number_list[idx]:
                    label = [1]
                    T_Num_Total_s1 += 1
                else:
                    label = [args.zero_label]
                    is_zero = 'O'
                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                d['boxes'] = gt_bbox
                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                targets_s1.append(d)  
                d = {}   

                number_slice = number_list[idx]   
                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                slice_name = number_slice.split("_")[2]
                
                Patient_S1.append(patient)
                Slice_S1.append(slice_name)
                Iszero_S1.append(is_zero)

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
                t_sizes.append([512, 512])
            orig_target_sizes = torch.tensor(t_sizes, dtype=torch.int64, device=device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            is_s1_save = True
            is_noc_save = True
            for idx_all in range(len(results)):
                is_pred_zero = False
                best_idx = int(torch.argmax(results[idx_all]['scores']))
                boxes = results[idx_all]['boxes']
                scores = results[idx_all]['scores']
                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                scores_cpu_ori = scores.cpu().numpy()
                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                score = round(float(scores_cpu[0]), 4)
                score_2 = round(float(scores_cpu[1]), 4)
                Score_1st_s1.append(score)
                Score_2nd_s1.append(score_2)

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

                    box_list = []
                    for i in range(args.box_usenum):
                        box_list.append(boxes_cpu[i].reshape(2,2))
      
                    score_list = scores_cpu[0:4]
                    score_weight_list = [0, 0, 0, 0]
                    select_score_list = score_list
                    select_score_weihgt_list = score_list

                else:
                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                    box_list, score_list, score_weight_list, select_score_list, select_score_weihgt_list = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)

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
                    
                Remain_boxes_s1.append(box_list)
                Remain_scores_s1.append(score_list)
                Remain_score_weights_s1.append(score_weight_list)
                if len(select_score_list) > 0:
                    Select_use_scores_s1.extend(select_score_list)
                    Select_use_score_weights_s1.extend(select_score_weihgt_list)

                if max_y < 80 and "ZERO" in number_list[idx_all]:                      ## zero detect
                    N_Num_Detect_s1 += 1
                    
                elif max_y > 80 and "ZERO" not in number_list[idx_all]:
                    T_Num_Detect_s1 += 1
                    
                if max_y < 80:
                    is_pred_zero = True

                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                S1_GT_marks.append(gt_landmarks_cpu[idx_all])
                S1_PD_marks.append(pred_landmarks)

                TP, FP, TN, FN = cal_score(gt_landmarks_cpu[idx_all], pred_landmarks, (512, 512))
                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
                Iou_s1.append(iou)
                Dice_s1.append(dice)
                Speciticity_s1.append(specitificity)
                Sensitivity_s1.append(sensitivity)
                Precision_s1.append(precision)

                if "ZERO" in number_list[idx_all]:
                    Zero_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_zero_s1.extend(select_score_list)
                    Select_scores_weight_zero_s1.extend(select_score_weihgt_list)
                else:
                    Tumor_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_tumor_s1.extend(select_score_list)
                    Select_scores_weight_tumor_s1.extend(select_score_weihgt_list)
                
                patient = f"{number_list[idx_all].split('_')[0]}_{number_list[idx_all].split('_')[1]}"
                slice_num = int(number_list[idx_all].split('_')[2])
                try:
                    min_x = int(colon_info[patient][str(slice_num)][0])
                    min_y = int(colon_info[patient][str(slice_num)][1])
                    max_x = int(colon_info[patient][str(slice_num)][2])
                    max_y = int(colon_info[patient][str(slice_num)][3])
                except Exception as e:
                    print("")
                    print(f"{patient} _ {slice_num} Check")
                    print(e)
                    exit()
                gt_segmark = [[min_x, min_y],[max_x, max_y]]
                if max_y > 10:
                    gt_segmark = np.array([[max_segmark[0][0],max_segmark[0][1]], [max_segmark[1][0],max_segmark[1][1]]])
                
                TP, _, _, _ = cal_score(gt_segmark, pred_landmarks, (shape_x, shape_y))
                is_next = False
                if gt_segmark[0][1] > 50: ## miny >50 이 colon seg 있는 slice 라고 가정 
                    correct_per = TP/((gt_segmark[1][0]-gt_segmark[0][0])*(gt_segmark[1][1]-gt_segmark[0][1]))
                    idx = idx_all

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
                    
                    if min_x_cut < 20:
                        cut_margin_min_x = 0
                    if min_y_cut < 20:              
                        cut_margin_min_y = 0
                    if max_x_cut > 491:  
                        cut_margin_max_x = 0
                    if max_y_cut > 491:  
                        cut_margin_max_y = 0
                    
                    cut_img = input_tensor_cpu[idx][0][min_y_cut-cut_margin_min_y:max_y_cut+cut_margin_max_y,min_x_cut-cut_margin_min_x:max_x_cut+cut_margin_max_x]
       
                    if (correct_per >= 0.01 and "ZERO" not in number_list[idx]) or is_pred_zero:
                        is_next = True
                    
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

                        s1_cut_marks = [[min_x_cut-cut_margin_min_x, min_y_cut-cut_margin_min_y],[max_x_cut+cut_margin_max_x, max_y_cut+cut_margin_max_y]]
                        
                        if img_save_count < img_max_num and is_s1_save:
                            img_save_count += 1
                            is_s1_save = False
                            img_save_name.append(number_list[idx])  
                            
                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            target_seg = seg_data[:,:,slice_num]
                            target_seg = np.flip(target_seg, axis=0)
                            target_seg = np.rot90(target_seg, 3)
                    
                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_S1}E{epoch}_{number_list[idx]}.tiff"
                            gt_cut_marks = [[gt_cut_min_x, gt_cut_min_y], [gt_cut_max_x, gt_cut_max_y]]
                            
                            save_1st(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks, gt_cut_marks, s1_cut_marks, dir_save)
                        
                        shape_y_cut, shape_x_cut = cut_img.shape
                        cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                        gt_cut_min_x *= (256 / shape_x_cut)
                        gt_cut_min_y *= (256 / shape_y_cut)
                        gt_cut_max_x *= (256 / shape_x_cut)
                        gt_cut_max_y *= (256 / shape_y_cut)

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
                                
                                is_zero = 'X'
                                if "ZERO" not in Cut_Number_list[s_num]:
                                    label = [1]
                                    T_Num_Total_s2 += 1
                                else:
                                    label = [args.zero_label]
                                    is_zero = 'O'
                                    
                                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                                d['boxes'] = gt_bbox
                                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                                targets.append(d)     
                                d = {}   

                                number_slice = Cut_Number_list[s_num]
                                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                                slice_name = number_slice.split("_")[2]
                                
                                Patient_S2.append(patient)
                                Slice_S2.append(slice_name)
                                Iszero_S2.append(is_zero)
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
                                boxes = results_2[idx]['boxes']
                                scores = results_2[idx]['scores']
                                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                                scores_cpu_ori = scores.cpu().numpy()
                                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                                score = round(float(scores_cpu[0]), 4)
                                score_2 = round(float(scores_cpu[1]), 4)
                                Score_1st_s2.append(score)
                                Score_2nd_s2.append(score_2)

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

                                    score_list = scores_cpu[0:4]
                                    score_weight_list = [0, 0, 0, 0]
                                    select_scores = score_list
                                    select_score_weihgts = score_list
                                else:
                                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                                    box_list, score_list, score_weight_list, select_scores, select_score_weihgts = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)
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
                                
                                Remain_boxes_s2.append(box_list)
                                Remain_scores_s2.append(score_list)
                                Remain_score_weights_s2.append(score_weight_list)
                                if len(select_score_list) > 0:
                                    Select_use_scores_s2.extend(select_scores)
                                    Select_use_score_weights_s2.extend(select_score_weihgts)
                                    
                                if max_y < 10 and "ZERO" in Cut_Number_list[idx]:                      ## zero detect
                                    N_Num_Detect_s2 += 1
                                elif max_y > 10 and "ZERO" not in Cut_Number_list[idx]:
                                    T_Num_Detect_s2 += 1

                                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                                S2_GT_marks.append(Cut_GT_marks[idx])
                                S2_PD_marks.append(pred_landmarks)

                                TP, FP, TN, FN = cal_score(Cut_GT_marks[idx], pred_landmarks, (256, 256))
                                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)

                                Iou_s2.append(iou)
                                Dice_s2.append(dice)
                                Speciticity_s2.append(specitificity)
                                Sensitivity_s2.append(sensitivity)
                                Precision_s2.append(precision)

                                if "ZERO" in Cut_Number_list[idx]:
                                    Zero_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                    Select_scores_zero_s2.extend(select_score_list)
                                    Select_scores_weight_zero_s2.extend(select_score_weihgt_list)   
                                else:
                                    Tumor_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                    Select_scores_tumor_s2.extend(select_score_list)
                                    Select_scores_weight_tumor_s2.extend(select_score_weihgt_list)

                                if Cut_Number_list[idx] in img_save_name:
                                    img_list = [CUT_Imgs[idx], Cut_Imgs_ori[idx]]
                                    dir_save = f"{DIR_Val_S2}E{epoch}_{Cut_Number_list[idx]}.tiff"
                                    
                                    save_2nd(img_list, Cut_GT_marks[idx], pred_landmarks, Cut_Colon_marks_ori[idx], Cut_GT_marks_ori[idx], Cut_PD_marks_ori[idx], Cut_marks[idx],
                                                Cut_length[idx], Cut_ratio[idx], dir_save)
                                
                                if pred_landmarks[1][1] > 10: 
                                    pd_min_x = (pred_landmarks[0][0] / Cut_ratio[idx][0]) + Cut_length[idx][0][0]
                                    pd_min_y = (pred_landmarks[0][1] / Cut_ratio[idx][1]) + Cut_length[idx][0][1]
                                    pd_max_x = (pred_landmarks[1][0] / Cut_ratio[idx][0]) + Cut_length[idx][1][0]
                                    pd_max_y = (pred_landmarks[1][1] / Cut_ratio[idx][1]) + Cut_length[idx][1][1]
                                else:
                                    pd_min_x = 0
                                    pd_min_y = 0
                                    pd_max_x = 0
                                    pd_max_y = 0

                                TP_ori, FP_ori, TN_ori, FN_ori = cal_score(Cut_GT_marks_ori[idx], [[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]], (512, 512))
                                iou_ori, dice_ori, specitificity_ori, sensitivity_ori, precision_ori = Cal_Result(TP_ori, FP_ori, TN_ori, FN_ori)

                                Iou_s2_ori.append(iou_ori)
                                Dice_s2_ori.append(dice_ori)
                                Speciticity_s2_ori.append(specitificity_ori)
                                Sensitivity_s2_ori.append(sensitivity_ori)
                                Precision_s2_ori.append(precision_ori)
                                S2_GT_marks_ori.append(Cut_GT_marks_ori[idx])
                                S2_PD_marks_ori.append([[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]])

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
                    else:
                        if img_save_count_noc < img_max_num_noc and is_noc_save:
                            img_save_count_noc += 1
                            is_noc_save = False
                            
                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            target_seg = seg_data[:,:,slice_num]
                            target_seg = np.flip(target_seg, axis=0)
                            target_seg = np.rot90(target_seg, 3)
                            
                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_NoC}E{epoch}_{number_list[idx]}.tiff"
                            save_NoC(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks,  dir_save)

                IsNext.append(is_next)
   
    N_Num_Total_s1 = len(data_loader.dataset) - T_Num_Total_s1  
    N_Num_Total_s2 = len(S2_GT_marks) - T_Num_Total_s2
    results_static_s1 = [Iou_s1, Dice_s1, Sensitivity_s1, Precision_s1, Speciticity_s1, T_Num_Total_s1, T_Num_Detect_s1, N_Num_Total_s1, N_Num_Detect_s1]
    results_static_s2 = [Iou_s2, Dice_s2, Sensitivity_s2, Precision_s2, Speciticity_s2, T_Num_Total_s2, T_Num_Detect_s2, N_Num_Total_s2, N_Num_Detect_s2]
    results_box_s1 = [Remain_boxes_s1, Remain_scores_s1, Remain_score_weights_s1, Select_scores_tumor_s1, Select_scores_weight_tumor_s1, Select_scores_zero_s1, 
                        Select_scores_weight_zero_s1, Select_use_scores_s1, Select_use_score_weights_s1, Score_1st_s1, Score_2nd_s1]
    results_box_s2 = [Remain_boxes_s2, Remain_scores_s2, Remain_score_weights_s2, Select_scores_tumor_s2, Select_scores_weight_tumor_s2, Select_scores_zero_s2, 
                        Select_scores_weight_zero_s2, Select_use_scores_s2, Select_use_score_weights_s2, Score_1st_s2, Score_2nd_s2]

    results_s1 = [results_static_s1, S1_GT_marks, S1_PD_marks, results_box_s1, Tumor_Results_S1, Zero_Results_S1]
    results_s2 = [results_static_s2, S2_GT_marks, S2_PD_marks, results_box_s2, Tumor_Results_S2, Zero_Results_S2]
    results_slice = [Patient_S1, Slice_S1, Iszero_S1, IsNext, Patient_S2, Slice_S2, Iszero_S2]
    results_s2_ori = [Patient_S2, Slice_S2, Iszero_S2, S2_GT_marks_ori, S2_PD_marks_ori, Iou_s2_ori, Dice_s2_ori, Speciticity_s2_ori, Sensitivity_s2_ori, Precision_s2_ori]
    print("")
    print(f"{epoch} Epoch Validation Fnish, Excel Write Start")
    return  results_s1, len(S2_GT_marks), results_s2, results_slice, results_s2_ori

@torch.no_grad()
def val_s2OnlyT(args, model, postprocessors, criterion, data_loader, model_2, postprocessors_2, criterion_2, epoch, device, colon_info):
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
    header = 'Val:'

    output_dir = f"{args.output_dir}Epoch_{epoch}/"
    DIR_Val_S1 = f"{output_dir}IMG_Result/Val_S1/"      
    os.makedirs(DIR_Val_S1, exist_ok=True)
    DIR_Val_S2 = f"{output_dir}IMG_Result/Val_S2/"      
    os.makedirs(DIR_Val_S2, exist_ok=True)
    DIR_Val_NoC = f"{output_dir}IMG_Result/Val_Noc/"      
    os.makedirs(DIR_Val_NoC, exist_ok=True)

    print_freq = 10
    idx_data = 0
    idx_data_2 = 0
    img_save_count = 0
    img_save_count_noc = 0
    img_max_num = 100
    img_max_num_noc = 15
    img_save_name = []

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

    Score_1st_s1 = []
    Score_2nd_s1 = []
    Score_1st_s2 = []
    Score_2nd_s2 = []

    S1_GT_marks = []
    S1_PD_marks = []
    S2_GT_marks = []
    S2_PD_marks = []

    T_Num_Total_s1 = 0
    T_Num_Detect_s1 = 0
    N_Num_Detect_s1 = 0
    T_Num_Total_s2 = 0
    T_Num_Detect_s2 = 0
    N_Num_Detect_s2 = 0

    Tumor_Results_S1 = []
    Zero_Results_S1 = []
    Tumor_Results_S2 = []
    Zero_Results_S2 = []

    Patient_S1 = []
    Slice_S1 = []
    Iszero_S1 = []
    IsNext = []
    Patient_S2 = []
    Slice_S2 = []
    Iszero_S2 = []

    Iou_s1 = []
    Dice_s1 = []
    Speciticity_s1 = []
    Sensitivity_s1 = []
    Precision_s1 = []
    Iou_s2 = []
    Dice_s2 = []
    Speciticity_s2 = []
    Sensitivity_s2 = []
    Precision_s2 = []

    Remain_boxes_s1 = []
    Remain_scores_s1 = []
    Remain_score_weights_s1 = []
    Select_scores_tumor_s1 = []
    Select_scores_weight_tumor_s1 = []
    Select_scores_zero_s1 = []
    Select_scores_weight_zero_s1 = []
    Select_use_scores_s1 = []
    Select_use_score_weights_s1 = []
    Remain_boxes_s2 = []
    Remain_scores_s2 = []
    Remain_score_weights_s2 = []
    Select_scores_tumor_s2 = []
    Select_scores_weight_tumor_s2 = []
    Select_scores_zero_s2 = []
    Select_scores_weight_zero_s2 = []
    Select_use_scores_s2 = []
    Select_use_score_weights_s2 = []

    Iou_s2_ori = []
    Dice_s2_ori = []
    Speciticity_s2_ori = []
    Sensitivity_s2_ori = []
    Precision_s2_ori = []
    S2_GT_marks_ori = []
    S2_PD_marks_ori = []

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

                is_zero = 'X'
                if "ZERO" not in number_list[idx]:
                    label = [1]
                    T_Num_Total_s1 += 1
                else:
                    label = [args.zero_label]
                    is_zero = 'O'
                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                d['boxes'] = gt_bbox
                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                targets_s1.append(d)  
                d = {}   

                number_slice = number_list[idx]   
                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                slice_name = number_slice.split("_")[2]
                
                Patient_S1.append(patient)
                Slice_S1.append(slice_name)
                Iszero_S1.append(is_zero)

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
                t_sizes.append([512, 512])
            orig_target_sizes = torch.tensor(t_sizes, dtype=torch.int64, device=device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            is_s1_save = True
            is_noc_save = True
            for idx_all in range(len(results)):
                is_pred_zero = False
                best_idx = int(torch.argmax(results[idx_all]['scores']))
                boxes = results[idx_all]['boxes']
                scores = results[idx_all]['scores']
                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                scores_cpu_ori = scores.cpu().numpy()
                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                score = round(float(scores_cpu[0]), 4)
                score_2 = round(float(scores_cpu[1]), 4)
                Score_1st_s1.append(score)
                Score_2nd_s1.append(score_2)

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

                    box_list = []
                    for i in range(args.box_usenum):
                        box_list.append(boxes_cpu[i].reshape(2,2))
      
                    score_list = scores_cpu[0:4]
                    score_weight_list = [0, 0, 0, 0]
                    select_score_list = score_list
                    select_score_weihgt_list = score_list

                else:
                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                    box_list, score_list, score_weight_list, select_score_list, select_score_weihgt_list = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)

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
                    
                Remain_boxes_s1.append(box_list)
                Remain_scores_s1.append(score_list)
                Remain_score_weights_s1.append(score_weight_list)
                if len(select_score_list) > 0:
                    Select_use_scores_s1.extend(select_score_list)
                    Select_use_score_weights_s1.extend(select_score_weihgt_list)

                if max_y < 80 and "ZERO" in number_list[idx_all]:                      ## zero detect
                    N_Num_Detect_s1 += 1
                elif max_y > 80 and "ZERO" not in number_list[idx_all]:
                    T_Num_Detect_s1 += 1
                
                if max_y < 80:
                    is_pred_zero = True

                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                S1_GT_marks.append(gt_landmarks_cpu[idx_all])
                S1_PD_marks.append(pred_landmarks)

                TP, FP, TN, FN = cal_score(gt_landmarks_cpu[idx_all], pred_landmarks, (512, 512))
                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
                Iou_s1.append(iou)
                Dice_s1.append(dice)
                Speciticity_s1.append(specitificity)
                Sensitivity_s1.append(sensitivity)
                Precision_s1.append(precision)

                if "ZERO" in number_list[idx_all]:
                    Zero_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_zero_s1.extend(select_score_list)
                    Select_scores_weight_zero_s1.extend(select_score_weihgt_list)
                else:
                    Tumor_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_tumor_s1.extend(select_score_list)
                    Select_scores_weight_tumor_s1.extend(select_score_weihgt_list)
                    
                patient = f"{number_list[idx_all].split('_')[0]}_{number_list[idx_all].split('_')[1]}"
                slice_num = int(number_list[idx_all].split('_')[2])
                try:
                    min_x = int(colon_info[patient][str(slice_num)][0])
                    min_y = int(colon_info[patient][str(slice_num)][1])
                    max_x = int(colon_info[patient][str(slice_num)][2])
                    max_y = int(colon_info[patient][str(slice_num)][3])
                except Exception as e:
                    print("")
                    print(f"{patient} _ {slice_num} Check")
                    print(e)
                    exit()
                gt_segmark = [[min_x, min_y],[max_x, max_y]]
                
                TP, _, _, _ = cal_score(gt_segmark, pred_landmarks, (shape_x, shape_y))
                is_next = False
                
                if gt_segmark[0][1] > 50 and "ZERO" not in number_list[idx_all]: ## miny >50 이 colon 있는 slice 라고 가정
                    correct_per = TP/((gt_segmark[1][0]-gt_segmark[0][0])*(gt_segmark[1][1]-gt_segmark[0][1]))
                    idx = idx_all

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
                    
                    if min_x_cut < 20:
                        cut_margin_min_x = 0
                    if min_y_cut < 20:              
                        cut_margin_min_y = 0
                    if max_x_cut > 491:  
                        cut_margin_max_x = 0
                    if max_y_cut > 491:  
                        cut_margin_max_y = 0
                    
                    cut_img = input_tensor_cpu[idx][0][min_y_cut-cut_margin_min_y:max_y_cut+cut_margin_max_y,min_x_cut-cut_margin_min_x:max_x_cut+cut_margin_max_x]
                    
                    if correct_per >= 0.01:
                        is_next = True
                        
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

                        s1_cut_marks = [[min_x_cut-cut_margin_min_x, min_y_cut-cut_margin_min_y],[max_x_cut+cut_margin_max_x, max_y_cut+cut_margin_max_y]]
                        
                        if img_save_count < img_max_num and is_s1_save:
                            img_save_count += 1
                            is_s1_save = False
                            img_save_name.append(number_list[idx])
                            
                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            target_seg = seg_data[:,:,slice_num]
                            target_seg = np.flip(target_seg, axis=0)
                            target_seg = np.rot90(target_seg, 3)
                            
                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_S1}E{epoch}_{number_list[idx]}.tiff"
                            gt_cut_marks = [[gt_cut_min_x, gt_cut_min_y], [gt_cut_max_x, gt_cut_max_y]]
                            
                            save_1st(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks, gt_cut_marks, s1_cut_marks, dir_save)
                        
                        shape_y_cut, shape_x_cut = cut_img.shape
                        cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                        gt_cut_min_x *= (256 / shape_x_cut)
                        gt_cut_min_y *= (256 / shape_y_cut)
                        gt_cut_max_x *= (256 / shape_x_cut)
                        gt_cut_max_y *= (256 / shape_y_cut)

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
                                
                                is_zero = 'X'
                                if "ZERO" not in Cut_Number_list[s_num]:
                                    label = [1]
                                    T_Num_Total_s2 += 1
                                else:
                                    label = [args.zero_label]
                                    is_zero = 'O'
                                    
                                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                                d['boxes'] = gt_bbox
                                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                                targets.append(d)     
                                d = {}   

                                number_slice = Cut_Number_list[s_num]
                                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                                slice_name = number_slice.split("_")[2]
                                
                                Patient_S2.append(patient)
                                Slice_S2.append(slice_name)
                                Iszero_S2.append(is_zero)
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
                                boxes = results_2[idx]['boxes']
                                scores = results_2[idx]['scores']
                                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                                scores_cpu_ori = scores.cpu().numpy()
                                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                                score = round(float(scores_cpu[0]), 4)
                                score_2 = round(float(scores_cpu[1]), 4)
                                Score_1st_s2.append(score)
                                Score_2nd_s2.append(score_2)

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

                                    score_list = scores_cpu[0:4]
                                    score_weight_list = [0, 0, 0, 0]
                                    select_scores = score_list
                                    select_score_weihgts = score_list
                                else:
                                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                                    box_list, score_list, score_weight_list, select_scores, select_score_weihgts = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)
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
                                
                                Remain_boxes_s2.append(box_list)
                                Remain_scores_s2.append(score_list)
                                Remain_score_weights_s2.append(score_weight_list)
                                if len(select_score_list) > 0:
                                    Select_use_scores_s2.extend(select_scores)
                                    Select_use_score_weights_s2.extend(select_score_weihgts)
                                    
                                if max_y < 10 and "ZERO" in Cut_Number_list[idx]:                      ## zero detect
                                    N_Num_Detect_s2 += 1
                                elif max_y > 10 and "ZERO" not in Cut_Number_list[idx]:
                                    T_Num_Detect_s2 += 1

                                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                                S2_GT_marks.append(Cut_GT_marks[idx])
                                S2_PD_marks.append(pred_landmarks)

                                TP, FP, TN, FN = cal_score(Cut_GT_marks[idx], pred_landmarks, (256, 256))
                                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)

                                Iou_s2.append(iou)
                                Dice_s2.append(dice)
                                Speciticity_s2.append(specitificity)
                                Sensitivity_s2.append(sensitivity)
                                Precision_s2.append(precision)

                                if "ZERO" in Cut_Number_list[idx]:
                                    Zero_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                    Select_scores_zero_s2.extend(select_score_list)
                                    Select_scores_weight_zero_s2.extend(select_score_weihgt_list)   
                                else:
                                    Tumor_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                    Select_scores_tumor_s2.extend(select_score_list)
                                    Select_scores_weight_tumor_s2.extend(select_score_weihgt_list)

                                if Cut_Number_list[idx] in img_save_name:
                                    img_list = [CUT_Imgs[idx], Cut_Imgs_ori[idx]]
                                    dir_save = f"{DIR_Val_S2}E{epoch}_{Cut_Number_list[idx]}.tiff"
                                    
                                    save_2nd(img_list, Cut_GT_marks[idx], pred_landmarks, Cut_Colon_marks_ori[idx], Cut_GT_marks_ori[idx], Cut_PD_marks_ori[idx], Cut_marks[idx],
                                                Cut_length[idx], Cut_ratio[idx], dir_save)
                                
                                if pred_landmarks[1][1] > 10: 
                                    pd_min_x = (pred_landmarks[0][0] / Cut_ratio[idx][0]) + Cut_length[idx][0][0]
                                    pd_min_y = (pred_landmarks[0][1] / Cut_ratio[idx][1]) + Cut_length[idx][0][1]
                                    pd_max_x = (pred_landmarks[1][0] / Cut_ratio[idx][0]) + Cut_length[idx][1][0]
                                    pd_max_y = (pred_landmarks[1][1] / Cut_ratio[idx][1]) + Cut_length[idx][1][1]
                                else:
                                    pd_min_x = 0
                                    pd_min_y = 0
                                    pd_max_x = 0
                                    pd_max_y = 0

                                TP_ori, FP_ori, TN_ori, FN_ori = cal_score(Cut_GT_marks_ori[idx], [[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]], (512, 512))
                                iou_ori, dice_ori, specitificity_ori, sensitivity_ori, precision_ori = Cal_Result(TP_ori, FP_ori, TN_ori, FN_ori)

                                Iou_s2_ori.append(iou_ori)
                                Dice_s2_ori.append(dice_ori)
                                Speciticity_s2_ori.append(specitificity_ori)
                                Sensitivity_s2_ori.append(sensitivity_ori)
                                Precision_s2_ori.append(precision_ori)
                                S2_GT_marks_ori.append(Cut_GT_marks_ori[idx])
                                S2_PD_marks_ori.append([[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]])

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
                    else:
                        if img_save_count_noc < img_max_num_noc and is_noc_save:
                            img_save_count_noc += 1
                            is_noc_save = False
                            
                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            target_seg = seg_data[:,:,slice_num]
                            target_seg = np.flip(target_seg, axis=0)
                            target_seg = np.rot90(target_seg, 3)
                            
                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_NoC}E{epoch}_{number_list[idx]}.tiff"
                            save_NoC(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks,  dir_save)

                IsNext.append(is_next)
   
    N_Num_Total_s1 = len(data_loader.dataset) - T_Num_Total_s1  
    N_Num_Total_s2 = len(S2_GT_marks) - T_Num_Total_s2
    results_static_s1 = [Iou_s1, Dice_s1, Sensitivity_s1, Precision_s1, Speciticity_s1, T_Num_Total_s1, T_Num_Detect_s1, N_Num_Total_s1, N_Num_Detect_s1]
    results_static_s2 = [Iou_s2, Dice_s2, Sensitivity_s2, Precision_s2, Speciticity_s2, T_Num_Total_s2, T_Num_Detect_s2, N_Num_Total_s2, N_Num_Detect_s2]
    results_box_s1 = [Remain_boxes_s1, Remain_scores_s1, Remain_score_weights_s1, Select_scores_tumor_s1, Select_scores_weight_tumor_s1, Select_scores_zero_s1, 
                        Select_scores_weight_zero_s1, Select_use_scores_s1, Select_use_score_weights_s1, Score_1st_s1, Score_2nd_s1]
    results_box_s2 = [Remain_boxes_s2, Remain_scores_s2, Remain_score_weights_s2, Select_scores_tumor_s2, Select_scores_weight_tumor_s2, Select_scores_zero_s2, 
                        Select_scores_weight_zero_s2, Select_use_scores_s2, Select_use_score_weights_s2, Score_1st_s2, Score_2nd_s2]

    results_s1 = [results_static_s1, S1_GT_marks, S1_PD_marks, results_box_s1, Tumor_Results_S1, Zero_Results_S1]
    results_s2 = [results_static_s2, S2_GT_marks, S2_PD_marks, results_box_s2, Tumor_Results_S2, Zero_Results_S2]
    results_slice = [Patient_S1, Slice_S1, Iszero_S1, IsNext, Patient_S2, Slice_S2, Iszero_S2]
    results_s2_ori = [Patient_S2, Slice_S2, Iszero_S2, S2_GT_marks_ori, S2_PD_marks_ori, Iou_s2_ori, Dice_s2_ori, Speciticity_s2_ori, Sensitivity_s2_ori, Precision_s2_ori]
    print("")
    print(f"{epoch} Epoch Validation Fnish, Excel Write Start")
    return  results_s1, len(S2_GT_marks), results_s2, results_slice, results_s2_ori

@torch.no_grad()
def val_s2OnlyT_NSq(args, model, postprocessors, criterion, data_loader, model_2, postprocessors_2, criterion_2, epoch, device, colon_info):
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
    header = 'Val:'

    output_dir = f"{args.output_dir}Epoch_{epoch}/"
    DIR_Val_S1 = f"{output_dir}IMG_Result/Val_S1/"      
    os.makedirs(DIR_Val_S1, exist_ok=True)
    DIR_Val_S2 = f"{output_dir}IMG_Result/Val_S2/"      
    os.makedirs(DIR_Val_S2, exist_ok=True)
    DIR_Val_NoC = f"{output_dir}IMG_Result/Val_Noc/"      
    os.makedirs(DIR_Val_NoC, exist_ok=True)

    print_freq = 10
    idx_data = 0
    idx_data_2 = 0
    img_save_count = 0
    img_save_count_noc = 0
    img_max_num = 100
    img_max_num_noc = 15
    img_save_name = []

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

    Score_1st_s1 = []
    Score_2nd_s1 = []
    Score_1st_s2 = []
    Score_2nd_s2 = []

    S1_GT_marks = []
    S1_PD_marks = []
    S2_GT_marks = []
    S2_PD_marks = []

    T_Num_Total_s1 = 0
    T_Num_Detect_s1 = 0
    N_Num_Detect_s1 = 0
    T_Num_Total_s2 = 0
    T_Num_Detect_s2 = 0
    N_Num_Detect_s2 = 0

    Tumor_Results_S1 = []
    Zero_Results_S1 = []
    Tumor_Results_S2 = []
    Zero_Results_S2 = []

    Patient_S1 = []
    Slice_S1 = []
    Iszero_S1 = []
    IsNext = []
    Patient_S2 = []
    Slice_S2 = []
    Iszero_S2 = []

    Iou_s1 = []
    Dice_s1 = []
    Speciticity_s1 = []
    Sensitivity_s1 = []
    Precision_s1 = []
    Iou_s2 = []
    Dice_s2 = []
    Speciticity_s2 = []
    Sensitivity_s2 = []
    Precision_s2 = []

    Remain_boxes_s1 = []
    Remain_scores_s1 = []
    Remain_score_weights_s1 = []
    Select_scores_tumor_s1 = []
    Select_scores_weight_tumor_s1 = []
    Select_scores_zero_s1 = []
    Select_scores_weight_zero_s1 = []
    Select_use_scores_s1 = []
    Select_use_score_weights_s1 = []
    Remain_boxes_s2 = []
    Remain_scores_s2 = []
    Remain_score_weights_s2 = []
    Select_scores_tumor_s2 = []
    Select_scores_weight_tumor_s2 = []
    Select_scores_zero_s2 = []
    Select_scores_weight_zero_s2 = []
    Select_use_scores_s2 = []
    Select_use_score_weights_s2 = []

    Iou_s2_ori = []
    Dice_s2_ori = []
    Speciticity_s2_ori = []
    Sensitivity_s2_ori = []
    Precision_s2_ori = []
    S2_GT_marks_ori = []
    S2_PD_marks_ori = []

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

                is_zero = 'X'
                if "ZERO" not in number_list[idx]:
                    label = [1]
                    T_Num_Total_s1 += 1
                else:
                    label = [args.zero_label]
                    is_zero = 'O'
                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                d['boxes'] = gt_bbox
                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                targets_s1.append(d)  
                d = {}   

                number_slice = number_list[idx]   
                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                slice_name = number_slice.split("_")[2]
                
                Patient_S1.append(patient)
                Slice_S1.append(slice_name)
                Iszero_S1.append(is_zero)

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
                t_sizes.append([512, 512])
            orig_target_sizes = torch.tensor(t_sizes, dtype=torch.int64, device=device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            is_s1_save = True
            is_noc_save = True
            for idx_all in range(len(results)):
                best_idx = int(torch.argmax(results[idx_all]['scores']))
                boxes = results[idx_all]['boxes']
                scores = results[idx_all]['scores']
                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                scores_cpu_ori = scores.cpu().numpy()
                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                score = round(float(scores_cpu[0]), 4)
                score_2 = round(float(scores_cpu[1]), 4)
                Score_1st_s1.append(score)
                Score_2nd_s1.append(score_2)

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

                    box_list = []
                    for i in range(args.box_usenum):
                        box_list.append(boxes_cpu[i].reshape(2,2))
      
                    score_list = scores_cpu[0:4]
                    score_weight_list = [0, 0, 0, 0]
                    select_score_list = score_list
                    select_score_weihgt_list = score_list

                else:
                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                    box_list, score_list, score_weight_list, select_score_list, select_score_weihgt_list = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)

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
                    
                Remain_boxes_s1.append(box_list)
                Remain_scores_s1.append(score_list)
                Remain_score_weights_s1.append(score_weight_list)
                if len(select_score_list) > 0:
                    Select_use_scores_s1.extend(select_score_list)
                    Select_use_score_weights_s1.extend(select_score_weihgt_list)

                if max_y < 80 and "ZERO" in number_list[idx_all]:                      ## zero detect
                    N_Num_Detect_s1 += 1
                elif max_y > 80 and "ZERO" not in number_list[idx_all]:
                    T_Num_Detect_s1 += 1

                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                S1_GT_marks.append(gt_landmarks_cpu[idx_all])
                S1_PD_marks.append(pred_landmarks)

                TP, FP, TN, FN = cal_score(gt_landmarks_cpu[idx_all], pred_landmarks, (512, 512))
                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
                Iou_s1.append(iou)
                Dice_s1.append(dice)
                Speciticity_s1.append(specitificity)
                Sensitivity_s1.append(sensitivity)
                Precision_s1.append(precision)

                if "ZERO" in number_list[idx_all]:
                    Zero_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_zero_s1.extend(select_score_list)
                    Select_scores_weight_zero_s1.extend(select_score_weihgt_list)
                else:
                    Tumor_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_tumor_s1.extend(select_score_list)
                    Select_scores_weight_tumor_s1.extend(select_score_weihgt_list)
                
                patient = f"{number_list[idx_all].split('_')[0]}_{number_list[idx_all].split('_')[1]}"
                slice_num = int(number_list[idx_all].split('_')[2])
                try:
                    min_x = int(colon_info[patient][str(slice_num)][0])
                    min_y = int(colon_info[patient][str(slice_num)][1])
                    max_x = int(colon_info[patient][str(slice_num)][2])
                    max_y = int(colon_info[patient][str(slice_num)][3])
                except Exception as e:
                    print("")
                    print(f"{patient} _ {slice_num} Check")
                    print(e)
                    exit()
                gt_segmark = [[min_x, min_y],[max_x, max_y]]
                
                TP, _, _, _ = cal_score(gt_segmark, pred_landmarks, (shape_x, shape_y))
                is_next = False
                
                if gt_segmark[0][1] > 50 and "ZERO" not in number_list[idx_all]: ## miny >50 이 colon 있는 slice 라고 가정
                    correct_per = TP/((gt_segmark[1][0]-gt_segmark[0][0])*(gt_segmark[1][1]-gt_segmark[0][1]))
                    idx = idx_all

                    cut_margin_min_x = 20
                    cut_margin_min_y = 20
                    cut_margin_max_x = 20
                    cut_margin_max_y = 20

                    min_x_cut = gt_segmark[0][0]
                    min_y_cut = gt_segmark[0][1]
                    max_x_cut = gt_segmark[1][0]
                    max_y_cut = gt_segmark[1][1]
                    
                    if min_x_cut < 20:
                        cut_margin_min_x = 0
                    if min_y_cut < 20:              
                        cut_margin_min_y = 0
                    if max_x_cut > 491:  
                        cut_margin_max_x = 0
                    if max_y_cut > 491:  
                        cut_margin_max_y = 0
                    
                    cut_img = input_tensor_cpu[idx][0][min_y_cut-cut_margin_min_y:max_y_cut+cut_margin_max_y,min_x_cut-cut_margin_min_x:max_x_cut+cut_margin_max_x]
                    
                    if correct_per >= 0.01:
                        is_next = True
                        
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

                        s1_cut_marks = [[min_x_cut-cut_margin_min_x, min_y_cut-cut_margin_min_y],[max_x_cut+cut_margin_max_x, max_y_cut+cut_margin_max_y]]
                        
                        if img_save_count < img_max_num and is_s1_save:
                            img_save_count += 1
                            is_s1_save = False
                            img_save_name.append(number_list[idx])
                            
                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            target_seg = seg_data[:,:,slice_num]
                            target_seg = np.flip(target_seg, axis=0)
                            target_seg = np.rot90(target_seg, 3)
                            
                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_S1}E{epoch}_{number_list[idx]}.tiff"
                            gt_cut_marks = [[gt_cut_min_x, gt_cut_min_y], [gt_cut_max_x, gt_cut_max_y]]
                            
                            save_1st(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks, gt_cut_marks, s1_cut_marks, dir_save)
                        
                        shape_y_cut, shape_x_cut = cut_img.shape
                        cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                        gt_cut_min_x *= (256 / shape_x_cut)
                        gt_cut_min_y *= (256 / shape_y_cut)
                        gt_cut_max_x *= (256 / shape_x_cut)
                        gt_cut_max_y *= (256 / shape_y_cut)

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
                                
                                is_zero = 'X'
                                if "ZERO" not in Cut_Number_list[s_num]:
                                    label = [1]
                                    T_Num_Total_s2 += 1
                                else:
                                    label = [args.zero_label]
                                    is_zero = 'O'
                                    
                                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                                d['boxes'] = gt_bbox
                                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                                targets.append(d)     
                                d = {}   

                                number_slice = Cut_Number_list[s_num]
                                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                                slice_name = number_slice.split("_")[2]
                                
                                Patient_S2.append(patient)
                                Slice_S2.append(slice_name)
                                Iszero_S2.append(is_zero)
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
                                boxes = results_2[idx]['boxes']
                                scores = results_2[idx]['scores']
                                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                                scores_cpu_ori = scores.cpu().numpy()
                                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                                score = round(float(scores_cpu[0]), 4)
                                score_2 = round(float(scores_cpu[1]), 4)
                                Score_1st_s2.append(score)
                                Score_2nd_s2.append(score_2)

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

                                    score_list = scores_cpu[0:4]
                                    score_weight_list = [0, 0, 0, 0]
                                    select_scores = score_list
                                    select_score_weihgts = score_list
                                else:
                                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                                    box_list, score_list, score_weight_list, select_scores, select_score_weihgts = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)
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
                                
                                Remain_boxes_s2.append(box_list)
                                Remain_scores_s2.append(score_list)
                                Remain_score_weights_s2.append(score_weight_list)
                                if len(select_score_list) > 0:
                                    Select_use_scores_s2.extend(select_scores)
                                    Select_use_score_weights_s2.extend(select_score_weihgts)
                                    
                                if max_y < 10 and "ZERO" in Cut_Number_list[idx]:                      ## zero detect
                                    N_Num_Detect_s2 += 1
                                elif max_y > 10 and "ZERO" not in Cut_Number_list[idx]:
                                    T_Num_Detect_s2 += 1

                                pred_landmarks = [[min_x, min_y],[max_x, max_y]]
                                S2_GT_marks.append(Cut_GT_marks[idx])
                                S2_PD_marks.append(pred_landmarks)

                                TP, FP, TN, FN = cal_score(Cut_GT_marks[idx], pred_landmarks, (256, 256))
                                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)

                                Iou_s2.append(iou)
                                Dice_s2.append(dice)
                                Speciticity_s2.append(specitificity)
                                Sensitivity_s2.append(sensitivity)
                                Precision_s2.append(precision)

                                if "ZERO" in Cut_Number_list[idx]:
                                    Zero_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                    Select_scores_zero_s2.extend(select_score_list)
                                    Select_scores_weight_zero_s2.extend(select_score_weihgt_list)   
                                else:
                                    Tumor_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                    Select_scores_tumor_s2.extend(select_score_list)
                                    Select_scores_weight_tumor_s2.extend(select_score_weihgt_list)

                                if Cut_Number_list[idx] in img_save_name:
                                    img_list = [CUT_Imgs[idx], Cut_Imgs_ori[idx]]
                                    dir_save = f"{DIR_Val_S2}E{epoch}_{Cut_Number_list[idx]}.tiff"
                                    
                                    save_2nd(img_list, Cut_GT_marks[idx], pred_landmarks, Cut_Colon_marks_ori[idx], Cut_GT_marks_ori[idx], Cut_PD_marks_ori[idx], Cut_marks[idx],
                                                Cut_length[idx], Cut_ratio[idx], dir_save)
                                
                                if pred_landmarks[1][1] > 10: 
                                    pd_min_x = (pred_landmarks[0][0] / Cut_ratio[idx][0]) + Cut_length[idx][0][0]
                                    pd_min_y = (pred_landmarks[0][1] / Cut_ratio[idx][1]) + Cut_length[idx][0][1]
                                    pd_max_x = (pred_landmarks[1][0] / Cut_ratio[idx][0]) + Cut_length[idx][1][0]
                                    pd_max_y = (pred_landmarks[1][1] / Cut_ratio[idx][1]) + Cut_length[idx][1][1]
                                else:
                                    pd_min_x = 0
                                    pd_min_y = 0
                                    pd_max_x = 0
                                    pd_max_y = 0

                                TP_ori, FP_ori, TN_ori, FN_ori = cal_score(Cut_GT_marks_ori[idx], [[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]], (512, 512))
                                iou_ori, dice_ori, specitificity_ori, sensitivity_ori, precision_ori = Cal_Result(TP_ori, FP_ori, TN_ori, FN_ori)

                                Iou_s2_ori.append(iou_ori)
                                Dice_s2_ori.append(dice_ori)
                                Speciticity_s2_ori.append(specitificity_ori)
                                Sensitivity_s2_ori.append(sensitivity_ori)
                                Precision_s2_ori.append(precision_ori)
                                S2_GT_marks_ori.append(Cut_GT_marks_ori[idx])
                                S2_PD_marks_ori.append([[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]])

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
                    else:
                        if img_save_count_noc < img_max_num_noc and is_noc_save:
                            img_save_count_noc += 1
                            is_noc_save = False
                            
                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            seg_data, header = nrrd.read(f"{args.data_seg}{seg_colon}.seg.nrrd")
                            target_seg = seg_data[:,:,slice_num]
                            target_seg = np.flip(target_seg, axis=0)
                            target_seg = np.rot90(target_seg, 3)
                            
                            img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                            dir_save = f"{DIR_Val_NoC}E{epoch}_{number_list[idx]}.tiff"
                            save_NoC(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks,  dir_save)

                IsNext.append(is_next)
   
    N_Num_Total_s1 = len(data_loader.dataset) - T_Num_Total_s1  
    N_Num_Total_s2 = len(S2_GT_marks) - T_Num_Total_s2
    results_static_s1 = [Iou_s1, Dice_s1, Sensitivity_s1, Precision_s1, Speciticity_s1, T_Num_Total_s1, T_Num_Detect_s1, N_Num_Total_s1, N_Num_Detect_s1]
    results_static_s2 = [Iou_s2, Dice_s2, Sensitivity_s2, Precision_s2, Speciticity_s2, T_Num_Total_s2, T_Num_Detect_s2, N_Num_Total_s2, N_Num_Detect_s2]
    results_box_s1 = [Remain_boxes_s1, Remain_scores_s1, Remain_score_weights_s1, Select_scores_tumor_s1, Select_scores_weight_tumor_s1, Select_scores_zero_s1, 
                        Select_scores_weight_zero_s1, Select_use_scores_s1, Select_use_score_weights_s1, Score_1st_s1, Score_2nd_s1]
    results_box_s2 = [Remain_boxes_s2, Remain_scores_s2, Remain_score_weights_s2, Select_scores_tumor_s2, Select_scores_weight_tumor_s2, Select_scores_zero_s2, 
                        Select_scores_weight_zero_s2, Select_use_scores_s2, Select_use_score_weights_s2, Score_1st_s2, Score_2nd_s2]

    results_s1 = [results_static_s1, S1_GT_marks, S1_PD_marks, results_box_s1, Tumor_Results_S1, Zero_Results_S1]
    results_s2 = [results_static_s2, S2_GT_marks, S2_PD_marks, results_box_s2, Tumor_Results_S2, Zero_Results_S2]
    results_slice = [Patient_S1, Slice_S1, Iszero_S1, IsNext, Patient_S2, Slice_S2, Iszero_S2]
    results_s2_ori = [Patient_S2, Slice_S2, Iszero_S2, S2_GT_marks_ori, S2_PD_marks_ori, Iou_s2_ori, Dice_s2_ori, Speciticity_s2_ori, Sensitivity_s2_ori, Precision_s2_ori]
    print("")
    print(f"{epoch} Epoch Validation Fnish, Excel Write Start")
    return  results_s1, len(S2_GT_marks), results_s2, results_slice, results_s2_ori

@torch.no_grad()
def val_contour(args, model, postprocessors, criterion, data_loader, model_2, postprocessors_2, criterion_2, epoch, device):
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
    header = 'Val:'

    output_dir = f"{args.output_dir}Epoch_{epoch}/"
    DIR_Val_S1 = f"{output_dir}IMG_Result/Val_S1/"      
    os.makedirs(DIR_Val_S1, exist_ok=True)
    DIR_Val_S2 = f"{output_dir}IMG_Result/Val_S2/"      
    os.makedirs(DIR_Val_S2, exist_ok=True)
    DIR_Val_NoC = f"{output_dir}IMG_Result/Val_Nooverlap/"      
    os.makedirs(DIR_Val_NoC, exist_ok=True)
    DIR_Val_NoColonMask = f"{output_dir}IMG_Result/Val_Noc/"      
    os.makedirs(DIR_Val_NoColonMask, exist_ok=True)

    print_freq = 10
    idx_data = 0
    idx_data_2 = 0
    img_save_count = 0
    img_save_count_noc = 0
    img_max_num = 100
    img_normal_num = 0
    img_max_normal_num = 30
    img_save_only_tumor = False
    is_stop_normal = True
    img_max_num_noc = 30
    img_save_name = []

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

    Score_1st_s1 = []
    Score_2nd_s1 = []
    Score_1st_s2 = []
    Score_2nd_s2 = []

    S1_GT_marks = []
    S1_PD_marks = []
    S2_GT_marks = []
    S2_PD_marks = []

    T_Num_Total_s1 = 0
    T_Num_Detect_s1 = 0
    N_Num_Detect_s1 = 0
    T_Num_Total_s2 = 0
    T_Num_Detect_s2 = 0
    N_Num_Detect_s2 = 0

    Tumor_Results_S1 = []
    Zero_Results_S1 = []
    Tumor_Results_S2 = []
    Zero_Results_S2 = []

    Patient_S1 = []
    Slice_S1 = []
    Iszero_S1 = []
    IsNext = []
    Patient_S2 = []
    Slice_S2 = []
    Iszero_S2 = []

    Iou_s1 = []
    Dice_s1 = []
    Speciticity_s1 = []
    Sensitivity_s1 = []
    Precision_s1 = []
    Iou_s2 = []
    Dice_s2 = []
    Speciticity_s2 = []
    Sensitivity_s2 = []
    Precision_s2 = []

    Remain_boxes_s1 = []
    Remain_scores_s1 = []
    Remain_score_weights_s1 = []
    Select_scores_tumor_s1 = []
    Select_scores_weight_tumor_s1 = []
    Select_scores_zero_s1 = []
    Select_scores_weight_zero_s1 = []
    Select_use_scores_s1 = []
    Select_use_score_weights_s1 = []
    Remain_boxes_s2 = []
    Remain_scores_s2 = []
    Remain_score_weights_s2 = []
    Select_scores_tumor_s2 = []
    Select_scores_weight_tumor_s2 = []
    Select_scores_zero_s2 = []
    Select_scores_weight_zero_s2 = []
    Select_use_scores_s2 = []
    Select_use_score_weights_s2 = []

    Iou_s2_ori = []
    Dice_s2_ori = []
    Speciticity_s2_ori = []
    Sensitivity_s2_ori = []
    Precision_s2_ori = []
    S2_GT_marks_ori = []
    S2_PD_marks_ori = []
    Patient_ori = []
    Slice_ori = []
    Is_normal_ori = []
    Cut_marks_ori = []

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

                is_zero = 'X'
                if "ZERO" not in number_list[idx]:
                    label = [1]
                    T_Num_Total_s1 += 1
                else:
                    label = [args.zero_label]
                    is_zero = 'O'
                gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                d['boxes'] = gt_bbox
                d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                targets_s1.append(d)  
                d = {}   

                number_slice = number_list[idx]   
                patient = number_slice.split("_")[0][0] + "_" + number_slice.split("_")[1]
                slice_name = number_slice.split("_")[2]
                
                Patient_S1.append(patient)
                Slice_S1.append(slice_name)
                Iszero_S1.append(is_zero)

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
                t_sizes.append([512, 512])
            orig_target_sizes = torch.tensor(t_sizes, dtype=torch.int64, device=device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            is_s1_save = True
            is_noc_save = True
            for idx_all in range(len(results)):
                is_pred_zero = False
                best_idx = int(torch.argmax(results[idx_all]['scores']))
                boxes = results[idx_all]['boxes']
                scores = results[idx_all]['scores']
                boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                scores_cpu_ori = scores.cpu().numpy()
                boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                score = round(float(scores_cpu[0]), 4)
                score_2 = round(float(scores_cpu[1]), 4)
                Score_1st_s1.append(score)
                Score_2nd_s1.append(score_2)

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

                    box_list = []
                    for i in range(args.box_usenum):
                        box_list.append(boxes_cpu[i].reshape(2,2))
      
                    score_list = scores_cpu[0:4]
                    score_weight_list = [0, 0, 0, 0]
                    select_score_list = score_list
                    select_score_weihgt_list = score_list

                else:
                    zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                    box_list, score_list, score_weight_list, select_score_list, select_score_weihgt_list = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)

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
                    
                Remain_boxes_s1.append(box_list)
                Remain_scores_s1.append(score_list)
                Remain_score_weights_s1.append(score_weight_list)
                if len(select_score_list) > 0:
                    Select_use_scores_s1.extend(select_score_list)
                    Select_use_score_weights_s1.extend(select_score_weihgt_list)

                if max_y < 80 and "ZERO" in number_list[idx_all]:                      ## zero detect
                    N_Num_Detect_s1 += 1
                elif max_y > 80 and "ZERO" not in number_list[idx_all]:
                    T_Num_Detect_s1 += 1

                pred_landmarks_1st = [[min_x, min_y],[max_x, max_y]]
                S1_GT_marks.append(gt_landmarks_cpu[idx_all])
                S1_PD_marks.append(pred_landmarks_1st)
                
                patient = f"{number_list[idx_all].split('_')[0]}_{number_list[idx_all].split('_')[1]}"
                # patient = f"{number_list[idx_all].split('_')[1]}.seg"
                slice_num = int(number_list[idx_all].split('_')[2])
                
                if max_y > 80: #### pd tumor
                    contor_list, contour_num = get_include_landmarks(args.data_seg_contour, patient, slice_num, pred_landmarks_1st)
                else:          #### pd normal
                    contor_list, contour_num = get_noraml_landmarks(args.data_seg_contour, patient, slice_num)
                    is_pred_zero = True

                TP, FP, TN, FN = cal_score(gt_landmarks_cpu[idx_all], pred_landmarks_1st, (512, 512))
                iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)
                Iou_s1.append(iou)
                Dice_s1.append(dice)
                Speciticity_s1.append(specitificity)
                Sensitivity_s1.append(sensitivity)
                Precision_s1.append(precision)

                if "ZERO" in number_list[idx_all]:
                    Zero_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_zero_s1.extend(select_score_list)
                    Select_scores_weight_zero_s1.extend(select_score_weihgt_list)
                else:
                    Tumor_Results_S1.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                    Select_scores_tumor_s1.extend(select_score_list)
                    Select_scores_weight_tumor_s1.extend(select_score_weihgt_list)

                is_next = False
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
                                 
                    TP, _, _, _ = cal_score(gt_segmark, pred_landmarks_1st, (shape_x, shape_y))
                    
                    if max_y_over > 50: ## miny >50 이 colon seg 있는 slice 라고 가정 
                        correct_per = TP/((gt_segmark[1][0]-gt_segmark[0][0])*(gt_segmark[1][1]-gt_segmark[0][1]))
                        idx = idx_all

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
                        
                        if min_x_cut < 20:
                            cut_margin_min_x = 0
                        if min_y_cut < 20:              
                            cut_margin_min_y = 0
                        if max_x_cut > 491:  
                            cut_margin_max_x = 0
                        if max_y_cut > 491:  
                            cut_margin_max_y = 0
                        
                        cut_img = input_tensor_cpu[idx][0][min_y_cut-cut_margin_min_y:max_y_cut+cut_margin_max_y,min_x_cut-cut_margin_min_x:max_x_cut+cut_margin_max_x]
                        
                        if (correct_per >= 0.01 and "ZERO" not in number_list[idx]) or is_pred_zero:
                        # if (correct_per >= 0.01) or is_pred_zero:
                            is_next = True

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
                            
                            if img_save_only_tumor:
                                if "ZERO" not in number_list[idx]:
                                    is_stop_normal = True
                                else:
                                    is_stop_normal = False
                            
                            if img_save_count < img_max_num and is_s1_save and is_stop_normal:
                                if img_save_count < img_max_num and is_s1_save:
                                    if "ZERO" in number_list[idx]:
                                        img_normal_num += 1
                                    if img_max_normal_num >= img_normal_num:
                                        img_save_only_tumor = True
                                img_save_count += 1
                                is_s1_save = False
                                img_save_name.append(f"{number_list[idx]}_{c_idx}")  
                                
                                seg_colon = f"{number_list[idx].split('_')[1]}"
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
                                    target_seg = seg_data[:,:,slice_num]
                                    exit()
                        
                                img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                                dir_save = f"{DIR_Val_S1}E{epoch}_{number_list[idx]}_{c_idx}.tiff"
                                gt_cut_marks = [[gt_cut_min_x, gt_cut_min_y], [gt_cut_max_x, gt_cut_max_y]]
                                
                                save_1st(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks_1st, gt_cut_marks, s1_cut_marks, dir_save)
                            
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
                                Cut_GT_marks_ori.append(np.array([[0,0],[1,1]]))  
                            else:
                                Cut_GT_marks_ori.append(gt_landmarks_cpu[idx])  
                           
                            CUT_Imgs.append(cut_img)
                            Cut_GT_marks.append(np.array([[gt_cut_min_x,gt_cut_min_y],[gt_cut_max_x,gt_cut_max_y]]))         
                            Cut_Number_list.append(f"{number_list[idx]}_{c_idx}")       
                            Cut_Colon_marks_ori.append(gt_segmark)
                            Cut_Imgs_ori.append(input_tensor_cpu[idx][0])
                            
                            Cut_length.append([[(min_x_cut-cut_margin_min_x), (min_y_cut-cut_margin_min_y)],[(gt_landmarks_cpu[idx][0][0] - (gt_landmarks_cpu[idx][0][0] - (min_x_cut - cut_margin_min_x))), 
                                                (gt_landmarks_cpu[idx][0][1] - (gt_landmarks_cpu[idx][0][1] - (min_y_cut - cut_margin_min_y)))]])   
                            Cut_ratio.append([(256 / shape_x_cut), (256 / shape_y_cut)])
                            Cut_PD_marks_ori.append(pred_landmarks_1st)
                            Cut_marks.append(s1_cut_marks)
                                                    
                            ## 2nd stage
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
                                    
                                    is_zero = 'X'
                                    if "ZERO" not in Cut_Number_list[s_num]:
                                        label = [1]
                                        T_Num_Total_s2 += 1
                                    else:
                                        label = [args.zero_label]
                                        is_zero = 'O'
                                        
                                    gt_bbox = torch.tensor([bbox], dtype=torch.float64, device=device)
                                    d['boxes'] = gt_bbox
                                    d['labels'] = torch.tensor(label, dtype=torch.int64, device=device) 
                                    targets.append(d)     
                                    d = {}   

                                    number_slice = Cut_Number_list[s_num]
                                    patient = f"{number_slice.split('_')[0][0]}_{number_slice.split('_')[1]}"
                                    slice_name = f"{number_slice.split('_')[2]}_{number_slice.split('_')[-1]}"
                                    
                                    Patient_S2.append(patient)
                                    Slice_S2.append(slice_name)
                                    Iszero_S2.append(is_zero)
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
                                    boxes = results_2[idx]['boxes']
                                    scores = results_2[idx]['scores']
                                    boxes_cpu_ori = boxes.cpu().numpy().astype(np.int64)
                                    scores_cpu_ori = scores.cpu().numpy()
                                    boxes_cpu, scores_cpu = arrage_result(boxes_cpu_ori, scores_cpu_ori)

                                    score = round(float(scores_cpu[0]), 4)
                                    score_2 = round(float(scores_cpu[1]), 4)
                                    Score_1st_s2.append(score)
                                    Score_2nd_s2.append(score_2)

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

                                        score_list = scores_cpu[0:4]
                                        score_weight_list = [0, 0, 0, 0]
                                        select_scores = score_list
                                        select_score_weihgts = score_list
                                    else:
                                        zero_img, create_box_points, select_boxes = create_box(boxes_cpu, scores_cpu, args.box_th, args.box_usenum, shape_x, shape_y)
                                        box_list, score_list, score_weight_list, select_scores, select_score_weihgts = calculate_box_score(args.box_usenum, create_box_points, boxes_cpu, scores_cpu)
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
                                    
                                    Remain_boxes_s2.append(box_list)
                                    Remain_scores_s2.append(score_list)
                                    Remain_score_weights_s2.append(score_weight_list)
                                    if len(select_score_list) > 0:
                                        Select_use_scores_s2.extend(select_scores)
                                        Select_use_score_weights_s2.extend(select_score_weihgts)
                                        
                                    if max_y < 10 and "ZERO" in Cut_Number_list[idx]:                      ## zero detect
                                        N_Num_Detect_s2 += 1
                                    elif max_y > 10 and "ZERO" not in Cut_Number_list[idx]:
                                        T_Num_Detect_s2 += 1

                                    pred_landmarks = [[min_x, min_y],[max_x, max_y]]

                                    S2_GT_marks.append(Cut_GT_marks[idx])
                                    S2_PD_marks.append(pred_landmarks)

                                    TP, FP, TN, FN = cal_score(Cut_GT_marks[idx], pred_landmarks, (256, 256))
                                    iou, dice, specitificity, sensitivity, precision = Cal_Result(TP, FP, TN, FN)

                                    Iou_s2.append(iou)
                                    Dice_s2.append(dice)
                                    Speciticity_s2.append(specitificity)
                                    Sensitivity_s2.append(sensitivity)
                                    Precision_s2.append(precision)

                                    if "ZERO" in Cut_Number_list[idx]:
                                        Zero_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                        Select_scores_zero_s2.extend(select_score_list)
                                        Select_scores_weight_zero_s2.extend(select_score_weihgt_list)   
                                    else:
                                        Tumor_Results_S2.append([iou, dice, sensitivity, precision, specitificity,  score, score_2])
                                        Select_scores_tumor_s2.extend(select_score_list)
                                        Select_scores_weight_tumor_s2.extend(select_score_weihgt_list)

                                    if Cut_Number_list[idx] in img_save_name:
                                        img_list = [CUT_Imgs[idx], Cut_Imgs_ori[idx]]
                                        dir_save = f"{DIR_Val_S2}E{epoch}_{Cut_Number_list[idx]}.tiff"
                                        
                                        save_2nd(img_list, Cut_GT_marks[idx], pred_landmarks, Cut_Colon_marks_ori[idx], Cut_GT_marks_ori[idx], Cut_PD_marks_ori[idx], Cut_marks[idx],
                                                    Cut_length[idx], Cut_ratio[idx], dir_save)
                                    
                                    if pred_landmarks[1][1] > 10: 
                                        pd_min_x = int(int((pred_landmarks[0][0] / Cut_ratio[idx][0])) + Cut_length[idx][0][0])
                                        pd_min_y = int(int((pred_landmarks[0][1] / Cut_ratio[idx][1])) + Cut_length[idx][0][1])
                                        pd_max_x = int(int((pred_landmarks[1][0] / Cut_ratio[idx][0])) + Cut_length[idx][1][0])
                                        pd_max_y = int(int((pred_landmarks[1][1] / Cut_ratio[idx][1])) + Cut_length[idx][1][1])
                                    else:
                                        pd_min_x = 0
                                        pd_min_y = 0
                                        pd_max_x = 1
                                        pd_max_y = 1

                                    # if "ZERO" not in Cut_Number_list[idx]:
                                    TP_ori, FP_ori, TN_ori, FN_ori = cal_score(Cut_GT_marks_ori[idx], [[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]], (512, 512))
                                    iou_ori, dice_ori, specitificity_ori, sensitivity_ori, precision_ori = Cal_Result(TP_ori, FP_ori, TN_ori, FN_ori)

                                    Iou_s2_ori.append(iou_ori)
                                    Dice_s2_ori.append(dice_ori)
                                    Speciticity_s2_ori.append(specitificity_ori)
                                    Sensitivity_s2_ori.append(sensitivity_ori)
                                    Precision_s2_ori.append(precision_ori)
                                    S2_GT_marks_ori.append(Cut_GT_marks_ori[idx])
                                    S2_PD_marks_ori.append([[pd_min_x,pd_min_y],[pd_max_x,pd_max_y]])
                                    p_ori = f"{Cut_Number_list[idx].split('_')[0][0]}_{Cut_Number_list[idx].split('_')[1]}"
                                    s_name_ori = f"{Cut_Number_list[idx].split('_')[2]}_{Cut_Number_list[idx].split('_')[-1]}"
                                    
                                    Patient_ori.append(p_ori)
                                    Slice_ori.append(s_name_ori)
                                    Cut_marks_ori.append(Cut_marks[idx])
                                    if "ZERO" not in Cut_Number_list[idx]:
                                        Is_normal_ori.append("X")
                                    else:
                                        Is_normal_ori.append("O")
                                    

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
                        else:
                            if img_save_count_noc <= img_max_num_noc and is_noc_save:
                                img_save_count_noc += 1
                                is_noc_save = False
                                seg_colon = f"{number_list[idx].split('_')[1]}"
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
                                    target_seg = seg_data[:,:,slice_num]
                                    exit()
                                img_list = [input_tensor_cpu[idx][0], target_seg, cut_img]
                                dir_save = f"{DIR_Val_NoC}E{epoch}_{number_list[idx]}_{c_idx}.tiff"
                                save_NoC(img_list, gt_segmark, gt_landmarks_cpu[idx], pred_landmarks_1st, dir_save)
                    else: ## max y < 50    
                        if img_save_count_noc <= img_max_num_noc and is_noc_save:     
                            img_save_count_noc += 1
                            is_noc_save = False

                            seg_colon = f"{number_list[idx_all].split('_')[1]}"
                            slice_num = int(number_list[idx_all].split('_')[2])
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
                                target_seg = seg_data[:,:,slice_num]
                                exit()

                            img_list = [input_tensor_cpu[idx_all][0]]
                            dir_save = f"{DIR_Val_NoColonMask}E{epoch}_{number_list[idx_all]}_{c_idx}.tiff"
                            
                            save_NoColonMask(input_tensor_cpu[idx_all][0], target_seg, gt_landmarks_cpu[idx_all], pred_landmarks_1st, dir_save)
                IsNext.append(is_next)
   
    N_Num_Total_s1 = len(data_loader.dataset) - T_Num_Total_s1  
    N_Num_Total_s2 = len(S2_GT_marks) - T_Num_Total_s2
    results_static_s1 = [Iou_s1, Dice_s1, Sensitivity_s1, Precision_s1, Speciticity_s1, T_Num_Total_s1, T_Num_Detect_s1, N_Num_Total_s1, N_Num_Detect_s1]
    results_static_s2 = [Iou_s2, Dice_s2, Sensitivity_s2, Precision_s2, Speciticity_s2, T_Num_Total_s2, T_Num_Detect_s2, N_Num_Total_s2, N_Num_Detect_s2]
    results_box_s1 = [Remain_boxes_s1, Remain_scores_s1, Remain_score_weights_s1, Select_scores_tumor_s1, Select_scores_weight_tumor_s1, Select_scores_zero_s1, 
                        Select_scores_weight_zero_s1, Select_use_scores_s1, Select_use_score_weights_s1, Score_1st_s1, Score_2nd_s1]
    results_box_s2 = [Remain_boxes_s2, Remain_scores_s2, Remain_score_weights_s2, Select_scores_tumor_s2, Select_scores_weight_tumor_s2, Select_scores_zero_s2, 
                        Select_scores_weight_zero_s2, Select_use_scores_s2, Select_use_score_weights_s2, Score_1st_s2, Score_2nd_s2]

    results_s1 = [results_static_s1, S1_GT_marks, S1_PD_marks, results_box_s1, Tumor_Results_S1, Zero_Results_S1]
    results_s2 = [results_static_s2, S2_GT_marks, S2_PD_marks, results_box_s2, Tumor_Results_S2, Zero_Results_S2]
    results_slice = [Patient_S1, Slice_S1, Iszero_S1, IsNext, Patient_S2, Slice_S2, Iszero_S2]
    results_s2_ori = [Patient_ori, Slice_ori, Is_normal_ori, S2_GT_marks_ori, S2_PD_marks_ori, Iou_s2_ori, Dice_s2_ori, Sensitivity_s2_ori, Precision_s2_ori, Speciticity_s2_ori, Cut_marks_ori]
    print("")
    print(f"{epoch} Epoch Validation Fnish, Excel Write Start")
    return  results_s1, len(S2_GT_marks), results_s2, results_slice, results_s2_ori