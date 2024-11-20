import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cv2

def save_1st(img_list, gt_segmarks, gt_landmarks, pd_landmarks, gt_cut_marks, s1_cut_marks, dir_save):
    fig = plt.figure(figsize=(6,8)) 
    gs = gridspec.GridSpec(nrows=3, ncols=2)
    
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(img_list[0], cmap='gray')
    ax.set_title("Input")
    ax.axis('off')
    ax = fig.add_subplot(gs[0,1])
    ax.imshow(img_list[1], cmap='gray')
    ax.set_title("GT_Colon")
    ax.add_patch(patches.Rectangle((gt_segmarks[0][0]-2, gt_segmarks[0][1]-2), gt_segmarks[1][0] - (gt_segmarks[0][0]-2), gt_segmarks[1][1] - (gt_segmarks[0][1]-2), edgecolor = 'g', fill=False))
    if "Contour" in dir_save:
        ax.add_patch(patches.Rectangle((pd_landmarks[0][0], pd_landmarks[0][1]), pd_landmarks[1][0] - pd_landmarks[0][0], pd_landmarks[1][1] - pd_landmarks[0][1], edgecolor = 'y', fill=False))
    ax.axis('off')
    ax = fig.add_subplot(gs[1,0])
    ax.imshow(img_list[0], cmap='gray')
    ax.add_patch(patches.Rectangle((gt_segmarks[0][0]-2, gt_segmarks[0][1]-2), gt_segmarks[1][0] - (gt_segmarks[0][0]-2), gt_segmarks[1][1] - (gt_segmarks[0][1]-2), edgecolor = 'g', fill=False))
    ax.add_patch(patches.Rectangle((gt_landmarks[0][0], gt_landmarks[0][1]), gt_landmarks[1][0] - gt_landmarks[0][0], gt_landmarks[1][1] - gt_landmarks[0][1], edgecolor = 'b', fill=False))
    ax.add_patch(patches.Rectangle((pd_landmarks[0][0], pd_landmarks[0][1]), pd_landmarks[1][0] - pd_landmarks[0][0], pd_landmarks[1][1] - pd_landmarks[0][1], edgecolor = 'y', fill=False))
    ax.set_title("Predicted")
    ax.axis('off')
    ax = fig.add_subplot(gs[1,1])
    ax.imshow(img_list[0], cmap='gray')
    ax.add_patch(patches.Rectangle((s1_cut_marks[0][0], s1_cut_marks[0][1]), s1_cut_marks[1][0] - s1_cut_marks[0][0], s1_cut_marks[1][1] - s1_cut_marks[0][1], edgecolor = 'c', fill=False))
    ax.set_title("CUT_Area")
    ax.axis('off')
    ax = fig.add_subplot(gs[2,0])
    ax.imshow(img_list[2], cmap='gray')
    ax.add_patch(patches.Rectangle((gt_cut_marks[0][0], gt_cut_marks[0][1]), gt_cut_marks[1][0] - gt_cut_marks[0][0], gt_cut_marks[1][1] - gt_cut_marks[0][1], edgecolor = 'b', fill=False))
    ax.set_title("CUT")
    ax.axis('off')
    shape_y_cut, shape_x_cut = img_list[2].shape
    img_resize = cv2.resize(img_list[2], (256, 256), interpolation=cv2.INTER_CUBIC)
    gt_cut_min_x = gt_cut_marks[0][0] * (256 / shape_x_cut) 
    gt_cut_min_y = gt_cut_marks[0][1] * (256 / shape_y_cut)
    gt_cut_max_x = gt_cut_marks[1][0] * (256 / shape_x_cut)
    gt_cut_max_y = gt_cut_marks[1][1] * (256 / shape_y_cut)
    ax = fig.add_subplot(gs[2,1])
    ax.imshow(img_resize, cmap='gray')
    ax.add_patch(patches.Rectangle((gt_cut_min_x, gt_cut_min_y), gt_cut_max_x - gt_cut_min_x, gt_cut_max_y - gt_cut_min_y, edgecolor = 'b', fill=False))
    ax.set_title("2nd Input")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(dir_save)
    plt.axis('off')
    plt.close()

def save_2nd(img_list, Cut_GT_marks, pred_landmarks, Colon_marks, GT_marks_ori, PD_marks_ori, Cut_marks, Cut_length, Cut_ratio, dir_save):
    fig = plt.figure(figsize=(6,6)) 
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(img_list[0], cmap='gray')
    ax.set_title("Input")
    ax.axis('off')
    ax = fig.add_subplot(gs[0,1])
    ax.imshow(img_list[0], cmap='gray')
    ax.set_title("Pred")
    ax.add_patch(patches.Rectangle((int(pred_landmarks[0][0]), int(pred_landmarks[0][1])), int(pred_landmarks[1][0] - pred_landmarks[0][0]), int(pred_landmarks[1][1] - pred_landmarks[0][1]), edgecolor = 'r', fill=False))
    ax.add_patch(patches.Rectangle((Cut_GT_marks[0][0], Cut_GT_marks[0][1]), Cut_GT_marks[1][0] - Cut_GT_marks[0][0], Cut_GT_marks[1][1] - Cut_GT_marks[0][1], edgecolor = 'b', fill=False))
    ax.axis('off')
    ax = fig.add_subplot(gs[1,0])
    ax.imshow(img_list[1], cmap='gray')
   
    if pred_landmarks[1][1] > 10: 
        pd_min_x = (pred_landmarks[0][0] / Cut_ratio[0]) + Cut_length[0][0]
        pd_min_y = (pred_landmarks[0][1] / Cut_ratio[1]) + Cut_length[0][1]
        pd_max_x = (pred_landmarks[1][0] / Cut_ratio[0]) + Cut_length[1][0]
        pd_max_y = (pred_landmarks[1][1] / Cut_ratio[1]) + Cut_length[1][1]
    else:
        pd_min_x = 0
        pd_min_y = 0
        pd_max_x = 0
        pd_max_y = 0

    ax.add_patch(patches.Rectangle((pd_min_x, pd_min_y), pd_max_x - pd_min_x, pd_max_y - pd_min_y, edgecolor = 'r', fill=False))
    ax.add_patch(patches.Rectangle((GT_marks_ori[0][0], GT_marks_ori[0][1]), GT_marks_ori[1][0] - GT_marks_ori[0][0], GT_marks_ori[1][1] - GT_marks_ori[0][1], edgecolor = 'b', fill=False))
    ax.set_title("Ori size")
    ax.axis('off')
    ax = fig.add_subplot(gs[1,1])
    ax.imshow(img_list[1], cmap='gray')
    ax.add_patch(patches.Rectangle((pd_min_x, pd_min_y), pd_max_x - pd_min_x, pd_max_y - pd_min_y, edgecolor = 'r', fill=False))
    ax.add_patch(patches.Rectangle((GT_marks_ori[0][0], GT_marks_ori[0][1]), GT_marks_ori[1][0] - GT_marks_ori[0][0], GT_marks_ori[1][1] - GT_marks_ori[0][1], edgecolor = 'b', fill=False))
    ax.add_patch(patches.Rectangle((Colon_marks[0][0], Colon_marks[0][1]), Colon_marks[1][0] - Colon_marks[0][0], Colon_marks[1][1] - Colon_marks[0][1], edgecolor = 'g', fill=False))
    ax.add_patch(patches.Rectangle((PD_marks_ori[0][0], PD_marks_ori[0][1]), PD_marks_ori[1][0] - PD_marks_ori[0][0], PD_marks_ori[1][1] - PD_marks_ori[0][1], edgecolor = 'y', fill=False))
    ax.add_patch(patches.Rectangle((Cut_marks[0][0], Cut_marks[0][1]), Cut_marks[1][0] - Cut_marks[0][0], Cut_marks[1][1] - Cut_marks[0][1], edgecolor = 'c', fill=False))
    ax.set_title("Output")
    ax.axis('off')
  
    plt.tight_layout()
    plt.savefig(dir_save)
    plt.axis('off')
    plt.close()

    return [[pd_min_x, pd_min_y], [pd_max_x, pd_max_y]]

def save_NoC(img_list, gt_segmarks, gt_landmarks, pd_landmarks, dir_save):
    fig = plt.figure(figsize=(6,6)) 
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(img_list[0], cmap='gray')
    ax.set_title("Input")
    ax.axis('off')
    ax = fig.add_subplot(gs[0,1])
    ax.imshow(img_list[1], cmap='gray')
    ax.set_title("GT_Colon")
    ax.add_patch(patches.Rectangle((gt_segmarks[0][0]-2, gt_segmarks[0][1]-2), gt_segmarks[1][0] - (gt_segmarks[0][0]-2), gt_segmarks[1][1] - (gt_segmarks[0][1]-2), edgecolor = 'g', fill=False))
    ax.axis('off')
    ax = fig.add_subplot(gs[1,0])
    ax.imshow(img_list[0], cmap='gray')
    ax.add_patch(patches.Rectangle((gt_segmarks[0][0]-2, gt_segmarks[0][1]-2), gt_segmarks[1][0] - (gt_segmarks[0][0]-2), gt_segmarks[1][1] - (gt_segmarks[0][1]-2), edgecolor = 'g', fill=False))
    ax.add_patch(patches.Rectangle((pd_landmarks[0][0], pd_landmarks[0][1]), pd_landmarks[1][0] - pd_landmarks[0][0], pd_landmarks[1][1] - pd_landmarks[0][1], edgecolor = 'y', fill=False))
    ax.set_title("Predicted")
    ax.axis('off')
    ax = fig.add_subplot(gs[1,1])
    ax.imshow(img_list[0], cmap='gray')
    ax.add_patch(patches.Rectangle((gt_segmarks[0][0]-2, gt_segmarks[0][1]-2), gt_segmarks[1][0] - (gt_segmarks[0][0]-2), gt_segmarks[1][1] - (gt_segmarks[0][1]-2), edgecolor = 'g', fill=False))
    ax.add_patch(patches.Rectangle((gt_landmarks[0][0], gt_landmarks[0][1]), gt_landmarks[1][0] - gt_landmarks[0][0], gt_landmarks[1][1] - gt_landmarks[0][1], edgecolor = 'b', fill=False))
    ax.add_patch(patches.Rectangle((pd_landmarks[0][0], pd_landmarks[0][1]), pd_landmarks[1][0] - pd_landmarks[0][0], pd_landmarks[1][1] - pd_landmarks[0][1], edgecolor = 'y', fill=False))
    ax.set_title("Predicted")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(dir_save)
    plt.axis('off')
    plt.close()

def save_NoColonMask(input_img, target_seg, gt_landmarks, pd_landmarks, dir_save):
    fig = plt.figure(figsize=(6,6)) 
    gs = gridspec.GridSpec(nrows=1, ncols=3)
    
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(input_img, cmap='gray')
    ax.set_title("Input")
    ax.axis('off')
    ax = fig.add_subplot(gs[0,1])
    ax.imshow(input_img, cmap='gray')
    ax.add_patch(patches.Rectangle((gt_landmarks[0][0], gt_landmarks[0][1]), gt_landmarks[1][0] - gt_landmarks[0][0], gt_landmarks[1][1] - gt_landmarks[0][1], edgecolor = 'b', fill=False))
    ax.add_patch(patches.Rectangle((pd_landmarks[0][0], pd_landmarks[0][1]), pd_landmarks[1][0] - pd_landmarks[0][0], pd_landmarks[1][1] - pd_landmarks[0][1], edgecolor = 'y', fill=False))
    ax.set_title("Predicted")
    ax.axis('off')
    ax = fig.add_subplot(gs[0,2])
    ax.imshow(target_seg, cmap='gray')
    ax.set_title("GT_Colon")
    ax.add_patch(patches.Rectangle((pd_landmarks[0][0], pd_landmarks[0][1]), pd_landmarks[1][0] - pd_landmarks[0][0], pd_landmarks[1][1] - pd_landmarks[0][1], edgecolor = 'y', fill=False))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(dir_save)
    plt.axis('off')
    plt.close()