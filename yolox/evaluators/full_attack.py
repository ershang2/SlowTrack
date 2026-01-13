from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import cv2
learning_rate = 0.01 #0.07
epochs = 220
import torchvision


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def my_postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):

    # prediction shape: [Batch, N, 85] (cx, cy, w, h, obj, cls...)
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(0)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    

    output_pred = prediction.clone()
    output_pred[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(output_pred))]
    for i, image_pred in enumerate(output_pred):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def run_attack(outputs, outputs_2, bx, strategy, max_tracker_num, adam_opt, num_classes, confthre, nmsthre, itera):

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num


    idx_b_start = int(100*180+50*90+(strategy)*per_num_b)
    idx_b_end   = int(100*180+50*90+(strategy+1)*per_num_b)
    idx_m_start = int(100*180+(strategy)*per_num_m)
    idx_m_end   = int(100*180+(strategy+1)*per_num_m)
    idx_s_start = int((strategy)*per_num_s)
    idx_s_end   = int((strategy+1)*per_num_s)


    scores = outputs[:,5] * outputs[:,4]
    scores_2 = outputs_2[:,5] * outputs_2[:,4]
    
    sel_scores_b = scores[idx_b_start:idx_b_end]
    sel_scores_m = scores[idx_m_start:idx_m_end]
    sel_scores_s = scores[idx_s_start:idx_s_end]
    
    sel_scores_b_2 = scores_2[idx_b_start:idx_b_end]
    sel_scores_m_2 = scores_2[idx_m_start:idx_m_end]
    sel_scores_s_2 = scores_2[idx_s_start:idx_s_end]

    sel_dets = torch.cat((sel_scores_b, sel_scores_m, sel_scores_s), dim=0)
    sel_dets_2 = torch.cat((sel_scores_b_2, sel_scores_m_2, sel_scores_s_2), dim=0)
    
    targets = torch.ones_like(sel_dets)
    loss1 = 1e-2*(F.mse_loss(sel_dets, targets, reduction='sum')+F.mse_loss(sel_dets_2, targets, reduction='sum'))

    targets_full = torch.ones_like(scores)
    loss3 = 1.0*(F.mse_loss(scores, targets_full, reduction='sum')+F.mse_loss(scores_2, targets_full, reduction='sum'))


    wh_b = outputs[idx_b_start:idx_b_end, 2:4]/100
    wh_m = outputs[idx_m_start:idx_m_end, 2:4]/100
    wh_s = outputs[idx_s_start:idx_s_end, 2:4]/100
    sel_wh = torch.cat((wh_b, wh_m, wh_s), dim=0)
    

    wh_b_2 = outputs_2[idx_b_start:idx_b_end, 2:4]/100
    wh_m_2 = outputs_2[idx_m_start:idx_m_end, 2:4]/100
    wh_s_2 = outputs_2[idx_s_start:idx_s_end, 2:4]/100
    sel_wh_2 = torch.cat((wh_b_2, wh_m_2, wh_s_2), dim=0)


    area_1 = sel_wh[:, 0] * sel_wh[:, 1]
    area_2 = sel_wh_2[:, 0] * sel_wh_2[:, 1]
    

    weight_area = 1e-2
    loss_area = weight_area * (area_1.sum() + area_2.sum())


    final_dets_1 = my_postprocess(outputs, num_classes, confthre, nmsthre)
    final_dets_2 = my_postprocess(outputs_2, num_classes, confthre, nmsthre)
    loss_match = torch.tensor(0.0).to(bx.device)
    ioucount = torch.tensor(0).to(bx.device)
    if final_dets_1[0] is not None and final_dets_2[0] is not None:
        
        boxes1 = final_dets_1[0][:, :4]
        boxes2 = final_dets_2[0][:, :4]
        iou_matrix = torchvision.ops.box_iou(boxes1, boxes2)
        
        max_ious, _ = iou_matrix.max(dim=1)
        ioucount = (max_ious > 0.7).sum()
        loss_match = 5 * (1.0 - max_ious.mean())**2



    if itera < 50:
        loss = loss1
    else:
        loss = loss1 + loss_match + loss_area # + loss2
    
    loss.requires_grad_(True)
    adam_opt.zero_grad()
    loss.backward(retain_graph=True)
    

    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -1.5 * bx.grad+ bx.data
    count = (scores > 0.3).sum()
    count_2 = (scores_2 > 0.3).sum()
    if itera % 100 == 0:

        print('loss',loss.item(),'loss_1',loss1.item(), 'loss_area', loss_area.item(), 'loss_match', loss_match.item(),'loss_3',loss3.item(),'count:',count.item(), 'count2:',count_2.item(),'iou_cnt_0.7:',ioucount.item())
    
    return bx



class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        device,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        frame_id = 0
        total_l1 = 0
        total_l2 = 0
        strategy = 0
        max_tracker_num = int(10)
        rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
        std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
        bx_bank = {}
        for cur_iter, (imgs,path,imgs_2,path_2) in enumerate(
            progress_bar(self.dataloader)
            ):
            print('strategy:',strategy)
            print(path,path_2)

            frame_id += 1
            current_epochs = epochs # Default epochs 220
            if strategy in bx_bank:
                # Retrieve stored perturbation
                stored_bx = bx_bank[strategy]
                
                # Handle size mismatch via padding or cropping
                # imgs shape is [batch, 3, H, W]
                target_h, target_w = imgs.shape[2], imgs.shape[3]
                src_h, src_w = stored_bx.shape[2], stored_bx.shape[3]
                
                # Crop if stored is larger
                if src_h > target_h: stored_bx = stored_bx[:, :, :target_h, :]
                if src_w > target_w: stored_bx = stored_bx[:, :, :, :target_w]
                
                # Pad if stored is smaller
                curr_h, curr_w = stored_bx.shape[2], stored_bx.shape[3]
                pad_h = target_h - curr_h
                pad_w = target_w - curr_w
                if pad_h > 0 or pad_w > 0:
                    stored_bx = F.pad(stored_bx, (0, pad_w, 0, pad_h), mode='constant', value=0)
                
                bx = stored_bx.clone().detach()
                current_epochs = 100 # Reduced epochs for fine-tuning
            else:
                # Initialize new zero perturbation
                bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
                bx = bx.astype(np.float32)
                bx = torch.from_numpy(bx).unsqueeze(0)
                current_epochs = 220 # Full training epochs
            # bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
            # bx = bx.astype(np.float32)
            # bx = torch.from_numpy(bx).to(device).unsqueeze(0)
            bx = bx.to(device)
            bx = bx.data.requires_grad_(True)
            adam_opt = Adam([bx], lr=learning_rate, amsgrad=True)
            imgs = imgs.type(tensor_type)
            imgs = imgs.to(device)
            imgs_2 = imgs_2.type(tensor_type)
            imgs_2 = imgs_2.to(device)
            #(1,23625,6)
            
            for iter in range(epochs):
                added_imgs = imgs+bx
                added_imgs_2 = imgs_2+bx
                
                l2_norm = torch.sqrt(torch.mean(bx ** 2))
                l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])

                added_imgs.clamp_(min=0, max=1)
                added_imgs_2.clamp_(min=0, max=1)
                input_imgs = (added_imgs - rgb_means)/std
                input_imgs_2 = (added_imgs_2 - rgb_means)/std
                if half:
                    input_imgs = input_imgs.half()
                    input_imgs_2 = input_imgs_2.half()
                outputs = model(input_imgs)[0]
                outputs_2 = model(input_imgs_2)[0]
                bx = run_attack(outputs, outputs_2, bx, strategy, max_tracker_num, adam_opt, self.num_classes, self.confthre, self.nmsthre, iter)

            if strategy == max_tracker_num-1:
                strategy = 0
            else:
                strategy += 1
            

            added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            added_blob = added_blob[..., ::-1]
            added_blob_2 = torch.clamp(added_imgs_2*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            added_blob_2 = added_blob_2[..., ::-1]
            
            save_dir = path[0].replace("dataset", "bytetrack_uni+match_3")
            save_dir_2 = path_2[0].replace("dataset", "bytetrack_uni+match_3")
            result_dir = os.path.dirname(save_dir)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                print(save_dir)
            cv2.imwrite(save_dir, added_blob)
            cv2.imwrite(save_dir_2, added_blob_2)
            print(l1_norm,l2_norm)
            total_l1 += l1_norm.item()
            total_l2 += l2_norm.item()
            mean_l1 = total_l1/frame_id
            mean_l2 = total_l2/frame_id
            print(mean_l1,mean_l2)
            del bx
            del adam_opt
            del outputs
            del outputs_2
            del imgs
            del imgs_2

        return mean_l1,mean_l2

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xywh2xyxy(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

