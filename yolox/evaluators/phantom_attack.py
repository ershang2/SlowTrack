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
import torch.nn as nn
import torchvision
learning_rate = 0.02 #0.07
epochs = 20
iter_eps = 0.0002
epsilon = 60
lambda_1 = 1
lambda_2 = 10

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x_out = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x_out.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x_out = torch.cat((x_out, v), 0)

        # If none remain process next image
        if not x_out.shape[0]:
            continue

        # Compute conf
        x_out[:, 5:] = x[xc[xi]][:, 5:] * x[xc[xi]][:, 4:5]#x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x_out[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x_out[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x_out = torch.cat((box[i], x_out[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x_out[:, 5:].max(1, keepdim=True)
            x_out = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x_out = x_out[(x_out[:, 5:6] == torch.tensor(classes, device=x_out.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x_out.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x_out = x_out[x_out[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x_out[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x_out[:, :4] + c, x_out[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x_out[i, :4] = torch.mm(weights, x_out[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x_out[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

class IoU(nn.Module):
    def __init__(self, conf_threshold, iou_threshold, img_size, device) -> None:
        super(IoU, self).__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device

    def forward(self, output_clean, output_patch):
        batch_loss = []

        gn = torch.tensor(self.img_size)[[1, 0, 1, 0]]
        gn = gn.to(self.device)
        pred_clean_bboxes = non_max_suppression(output_clean, self.conf_threshold, self.iou_threshold, classes=None,
                                                max_det=1000)
        patch_conf = 0.001
        pred_patch_bboxes = non_max_suppression(output_patch, patch_conf, self.iou_threshold, classes=None,
                                                max_det=30000)

        # print final amount of predictions
        final_preds_batch = 0
        for img_preds in non_max_suppression(output_patch, self.conf_threshold, self.iou_threshold, classes=None,
                                             max_det=30000):
            final_preds_batch += len(img_preds)

        for (img_clean_preds, img_patch_preds) in zip(pred_clean_bboxes, pred_patch_bboxes):  # per image

            for clean_det in img_clean_preds:

                clean_clss = clean_det[5]

                clean_xyxy = torch.stack([clean_det])  # .clone()
                clean_xyxy_out = (clean_xyxy[..., :4] / gn).to(
                    self.device)

                img_patch_preds_out = img_patch_preds[img_patch_preds[:, 5].view(-1) == clean_clss]

                patch_xyxy_out = (img_patch_preds_out[..., :4] / gn).to(self.device)

                if len(clean_xyxy_out) != 0:
                    target = self.get_iou(patch_xyxy_out, clean_xyxy_out)
                    if len(target) != 0:
                        target_m, _ = target.max(dim=0)
                    else:
                        target_m = torch.zeros(1).to(self.device)

                    batch_loss.append(target_m)

        one = torch.tensor(1.0).to(self.device)
        if len(batch_loss) == 0:
            return one

        return (one - torch.stack(batch_loss).mean())

    def get_iou(self, bbox1, bbox2):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            bbox1: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
            bbox2: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """

        inter = self.intersect(bbox1, bbox2)
        area_a = ((bbox1[:, 2] - bbox1[:, 0]) *
                  (bbox1[:, 3] - bbox1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((bbox2[:, 2] - bbox2[:, 0]) *
                  (bbox2[:, 3] - bbox2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

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

def max_objects(output_patch, conf_thres=0.25, target_class=0):

    x2 = output_patch[:, :, 5:] * output_patch[:, :, 4:5]

    conf, j = x2.max(2, keepdim=False)

    all_target_conf = x2[:, :, target_class]
    under_thr_target_conf = all_target_conf[conf < conf_thres]

    conf_avg = len(conf.view(-1)[conf.view(-1) > conf_thres]) / len(output_patch)
    print(f"pass to NMS: {conf_avg}")

    zeros = torch.zeros(under_thr_target_conf.size()).to(output_patch.device)
    zeros.requires_grad = True
    x3 = torch.maximum(-under_thr_target_conf + conf_thres, zeros)
    mean_conf = torch.sum(x3, dim=0) #/ (output_patch.size()[0] * output_patch.size()[1])

    return mean_conf

def bboxes_area(output_clean, output_patch, patch_size, conf_thres=0.25):

    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    t_loss = 0.0
    preds_num = 0

    xc_patch = output_patch[..., 4] > conf_thres
    not_nan_count = 0

    # For each img in the batch
    for (xi, x), (li, l) in (zip(enumerate(output_patch), enumerate(output_clean))):  # image index, image inference

        x1 = x[xc_patch[xi]]  # .clone()
        x2 = x1[:, 5:] * x1[:, 4:5]  # x1[:, 5:] *= x1[:, 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box_x1 = xywh2xyxy(x1[:, :4])

        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        agnostic = True

        conf_x1, j_x1 = x2.max(1, keepdim=True)
        x1_full = torch.cat((box_x1, conf_x1, j_x1.float()), 1)[conf_x1.view(-1) > conf_thres]
        c_x1 = x1_full[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes_x1, scores_x1 = x1_full[:, :4] + c_x1, x1_full[:, 4]  # boxes (offset by class), scores
        final_preds_num = len(torchvision.ops.nms(boxes_x1, scores_x1, conf_thres))
        preds_num += final_preds_num

        # calculate bboxes' area avg
        bboxes_x1_wh = xyxy2xywh(boxes_x1)[:, 2:]
        bboxes_x1_area = bboxes_x1_wh[:, 0] * bboxes_x1_wh[:, 1]
        img_loss = bboxes_x1_area.mean() / (patch_size[1] * patch_size[2])
        if not torch.isnan(img_loss):
            t_loss += img_loss
            not_nan_count += 1

    if not_nan_count == 0:
        t_loss_f = torch.tensor(torch.nan)
    else:
        t_loss_f = t_loss / not_nan_count

    return t_loss_f
def bboxes_area(output_clean, output_patch, patch_size, conf_thres=0.25):

    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    t_loss = 0.0
    preds_num = 0

    xc_patch = output_patch[..., 4] > conf_thres
    not_nan_count = 0

    # For each img in the batch
    for (xi, x), (li, l) in (zip(enumerate(output_patch), enumerate(output_clean))):  # image index, image inference

        x1 = x[xc_patch[xi]]  # .clone()
        x2 = x1[:, 5:] * x1[:, 4:5]  # x1[:, 5:] *= x1[:, 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box_x1 = xywh2xyxy(x1[:, :4])

        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        agnostic = True

        conf_x1, j_x1 = x2.max(1, keepdim=True)
        x1_full = torch.cat((box_x1, conf_x1, j_x1.float()), 1)[conf_x1.view(-1) > conf_thres]
        c_x1 = x1_full[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes_x1, scores_x1 = x1_full[:, :4] + c_x1, x1_full[:, 4]  # boxes (offset by class), scores
        final_preds_num = len(torchvision.ops.nms(boxes_x1, scores_x1, conf_thres))
        preds_num += final_preds_num

        # calculate bboxes' area avg
        bboxes_x1_wh = xyxy2xywh(boxes_x1)[:, 2:]
        bboxes_x1_area = bboxes_x1_wh[:, 0] * bboxes_x1_wh[:, 1]
        img_loss = bboxes_x1_area.mean() / (patch_size[1] * patch_size[2])
        if not torch.isnan(img_loss):
            t_loss += img_loss
            not_nan_count += 1

    if not_nan_count == 0:
        t_loss_f = torch.tensor(torch.nan)
    else:
        t_loss_f = t_loss / not_nan_count

    return t_loss_f
    
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

def run_attack(outputs,bx, strategy, max_tracker_num, mask):

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    scores = outputs[:,5] * outputs[:,4]

    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores)
    loss3 = F.mse_loss(scores, targets, reduction='sum')
    loss = loss3#+loss2
    
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -5.5 * mask * bx.grad+ bx.data
    count = (scores > 0.3).sum()
    print('loss',loss.item(),'loss_2',loss2.item(),'loss_3',loss3.item(),'count:',count.item())
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
        self.confthre = 0.6
        self.nmsthre = 0.6
        self.num_classes = num_classes
        self.args = args
        self.current_train_loss = 0.0
        self.current_max_objects_loss = 0.0
        self.current_orig_classification_loss = 0.0
        self.min_bboxes_added_preds_loss = 0.0

    def loss_function_gradient(self, applied_patch, init_images, adv_patch, device, half):
        
        iou = IoU(conf_threshold=self.confthre, iou_threshold=self.nmsthre, img_size=init_images.shape[1:], device=device)
        init_images = init_images.to(device)
        applied_patch = applied_patch.to(device)
        # r = random.randint(0, len(self.models)-1) # choose a random model
        if init_images.ndimension()==3:
            init_images = init_images.unsqueeze(0)
        if applied_patch.ndimension()==3:
            applied_patch = applied_patch.unsqueeze(0)
        init_images.clamp_(min=0, max=1)
        init_images = (init_images - self.rgb_means)/self.std
        if half:
            init_images = init_images.half()
            applied_patch = applied_patch.half()
        with torch.no_grad():
            # output_clean = model(init_images)[0].detach()
            output_clean = self.model(init_images).detach()
        output_patch = self.model(applied_patch)

        max_objects_loss = max_objects(output_patch,conf_thres=self.confthre)
        
        bboxes_area_loss = bboxes_area(output_clean, output_patch, init_images.shape[1:])

        iou_loss = iou(output_clean, output_patch)
        loss = max_objects_loss * lambda_1

        if not torch.isnan(iou_loss):
            loss += (iou_loss * (1 - lambda_1))
            self.current_orig_classification_loss += ((1 - lambda_1) * iou_loss.item())

        if not torch.isnan(bboxes_area_loss):
            loss += (bboxes_area_loss * lambda_2)

        self.current_train_loss += loss.item()
        self.current_max_objects_loss += (lambda_1 * max_objects_loss.item())

        loss = loss.to(device)

        self.model.zero_grad()
        data_grad = torch.autograd.grad(loss, adv_patch)[0]
        return data_grad

    def fastGradientSignMethod(self, adv_patch, images, device, half, epsilon=0.3):

        # image_attack = image
        applied_patch = torch.clamp(images[:] + adv_patch, 0, 1)
        # image_attack = image+ (torch.rand(image.size())/255)

        # torch.autograd.set_detect_anomaly(True)
        data_grad = self.loss_function_gradient(applied_patch, images, adv_patch, device, half)  # init_image, penalty_term, adv_patch)

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_patch = adv_patch - epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_patch_c = torch.clamp(perturbed_patch, 0, 1).detach()
        # Return the perturbed image
        return perturbed_patch_c, applied_patch

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
        self.model = model.eval()
        if half:
            self.model = self.model.half()
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
        max_tracker_num = int(15)
        self.rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
        self.std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
        bx = np.zeros((3, self.img_size[0], self.img_size[1]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        adv_patch = bx
        for iter in tqdm(range(epochs)):

            frame_id = 0
            
            #(1,23625,6)
            for cur_iter, (imgs, path) in enumerate(
                progress_bar(self.dataloader)
                ):
                print(path)
                frame_id += 1
                imgs = imgs.type(tensor_type)
                imgs = imgs.to(device)
                
                adv_patch, applied_patch = self.fastGradientSignMethod(adv_patch, imgs, device, half, epsilon=iter_eps)
                perturbation = adv_patch - bx
                norm = torch.sum(torch.square(perturbation))
                norm = torch.sqrt(norm)
                factor = min(1, epsilon / norm.item())  # torch.divide(epsilon, norm.numpy()[0]))
                adv_patch = (torch.clip(bx + perturbation * factor, 0.0, 1.0))  # .detach()

                l2_norm = torch.sqrt(torch.mean(adv_patch ** 2))
                l1_norm = torch.norm(adv_patch, p=1)/(bx.shape[3]*bx.shape[2])
                if iter == epochs-1:
                    save_dir = path[0].replace("dataset", "bytetrack_phantom")
                    result_dir = os.path.dirname(save_dir)
                    added_blob = torch.clamp(applied_patch*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                        print(save_dir)
                    added_blob = added_blob[..., ::-1]
                    cv2.imwrite(save_dir, added_blob)
                print(l1_norm.item(),l2_norm.item())
                    

            
            # added_blob = added_blob[..., ::-1]
            # added_blob_2 = added_blob_2[..., ::-1]
            
            
            
            total_l1 += l1_norm
            total_l2 += l2_norm
            mean_l1 = total_l1/frame_id
            mean_l2 = total_l2/frame_id
            print(mean_l1.item(),mean_l2.item())

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

