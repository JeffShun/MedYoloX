import torch
import torch.nn as nn

class Detection_Loss(nn.Module):
    def __init__(self, gamma=2, ota_top_k=10, ota_radius=5, ota_iou_weight=3, img_size=[64, 224, 224]):
        super(Detection_Loss, self).__init__()
        self.gamma = gamma
        self.ota_top_k = ota_top_k
        self.ota_radius = ota_radius
        self.ota_iou_weight = ota_iou_weight
        self.img_size = img_size

    def forward(self, classifications, regressions, anchors, labels):
        device = regressions.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        for j in range(batch_size):

            classification = classifications[j, :]
            regression = regressions[j, :, :]
            anchors = anchors.to(device)

            bbox_annotation = labels[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 0] != -1]
            classification = torch.clamp(classification, 1e-7, 1.0 - 1e-7)

            # compute the loss for classification
            targets = torch.zeros_like(classification, device=device)
            matcher = SimOTA(topk=self.ota_top_k, radius=self.ota_radius, iou_weight=self.ota_iou_weight)
            pred_boxes = self.decode(regression, anchors)
            pos_index = matcher(anchors, pred_boxes, classification, bbox_annotation)
            assigned_box = bbox_annotation[pos_index[:,0]]
            pos_box = pred_boxes[pos_index[:,1], :]
            targets[pos_index[:,1]] = 1

            classification_softmax = nn.functional.softmax(classification)
            p = (classification_softmax * targets).sum()
            classification_loss = -((1-p)**self.gamma)*torch.log(p+1e-24)
            classification_losses.append(classification_loss)

            if pos_box.shape[0] > 0:
                regression_loss = self.ciou_loss(pos_box, assigned_box)
                regression_losses.append(regression_loss)
            else:
                regression_losses.append(torch.tensor(0, device=device).float())

        return {"c_loss":torch.stack(classification_losses).mean(), "r_loss":torch.stack(regression_losses).mean()} 
    
    def ciou_loss(self, preds, bbox, eps=1e-7, dice_for_iou=False, reduction='mean'):
        '''
        :param preds: [[x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2],,,]
        :param bbox: [[x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2],,,]
        :param eps: eps to avoid divide 0
        :param reduction: mean or sum
        :return: ciou-loss
        '''
        ix1 = torch.max(preds[:, 0], bbox[:, 0])
        iy1 = torch.max(preds[:, 1], bbox[:, 1])
        iz1 = torch.max(preds[:, 2], bbox[:, 2])
        ix2 = torch.min(preds[:, 3], bbox[:, 3])
        iy2 = torch.min(preds[:, 4], bbox[:, 4])
        iz2 = torch.min(preds[:, 5], bbox[:, 5])

        iw = (ix2 - ix1).clamp(min=0.)
        ih = (iy2 - iy1).clamp(min=0.)
        id = (iz2 - iz1).clamp(min=0.)

        inters = iw * ih * id
        volume_preds = (preds[:, 3] - preds[:, 0]) * (preds[:, 4] - preds[:, 1]) * (preds[:, 5] - preds[:, 2])
        volume_bbox = (bbox[:, 3] - bbox[:, 0]) * (bbox[:, 4] - bbox[:, 1]) * (bbox[:, 5] - bbox[:, 2])

        if dice_for_iou:
            union = volume_preds + volume_bbox
            iou = 2 * inters / (union + eps)
        else:
            union = volume_preds + volume_bbox - inters
            iou = inters / (union + eps)            

        # inter_diag
        cxpreds = (preds[:, 3] + preds[:, 0]) / 2
        cypreds = (preds[:, 4] + preds[:, 1]) / 2
        czpreds = (preds[:, 5] + preds[:, 2]) / 2

        cxbbox = (bbox[:, 3] + bbox[:, 0]) / 2
        cybbox = (bbox[:, 4] + bbox[:, 1]) / 2
        czbbox = (bbox[:, 5] + bbox[:, 2]) / 2

        inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2 + (czbbox - czpreds) ** 2

        # outer_diag
        ox1 = torch.min(preds[:, 0], bbox[:, 0])
        oy1 = torch.min(preds[:, 1], bbox[:, 1])
        oz1 = torch.min(preds[:, 2], bbox[:, 2])
        ox2 = torch.max(preds[:, 3], bbox[:, 3])
        oy2 = torch.max(preds[:, 4], bbox[:, 4])
        oz2 = torch.max(preds[:, 5], bbox[:, 5])

        outer_diag = (ox2 - ox1) ** 2 + (oy2 - oy1) ** 2 + (oz2 - oz1) ** 2

        diou = iou - inter_diag / (outer_diag + eps)

        # calculate v, alpha
        wpreds = preds[:, 3] - preds[:, 0]
        hpreds = preds[:, 4] - preds[:, 1]
        dpreds = preds[:, 5] - preds[:, 2]

        wbbox = bbox[:, 3] - bbox[:, 0]
        hbbox = bbox[:, 4] - bbox[:, 1]
        dbbox = bbox[:, 5] - bbox[:, 2]

        v = (4 / (torch.pi ** 2)) * (torch.pow(torch.atan(wbbox / (hbbox + eps)) - torch.atan(wpreds / (hpreds + eps)), 2) +
                                     torch.pow(torch.atan(hbbox / (dbbox + eps)) - torch.atan(hpreds / (dpreds + eps)), 2) +
                                     torch.pow(torch.atan(dbbox / (wbbox + eps)) - torch.atan(dpreds / (wpreds + eps)), 2))
        alpha = v / (1 - iou + v + eps)
        ciou = diou - alpha * v

        ciou_loss = 1 - torch.clamp(ciou, min=-1.0, max=1.0)
        if reduction == 'mean':
            loss = torch.mean(ciou_loss)
        elif reduction == 'sum':
            loss = torch.sum(ciou_loss)
        else:
            raise NotImplementedError
        return loss

    def decode(self, regression, anchor):
        """
        Decode regression outputs to bounding box coordinates.
        """
        # Decode the regression predictions
        pred_x_center = regression[:, 0] * anchor[:, 3] + anchor[:, 0]
        pred_y_center = regression[:, 1] * anchor[:, 4] + anchor[:, 1]
        pred_z_center = regression[:, 2] * anchor[:, 5] + anchor[:, 2]
        pred_w = torch.exp(regression[:, 3]) * anchor[:, 3]
        pred_h = torch.exp(regression[:, 4]) * anchor[:, 4]
        pred_d = torch.exp(regression[:, 5]) * anchor[:, 5] 

        # Convert center coordinates to corner coordinates
        x1 = pred_x_center - pred_w / 2
        y1 = pred_y_center - pred_h / 2
        z1 = pred_z_center - pred_d / 2
        x2 = pred_x_center + pred_w / 2
        y2 = pred_y_center + pred_h / 2
        z2 = pred_z_center + pred_d / 2

        decoded_boxes = torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)

        return decoded_boxes
    
class SimOTA:
    def __init__(self, topk: int = 10, radius: float = 2.5, iou_weight: float = 3.0):
        self.topk = topk
        self.radius = radius
        self.iou_weight = iou_weight
        self.bce_loss = nn.BCELoss(reduce=False)

    def candidates(self, anchor_points, gt_boxes):
        gt_boxes = gt_boxes.unsqueeze(1)
        gt_centers = (gt_boxes[:, :, 0:3] + gt_boxes[:, :, 3:6]) / 2.0

        is_in_box = (
            (gt_boxes[:, :, 0] <= anchor_points[:, 0])
            & (anchor_points[:, 0] <= gt_boxes[:, :, 3])
            & (gt_boxes[:, :, 1] <= anchor_points[:, 1])
            & (anchor_points[:, 1] <= gt_boxes[:, :, 4])
            & (gt_boxes[:, :, 2] <= anchor_points[:, 2])
            & (anchor_points[:, 2] <= gt_boxes[:, :, 5])
            ) 
         
        gt_center_lbound = gt_centers - self.radius * anchor_points[:,3:]
        gt_center_ubound = gt_centers + self.radius * anchor_points[:,3:]

        is_in_center = (
            (gt_center_lbound[:, :, 0] <= anchor_points[:, 0])
            & (anchor_points[:, 0] <= gt_center_ubound[:, :, 0])
            & (gt_center_lbound[:, :, 1] <= anchor_points[:, 1])
            & (anchor_points[:, 1] <= gt_center_ubound[:, :, 1])
            & (gt_center_lbound[:, :, 2] <= anchor_points[:, 2])
            & (anchor_points[:, 2] <= gt_center_ubound[:, :, 2])
            )  
        fg_mask = is_in_box.any(dim=0) | is_in_center.any(dim=0)
        center_mask = is_in_box[:, fg_mask] & is_in_center[:, fg_mask]
        return fg_mask, center_mask

    @torch.no_grad()
    def __call__(self, anchor_points, pred_boxes, pred_objs, gt_boxes):
        device = pred_boxes.device
        gt_count = gt_boxes.shape[0]
        pred_count = pred_boxes.shape[0]
        fg_mask, center_mask = self.candidates(anchor_points=anchor_points, gt_boxes=gt_boxes)
        pred_objs = pred_objs[fg_mask]
        pred_boxes = pred_boxes[fg_mask]

        num_fg = pred_objs.size(0)
        obj_matrix = self.bce_loss(pred_objs, torch.ones(num_fg).to(device))
        iou_matrix = self.box_iou(gt_boxes, pred_boxes)
        matrix = (
            -obj_matrix
            + self.iou_weight * torch.log(iou_matrix + 1e-8)
            + center_mask * 10000
            )
        topk = min(self.topk, iou_matrix.size(1))
        topk_ious, _ = torch.topk(iou_matrix, topk, dim=1)
        dynamic_ks = (topk_ious.sum(1)).int().clamp(min=1, max=iou_matrix.size(1))
        matching_matrix = torch.zeros((gt_count, pred_count), dtype=torch.long)
        fg_mask_idx = fg_mask.nonzero().view(-1)
        for (row, dynamic_topk, matching_row) in zip(matrix, dynamic_ks, matching_matrix):
            _, pos_idx = torch.topk(row, k=dynamic_topk)
            matching_row[fg_mask_idx[pos_idx]] = 1

        # Fix columns where multiple matches are present
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            multiple_match_indices = torch.nonzero(anchor_matching_gt > 1).view(-1)
            for idx in multiple_match_indices:
                overlapping_matches = matrix[:, fg_mask_idx == idx]
                max_val, matrix_argmax = torch.max(overlapping_matches, dim=0)
                matching_matrix[:, idx] = 0
                matching_matrix[matrix_argmax, idx] = 1

        return matching_matrix.nonzero()


    def box_iou(self, a, b):
        # unsqueeze to enable broadcasting
        a = a.unsqueeze(-2)
        inter_d = torch.min(a[..., 3], b[:, 3]) - torch.max(a[..., 0], b[:, 0])
        inter_w = torch.min(a[..., 4], b[:, 4]) - torch.max(a[..., 1], b[:, 1])
        inter_h = torch.min(a[..., 5], b[:, 5]) - torch.max(a[..., 2], b[:, 2])

        inter_d = torch.clamp(inter_d, min=0)
        inter_w = torch.clamp(inter_w, min=0)
        inter_h = torch.clamp(inter_h, min=0)

        area_a = (a[..., 3] - a[..., 0]) * (a[..., 4] - a[..., 1]) * (a[..., 5] - a[..., 2])
        area_b = (b[:, 3] - b[:, 0]) * (b[:, 4] - b[:, 1]) * (b[:, 5] - b[:, 2])

        intersection = inter_d * inter_w * inter_h
        union = area_a + area_b - intersection
        union = torch.clamp(union, min=1e-8)
        
        IoU = intersection / union
        return IoU