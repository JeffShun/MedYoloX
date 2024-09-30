import torch
import torch.nn as nn

class SDecoder(nn.Module):
    def __init__(self, img_size):
        super(SDecoder, self).__init__()
        self.img_size = img_size 

    def forward(self, cls_heads, reg_heads, batch_anchors):
        device = cls_heads.device 
        batch_anchors = batch_anchors[None].repeat(cls_heads.shape[0], 1, 1).to(device)
        # cls_heads: [batch, nbox]
        # reg_heads: [batch, nbox, 6]
        # batch_anchors: [batch, nbox, 6]
        with torch.no_grad():
            # 选择分数最高的框
            batchsize = cls_heads.shape[0]
            max_scores, max_indices = torch.max(cls_heads, dim=1)  
            max_indices = max_indices.long()
            max_reg_heads = reg_heads[torch.arange(batchsize), max_indices]           # shape: [batch, 6]
            max_batch_anchors = batch_anchors[torch.arange(batchsize), max_indices]   # shape: [batch, 6]
            # 将回归头的预测值转换为边界框坐标
            image_pred_bboxes = self.decode(max_reg_heads, max_batch_anchors)
            return max_scores, image_pred_bboxes

    def decode(self, regression, anchor):
        """
        Decode regression outputs to bounding box coordinates.
        """
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

        # limitied the bbox in volumes boundarys
        x1 = torch.clamp(x1, min=0)
        y1 = torch.clamp(y1, min=0)
        z1 = torch.clamp(z1, min=0)
        x2 = torch.clamp(x2, max=self.img_size[0] - 1)
        y2 = torch.clamp(y2, max=self.img_size[1] - 1)
        z2 = torch.clamp(z2, max=self.img_size[2] - 1)

        # Stack the coordinates to get the final bounding boxes
        decoded_boxes = torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)
        return decoded_boxes

class Decoder(nn.Module):
    def __init__(self, img_size, top_k, min_score_threshold, min_volume, nms_threshold, max_detection_num):
        super(Decoder, self).__init__()
        self.img_size = img_size 
        self.top_k = top_k
        self.min_score_threshold = min_score_threshold
        self.min_volume = min_volume
        self.nms_threshold = nms_threshold
        self.max_detection_num = max_detection_num


    def forward(self, cls_heads, reg_heads, batch_anchors):
        # cls_heads: [batch, nbox]
        # reg_heads: [batch, nbox, 6]
        # batch_anchors: [batch, nbox, 6]
        device = cls_heads[0].device 
        batch_anchors = batch_anchors[None].repeat(cls_heads.shape[0], 1, 1).to(device)
        with torch.no_grad():       
            # select the top_k bbox 
            filter_scores, indexes = torch.topk(cls_heads, self.top_k, dim=1, largest=True, sorted=True)  
            filter_reg_heads = torch.gather(reg_heads, 1, indexes.unsqueeze(-1).repeat(1, 1, 6))              # shape: [batch, top_n, 6]
            filter_batch_anchors = torch.gather(batch_anchors, 1, indexes.unsqueeze(-1).repeat(1, 1, 6))      # shape: [batch, top_n, 6]     
            batch_scores, batch_pred_bboxes = [], []
            for per_image_scores, per_image_reg_heads, per_image_anchors in zip(filter_scores, filter_reg_heads, filter_batch_anchors):
                # convert the regression value to bbox coordinates
                pred_bboxes = self.decode(per_image_reg_heads, per_image_anchors)

                # filter the predicted bbox by score threshold
                pred_bboxes = pred_bboxes[per_image_scores > self.min_score_threshold].float()
                scores = per_image_scores[per_image_scores > self.min_score_threshold].float()

                # filter the predicted bbox by min_area
                volumes = (pred_bboxes[:, 3] - pred_bboxes[:, 0]) * (pred_bboxes[:, 4] - pred_bboxes[:, 1]) * (pred_bboxes[:, 5] - pred_bboxes[:, 2])
                valid_indices = volumes > self.min_volume
                pred_bboxes = pred_bboxes[valid_indices]
                scores = scores[valid_indices]                

                one_image_scores = (-1) * torch.ones((self.max_detection_num,), device=device)
                one_image_pred_bboxes = (-1) * torch.ones((self.max_detection_num, 6), device=device)         

                if scores.size(0) != 0:
                    # nms process
                    sorted_scores, sorted_indexes = torch.sort(scores, descending=True)
                    sorted_pred_bboxes = pred_bboxes[sorted_indexes]
                    keep_pred_bboxes, keep_scores = self.nms3d(sorted_pred_bboxes, sorted_scores, torch.tensor(self.nms_threshold, device=device))   

                    final_detection_num = min(self.max_detection_num, keep_scores.shape[0]) 
                    one_image_scores[:final_detection_num] = keep_scores[:final_detection_num]
                    one_image_pred_bboxes[:final_detection_num, :] = keep_pred_bboxes[:final_detection_num, :]

                one_image_scores = one_image_scores.unsqueeze(0)
                one_image_pred_bboxes = one_image_pred_bboxes.unsqueeze(0)

                batch_scores.append(one_image_scores)
                batch_pred_bboxes.append(one_image_pred_bboxes)

            batch_scores = torch.cat(batch_scores, dim=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, dim=0)

            return batch_scores, batch_pred_bboxes
        

    def decode(self, regression, anchor):
        """
        Decode regression outputs to bounding box coordinates.
        """
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

        # limitied the bbox in volumes boundarys
        x1 = torch.clamp(x1, min=0)
        y1 = torch.clamp(y1, min=0)
        z1 = torch.clamp(z1, min=0)
        x2 = torch.clamp(x2, max=self.img_size[0] - 1)
        y2 = torch.clamp(y2, max=self.img_size[1] - 1)
        z2 = torch.clamp(z2, max=self.img_size[2] - 1)

        # Stack the coordinates to get the final bounding boxes
        decoded_boxes = torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)
        return decoded_boxes


    def nms3d(self, boxes, scores, iou_threshold):
        if boxes.shape[0] == 0:
            return boxes, scores

        # coordinates of bounding boxes
        front_bottom_left_z = boxes[:, 0]
        front_bottom_left_y = boxes[:, 1]
        front_bottom_left_x = boxes[:, 2]
        back_top_right_z = boxes[:, 3]
        back_top_right_y = boxes[:, 4]
        back_top_right_x = boxes[:, 5]

        # compute volume of bounding boxes
        volumes = (back_top_right_x - front_bottom_left_x) * (back_top_right_y - front_bottom_left_y) * (back_top_right_z - front_bottom_left_z)

        # sort score in order to extract the most potential bounding box
        order = torch.argsort(scores, descending=True)

        # initialize result bounding box & its score
        res_boxes = []
        res_score = []

        while order.numel() > 0:
            # extract the index of the bounding box with the highest score
            max_index = order[0]  

            # extract result bounding box
            res_boxes.append(boxes[max_index].unsqueeze(0))
            res_score.append(scores[max_index].unsqueeze(0))

            # compute the coordinates of the intersection regions (of the res_box and all other boxes)
            x1 = torch.max(front_bottom_left_x[max_index], front_bottom_left_x[order[1:]])
            x2 = torch.min(back_top_right_x[max_index], back_top_right_x[order[1:]])

            y1 = torch.max(front_bottom_left_y[max_index], front_bottom_left_y[order[1:]])
            y2 = torch.min(back_top_right_y[max_index], back_top_right_y[order[1:]])

            z1 = torch.max(front_bottom_left_z[max_index], front_bottom_left_z[order[1:]])
            z2 = torch.min(back_top_right_z[max_index], back_top_right_z[order[1:]])

            # compute the volume of intersection region
            w = torch.clamp(x2 - x1, min=0)
            h = torch.clamp(y2 - y1, min=0)
            d = torch.clamp(z2 - z1, min=0)

            intersection_volume = w * h * d

            # compute the volume ratio between intersection region and the union of the two bounding boxes
            ratio = intersection_volume / (volumes[max_index] + volumes[order[1:]] - intersection_volume)

            # delete the bounding boxes with a higher intersection ratio than iou_threshold
            order = order[1:][ratio < iou_threshold]

        return torch.cat(res_boxes, dim=0), torch.cat(res_score, dim=0)