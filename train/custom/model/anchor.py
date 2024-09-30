import torch
import torch.nn as nn

class Anchors(nn.Module):
    def __init__(self, img_size, pyramid_strides):
        super(Anchors, self).__init__()
        self.img_size = img_size
        self.pyramid_strides = pyramid_strides

    
    def forward(self):
        with torch.no_grad():
            all_anchors = []
            for stride in self.pyramid_strides:   
                img_size_p = [x // stride[j] for j, x in enumerate(self.img_size)]
                anchor_cx = (torch.arange(0, img_size_p[0], dtype=torch.float32) + 0.5) * stride[0]
                anchor_cy = (torch.arange(0, img_size_p[1], dtype=torch.float32) + 0.5) * stride[1]
                anchor_cz = (torch.arange(0, img_size_p[2], dtype=torch.float32) + 0.5) * stride[2]
                anchor_cx, anchor_cy, anchor_cz = torch.meshgrid(anchor_cx, anchor_cy, anchor_cz)

                anchor_cx = anchor_cx.flatten()
                anchor_cy = anchor_cy.flatten()
                anchor_cz = anchor_cz.flatten()
                stride_x = torch.ones_like(anchor_cx)*stride[0]
                stride_y = torch.ones_like(anchor_cx)*stride[1]
                stride_z = torch.ones_like(anchor_cx)*stride[2]
                anchors_cxcycz = torch.stack((anchor_cx, anchor_cy, anchor_cz, stride_x, stride_y, stride_z), dim=-1)
                all_anchors.append(anchors_cxcycz) 
        return torch.cat(all_anchors, 0)



    
    