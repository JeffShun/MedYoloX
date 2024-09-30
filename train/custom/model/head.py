import torch
import torch.nn as nn
from typing import List

class Detection_Head(nn.Module):
    def __init__(self, feature_size: int):
        super(Detection_Head, self).__init__()
        self.regression_head = RegressionModel(feature_size)
        self.classification_head = ClassificationModel(feature_size)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        regressions = torch.cat([self.regression_head(feature) for feature in features], dim=1)
        classifications = torch.cat([self.classification_head(feature) for feature in features], dim=1)
        return [regressions, classifications]


class RegressionModel(nn.Module):
    def __init__(self, feature_size=64):
        super(RegressionModel, self).__init__()

        self.reg_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, 6, kernel_size=3, padding=1)           
        )

    def forward(self, input):
        out = self.reg_head(input)
        out = out.permute(0, 2, 3, 4, 1)
        return out.contiguous().view(out.shape[0], -1, 6)


class ClassificationModel(nn.Module):
    def __init__(self, feature_size=64):
        super(ClassificationModel, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, 1, kernel_size=3, padding=1)     
        )

    def forward(self, input):
        out = self.cls_head(input)
        return out.contiguous().view(out.shape[0], -1)


if __name__ == "__main__":
    features = [torch.rand(1,256,8,32,32).cuda(), torch.rand(1,256,4,16,16).cuda(), torch.rand(1,256,2,8,8).cuda()]
    head=Detection_Head(
        num_features_in=256,
        num_anchors=3,
        feature_size=256
        ).cuda()
    regressions, classifications = head(features)
    print(regressions.shape, classifications.shape)


