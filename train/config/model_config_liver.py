import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
import torch
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones import ResNet3D
from custom.model.neck import FPN
from custom.model.anchor import Anchors
from custom.model.loss import Detection_Loss
from custom.model.head import Detection_Head
from custom.model.decoder import SDecoder
from custom.model.network import Detection_Network

class liver_detection_cfg:

    device = torch.device('cuda')
    dist_backend = 'nccl'
    dist_url = 'env://'

    # img
    img_size = [64, 224, 224]
    
    # network
    in_channel = 1
    base_channel = 32
    pyramid_strides = [[2**3]*3, [2**4]*3, [2**5]*3]
    fpn_sizes = [base_channel*2, base_channel*4, base_channel*8]
    feature_size = 64

    network = Detection_Network(
        backbone = ResNet3D(
            in_channel=in_channel, 
            base_channel=base_channel,
            layers=[3, 4, 6, 3]
        ),       
        neck = FPN(
            fpn_sizes = fpn_sizes, 
            feature_size=feature_size
        ),
        anchor_generator = Anchors(
            img_size=img_size,
            pyramid_strides=pyramid_strides 
        ),
        head = Detection_Head(
            feature_size=feature_size
        ),
        decoder = SDecoder(
            img_size=img_size
        )
    )


    # loss function
    loss_f = Detection_Loss(gamma=2, ota_top_k=10, ota_radius=5, ota_iou_weight=3, img_size=img_size)

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            resize(img_size),
            random_gamma_transform(gamma_range=[0.8, 1.2], prob=0.5),
            random_cutout(max_cutsize=30, cutnum=5, overlap_trhesh=0.2, prob=0.5),
            # random_flip(axis=1, prob=0.5),
            # random_flip(axis=2, prob=0.5),
            # random_flip(axis=3, prob=0.5),
            random_add_noise(sigma_range=[0.01, 0.03], prob=0.5),
            label_alignment(max_box_num=1, pad_val=-1)
            ])
        )
    
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            resize(img_size),
            label_alignment(max_box_num=1, pad_val=-1)
            ])
        )
    
    # train dataloader
    batchsize = 4
    shuffle = True
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 5e-4

    # scheduler
    milestones = [50,100,150]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    total_epochs = 200
    valid_interval = 1
    checkpoint_save_interval = 1
    log_dir = work_dir + "/Logs/liver/ResNet3D"
    checkpoints_dir = work_dir + '/checkpoints/liver/ResNet3D'
    load_from = work_dir + '/checkpoints/liver/ResNet3D/none.pth'
