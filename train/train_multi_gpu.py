import warnings
warnings.filterwarnings('ignore')
import os
import argparse
from config.model_config_liver import liver_detection_cfg
from config.model_config_kidney import kidney_detection_cfg
import torch
from torch import optim
from torch.autograd import Variable as V
import time
from custom.utils.logger import Logger
from custom.utils.model_backup import model_backup
from custom.utils.lr_scheduler import WarmupMultiStepLR
from custom.utils.dataloaderX import DataLoaderX
from custom.utils.distributed_utils import *
import torch.distributed as dist
from custom.utils.tensorboad_utils import get_writer
from custom.utils.validation_show import save_validation_result
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="liver", type=str)
    args = parser.parse_args()
    return args


def train(task):  
    if task == "liver":
        network_cfg = liver_detection_cfg
    elif task == "kidney":
        network_cfg = kidney_detection_cfg
    else:
        raise ValueError("Value must be liver or kidney")
    
    # 分布式训练初始化
    init_distributed_mode(network_cfg)
    rank = network_cfg.rank
    device = network_cfg.device

    # 训练准备
    logger_dir = network_cfg.log_dir
    tensorboad_dir = logger_dir + "/tf_logs"
    os.makedirs(network_cfg.checkpoints_dir,exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    if rank == 0:
        model_backup(logger_dir+"/backup.tar")
        if os.path.exists(tensorboad_dir): 
            shutil.rmtree(tensorboad_dir)
    logger = Logger(logger_dir+"/trainlog.txt", level='debug').logger
    writer = get_writer(tensorboad_dir)

    # 网络定义
    DetectionNet = network_cfg.network.to(device)

    # 定义损失函数
    loss_f = network_cfg.loss_f

    # 学习率要根据并行GPU的数量进行倍增
    network_cfg.lr *= network_cfg.world_size  
    DetectionNet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(DetectionNet)
    init_weight = os.path.join(network_cfg.checkpoints_dir, "initial_weights.pth")
    # 如果存在预训练权重则载入
    if os.path.exists(network_cfg.load_from):
        if rank == 0:
            print("Load pretrain weight from: " + network_cfg.load_from)
        DetectionNet.load_state_dict(torch.load(network_cfg.load_from, map_location=network_cfg.device))
    else:
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(DetectionNet.state_dict(), init_weight)
        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        DetectionNet.load_state_dict(torch.load(init_weight, map_location=device))

    # 转为DDP模型
    DetectionNet = torch.nn.parallel.DistributedDataParallel(DetectionNet, device_ids=[network_cfg.gpu])

    train_dataset = network_cfg.train_dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=network_cfg.shuffle)
    train_dataloader = DataLoaderX(dataset = train_dataset, 
                                batch_size = network_cfg.batchsize,
                                num_workers=network_cfg.num_workers,
                                sampler = train_sampler,
                                drop_last=network_cfg.drop_last,
                                pin_memory = False
                                )               
                    
    valid_dataset = network_cfg.valid_dataset
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
    valid_dataloader = DataLoaderX(dataset = valid_dataset, 
                                batch_size = network_cfg.batchsize,                                
                                num_workers=network_cfg.num_workers, 
                                sampler = valid_sampler,
                                drop_last=False,
                                pin_memory = False
                                )

    optimizer = optim.AdamW(params=DetectionNet.parameters(), lr=network_cfg.lr, weight_decay=network_cfg.weight_decay)
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                milestones=network_cfg.milestones,
                                gamma=network_cfg.gamma,
                                warmup_factor=network_cfg.warmup_factor,
                                warmup_iters=network_cfg.warmup_iters,
                                warmup_method=network_cfg.warmup_method,
                                last_epoch=network_cfg.last_epoch)
    

    time_start=time.time()
    for epoch in range(network_cfg.total_epochs): 
        train_sampler.set_epoch(epoch)
        #Training Step!
        DetectionNet.train()
        for ii, (image, label) in enumerate(train_dataloader):
            image = V(image).to(device)
            label = V(label).to(device)
            regressions, classifications, anchors = DetectionNet.module.forward_train(image)
            t_loss = loss_f(classifications, regressions, anchors, label)
            t_losses = V(torch.zeros(1)).to(device)
            loss_info = ""
            for loss_item, loss_val in t_loss.items():
                t_losses += loss_val
                loss_info += "{}={:.4f}\t ".format(loss_item,loss_val.item())
                if rank == 0:
                    writer.add_scalar('TrainLoss/{}'.format(loss_item),loss_val.item(), epoch*len(train_dataloader)+ii+1)
            time_temp=time.time()
            eta = ((network_cfg.total_epochs-epoch)+(1-(ii+1)/len(train_dataloader)))/(epoch+(ii+1)/len(train_dataloader))*(time_temp-time_start)/60
            if eta < 60:
                eta = "{:.1f}min".format(eta)
            else:
                eta = "{:.1f}h".format(eta/60.0)
            if rank == 0:
                logger.info('Epoch:[{}/{}]\t Iter:[{}/{}]\t Eta:{}\t {}'.format(epoch+1 ,network_cfg.total_epochs, ii+1, len(train_dataloader), eta, loss_info))
            optimizer.zero_grad()
            t_losses.backward()
            optimizer.step()

        if rank == 0:
            writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], epoch)   
        scheduler.step()

        # Valid Step!
        if (epoch+1) % network_cfg.valid_interval == 0:
            valid_loss = dict()
            DetectionNet.eval()
            for ii, (image, label) in enumerate(valid_dataloader):
                image = V(image).to(device)
                label = V(label).to(device)
                with torch.no_grad():
                    regressions, classifications, anchors = DetectionNet.module.forward_train(image)
                    v_loss = loss_f(classifications, regressions, anchors, label)    
                    save_dir = network_cfg.log_dir+"/sample/{}_{}".format(epoch+1, ii+1)
                    os.makedirs(save_dir, exist_ok=True)
                    scores, pred_boxes = DetectionNet(image)
                    if rank == 0:
                        save_validation_result(image, label, pred_boxes, save_dir)

                for loss_item, loss_val in v_loss.items():
                    if loss_item not in valid_loss:
                        valid_loss[loss_item] = loss_val.item()
                    else:
                        valid_loss[loss_item] += loss_val.item()  
            loss_info = ""              
            for loss_item, loss_val in valid_loss.items():
                valid_loss[loss_item] /= (ii+1)
                loss_info += "{}={:.4f}\t ".format(loss_item, valid_loss[loss_item])
                if rank == 0:
                    writer.add_scalar('ValidLoss/{}'.format(loss_item),valid_loss[loss_item], (epoch+1)*len(train_dataloader))

            if rank == 0:
                logger.info('Validating Step:\t {}'.format(loss_info))
        
        if (epoch+1) % network_cfg.checkpoint_save_interval == 0:
            torch.save(DetectionNet.module.state_dict(), network_cfg.checkpoints_dir+"/{}.pth".format(epoch+1))
    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(init_weight):
            os.remove(init_weight)
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    train(task=args.task)
