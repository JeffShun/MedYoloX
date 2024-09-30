"""生成模型输入数据."""

import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import json
import SimpleITK as sitk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--save_path', type=str, default='./train_data/processed_data')
    parser.add_argument('--task', type=str, default='liver')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    return args


def gen_lst(save_path, task, all_pids):
    save_file = os.path.join(save_path, task+'.txt')
    data_list = glob.glob(os.path.join(save_path, '*.npz'))
    num = 0
    with open(save_file, 'w') as f:
        for pid in all_pids:
            data = os.path.join(save_path, pid+".npz")
            if data in data_list:
                num+=1
                f.writelines(data.replace("\\","/") + '\n')
    print('num of data: ', num)


def find_outer_bounding_box3D(boxes):
    if not boxes:
        return None
    
    x_min = boxes[0][0]
    y_min = boxes[0][1]
    x_max = boxes[0][2]
    y_max = boxes[0][3]
    z_min =  boxes[0][4]
    z_max =  boxes[0][4]
    
    for box in boxes:
        x1, y1, x2, y2, z = box
        if x1 < x_min:
            x_min = x1
        if y1 < y_min:
            y_min = y1
        if x2 > x_max:
            x_max = x2
        if y2 > y_max:
            y_max = y2
        if z < z_min:
            z_min = z    
        if z > z_max:
            z_max = z   
    
    return [[z_min, y_min, x_min, z_max, y_max, x_max]]


def process_single(input):
    volume_path, label_jsons, save_path, sample, task = input
    volume_itk = sitk.ReadImage(volume_path)
    volume = sitk.GetArrayFromImage(volume_itk)
    boxes2D = [] 
    for label_json in label_jsons:       
        with open(label_json) as f:
            json_data = json.load(f)          
        for shape in json_data["shapes"]: 
            if shape["label"] == task:
                points = shape["points"]
                # 提取矩形框的左上角和右下角坐标以及切片索引
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                z = int(os.path.basename(label_json).split(".")[0][-4:])
                boxes2D.append((x1, y1, x2, y2, z))            

    bounding_box3D = find_outer_bounding_box3D(boxes2D)
    if bounding_box3D:
        bounding_box3D = np.array(bounding_box3D)  
        np.savez_compressed(os.path.join(save_path, f'{sample}.npz'), volume=volume, box3D=bounding_box3D)


    # box_mask = np.zeros_like(volume)
    # z_min, y_min, x_min, z_max, y_max, x_max = bounding_box3D
    # # SimpleITK读取的array各轴顺序为[z,y,x]
    # box_mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1
    # box_itk = sitk.GetImageFromArray(box_mask)
    # box_itk.CopyInformation(volume_itk)
    # debug_dir = "./train_data/debug"
    # os.makedirs(debug_dir, exist_ok=True)
    # sitk.WriteImage(volume_itk, os.path.join(debug_dir, f'{sample}.dcm.nii.gz'))
    # sitk.WriteImage(box_itk, os.path.join(debug_dir, f'{sample}.box.nii.gz'))



if __name__ == '__main__':
    args = parse_args()
    threads = args.threads
    task = args.task
    data_path = args.data_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    for dataset in ["train","valid"]:
        print("\nBegin gen %s data!"%(dataset))
        data_dir = os.path.join(data_path, dataset)
        inputs = []
        all_samples = []
        for sample in tqdm(os.listdir(data_dir)):
            sample_dir = os.path.join(data_dir, sample)
            label_jsons = glob.glob(os.path.join(sample_dir, '*.json'))
            volume_path = os.path.join(sample_dir, f"{sample}.volume.nii.gz")  
            inputs.append([volume_path, label_jsons, save_path, sample, task])
            all_samples.append(sample)
        pool = Pool(threads)
        pool.map(process_single, inputs)
        pool.close()
        pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(save_path, dataset, all_samples)


    