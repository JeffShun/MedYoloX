import argparse
import os
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm_path', type=str, default='./train_data/dcms')
    parser.add_argument('--save_path', type=str, default='./train_data/prepared_data')
    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    sitk_img = reader.Execute()
    return sitk_img

if __name__ == '__main__':
    args = parse_args()
    dcm_path = args.dcm_path
    save_path = args.save_path
    for sample in tqdm(os.listdir(dcm_path)):
        sample_dir = os.path.join(dcm_path, sample)
        sitk_img = load_scans(sample_dir)
        volume = sitk.GetArrayFromImage(sitk_img)
        save_dir = os.path.join(save_path, sample)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(volume.shape[0]):
            slice = volume[i]
            img = (slice - slice.min())/(slice.max() - slice.min())
            img = img*255
            img = Image.fromarray(img.astype("uint8"))
            img.save(os.path.join(save_dir, f"{sample}{i:04}.png"))

    sitk.WriteImage(sitk_img, os.path.join(save_dir, f"{sample}.volume.nii.gz"))    