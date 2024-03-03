import cv2
import numpy as np
import pandas as pd
import os
import pydicom
import glob
import imageio

from PIL import Image
from tqdm import tqdm


def sort_dcm_files(input_list, reverse=False):
    output_list = sorted(input_list, key = lambda x:int(x.split('/')[-1].split('.')[0]), reverse=reverse) # 文件名进行升序排序
    return output_list


def norm(image, hu_min=-500.0, hu_max=500.0):
    image = (np.clip(image.astype(np.float32), hu_min, hu_max) - hu_min) / float(hu_max - hu_min)
    image = image * 255
    # return (image - 0.5) * 2.0
    return image.astype('uint8')

def single_save(dcm_files, bboxes_dict, save_file):
    imgs = []
    for dcm_file in tqdm(dcm_files):
        instance_num = int(dcm_file.split('/')[-1].split('.')[0])
        if instance_num not in bboxes_dict:
            # print(instance_num)
            continue
        bbox = bboxes_dict[instance_num]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dcm_img = pydicom.dcmread(dcm_file)
        img_arr = dcm_img.pixel_array
      
        bit_shift = dcm_img.BitsAllocated - dcm_img.BitsStored
        dtype = img_arr.dtype 
        img_arr = (img_arr << bit_shift).astype(dtype) >>  bit_shift
        img_arr = pydicom.pixel_data_handlers.util.apply_modality_lut(img_arr, dcm_img)
        # print(img_arr.min(), img_arr.max())
        # img = norm(img_arr, img_arr.min(), img_arr.max())
        img = norm(img_arr, -150, 250)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        # cv2.imwrite('/data/wangfy/rsna-2023-abdominal-trauma-detection/tmp.png', img)
        # raise
        imgs.append(img)

    # ret = cv2.imwritemulti(save_file, imgs)
    gif = imageio.mimsave(save_file, imgs, 'GIF', duration=0.001)


def rect_demo(root_dir, bbox_csv, pid, uid, save_file):
    dcm_files = [os.path.join(root_dir, str(pid), str(uid), dcm_name) for dcm_name in os.listdir(os.path.join(root_dir, str(pid), str(uid)))]
    dcm_files = sort_dcm_files(dcm_files)
    df = pd.read_csv(bbox_csv)
    df = df[df['series_id'] == uid]
    bboxes_dict = {}
    for data in df.values:
        bbox = data[1:5]
        instance_num = int(data[7])
        bboxes_dict[instance_num] = bbox
    single_save(dcm_files, bboxes_dict, save_file)

def rect_all(root_dir, bbox_csv, save_dir):
    df = pd.read_csv(bbox_csv)
    uids = df['series_id'].drop_duplicates().values.tolist()
    
    for uid in tqdm(uids):
        items = df[df['series_id'] == uid]
        pid = items.values[0][5]
        dcm_files = [os.path.join(root_dir, str(pid), str(uid), dcm_name) for dcm_name in os.listdir(os.path.join(root_dir, str(pid), str(uid)))]
        dcm_files = sort_dcm_files(dcm_files)
        bboxes_dict = {}
        for data in items.values:
            bbox = data[1:5]
            # pid = data[5]
            instance_num = int(data[7])
            bboxes_dict[instance_num] = bbox

        save_file = os.path.join(save_dir, str(pid), str(uid) + '.gif')
        dirname = os.path.dirname(save_file)
        os.makedirs(dirname, exist_ok=True)
        single_save(dcm_files, bboxes_dict, save_file)

if __name__ == '__main__':
    # rect_demo(root_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images', 
    #           bbox_csv='/data/wangfy/rsna-2023-abdominal-trauma-detection/active_extravasation_bounding_boxes.csv', 
    #           pid=10004, 
    #           uid=21057, 
    #           save_file='/data/wangfy/rsna-2023-abdominal-trauma-detection/21057.gif')

    rect_all(root_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images', 
              bbox_csv='/data/wangfy/rsna-2023-abdominal-trauma-detection/active_extravasation_bounding_boxes.csv', 
              save_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_extravasation_bbox_vis')