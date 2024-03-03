import os
import numpy as np
import SimpleITK as sitk
import lmdb
import json
from tqdm import tqdm


def get3dboudingbox(mask_array):
    # x
    arr0 = np.max(mask_array, axis=0)
    arr01 = np.max(arr0, axis=0)
    itemindex01 = np.where(arr01 != 0)
    min01 = np.min(itemindex01)
    max01 = np.max(itemindex01)

    # y
    arr02 = np.max(arr0, axis=1)
    itemindex02 = np.where(arr02 != 0)
    min02 = np.min(itemindex02)
    max02 = np.max(itemindex02)

    # z
    arr1 = np.max(mask_array,axis = 1)
    arr12 = np.max(arr1, axis = 1)
    itemindex12 = np.where(arr12!=0)
    min12 = np.min(itemindex12)
    max12 = np.max(itemindex12)

    start_point = np.array([min01, min02, min12 ]) # x,y,z
    end_point = np.array([max01, max02, max12])
    return start_point, end_point


def get_min_max_mean(input_list):
    input_arr = np.array(input_list)
    min_v = np.min(input_arr)
    max_v = np.max(input_arr)
    mean_v = np.mean(input_arr)
    return min_v, max_v, mean_v


def get_range(train_db, rootDir_mask):
    MASK_NAMES = ['kidney_right.nii.gz', 'kidney_left.nii.gz', 
                  'spleen.nii.gz',
                  'liver.nii.gz',
                  'small_bowel.nii.gz', 
                  'colon.nii.gz', 'duodenum.nii.gz']
    
    env = lmdb.open(train_db)
    txn = env.begin()
   
    x_starts, x_ends = [], []
    y_starts, y_ends = [], []
    z_starts, z_ends = [], []

    for key, value in tqdm(txn.cursor()):
        mask_arr = None
        img_size = None
        key = str(key, encoding='utf-8')
        value = str(value, encoding='utf-8')
        maskDir_uid = os.path.join(rootDir_mask, key)
        for mask_name in MASK_NAMES:
            mask_file = os.path.join(maskDir_uid, mask_name)
            if not os.path.exists(mask_file):
                print(f'{mask_name} for {key} not exists!')
                continue
            mask = sitk.ReadImage(mask_file)
            if img_size is None:
                img_size = mask.GetSize()
            else:
                assert img_size == mask.GetSize()
            mask_arr_single = sitk.GetArrayFromImage(mask)
            if mask_arr is None:
                mask_arr = mask_arr_single
            else:
                mask_arr += mask_arr_single

        start_point, end_point = get3dboudingbox(mask_arr)
        x_s = start_point[0] / img_size[0]
        x_e = end_point[0] / img_size[0]
        x_starts.append(x_s)
        x_ends.append(x_e)
        y_s = start_point[1] / img_size[1]
        y_e = end_point[1] / img_size[1]
        y_starts.append(y_s)
        y_ends.append(y_e)
        z_s = start_point[2] / img_size[2]
        z_e = end_point[2] / img_size[2]
        z_starts.append(z_s)
        z_ends.append(z_e)

    xs_min, xs_max, xs_mean = get_min_max_mean(x_starts)
    xe_min, xe_max, xe_mean = get_min_max_mean(x_ends)
    print(f'xs_min: {xs_min}, xs_max: {xs_max}, xs_mean: {xs_mean}')
    print(f'xe_min: {xe_min}, xe_max: {xe_max}, xe_mean: {xe_mean}')
    
    ys_min, ys_max, ys_mean = get_min_max_mean(y_starts)
    ye_min, ye_max, ye_mean = get_min_max_mean(y_ends)
    print(f'ys_min: {ys_min}, ys_max: {ys_max}, ys_mean: {ys_mean}')
    print(f'ye_min: {ye_min}, ye_max: {ye_max}, ye_mean: {ye_mean}')
    
    zs_min, zs_max, zs_mean = get_min_max_mean(z_starts)
    ze_min, ze_max, ze_mean = get_min_max_mean(z_ends)
    print(f'zs_min: {zs_min}, zs_max: {zs_max}, zs_mean: {zs_mean}')
    print(f'ze_min: {ze_min}, ze_max: {ze_max}, ze_mean: {ze_mean}')


def crop_image_with_range(input_dir, output_dir):
    """
    range is calculated by the function get_range
    In this project, x range is 15% ~ 85%, y range is 20% ~70%, z range is 15% ~ 90%
    """
    x_range = [0.15, 0.85]
    y_range = [0.2, 0.7]
    z_range = [0.15, 0.9]

    filenames = os.listdir(input_dir)
    for filename in filenames:
        file = os.path.join(input_dir, filename)
        itk_image = sitk.ReadImage(file) # read raw nii image
        w, h, d = itk_image.GetSize() # get image size, shape(x, y, z)
        npy_image = sitk.GetArrayFromImage(itk_image) # get numpy array from image, shape(z, y, x)
        z_start, z_end = int(z_range[0] * d), int(z_range[1] * d)
        y_start, y_end = int(y_range[0] * h), int(y_range[1] * h)
        x_start, x_end = int(x_range[0] * w), int(x_range[1] * w)
        npy_image_crop = npy_image[z_start:z_end, y_start:y_end, x_start:x_end] # crop image with range
        new_image = sitk.GetImageFromArray(npy_image_crop)
        new_image.SetSpacing(itk_image.GetSpacing())
        new_image.SetDirection(itk_image.GetDirection())
        new_image.SetOrigin(itk_image.GetOrigin())
        
        # wirte itk image
        sitk.WriteImage(new_image, os.path.join(output_dir, filename))
        
if __name__ == '__main__':
    get_range(train_db='/data/wangfy/rsna-2023-abdominal-trauma-detection/split_csv/overall/db/train', 
              rootDir_mask='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_masks')

    crop_image_with_range(input_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_nii', 
                          output_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_nii_croped_ratio_range')