import os
import subprocess
# import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def single_seg_infer(input_files, output_dir, rank):
    '''
    rank means the gpu id
    input_file means the input ct image with nii.gz as the suffix
    save_dir means the output folder for saving generated masks
    '''
    for input_file in tqdm(input_files):
        uid = input_file.split('/')[-1].replace('.nii.gz', '')
        save_dir = os.path.join(output_dir, uid)
        if os.path.exists(save_dir):
            mask_niis = os.listdir(save_dir)
            if len(mask_niis) == 104:
                continue
        os.makedirs(save_dir, exist_ok=True)
        try:
            cmd = 'CUDA_VISIBLE_DEVICES=' + str(rank) + ' TotalSegmentator -i ' + input_file + ' -o ' + save_dir + ' --fast'
            print(cmd)
            subprocess.call(cmd, shell=True)
        except:
            print(uid)

def seg_infer_one(input_dir, output_dir):
    nii_names = os.listdir(input_dir)
    for nii_name in tqdm(nii_names):
        input_nii = os.path.join(input_dir, nii_name)
        uid = nii_name.replace('.nii.gz', '')
        save_dir = os.path.join(output_dir, uid)
        os.makedirs(save_dir, exist_ok=True)
        cmd = 'TotalSegmentator -i ' + input_nii + ' -o ' + save_dir + ' --fast'
        subprocess.call(cmd, shell=True)


def get_input_files(input_dir, done_uids):
    input_files = []
    nii_names = os.listdir(input_dir)
    for nii_name in tqdm(nii_names):
        input_nii = os.path.join(input_dir, nii_name)
        uid = nii_name.replace('.nii.gz', '')
        if uid in done_uids:
            continue
        input_files.append(input_nii)

    return input_files


def split_list(input_list, split_num=8):
    total_num = len(input_list)
    print('total_num: ', total_num)
    split_dict = {}
    per_num = int(total_num / split_num) + 1
    for i in range(split_num):
        split_dict[i] = input_list[i * per_num: (i + 1) * per_num]
        print('split {}, num: {}'.format(i, len(split_dict[i])))
    return split_dict


def seg_infer(input_dir, output_dir, pool_size=8):
    done_uids = []
    uids = os.listdir(output_dir)
    for uid in uids:
        uid_dir = os.path.join(output_dir, uid)
        mask_niis = os.listdir(uid_dir)
        if len(mask_niis) == 104:
            done_uids.append(uid)
    print('{} finished!'.format(len(done_uids)))

    input_files = get_input_files(input_dir, done_uids)
    split_dict = split_list(input_files)
    pool = Pool(pool_size)
    for i in range(8):
        pool.apply_async(single_seg_infer, (split_dict[i], output_dir, i))
    pool.close()
    pool.join()
    # single_seg_infer(split_dict[1], output_dir, 1)

if __name__ == '__main__':
    output_dir = '/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_masks'
    uids = os.listdir(output_dir)
    for uid in uids:
        uid_dir = os.path.join(output_dir, uid)
        mask_niis = os.listdir(uid_dir)
        if len(mask_niis) != 104:
            print(uid)

    seg_infer(input_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_nii', output_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_masks', pool_size=8)