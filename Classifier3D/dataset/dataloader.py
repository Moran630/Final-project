from binascii import Incomplete
import os
from re import T
import numpy as np
import lmdb
import json
import torch
import SimpleITK as sitk
import torch.utils.data as data
import pandas as pd
# from scipy.ndimage.interpolation import rotate
from scipy.ndimage import rotate
from scipy.ndimage import zoom
import copy

class DefaultLoader(data.Dataset):
    def __init__(self, input_db, args, logger, mode='train', input_size=(400, 400, 400)):
        self.input_db = input_db
        self.input_size = input_size
        self.mode = mode
        self.args = args
        self.logger = logger
        self.root_dir = args.root_dir
        self.data_region = args.data_region
        # self.aug = None
        self.remove_large_image = args.remove_large_image

        # self.hu_min = -500.0
        # self.hu_max = 500.0

        self.hu_min = -150.0
        self.hu_max = 250.0

        if self.remove_large_image:
            self.size_info = self.load_size_info(args.size_csv)
        if args.data_region == 'all':
            self.label_names = ['any_injury', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        else:
            self.label_names = [args.data_region]
        self.binary_names = ['bowel', 'extravasation']
        self.triple_names = ['kidney', 'liver', 'spleen']
        self.img_files, self.label_infos = self.db_loader()
        # self.img_files_ori = list(self.label_infos.keys())
        assert len(self.img_files) == len(self.label_infos)

        self.logger.info(f"{self.mode}: Num of images/labels = {len(self.img_files)}")
        self.label_info_printer()
        if args.data_resample and self.mode == 'train':
            self.resample_data()
            # self.img_files = self.resample_data_single()
            self.logger.info(f"After resample")
            self.label_info_printer()

    def reset(self):
        self.img_files = list(self.label_infos.keys())

    def load_size_info(self, size_csv):
        df = pd.read_csv(size_csv)
        size_info = {}
        for data in df.values:
            uid = str(data[0])
            z, y, x = data[1:]
            size_info[uid] = [z, y, x]
        return size_info
    
    def is_large_image(self, size):
        input_d, input_h, input_w = self.input_size
        img_d, img_h, img_w = size
        if input_d < img_d or input_h < img_h or input_w < img_w:
            return True
        else:
            return False

    def db_loader(self):
        hu_df = pd.read_csv('/data/wangfy/rsna-2023-abdominal-trauma-detection/info_preprocessed/train_hu.csv')
        hu_min_dict = {}
        hu_max_dict = {}
        for data in hu_df.values:
            uid = data[0]
            hu_min = data[1]
            hu_max = data[2]
            hu_min_dict[uid] = hu_min
            hu_max_dict[uid] = hu_max

        meta_df = pd.read_csv('/data/wangfy/rsna-2023-abdominal-trauma-detection/train_series_meta.csv')
        incomplete_organ_dict = {}
        for data in meta_df.values:
            uid = data[1]
            value = data[3]
            incomplete_organ_dict[uid] = value

            
        img_files = []
        label_infos = {}
        remove_num = 0
        env = lmdb.open(self.input_db)
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            hu_min = hu_min_dict[int(key)]
            hu_max = hu_max_dict[int(key)]
            incomplete_organ = incomplete_organ_dict[int(key)]
            if self.mode != 'test':
                if hu_min == 0:
                    continue
                if incomplete_organ == 1:
                    continue
            if self.remove_large_image:
                size = self.size_info[key]
                if self.is_large_image(size):
                    remove_num += 1
                    continue
            if self.data_region == 'all':
                img_file = os.path.join(self.root_dir, 'train_images_nii_croped_ratio_range', key + '_croped.nii.gz')
            else:
                img_file = os.path.join(self.root_dir, 'train_images_nii_croped_ratio_range_' + self.data_region, key + '_croped.nii.gz')
                if not os.path.exists(img_file):
                    continue
            value = str(value, encoding='utf-8')
            label_info = json.loads(value)
            original_info = label_info['original_info']
            cls_info = label_info['cls_info']
            label_names = list(cls_info.keys())
            assert len(label_names) == 6
            img_files.append(img_file)
            label_infos[img_file] = cls_info
        env.close()
        self.logger.info(f"{self.mode}: Remove images/labels = {remove_num}")
        return img_files, label_infos

    def label_info_printer(self):
        healthy_info = {}
        injury_info = {}
        injury_info_high = {}
        for name in self.label_names:
            healthy_info[name] = 0
            injury_info[name] = 0
            injury_info_high[name] = 0
        for img_file in self.img_files:
            label_info = self.label_infos[img_file]
            for name in label_info:
                if name not in self.label_names:
                    continue
                value = label_info[name]
                if value == 0:
                    healthy_info[name] += 1
                elif value == 1:
                    injury_info[name] += 1
                else:
                    assert value == 2, value
                    injury_info_high[name] += 1
        for name in self.label_names:
            self.logger.info(f"{self.mode}: {name}, healthy: {healthy_info[name]}, injury low: {injury_info[name]}, injury high: {injury_info_high[name]}")
    
    def get_label_num(self):
        binary_dict = {}
        triple_dict = {}
        all_injury = {}
        for img_file in self.img_files:
            label_info = self.label_infos[img_file]
            injury_num = 0
            for name, label in label_info.items():
                if name == 'any_injury':
                    continue
                if label > 0:
                    injury_num += 1
            if injury_num != 0:
                if injury_num in all_injury:
                    all_injury[injury_num].append(img_file)
                else:
                    all_injury[injury_num] = [img_file]
            else:
                if injury_num in all_injury:
                    all_injury[injury_num].append(img_file)
                else:
                    all_injury[injury_num] = [img_file]
            for name in self.binary_names:
                healthy_name = name + '_healthy'
                injury_name = name + '_injury'
                label = label_info[name]
                assert label in [0, 1]
                if label == 0:
                    if healthy_name not in binary_dict:
                        binary_dict[healthy_name] = [img_file]
                    else:
                        binary_dict[healthy_name].append(img_file)
                else:
                    if injury_name not in binary_dict:
                        binary_dict[injury_name] = [img_file]
                    else:
                        binary_dict[injury_name].append(img_file)
            
            for name in self.triple_names:
                healthy_name = name + '_healthy'
                low_name = name + '_low'
                high_name = name + '_high'
                label = label_info[name]
                assert label in [0, 1, 2]
                if label == 0:
                    if healthy_name not in triple_dict:
                        triple_dict[healthy_name] = [img_file]
                    else:
                        triple_dict[healthy_name].append(img_file)
                elif label == 1:
                    if low_name not in triple_dict:
                        triple_dict[low_name] = [img_file]
                    else:
                        triple_dict[low_name].append(img_file)
                else:
                    assert label == 2
                    if high_name not in triple_dict:
                        triple_dict[high_name] = [img_file]
                    else:
                        triple_dict[high_name].append(img_file)
        return binary_dict, triple_dict, all_injury
        
    def resample_data(self):
        binary_dict, triple_dict, all_injury = self.get_label_num()
        for name, img_files in all_injury.items():
            self.logger.info(f"number={name}: {len(img_files)}")
        self.logger.info(f"binary labels")
        for name, img_files in binary_dict.items():
            self.logger.info(f"{name}: {len(img_files)}")
        self.logger.info(f"triple labels")
        for name, img_files in triple_dict.items():
            self.logger.info(f"{name}: {len(img_files)}")

        # for injury_num, img_files in all_injury.items():
        #     if injury_num in [3, 4]:
        #         for _ in range(5):
        #             self.img_files.extend(img_files)
        #     elif injury_num == 2:
        #         for _ in range(2):
        #             self.img_files.extend(img_files)
        #     elif injury_num == 0:
        #         dowm_ratio = 0.5
        #         import random
        #         healthy_remove = random.sample(img_files, int(dowm_ratio * len(img_files)))
        #         for img_file in healthy_remove:
        #             self.img_files.remove(img_file)

        injury_resample_ratio = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 1}
        for name, img_files in binary_dict.items():
            if 'healthy' in name:
                continue
            name_prefix = name.split('_')[0]
            ratio = injury_resample_ratio[name_prefix]
            for _ in range(ratio):
                self.img_files.extend(img_files)
        for name, img_files in triple_dict.items():
            if 'healthy' in name:
                continue
            name_prefix = name.split('_')[0]
            ratio = injury_resample_ratio[name_prefix]
            for _ in range(ratio):
                self.img_files.extend(img_files)


    def get_num_single(self):
        files_dict = {}
        for img_file in self.img_files:
            label_info = self.label_infos[img_file]
            for name in label_info:
                if name not in self.label_names:
                    continue
                value = label_info[name]
                if value not in files_dict:
                    files_dict[value] = [img_file]
                else:
                    files_dict[value].append(img_file)
        return files_dict

    def resample_data_single(self):
        import random
        files_dict = self.get_num_single()
        num_dict = {}
        for label, img_file_list in files_dict.items():
            num_dict[label] = len(img_file_list)
        
        img_files_resample = []
        pos_num = num_dict[1] + num_dict[2]
        for label, img_file_list in files_dict.items():
            if label == 0:
                neg_resample = random.sample(img_file_list, pos_num*3)
                img_files_resample.extend(neg_resample)
            else:
                img_files_resample.extend(img_file_list)
        return img_files_resample




    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        label_info = self.label_infos[img_file]
        img = self.load_img(img_file)

        img = self.resize(img, pad_value=self.hu_min)
        img_pad = self.pad(img, pad_value=self.hu_min)
        if self.mode == 'train':
            img_pad = self.augment(img_pad)
        img_norm = self.norm(img_pad, hu_min=self.hu_min, hu_max=self.hu_max)
        if self.mode == 'test':
            uid = img_file.split('/')[-1].split('_')[0]
            uid = int(uid)
            return torch.from_numpy(img_norm).float(), label_info, torch.tensor(uid)
        else:
            return torch.from_numpy(img_norm).float(), label_info

    def load_img(self, path_to_img):
        if path_to_img.endswith('.nii.gz'):
            itk_image = sitk.ReadImage(path_to_img)
            img = sitk.GetArrayFromImage(itk_image)[np.newaxis, ...]
        else:
           img = np.load(path_to_img)[np.newaxis, ...]
        return img

    def pad(self, image, pad_value=-150.0):
        _, depth, height, width = image.shape
        input_d = self.input_size[0]
        input_h = self.input_size[1]
        input_w = self.input_size[2]
        assert input_d >= depth, ('depth:', depth)
        # assert input_h >= height, ('height:', height)
        assert input_w >= width, ('width:', width)
        if input_h < height:
            image = image[:,:,:input_h,:]
            pad = []
            pad.append([0, 0])
            pad.append([0, input_d - depth])
            pad.append([0, 0])
            pad.append([0, input_w - width])

        else:        
            pad = []
            pad.append([0, 0])
            pad.append([0, input_d - depth])
            pad.append([0, input_h - height])
            pad.append([0, input_w - width])

        image = np.pad(image, pad, 'constant', constant_values=pad_value)

        return image

    def resize(self, image, pad_value=-500.0):
        _, depth, height, width = image.shape
        # input_d = self.input_size[0]
        # input_h = self.input_size[1]
        # input_w = self.input_size[2]
        max_input = max(self.input_size)
        max_img = max(depth, height, width)
        scale = max_input / max_img
        image_resize = zoom(image, [1, scale, scale, scale], order=1, cval=pad_value)
        return image_resize

        # if max_img <= max_input:
        #     return image
        # else:
        #     scale = max_input / max_img
        #     image_resize = zoom(image, [1, scale, scale, scale], order=1, cval=pad_value)
        #     return image_resize

    
    def norm(self, image, hu_min=-500.0, hu_max=500.0):
        image = (np.clip(image.astype(np.float32), hu_min, hu_max) - hu_min) / float(hu_max - hu_min)
        return (image - 0.5) * 2.0

    def augment(self, sample, do_flip=True, do_rotate=True, do_swap=True):
        if do_rotate:
            angle = float(np.random.randint(0, 4) * 90)
            # angle = float(np.random.rand() * 180)
            rotate_mat = np.array([[np.cos(angle / 180 * np.pi), -np.sin(angle / 180 * np.pi)],
                                   [np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)]])

            # 璁＄畻rotate鍚庣殑target浣嶇疆
            sample = rotate(sample, angle, axes=(2, 3), reshape=False)

        if do_swap:
            axis_order = np.random.permutation(2)
            sample = np.transpose(sample, np.concatenate([[0, 1], axis_order + 2]))
            # axis_order = np.random.permutation(3)
            # sample = np.transpose(sample, np.concatenate([[0], axis_order + 1]))

        if do_flip:
            # only flip by x/y axis
            flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
            sample = np.ascontiguousarray(sample[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]])

        return sample


def collate(batch):
    imgs = []
    label_infos = []
    for sample in batch:
        imgs.append(sample[0])
        label_infos.append(sample[1])
    imgs = torch.stack(imgs, 0)

    return imgs, label_infos

def collate_test(batch):
    imgs = []
    label_infos = []
    uids = []
    for sample in batch:
        imgs.append(sample[0])
        label_infos.append(sample[1])
        uids.append(sample[2])
    imgs = torch.stack(imgs, 0)
    uids = torch.stack(uids, 0)
    return imgs, label_infos, uids


if __name__ == '__main__':
    pass
