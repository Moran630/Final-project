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
import pydicom
import glob

class DefaultLoader(data.Dataset):
    def __init__(self, input_db, args, logger, mode='train', input_size=(224, 224)):
        self.input_db = input_db
        self.input_size = input_size
        self.mode = mode
        self.args = args
        self.logger = logger
        self.root_dir = args.root_dir
        self.nc = args.nc
        self.dcm_files_dict = self.get_dcm_files(args.root_dir)
        self.pos_dict, self.pos_uids_dict = self.get_image_level_labels(args.image_level_labels_csv)
        

        # self.hu_min = -500.0
        # self.hu_max = 500.0
        self.hu_min = -150.0
        self.hu_max = 250.0


        self.label_names = ['bowel', 'extravasation']
        if mode != 'test':
            self.img_files, self.pos_img_files, self.neg_img_files, self.label_infos = self.db_loader_test()
        else:
            self.img_files, self.pos_img_files, self.neg_img_files, self.label_infos = self.db_loader_test()
        assert len(self.img_files) == len(self.label_infos), (len(self.img_files), len(self.label_infos))
        assert (len(self.pos_img_files) + len(self.neg_img_files)) == len(self.img_files)
        self.logger.info(f"{mode}: pos files: {len(self.pos_img_files)}, neg files: {len(self.neg_img_files)}")
        self.label_info_printer()

        if args.data_resample and mode == 'train':
            self.resample()
            self.logger.info(f"========== After resample ==========")
            self.label_info_printer()


    def resample(self, neg_rate=1):
        import random
        self.img_files = []
        pos_num = len(self.pos_img_files)
        self.img_files.extend(self.pos_img_files)
        resample_num = int(neg_rate * pos_num)
        neg_files = random.sample(self.neg_img_files, resample_num)
        self.img_files.extend(neg_files)

    def reset(self):
        self.img_files = list(self.label_infos.keys())

    def get_dcm_files(self, root_dir):
        dcm_files_dict = {}
        uids_dir = glob.glob(root_dir + '/*/*')
        for uid_dir in uids_dir:
            uid = uid_dir.split('/')[-1]
            dcm_files = glob.glob(uid_dir + '/*')
            dcm_files_dict[uid] = dcm_files
        return dcm_files_dict

    def get_image_level_labels(self, image_level_labels_csv):
        df = pd.read_csv(image_level_labels_csv)
        # pos_uids = df['series_id'].drop_duplicates().values.tolist()
        pos_uids_dict = {}
        pos_dict = {}
        for data in df.values:
            pid, uid, instance_number, injury_name = data
            pid = str(pid)
            uid = str(uid)
            instance_number = str(instance_number)
            img_file = os.path.join(self.root_dir, str(pid), str(uid), str(instance_number) + '.dcm')
            dcm_files = self.dcm_files_dict[uid]
            assert img_file in dcm_files
            if img_file not in pos_dict:
                pos_dict[img_file] = {'bowel_injury': 0, 'extravasation_injury': 0}
            if injury_name == 'Bowel':
                pos_dict[img_file]['bowel_injury'] = 1
            if injury_name == 'Active_Extravasation':
                pos_dict[img_file]['extravasation_injury'] = 1
            if uid not in pos_uids_dict:
                pos_uids_dict[uid] = [img_file]
            else:
                if img_file not in pos_uids_dict[uid]:
                    pos_uids_dict[uid].append(img_file)
        return pos_dict, pos_uids_dict

    def db_loader(self):

        img_files = []
        pos_img_files = []
        neg_img_files = []
        label_infos = {}

        env = lmdb.open(self.input_db)
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            uid = str(int(key))

            # hu_min = hu_min_dict[int(key)]
            # hu_max = hu_max_dict[int(key)]
            # incomplete_organ = incomplete_organ_dict[int(key)]
            # if self.mode != 'test':
            #     if hu_min == 0:
            #         continue
            #     if incomplete_organ == 1:
            #         continue

            if uid in self.pos_uids_dict:
                dcm_file_list = self.pos_uids_dict[uid]
                for dcm_file in dcm_file_list:
                    cls_info = self.pos_dict[dcm_file]
                    label_infos[dcm_file] = cls_info
                    pos_img_files.append(dcm_file)
                    img_files.append(dcm_file)
            else:
                # pure neg uid
                dcm_files = self.dcm_files_dict[uid]
                for dcm_file in dcm_files:
                    cls_info = {'bowel_injury': 0, 'extravasation_injury': 0}
                    label_infos[dcm_file] = cls_info
                    neg_img_files.append(dcm_file)
                    img_files.append(dcm_file)

        env.close()
        return img_files, pos_img_files, neg_img_files, label_infos

    def db_loader_test(self):
        img_files = []
        pos_img_files = []
        neg_img_files = []
        label_infos = {}

        env = lmdb.open(self.input_db)
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            uid = str(int(key))

            if uid in self.pos_uids_dict:
                dcm_file_list = self.pos_uids_dict[uid]
                dcm_file_list_all = self.dcm_files_dict[uid]
                for dcm_file in dcm_file_list_all:
                    if dcm_file in dcm_file_list:
                        cls_info = self.pos_dict[dcm_file]
                        label_infos[dcm_file] = cls_info
                        pos_img_files.append(dcm_file)                   
                    else:
                        cls_info = {'bowel_injury': 0, 'extravasation_injury': 0}
                        label_infos[dcm_file] = cls_info
                        neg_img_files.append(dcm_file)
                    img_files.append(dcm_file)

            else:
                # pure neg uid
                dcm_files = self.dcm_files_dict[uid]
                for dcm_file in dcm_files:
                    cls_info = {'bowel_injury': 0, 'extravasation_injury': 0}
                    label_infos[dcm_file] = cls_info
                    neg_img_files.append(dcm_file)
                    img_files.append(dcm_file)

        env.close()
        return img_files, pos_img_files, neg_img_files, label_infos

    def label_info_printer(self):
        info = {'bowel': {'healthy': 0, 'injury': 0}, 
                'extravasation': {'healthy': 0, 'injury': 0}}
        for img_file in self.img_files:
            label_info = self.label_infos[img_file]
            bowel_injury = label_info['bowel_injury']
            extravasation_injury = label_info['extravasation_injury']
            if bowel_injury == 1:
                info['bowel']['injury'] += 1
            else:
                info['bowel']['healthy'] += 1

            if extravasation_injury == 1:
                info['extravasation']['injury'] += 1
            else:
                info['extravasation']['healthy'] += 1
        for organ, label_info in info.items():
            for label, num in label_info.items():
                self.logger.info(f"{self.mode}: {organ}_{label}: {num}")
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        label_info = self.label_infos[img_file]
        img = self.load_img(img_file)

        img = self.resize(img, pad_value=self.hu_min)
        img_pad = self.pad(img, pad_value=self.hu_min)
        # if self.mode == 'train':
        #     img_pad = self.augment(img_pad)
        img_norm = self.norm(img_pad, hu_min=self.hu_min, hu_max=self.hu_max)
        if self.mode == 'test':
            # uid = img_file.split('/')[-1].split('_')[0]
            # uid = int(uid)
            return torch.from_numpy(img_norm).float(), label_info, img_file
        else:
            return torch.from_numpy(img_norm).float(), label_info

    def load_img(self, path_to_img):
        assert path_to_img.endswith('.dcm')
        dcm_img = pydicom.dcmread(path_to_img)
        img_arr = dcm_img.pixel_array
        img_arr = img_arr[np.newaxis, ...]
        return img_arr

    def pad(self, image, pad_value=-500.0):
        _, height, width = image.shape
        input_h = self.input_size[0]
        input_w = self.input_size[1]
        assert input_h >= height, ('height:', height)
        assert input_w >= width, ('width:', width)
        pad = []
        pad.append([0, 0])
        pad.append([0, input_h - height])
        pad.append([0, input_w - width])

        image = np.pad(image, pad, 'constant', constant_values=pad_value)

        return image

    def resize(self, image, pad_value=-500.0):
        _, height, width = image.shape
        max_input = max(self.input_size)
        max_img = max(height, width)
        scale = max_input / max_img
        image_resize = zoom(image, [1, scale, scale], order=1, cval=pad_value)
        return image_resize

    
    def norm(self, image, hu_min=-500.0, hu_max=500.0):
        image = (np.clip(image.astype(np.float32), hu_min, hu_max) - hu_min) / float(hu_max - hu_min)
        return (image - 0.5) * 2.0

    def augment(self, sample, do_flip=True, do_rotate=True, do_swap=True):
        if do_rotate:
            angle = float(np.random.randint(0, 4) * 90)
            # angle = float(np.random.rand() * 180)

            # 计算rotate后的target位置
            sample = rotate(sample, angle, axes=(1, 2), reshape=False)

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
    return imgs, label_infos, uids


if __name__ == '__main__':
    pass