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
from tqdm import tqdm
from torchvision import transforms
import random


def sort_dcm_files(input_list, reverse=False):
    output_list = sorted(input_list, key = lambda x:int(x.split('/')[-1].split('.')[0]), reverse=reverse) # 文件名进行升序排序
    return output_list
    

class DefaultLoaderCls5Rnn(data.Dataset):
    def __init__(self, input_db, args, logger, mode='train', input_size=(224, 224)):
        self.input_db = input_db
        self.input_size = input_size
        self.mode = mode
        self.args = args
        self.logger = logger
        self.root_dir = args.root_dir
        self.num_channels = args.num_channels
        self.time_step = args.time_step

        self.label_names = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury']
        self.label_names_binary = ['bowel', 'extravasation']
        self.label_names_triple = ['kidney', 'liver', 'spleen']
        self.injury_names_binary = ['Bowel', 'Active_Extravasation']

        self.hu_min = args.hu_min
        self.hu_max =  args.hu_max

        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.Resize((256, 256)),
                # transforms.CenterCrop(input_size),
                transforms.RandomResizedCrop(size=input_size, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5] * args.num_channels, std=[0.5] * args.num_channels)
            ])
        else:
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5] * args.num_channels, std=[0.5] * args.num_channels)
            ])

        self.pos_dict_binary = self.get_image_level_labels(args.image_level_labels_csv)
        self.organ_info_triple = self.get_3organ_info(args.organ_info_csv)
        self.cls_infos = self.db_loader()
        self.label_infos = self.merge_labels()
        self.img_files_dict, self.label_infos_rnn = self.get_rnn_labels()
        self.logger.info(f"{mode}: total files: {len(self.label_infos_rnn)}")

        self.label_info_printer()

        if args.data_resample and mode == 'train':
            self.resample()
            self.logger.info(f"========== After resample ==========")
            self.label_info_printer()

    def reset(self):
        self.indexes = list(self.img_files_dict.keys())

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
        pos_dict = {}
        pos_uids = df['series_id'].drop_duplicates().values.tolist()
        for uid in pos_uids:
            pos_dict[str(uid)] = {}
            items = df[df['series_id'] == uid]
            injury_names = items['injury_name'].drop_duplicates().values.tolist()
            for injury_name in injury_names:
                items_injury = items[items['injury_name'] == injury_name]
                if len(items_injury):
                    if injury_name == 'Bowel':
                        name = 'bowel'
                    elif injury_name == 'Active_Extravasation':
                        name = 'extravasation'
                    pos_dict[str(uid)][name] = []
                    for data in items_injury.values:
                        pid, uid, instance_number, injury_name = data
                        img_file = os.path.join(self.root_dir, str(pid), str(uid), str(instance_number) + '.dcm')
                       
                        pos_dict[str(uid)][name].append(img_file)
                    pos_dict[str(uid)][name] = sort_dcm_files(pos_dict[str(uid)][name]) # 文件名进行升序排序
        return pos_dict

    def get_3organ_info(self, organ_info_csv):
        df = pd.read_csv(organ_info_csv)
        organ_info = {}
        pos_uids = df['series_id'].drop_duplicates().values.tolist()
        for uid in pos_uids:
            organ_info[str(uid)] = {}
            items = df[df['series_id'] == uid]
            organ_names = items['organ'].drop_duplicates().values.tolist()
            for organ_name in organ_names:
                items_organ = items[items['organ'] == organ_name]
                if len(items_organ):
                    organ_info[str(uid)][organ_name] = []
                    for data in items_organ.values:
                        pid, uid, _, instance_number = data
                        img_file = os.path.join(self.root_dir, str(pid), str(uid), instance_number)
                        organ_info[str(uid)][organ_name].append(img_file)
                    organ_info[str(uid)][organ_name] = sort_dcm_files(organ_info[str(uid)][organ_name])
        return organ_info

    def db_loader(self):
        cls_infos = {}
        env = lmdb.open(self.input_db)
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')
            label_info = json.loads(value)
            original_info = label_info['original_info']
            cls_info = label_info['cls_info']
            cls_infos[key] = cls_info
        env.close()
        return cls_infos

    def get_consecutive_by_time_step(self, img_files, time_step):
        img_files_select = []
        if len(img_files) < time_step:
            img_files_select.extend(img_files)
            for _ in range(time_step - len(img_files)):
                img_files_select.append(img_files[-1])
        else:
            start_range = list(range(0, len(img_files) - time_step + 1))
            
            start_idx = random.sample(start_range, 1)[0]
            img_files_select = img_files[start_idx: start_idx + time_step]
        return img_files_select

    def merge_labels(self):
        # binary
        label_infos = {}
        for uid in self.pos_dict_binary:
            if uid not in self.cls_infos:
                continue
            cls_info = self.cls_infos[uid]
            injury_dict = self.pos_dict_binary[uid]
            for name in injury_dict:
                assert cls_info[name] == 1
                img_files = injury_dict[name]
                for img_file in img_files:
                    if img_file in label_infos:
                        label_infos[img_file][name] = 1
                    else:
                        label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                        label_info_tmp[name] = 1
                        label_infos[img_file] = label_info_tmp
        
        # triple
        for uid in self.organ_info_triple:
            if uid not in self.cls_infos:
                continue
            cls_info = self.cls_infos[uid]
            organ_dict = self.organ_info_triple[uid]
            for organ_name in organ_dict:
                cls_organ = cls_info[organ_name]
                img_files = organ_dict[organ_name]
                for img_file in img_files:
                    if img_file in label_infos:
                        label_infos[img_file][organ_name] = cls_organ
                    else:
                        label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                        label_info_tmp[organ_name] = cls_organ
                        label_infos[img_file] = label_info_tmp
        return label_infos

    def get_rnn_labels(self):
        idx = 0
        label_infos = {}
        img_files_dict = {}
        # binary
        
        for uid in self.pos_dict_binary:
            if uid not in self.cls_infos:
                continue
            cls_info = self.cls_infos[uid]
            injury_dict = self.pos_dict_binary[uid]
            for name in injury_dict:
                assert cls_info[name] == 1
                img_files = injury_dict[name]
                img_files_select = self.get_consecutive_by_time_step(img_files, self.time_step)
                label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                for img_file in img_files_select:
                    label_info = self.label_infos[img_file]
                    for name, value in label_info.items():
                        if value != 0:
                            label_info_tmp[name] = value
                
                label_infos[idx] = label_info_tmp
                img_files_dict[idx] = img_files_select
                idx += 1
        
        # triple
        for uid in self.organ_info_triple:
            if uid not in self.cls_infos:
                continue
            cls_info = self.cls_infos[uid]
            organ_dict = self.organ_info_triple[uid]
            for organ_name in organ_dict:
                cls_organ = cls_info[organ_name]
                img_files = organ_dict[organ_name]
                img_files_select = self.get_consecutive_by_time_step(img_files, self.time_step)
                label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                for img_file in img_files_select:
                    label_info = self.label_infos[img_file]
                    for name, value in label_info.items():
                        if value != 0:
                            label_info_tmp[name] = value
                label_infos[idx] = label_info_tmp
                img_files_dict[idx] = img_files_select
                idx += 1
        return img_files_dict, label_infos

    def label_info_printer(self):
        info = {'bowel': {'healthy': 0, 'injury': 0}, 
                'extravasation': {'healthy': 0, 'injury': 0}, 
                'kidney': {'healthy': 0, 'low': 0, 'high': 1}, 
                'liver': {'healthy': 0, 'low': 0, 'high': 1}, 
                'spleen': {'healthy': 0, 'low': 0, 'high': 1}}
        
        for img_file, cls_info in self.label_infos_rnn.items():
            for name in self.label_names_binary:
                if cls_info[name] == 0:
                    info[name]['healthy'] += 1
                elif cls_info[name] == 1:
                    info[name]['injury'] += 1
            for name in self.label_names_triple:
                if cls_info[name] == 0:
                    info[name]['healthy'] += 1
                elif cls_info[name] == 1:
                    info[name]['low'] += 1
                elif cls_info[name] == 2:
                    info[name]['high'] += 1

        for name in info:
            num_dict = info[name]
            for label_type in num_dict:
                num = num_dict[label_type]
                self.logger.info(f"{self.mode}: {name}_{label_type}: {num}")
    
    def __len__(self):
        # return len(self.img_files_dict)
        return len(self.label_infos_rnn)

    def __getitem__(self, index):
        img_files = self.img_files_dict[index]
        label_info = self.label_infos_rnn[index]
        img = self.load_imgs(img_files)
        
        if self.mode == 'test':
            # uid = img_file.split('/')[-1].split('_')[0]
            # uid = int(uid)
            dirname = os.path.dirname(img_files[0])
            names = []
            for img_file in img_files:
                num = img_file.split('/')[-1].split('.')[0]
                names.append(num)
            name = '_'.join(names)
            img_file_merge = os.path.join(dirname, name + '.dcm')
            return img_norm, label_info, img_file_merge
        else:
            return img, label_info

    def load_img_bak(self, img_file):
    
        assert img_file.endswith('.dcm')
        dcm_img = pydicom.dcmread(img_file)
        img_arr = dcm_img.pixel_array
        bit_shift = dcm_img.BitsAllocated - dcm_img.BitsStored
        dtype = img_arr.dtype 
        img_arr = (img_arr << bit_shift).astype(dtype) >>  bit_shift
        img_arr = pydicom.pixel_data_handlers.util.apply_modality_lut(img_arr, dcm_img)
        img_arr = img_arr[np.newaxis, ...]
        return img_arr

    def standardize_pixel_array(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
        """
        Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
        """
        # Correct DICOM pixel_array if PixelRepresentation == 1.
        pixel_array = dcm.pixel_array
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
    #         pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)

        intercept = float(dcm.RescaleIntercept)
        slope = float(dcm.RescaleSlope)
        center = int(dcm.WindowCenter)
        width = int(dcm.WindowWidth)
        low = center - width / 2
        high = center + width / 2    
        
        pixel_array = (pixel_array * slope) + intercept
        pixel_array = np.clip(pixel_array, low, high)

        return pixel_array

    def load_img(self, img_file):
        dcm_img = pydicom.dcmread(img_file)
        img = self.standardize_pixel_array(dcm_img)
        y_len, x_len = img.shape
        x_min = int(x_len*0.15)
        x_max = int(x_len*0.85)
        y_min = int(y_len*0.15)
        y_max = int(y_len*0.85)
        img = img[y_min:y_max, x_min:x_max]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if dcm_img.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        return img

    def load_imgs(self, img_files):
        imgs = []
        for img_file in img_files:
            img = self.load_img(img_file)
            img = img[np.newaxis, ...]
            img = torch.from_numpy(img).float()
            img = self.transforms(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        return imgs

class DefaultLoaderCls5Rnn_test(data.Dataset):
    def __init__(self, input_db, args, logger, mode='train', input_size=(224, 224)):
        self.input_db = input_db
        self.input_size = input_size
        self.mode = mode
        self.args = args
        self.logger = logger
        self.root_dir = args.root_dir
        self.num_channels = args.num_channels
        self.time_step = args.time_step
        self.reverse = args.reverse

        self.label_names = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury']
        self.label_names_binary = ['bowel', 'extravasation']
        self.label_names_triple = ['kidney', 'liver', 'spleen']
        self.injury_names_binary = ['Bowel', 'Active_Extravasation']

        self.hu_min = args.hu_min
        self.hu_max =  args.hu_max

        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.Resize((256, 256)),
                # transforms.CenterCrop(input_size),
                transforms.RandomResizedCrop(size=input_size, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5] * args.num_channels, std=[0.5] * args.num_channels)
            ])
        else:
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5] * args.num_channels, std=[0.5] * args.num_channels)
            ])

        self.pos_dict_binary = self.get_image_level_labels(args.image_level_labels_csv)
        self.organ_info_triple = self.get_3organ_info(args.organ_info_csv)
        self.cls_infos = self.db_loader()
        self.label_infos = self.merge_labels()
        self.dcm_files_dict = self.get_dcm_files(self.root_dir)
        self.img_files_dict, self.label_infos_rnn = self.get_rnn_labels()
        self.logger.info(f"{mode}: total files: {len(self.label_infos_rnn)}")

        self.label_info_printer()

        if args.data_resample and mode == 'train':
            self.resample()
            self.logger.info(f"========== After resample ==========")
            self.label_info_printer()

    def reset(self):
        self.indexes = list(self.img_files_dict.keys())

    def get_dcm_files(self, root_dir):
        dcm_files_dict = {}
        uids_dir = glob.glob(root_dir + '/*/*')
        for uid_dir in uids_dir:
            uid = uid_dir.split('/')[-1]
            dcm_files = glob.glob(uid_dir + '/*')
            dcm_files = sort_dcm_files(dcm_files, self.reverse)
            dcm_files_dict[uid] = dcm_files
        return dcm_files_dict

    def get_image_level_labels(self, image_level_labels_csv):
        df = pd.read_csv(image_level_labels_csv)
        pos_dict = {}
        pos_uids = df['series_id'].drop_duplicates().values.tolist()
        for uid in pos_uids:
            pos_dict[str(uid)] = {}
            items = df[df['series_id'] == uid]
            injury_names = items['injury_name'].drop_duplicates().values.tolist()
            for injury_name in injury_names:
                items_injury = items[items['injury_name'] == injury_name]
                if len(items_injury):
                    if injury_name == 'Bowel':
                        name = 'bowel'
                    elif injury_name == 'Active_Extravasation':
                        name = 'extravasation'
                    pos_dict[str(uid)][name] = []
                    for data in items_injury.values:
                        pid, uid, instance_number, injury_name = data
                        img_file = os.path.join(self.root_dir, str(pid), str(uid), str(instance_number) + '.dcm')
                       
                        pos_dict[str(uid)][name].append(img_file)
                    pos_dict[str(uid)][name] = sort_dcm_files(pos_dict[str(uid)][name]) # 文件名进行升序排序
        return pos_dict

    def get_3organ_info(self, organ_info_csv):
        df = pd.read_csv(organ_info_csv)
        organ_info = {}
        pos_uids = df['series_id'].drop_duplicates().values.tolist()
        for uid in pos_uids:
            organ_info[str(uid)] = {}
            items = df[df['series_id'] == uid]
            organ_names = items['organ'].drop_duplicates().values.tolist()
            for organ_name in organ_names:
                items_organ = items[items['organ'] == organ_name]
                if len(items_organ):
                    organ_info[str(uid)][organ_name] = []
                    for data in items_organ.values:
                        pid, uid, _, instance_number = data
                        img_file = os.path.join(self.root_dir, str(pid), str(uid), instance_number)
                        organ_info[str(uid)][organ_name].append(img_file)
                    organ_info[str(uid)][organ_name] = sort_dcm_files(organ_info[str(uid)][organ_name])
        return organ_info

    def db_loader(self):
        cls_infos = {}
        env = lmdb.open(self.input_db)
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')
            label_info = json.loads(value)
            original_info = label_info['original_info']
            cls_info = label_info['cls_info']
            cls_infos[key] = cls_info
        env.close()
        return cls_infos

    def get_consecutive_by_time_step(self, img_files, time_step):
        num_files = len(img_files)
        min_z = int(num_files * 0.1)
        max_z = int(num_files * 0.9)
        img_files = img_files[min_z:max_z]
        if len(img_files) > 300:
            z_min = int(len(img_files) / 2) - 150
            z_max = int(len(img_files) / 2) + 150
            img_files = img_files[z_min:z_max]

        img_files_list = []
        num = len(img_files) // time_step
        for i in range(num+1):
            if i == num:
                img_files_tmp = img_files[(len(img_files)-time_step):]
            else:
                img_files_tmp = img_files[i*time_step: (i+1)*time_step]
            img_files_list.append(img_files_tmp)
        return img_files_list

    def merge_labels(self):
        # binary
        label_infos = {}
        for uid in self.pos_dict_binary:
            if uid not in self.cls_infos:
                continue
            cls_info = self.cls_infos[uid]
            injury_dict = self.pos_dict_binary[uid]
            for name in injury_dict:
                assert cls_info[name] == 1
                img_files = injury_dict[name]
                for img_file in img_files:
                    if img_file in label_infos:
                        label_infos[img_file][name] = 1
                    else:
                        label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                        label_info_tmp[name] = 1
                        label_infos[img_file] = label_info_tmp
        
        # triple
        for uid in self.organ_info_triple:
            if uid not in self.cls_infos:
                continue
            cls_info = self.cls_infos[uid]
            organ_dict = self.organ_info_triple[uid]
            for organ_name in organ_dict:
                cls_organ = cls_info[organ_name]
                img_files = organ_dict[organ_name]
                for img_file in img_files:
                    if img_file in label_infos:
                        label_infos[img_file][organ_name] = cls_organ
                    else:
                        label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                        label_info_tmp[organ_name] = cls_organ
                        label_infos[img_file] = label_info_tmp
        return label_infos

    def get_rnn_labels(self):
        idx = 0
        label_infos = {}
        img_files_dict = {}
        for uid, cls_info in self.cls_infos.items():
            dcm_files = self.dcm_files_dict[uid]
            img_files_list = self.get_consecutive_by_time_step(dcm_files, self.time_step)
            for img_files_select in img_files_list:
                label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                for img_file in img_files_select:
                    if img_file in self.label_infos:
                        label_info = self.label_infos[img_file]
                        for name, value in label_info.items():
                            if value != 0:
                                label_info_tmp[name] = value
                
                label_infos[idx] = label_info_tmp
                img_files_dict[idx] = img_files_select
                idx += 1

        return img_files_dict, label_infos

    def label_info_printer(self):
        info = {'bowel': {'healthy': 0, 'injury': 0}, 
                'extravasation': {'healthy': 0, 'injury': 0}, 
                'kidney': {'healthy': 0, 'low': 0, 'high': 1}, 
                'liver': {'healthy': 0, 'low': 0, 'high': 1}, 
                'spleen': {'healthy': 0, 'low': 0, 'high': 1}}
        
        for img_file, cls_info in self.label_infos_rnn.items():
            for name in self.label_names_binary:
                if cls_info[name] == 0:
                    info[name]['healthy'] += 1
                elif cls_info[name] == 1:
                    info[name]['injury'] += 1
            for name in self.label_names_triple:
                if cls_info[name] == 0:
                    info[name]['healthy'] += 1
                elif cls_info[name] == 1:
                    info[name]['low'] += 1
                elif cls_info[name] == 2:
                    info[name]['high'] += 1

        for name in info:
            num_dict = info[name]
            for label_type in num_dict:
                num = num_dict[label_type]
                self.logger.info(f"{self.mode}: {name}_{label_type}: {num}")
    
    def __len__(self):
        # return len(self.img_files_dict)
        return len(self.label_infos_rnn)

    def __getitem__(self, index):
        img_files = self.img_files_dict[index]
        label_info = self.label_infos_rnn[index]
        img = self.load_imgs(img_files)
        
        if self.mode == 'test':
            # uid = img_file.split('/')[-1].split('_')[0]
            # uid = int(uid)
            dirname = os.path.dirname(img_files[0])
            names = []
            for img_file in img_files:
                num = img_file.split('/')[-1].split('.')[0]
                names.append(num)
            name = '_'.join(names)
            # uid = img_files[0].split('/')[-2]
            img_file_merge = os.path.join(dirname, name + '.dcm')
            return img, label_info, img_file_merge
        else:
            return img, label_info


    def standardize_pixel_array(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
        """
        Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
        """
        # Correct DICOM pixel_array if PixelRepresentation == 1.
        pixel_array = dcm.pixel_array
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
    #         pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)

        intercept = float(dcm.RescaleIntercept)
        slope = float(dcm.RescaleSlope)
        center = int(dcm.WindowCenter)
        width = int(dcm.WindowWidth)
        low = center - width / 2
        high = center + width / 2    
        
        pixel_array = (pixel_array * slope) + intercept
        pixel_array = np.clip(pixel_array, low, high)

        return pixel_array

    def load_img(self, img_file):
        dcm_img = pydicom.dcmread(img_file)
        img = self.standardize_pixel_array(dcm_img)
        y_len, x_len = img.shape
        x_min = int(x_len*0.15)
        x_max = int(x_len*0.85)
        y_min = int(y_len*0.20)
        y_max = int(y_len*0.70)
        img = img[y_min:y_max, x_min:x_max]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if dcm_img.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        return img

    def load_imgs(self, img_files):
        imgs = []
        for img_file in img_files:
            img = self.load_img(img_file)
            img = img[np.newaxis, ...]
            img = torch.from_numpy(img).float()
            img = self.transforms(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        return imgs

class DefaultLoaderCls5Rnn_train(data.Dataset):
    def __init__(self, input_db, args, logger, mode='train', input_size=(224, 224)):
        self.input_db = input_db
        self.input_size = input_size
        self.mode = mode
        self.args = args
        self.logger = logger
        self.root_dir = args.root_dir
        self.num_channels = args.num_channels
        self.time_step = args.time_step
        self.reverse = args.reverse

        self.label_names = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury']
        self.label_names_binary = ['bowel', 'extravasation']
        self.label_names_triple = ['kidney', 'liver', 'spleen']
        self.injury_names_binary = ['Bowel', 'Active_Extravasation']

        self.hu_min = args.hu_min
        self.hu_max =  args.hu_max
        self.shuffle_dcm = args.shuffle_dcm

        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.Resize((256, 256)),
                # transforms.CenterCrop(input_size),
                transforms.RandomResizedCrop(size=input_size, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5] * args.num_channels, std=[0.5] * args.num_channels)
            ])
        else:
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5] * args.num_channels, std=[0.5] * args.num_channels)
            ])

        self.pos_dict_binary = self.get_image_level_labels(args.image_level_labels_csv)
        self.organ_info_triple = self.get_3organ_info(args.organ_info_csv)
        self.cls_infos = self.db_loader()
        self.label_infos = self.merge_labels()
        self.dcm_files_dict = self.get_dcm_files(self.root_dir)
        self.img_files_dict, self.label_infos_rnn = self.get_rnn_labels()
        self.logger.info(f"{mode}: total files: {len(self.label_infos_rnn)}")

        self.label_info_printer()

        if args.data_resample and mode == 'train':
            self.resample()
            self.logger.info(f"========== After resample ==========")
            self.label_info_printer()

    def reset(self):
        self.indexes = list(self.img_files_dict.keys())

    def get_dcm_files(self, root_dir):
        dcm_files_dict = {}
        uids_dir = glob.glob(root_dir + '/*/*')
        for uid_dir in uids_dir:
            uid = uid_dir.split('/')[-1]
            dcm_files = glob.glob(uid_dir + '/*')
            dcm_files = sort_dcm_files(dcm_files, self.reverse)
            dcm_files_dict[uid] = dcm_files
        return dcm_files_dict

    def get_image_level_labels(self, image_level_labels_csv):
        df = pd.read_csv(image_level_labels_csv)
        pos_dict = {}
        pos_uids = df['series_id'].drop_duplicates().values.tolist()
        for uid in pos_uids:
            pos_dict[str(uid)] = {}
            items = df[df['series_id'] == uid]
            injury_names = items['injury_name'].drop_duplicates().values.tolist()
            for injury_name in injury_names:
                items_injury = items[items['injury_name'] == injury_name]
                if len(items_injury):
                    if injury_name == 'Bowel':
                        name = 'bowel'
                    elif injury_name == 'Active_Extravasation':
                        name = 'extravasation'
                    pos_dict[str(uid)][name] = []
                    for data in items_injury.values:
                        pid, uid, instance_number, injury_name = data
                        img_file = os.path.join(self.root_dir, str(pid), str(uid), str(instance_number) + '.dcm')
                       
                        pos_dict[str(uid)][name].append(img_file)
                    pos_dict[str(uid)][name] = sort_dcm_files(pos_dict[str(uid)][name]) # 文件名进行升序排序
        return pos_dict

    def get_3organ_info(self, organ_info_csv):
        df = pd.read_csv(organ_info_csv)
        organ_info = {}
        pos_uids = df['series_id'].drop_duplicates().values.tolist()
        for uid in pos_uids:
            organ_info[str(uid)] = {}
            items = df[df['series_id'] == uid]
            organ_names = items['organ'].drop_duplicates().values.tolist()
            for organ_name in organ_names:
                items_organ = items[items['organ'] == organ_name]
                if len(items_organ):
                    organ_info[str(uid)][organ_name] = []
                    for data in items_organ.values:
                        pid, uid, _, instance_number = data
                        img_file = os.path.join(self.root_dir, str(pid), str(uid), instance_number)
                        organ_info[str(uid)][organ_name].append(img_file)
                    organ_info[str(uid)][organ_name] = sort_dcm_files(organ_info[str(uid)][organ_name])
        return organ_info

    def db_loader(self):
        cls_infos = {}
        env = lmdb.open(self.input_db)
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')
            label_info = json.loads(value)
            original_info = label_info['original_info']
            cls_info = label_info['cls_info']
            cls_infos[key] = cls_info
        env.close()
        return cls_infos

    def get_consecutive_by_time_step(self, img_files, time_step):
        
        num_files = len(img_files)
        min_z = int(num_files * 0.1)
        max_z = int(num_files * 0.9)
        img_files = img_files[min_z:max_z]
        if self.mode == 'train':
            if (self.shuffle_dcm) and (random.random() > 0.8):
                random.shuffle(img_files)
        # if len(img_files) > 300:
        #     z_min = int(len(img_files) / 2) - 150
        #     z_max = int(len(img_files) / 2) + 150
        #     img_files = img_files[z_min:z_max]
  
        img_files_list = []
        num = len(img_files) // time_step
        for i in range(num+1):
            if i == num:
                img_files_tmp = img_files[(len(img_files)-time_step):]
            else:
                img_files_tmp = img_files[i*time_step: (i+1)*time_step]
            img_files_list.append(img_files_tmp)
        return img_files_list

    def merge_labels(self):
        # binary
        label_infos = {}
        for uid in self.pos_dict_binary:
            if uid not in self.cls_infos:
                continue
            cls_info = self.cls_infos[uid]
            injury_dict = self.pos_dict_binary[uid]
            for name in injury_dict:
                assert cls_info[name] == 1
                img_files = injury_dict[name]
                for img_file in img_files:
                    if img_file in label_infos:
                        label_infos[img_file][name] = 1
                    else:
                        label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                        label_info_tmp[name] = 1
                        label_infos[img_file] = label_info_tmp
        
        # triple
        for uid in self.organ_info_triple:
            if uid not in self.cls_infos:
                continue
            cls_info = self.cls_infos[uid]
            organ_dict = self.organ_info_triple[uid]
            for organ_name in organ_dict:
                cls_organ = cls_info[organ_name]
                img_files = organ_dict[organ_name]
                for img_file in img_files:
                    if img_file in label_infos:
                        label_infos[img_file][organ_name] = cls_organ
                    else:
                        label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                        label_info_tmp[organ_name] = cls_organ
                        label_infos[img_file] = label_info_tmp
        return label_infos

    def get_rnn_labels(self):
        idx = 0
        label_infos = {}
        img_files_dict = {}
        for uid, cls_info in self.cls_infos.items():
            dcm_files = self.dcm_files_dict[uid]
            img_files_list = self.get_consecutive_by_time_step(dcm_files, self.time_step)
            for img_files_select in img_files_list:
                label_info_tmp = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
                for img_file in img_files_select:
                    if img_file in self.label_infos:
                        label_info = self.label_infos[img_file]
                        for name, value in label_info.items():
                            if value != 0:
                                label_info_tmp[name] = value
                
                label_infos[idx] = label_info_tmp
                img_files_dict[idx] = img_files_select
                idx += 1

        return img_files_dict, label_infos

    def label_info_printer(self):
        info = {'bowel': {'healthy': 0, 'injury': 0}, 
                'extravasation': {'healthy': 0, 'injury': 0}, 
                'kidney': {'healthy': 0, 'low': 0, 'high': 1}, 
                'liver': {'healthy': 0, 'low': 0, 'high': 1}, 
                'spleen': {'healthy': 0, 'low': 0, 'high': 1}}
        
        for img_file, cls_info in self.label_infos_rnn.items():
            for name in self.label_names_binary:
                if cls_info[name] == 0:
                    info[name]['healthy'] += 1
                elif cls_info[name] == 1:
                    info[name]['injury'] += 1
            for name in self.label_names_triple:
                if cls_info[name] == 0:
                    info[name]['healthy'] += 1
                elif cls_info[name] == 1:
                    info[name]['low'] += 1
                elif cls_info[name] == 2:
                    info[name]['high'] += 1

        for name in info:
            num_dict = info[name]
            for label_type in num_dict:
                num = num_dict[label_type]
                self.logger.info(f"{self.mode}: {name}_{label_type}: {num}")
    
    def __len__(self):
        # return len(self.img_files_dict)
        return len(self.label_infos_rnn)

    def __getitem__(self, index):
        img_files = self.img_files_dict[index]
        label_info = self.label_infos_rnn[index]
        img = self.load_imgs(img_files)
        
        if self.mode == 'test':
            # uid = img_file.split('/')[-1].split('_')[0]
            # uid = int(uid)
            dirname = os.path.dirname(img_files[0])
            names = []
            for img_file in img_files:
                num = img_file.split('/')[-1].split('.')[0]
                names.append(num)
            name = '_'.join(names)
            # uid = img_files[0].split('/')[-2]
            img_file_merge = os.path.join(dirname, name + '.dcm')
            return img, label_info, img_file_merge
        else:
            return img, label_info


    def standardize_pixel_array(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
        """
        Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
        """
        # Correct DICOM pixel_array if PixelRepresentation == 1.
        pixel_array = dcm.pixel_array
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
    #         pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)

        intercept = float(dcm.RescaleIntercept)
        slope = float(dcm.RescaleSlope)
        center = int(dcm.WindowCenter)
        width = int(dcm.WindowWidth)
        low = center - width / 2
        high = center + width / 2    
        
        pixel_array = (pixel_array * slope) + intercept
        pixel_array = np.clip(pixel_array, low, high)

        return pixel_array

    def load_img(self, img_file):
        dcm_img = pydicom.dcmread(img_file)
        img = self.standardize_pixel_array(dcm_img)
        y_len, x_len = img.shape
        x_min = int(x_len*0.15)
        x_max = int(x_len*0.85)
        y_min = int(y_len*0.20)
        y_max = int(y_len*0.70)
        img = img[y_min:y_max, x_min:x_max]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if dcm_img.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        return img

    def load_imgs(self, img_files):
        imgs = []
        for img_file in img_files:
            img = self.load_img(img_file)
            img = img[np.newaxis, ...]
            img = torch.from_numpy(img).float()
            img = self.transforms(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        return imgs


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
    img_files = []
    for sample in batch:
        imgs.append(sample[0])
        label_infos.append(sample[1])
        img_files.append(sample[2])
    imgs = torch.stack(imgs, 0)
    return imgs, label_infos, img_files


if __name__ == '__main__':
    pass