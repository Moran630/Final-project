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

def sort_dcm_files(input_list, reverse=False):
    output_list = sorted(input_list, key = lambda x:int(x.split('/')[-1].split('.')[0]), reverse=reverse) # 文件名进行升序排序
    return output_list


class DefaultLoaderCls5(data.Dataset):
    def __init__(self, input_db, args, logger, mode='train', input_size=(224, 224)):
        self.input_db = input_db
        self.input_size = input_size
        self.mode = mode
        self.args = args
        self.logger = logger
        self.root_dir = args.root_dir
        self.num_channels = args.num_channels
        # self.add_neg_in_pos = args.add_neg_in_pos

        self.label_names = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury']
        self.label_names_binary = ['bowel', 'extravasation']
        self.label_names_triple = ['kidney', 'liver', 'spleen']
        self.injury_names_binary = ['Bowel', 'Active_Extravasation']

        # self.hu_min = -500.0
        # self.hu_max = 500.0
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
        self.logger.info(f"{mode}: total files: {len(self.label_infos)}")
        self.img_files = list(self.label_infos.keys())
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


    def label_info_printer(self):
        info = {'bowel': {'healthy': 0, 'injury': 0}, 
                'extravasation': {'healthy': 0, 'injury': 0}, 
                'kidney': {'healthy': 0, 'low': 0, 'high': 1}, 
                'liver': {'healthy': 0, 'low': 0, 'high': 1}, 
                'spleen': {'healthy': 0, 'low': 0, 'high': 1}}
        
        for img_file, cls_info in self.label_infos.items():
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
        return len(self.label_infos)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        label_info = self.label_infos[img_file]
        img = self.load_img(img_file)
        img_norm = img[np.newaxis, ...]
        # img_norm = self.norm(img, hu_min=self.hu_min, hu_max=self.hu_max)
        img_norm = torch.from_numpy(img_norm).float()
        img_norm = self.transforms(img_norm)
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
            return img_norm, label_info

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

    def pad(self, image, pad_value=-500.0):
        _, height, width = image.shape
        if self.mode == 'train':
            input_h = 256
            input_w = 256
        else:
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
        max_img = max(height, width)
        if self.mode == 'train':
            max_input = 256
        else:
            max_input = max(self.input_size)
        
        scale = max_input / max_img
        image_resize = zoom(image, [1, scale, scale], order=1, cval=pad_value)
        return image_resize

    
    def norm(self, image, hu_min=-500.0, hu_max=500.0):
        image = (np.clip(image.astype(np.float32), hu_min, hu_max) - hu_min) / float(hu_max - hu_min)
        # return (image - 0.5) * 2.0
        return image

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
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        if dcm_img.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        return img


class DefaultLoaderCls5_test(data.Dataset):
    def __init__(self, input_db, args, logger, mode='test', input_size=(224, 224)):
        self.input_db = input_db
        self.input_size = input_size
        self.mode = mode
        self.args = args
        self.logger = logger
        self.root_dir = args.root_dir
        self.num_channels = args.num_channels
        # self.add_neg_in_pos = args.add_neg_in_pos

        self.label_names = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury']
        self.label_names_binary = ['bowel', 'extravasation']
        self.label_names_triple = ['kidney', 'liver', 'spleen']
        self.injury_names_binary = ['Bowel', 'Active_Extravasation']

        self.hu_min = args.hu_min
        self.hu_max =  args.hu_max

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
        self.dcm_files_dict = self.get_dcm_files(self.root_dir)
        self.label_infos = self.gen_labels()
        self.logger.info(f"{mode}: total files: {len(self.label_infos)}")
        self.img_files = list(self.label_infos.keys())
        self.label_info_printer()

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

    def get_dcm_label(self, dcm_file, uid, cls_info):
        label_info = {'bowel': 0, 'extravasation': 0, 'kidney': 0, 'liver': 0, 'spleen': 0}
        if uid in self.pos_dict_binary:
            injury_dict = self.pos_dict_binary[uid]
            for name in injury_dict:
                dcm_files = injury_dict[name]
                if dcm_file in dcm_files:
                    label_info[name] = 1
        if uid in self.organ_info_triple:
            organ_dict = self.organ_info_triple[uid]
            for organ_name in organ_dict:
                dcm_files = organ_dict[organ_name]
                cls_organ = cls_info[organ_name]
                if dcm_file in dcm_files:
                    label_info[organ_name] = cls_organ
        return label_info

    def gen_labels(self):
        label_infos = {}
        for uid in self.cls_infos:
            cls_info = self.cls_infos[uid]
            dcm_files = self.dcm_files_dict[uid]
            for dcm_file in dcm_files:
                label_info = self.get_dcm_label(dcm_file, uid, cls_info)
                label_infos[dcm_file] = label_info
        return label_infos

    def label_info_printer(self):
        info = {'bowel': {'healthy': 0, 'injury': 0}, 
                'extravasation': {'healthy': 0, 'injury': 0}, 
                'kidney': {'healthy': 0, 'low': 0, 'high': 1}, 
                'liver': {'healthy': 0, 'low': 0, 'high': 1}, 
                'spleen': {'healthy': 0, 'low': 0, 'high': 1}}
        
        for img_file, cls_info in self.label_infos.items():
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
        return len(self.label_infos)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        label_info = self.label_infos[img_file]
        img = self.load_img(img_file)

        img_norm = self.norm(img, hu_min=self.hu_min, hu_max=self.hu_max)
        img_norm = torch.from_numpy(img_norm).float()
        img_norm = self.transforms(img_norm)
        
        uid = img_file.split('/')[-2]
        uid = int(uid)
        
        if self.mode == 'test':
            return img_norm, label_info, img_file
        else:
            return img_norm, label_info
        
    def load_img(self, img_file):
    
        assert img_file.endswith('.dcm')
        dcm_img = pydicom.dcmread(img_file)
        img_arr = dcm_img.pixel_array
        bit_shift = dcm_img.BitsAllocated - dcm_img.BitsStored
        dtype = img_arr.dtype 
        img_arr = (img_arr << bit_shift).astype(dtype) >>  bit_shift
        img_arr = pydicom.pixel_data_handlers.util.apply_modality_lut(img_arr, dcm_img)
        img_arr = img_arr[np.newaxis, ...]
        return img_arr

    def pad(self, image, pad_value=-500.0):
        _, height, width = image.shape
        if self.mode == 'train':
            input_h = 256
            input_w = 256
        else:
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
        max_img = max(height, width)
        if self.mode == 'train':
            max_input = 256
        else:
            max_input = max(self.input_size)
        
        scale = max_input / max_img
        image_resize = zoom(image, [1, scale, scale], order=1, cval=pad_value)
        return image_resize

    
    def norm(self, image, hu_min=-500.0, hu_max=500.0):
        image = (np.clip(image.astype(np.float32), hu_min, hu_max) - hu_min) / float(hu_max - hu_min)
        # return (image - 0.5) * 2.0
        return image



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