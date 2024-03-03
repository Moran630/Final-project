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

class DataInfo:
    def __init__(self, uid, bowel_injury_files=[], extravasation_injury_files=[], healthy_files=[], injury=False):
        self.uid = uid
        self.bowel_injury_files = bowel_injury_files
        self.extravasation_injury_files = extravasation_injury_files
        self.healthy_files = healthy_files
        self.injury = injury

       
    def updata_bowel(self, bowel_injury_files):
        self.bowel_injury_files = bowel_injury_files
    
    def updata_extravasation(self, extravasation_injury_files):
        self.extravasation_injury_files = extravasation_injury_files

    def update_healthy(self, healthy_files):
        self.healthy_files.extend(healthy_files)

    def updata_label(self):
         if (len(self.bowel_injury_files)) > 0 or (len(self.extravasation_injury_files) > 0):
            self.injury = True


class DefaultLoader_new(data.Dataset):
    def __init__(self, input_db, args, logger, mode='train', input_size=(224, 224)):
        self.input_db = input_db
        self.input_size = input_size
        self.mode = mode
        self.args = args
        self.logger = logger
        self.root_dir = args.root_dir
        self.num_channels = args.num_channels
        self.add_neg_in_pos = args.add_neg_in_pos

        self.label_names = ['bowel', 'extravasation']
        self.injury_names = ['Bowel', 'Active_Extravasation']

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
                transforms.Normalize(mean=[0.5] * args.num_channels, std=[0.5] * args.num_channels)
            ])
        else:
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * args.num_channels, std=[0.5] * args.num_channels)
            ])

        self.dcm_files_dict = self.get_dcm_files(args.root_dir)
        self.pos_dict = self.get_image_level_labels(args.image_level_labels_csv)
        
        self.data_infos = self.db_loader()
        self.img_files_dict, self.label_infos_dict = self.get_img_files_and_labels()
        assert len(self.img_files_dict) == len(self.label_infos_dict)
        self.logger.info(f"{mode}: total files: {len(self.img_files_dict)}")
        self.label_info_printer()

        self.indexes = list(self.img_files_dict.keys())
        if args.data_resample and mode == 'train':
            self.resample()
            self.logger.info(f"========== After resample ==========")
            self.label_info_printer()

    def label_count(self):
        count_dict = {'bowel_injury': [], 'extravasation_injury': [], 'pure_neg': []}
        for idx in list(self.img_files_dict.keys()):
            label_info = self.label_infos_dict[idx]
            bowel_injury = label_info['bowel_injury']
            extravasation_injury = label_info['extravasation_injury']
            if bowel_injury == 1:
                count_dict['bowel_injury'].append(idx)
            if extravasation_injury == 1:
                count_dict['extravasation_injury'].append(idx)
            if (bowel_injury == 0) and (extravasation_injury == 0):
                count_dict['pure_neg'].append(idx)
        return count_dict

    def resample(self, neg_rate=1):
        count_dict = self.label_count()
        import random

        self.indexes = []
        bowel_injury_num = len(count_dict['bowel_injury'])
        extravasation_injury_num = len(count_dict['extravasation_injury'])
        pos_num = bowel_injury_num + extravasation_injury_num
        pure_neg_num = len(count_dict['pure_neg'])
        resample_num = int(neg_rate * pos_num)

        self.indexes.extend(count_dict['bowel_injury'])
        self.indexes.extend(count_dict['extravasation_injury'])
        neg_idxes = random.sample(count_dict['pure_neg'], resample_num)
        self.indexes.extend(neg_idxes)

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
                    pos_dict[str(uid)][injury_name] = []
                    for data in items_injury.values:
                        pid, uid, instance_number, injury_name = data
                        img_file = os.path.join(self.root_dir, str(pid), str(uid), str(instance_number) + '.dcm')
                        pos_dict[str(uid)][injury_name].append(img_file)
                    pos_dict[str(uid)][injury_name] = sort_dcm_files(pos_dict[str(uid)][injury_name]) # 文件名进行升序排序
        return pos_dict

    def db_loader(self):
        data_infos = []
        env = lmdb.open(self.input_db)
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            uid = str(int(key))
            data_info = {'uid': uid, 
                         'bowel_injury_files': [], 
                         'extravasation_injury_files': [],
                         'healthy_files': [],
                         'injury': False}
            if uid in self.pos_dict:
                injury_info = self.pos_dict[uid]
                for injury_name, dcm_files_list in injury_info.items():
                    if injury_name == 'Bowel':
                        data_info['bowel_injury_files'].extend(dcm_files_list)
                        data_info['injury'] = True
                    if injury_name == 'Active_Extravasation':
                        data_info['extravasation_injury_files'].extend(dcm_files_list)
                        data_info['injury'] = True
                
                dcm_files_list_all = self.dcm_files_dict[uid]
                dcm_files_healthy = self.get_healthy_file_in_pos(dcm_files_list_all, data_info)
                data_info['healthy_files'].extend(dcm_files_healthy)
            else:
                # pure neg uid
                dcm_files_list_all = self.dcm_files_dict[uid]
                dcm_files_list_all_sort = sort_dcm_files(dcm_files_list_all)
                data_info['healthy_files'].extend(dcm_files_list_all_sort)
            data_infos.append(data_info)
        env.close()
        return data_infos

    def get_healthy_file_in_pos(self, dcm_files_list_all, data_info):
        injury_files = data_info['bowel_injury_files'] + data_info['extravasation_injury_files']
        healthy_files = list(set(dcm_files_list_all) - set(injury_files))
        healthy_files_sorted = sort_dcm_files(healthy_files)
        return healthy_files_sorted
    
    def get_img_files_and_labels(self):
        img_files_dict = {}
        label_infos_dict = {}
        if self.mode == 'train':
            idx = 0
            for data_info in tqdm(self.data_infos):
                # 阳性数据, 阴性数据根据injury进行判断
                # bowel_injury_files=[], extravasation_injury_files=[], healthy_files=[]
                img_files, label_infos = self.get_file_label_by_nc_train(data_info, self.num_channels, self.add_neg_in_pos)
                assert len(img_files) == len(label_infos)
                for img_file, label_info in zip(img_files, label_infos):
                    assert len(img_file) == self.num_channels, (len(img_file), self.num_channels)
                    img_files_dict[idx] = img_file
                    label_infos_dict[idx] = label_info
                    idx += 1

        else:
            # 验证/测试阶段, 根据num_channesl进行滑动读图
            idx = 0
            for data_info in tqdm(self.data_infos):
                img_files, label_infos = self.get_file_label_by_nc_valtest(data_info, self.num_channels)
                assert len(img_files) == len(label_infos)
                for img_file, label_info in zip(img_files, label_infos):
                    assert len(img_file) == self.num_channels, (len(img_file), self.num_channels)
                    img_files_dict[idx] = img_file
                    label_infos_dict[idx] = label_info
                    idx += 1
        return img_files_dict, label_infos_dict
    
    def get_file_label_by_nc_valtest(self, data_info, num_channels):
        uid = data_info['uid']
        bowel_injury_files = data_info['bowel_injury_files']
        extravasation_injury_files = data_info['extravasation_injury_files']
        # healthy_files = data_info['healthy_files']
        injury = data_info['injury']

        img_files = []
        label_infos = []
        dcm_files_all = self.dcm_files_dict[uid]
        num = len(dcm_files_all)
        if num < num_channels:
            img_files_tmp = dcm_files_all
            repeat_num = num_channels - num
            for _ in range(repeat_num):
                img_files_tmp.append(dcm_files_all[-1])
                label_info = {'bowel_injury': 0, 'extravasation_injury': 0}
                if injury:
                    if len(bowel_injury_files):
                        inter = list(set(img_files_tmp) & set(bowel_injury_files))
                        if len(inter):
                            label_info['bowel_injury'] = 1
                    if len(extravasation_injury_files):
                        inter = list(set(img_files_tmp) & set(extravasation_injury_files))
                        if len(inter):
                            label_info['extravasation_injury'] = 1
                img_files.append(img_files_tmp)
                label_infos.append(label_info)
        else:
            remainder = num % num_channels
            if remainder == 0:
                for i in range(0, num, num_channels):
                    img_files_tmp = dcm_files_all[i: i+num_channels]
                    label_info = {'bowel_injury': 0, 'extravasation_injury': 0}
                    if injury:
                        if len(bowel_injury_files):
                            inter = list(set(img_files_tmp) & set(bowel_injury_files))
                            if len(inter):
                                label_info['bowel_injury'] = 1
                        if len(extravasation_injury_files):
                            inter = list(set(img_files_tmp) & set(extravasation_injury_files))
                            if len(inter):
                                label_info['extravasation_injury'] = 1
                    img_files.append(img_files_tmp)
                    label_infos.append(label_info)
            else:
                # 如果num不能被num_channels整除, 最后一组数据开始的索引前移
                num_range = int(num / num_channels)
                for i in range(num_range):
                    img_files_tmp = dcm_files_all[i*num_channels: (i+1)*num_channels]
                    label_info = {'bowel_injury': 0, 'extravasation_injury': 0}
                    if injury:
                        if len(bowel_injury_files):
                            inter = list(set(img_files_tmp) & set(bowel_injury_files))
                            if len(inter):
                                label_info['bowel_injury'] = 1
                        if len(extravasation_injury_files):
                            inter = list(set(img_files_tmp) & set(extravasation_injury_files))
                            if len(inter):
                                label_info['extravasation_injury'] = 1
                    img_files.append(img_files_tmp)
                    label_infos.append(label_info)
            
                # 增加最后一组数据
                # offset_num = num_channels - remainder
                img_files_tmp = dcm_files_all[(num-num_channels): num]
                label_info = {'bowel_injury': 0, 'extravasation_injury': 0}
                if injury:
                    if len(bowel_injury_files):
                        inter = list(set(img_files_tmp) & set(bowel_injury_files))
                        if len(inter):
                            label_info['bowel_injury'] = 1
                    if len(extravasation_injury_files):
                        inter = list(set(img_files_tmp) & set(extravasation_injury_files))
                        if len(inter):
                            label_info['extravasation_injury'] = 1
                img_files.append(img_files_tmp)
                label_infos.append(label_info)
        return img_files, label_infos

    def get_file_label_by_nc_train(self, data_info, num_channels, add_neg_in_pos):
        bowel_injury_files = data_info['bowel_injury_files']
        extravasation_injury_files = data_info['extravasation_injury_files']
        healthy_files = data_info['healthy_files']
        injury = data_info['injury']
        # 逻辑验证
        if injury:
            assert ((len(bowel_injury_files) > 0) or (len(extravasation_injury_files) > 0))
        else:
            assert len(bowel_injury_files) == 0
            assert len(extravasation_injury_files) == 0
            assert len(healthy_files) > 0
        img_files = []
        label_infos = []
        if len(bowel_injury_files):
            num = len(bowel_injury_files) - num_channels + 1
            if num <= 0:
                # 总数不足 num_channels, 重复最后一张图
                repeat_num = abs(num) + 1
                img_files_tmp = bowel_injury_files
                for _ in range(repeat_num):
                    img_files_tmp.append(bowel_injury_files[-1])
                label_info = {'bowel_injury': 1, 'extravasation_injury': 0}
                inter = list(set(img_files_tmp) & set(extravasation_injury_files))
                if len(inter):
                    label_info['extravasation_injury'] = 1
                img_files.append(img_files_tmp)
                label_infos.append(label_info)
            else:
                for i in range(num):
                    img_files_tmp = bowel_injury_files[i: i+num_channels]
                    label_info = {'bowel_injury': 1, 'extravasation_injury': 0}
                    inter = list(set(img_files_tmp) & set(extravasation_injury_files))
                    if len(inter):
                        label_info['extravasation_injury'] = 1
                    img_files.append(img_files_tmp)
                    label_infos.append(label_info)

        if len(extravasation_injury_files):
            num = len(extravasation_injury_files) - num_channels + 1
            if num < 0:
                # 总数不足 num_channels, 重复最后一张图
                repeat_num = abs(num) + 1
                img_files_tmp = extravasation_injury_files
                for _ in range(repeat_num):
                    img_files_tmp.append(extravasation_injury_files[-1])
                label_info = {'bowel_injury': 0, 'extravasation_injury': 1}
                inter = list(set(img_files_tmp) & set(bowel_injury_files))
                if len(inter):
                    label_info['bowel_injury'] = 1
                img_files.append(img_files_tmp)
                label_infos.append(label_info)
            else:
                for i in range(num):
                    img_files_tmp = extravasation_injury_files[i: i+num_channels]
                    label_info = {'bowel_injury': 0, 'extravasation_injury': 1}
                    inter = list(set(img_files_tmp) & set(bowel_injury_files))
                    if len(inter):
                        label_info['bowel_injury'] = 1
                    img_files.append(img_files_tmp)
                    label_infos.append(label_info)

        if (add_neg_in_pos and injury) or (not injury):
            # 阴性dicom
            label_info = {'bowel_injury': 0, 'extravasation_injury': 0}
            num = len(healthy_files)
            if num < num_channels:
                img_files_tmp = healthy_files
                repeat_num = num_channels - num
                for _ in range(repeat_num):
                    img_files_tmp.append(healthy_files[-1])
                    img_files.append(img_files_tmp)
                    label_infos.append(label_info)
            else:
                remainder = num % num_channels
                if remainder == 0:
                    for i in range(0, num, num_channels):
                        img_files_tmp = healthy_files[i: i+num_channels]
                        img_files.append(img_files_tmp)
                        label_infos.append(label_info)
                else:
                    # 如果num不能被num_channels整除, 最后一组数据开始的索引前移
                    num_range = int(num / num_channels)
                    for i in range(num_range):
                        img_files_tmp = healthy_files[i*num_channels: (i+1)*num_channels]
                        img_files.append(img_files_tmp)
                        label_infos.append(label_info)
                
                    # 增加最后一组数据
                    # offset_num = num_channels - remainder
                    img_files_tmp = healthy_files[(num-num_channels): num]
                    img_files.append(img_files_tmp)
                    label_infos.append(label_info)

        return img_files, label_infos


    def label_info_printer(self):
        info = {'bowel': {'healthy': 0, 'injury': 0}, 
                'extravasation': {'healthy': 0, 'injury': 0}}
        two_injury = 0
        for idx in self.img_files_dict:
            label_info = self.label_infos_dict[idx]
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
            if bowel_injury and extravasation_injury:
                two_injury += 1
        for organ, label_info in info.items():
            for label, num in label_info.items():
                self.logger.info(f"{self.mode}: {organ}_{label}: {num}")
        self.logger.info(f"{self.mode}: two organ all injury: {two_injury}")
    
    def __len__(self):
        # return len(self.img_files_dict)
        return len(self.indexes)

    def __getitem__(self, index):
        idx = self.indexes[index]
        img_files = self.img_files_dict[idx]
        label_info = self.label_infos_dict[idx]
        img = self.load_img(img_files)

        # img = self.resize(img, pad_value=self.hu_min)
        # img_pad = self.pad(img, pad_value=self.hu_min)
        # img_norm = self.norm(img_pad, hu_min=self.hu_min, hu_max=self.hu_max)
        img_norm = self.norm(img, hu_min=self.hu_min, hu_max=self.hu_max)
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

    def load_img(self, img_files):
        img_arrs = []
        for img_file in img_files:
            assert img_file.endswith('.dcm')
            dcm_img = pydicom.dcmread(img_file)
            img_arr = dcm_img.pixel_array
            bit_shift = dcm_img.BitsAllocated - dcm_img.BitsStored
            dtype = img_arr.dtype 
            img_arr = (img_arr << bit_shift).astype(dtype) >>  bit_shift
            img_arr = pydicom.pixel_data_handlers.util.apply_modality_lut(img_arr, dcm_img)
            img_arr = img_arr[np.newaxis, ...]
            img_arrs.append(img_arr)
        img_arrs = np.concatenate(img_arrs, axis=0)
        return img_arrs

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
    uids = []
    for sample in batch:
        imgs.append(sample[0])
        label_infos.append(sample[1])
        uids.append(sample[2])
    imgs = torch.stack(imgs, 0)
    return imgs, label_infos, uids


if __name__ == '__main__':
    pass