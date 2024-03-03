import SimpleITK as sitk
import pandas as pd
import pydicom
from pydicom import dicomio
import numpy as np
import os
from tqdm import tqdm


def save_csv(final_columns, final_records, output_csv):
    data_save = pd.DataFrame(columns=final_columns, data=final_records)
    data_save.drop_duplicates(inplace=True)
    data_save.to_csv(output_csv, index=False)


def load_ct_from_dicom(dcm_path, sort_by_distance=True):
    class DcmInfo(object):
        def __init__(self, dcm_path, series_instance_uid, acquisition_number, sop_instance_uid, instance_number,
                     image_orientation_patient, image_position_patient):
            super(DcmInfo, self).__init__()

            self.dcm_path = dcm_path
            self.acquisition_number = acquisition_number
            self.sop_instance_uid = sop_instance_uid
            self.instance_number = instance_number
            self.image_orientation_patient = image_orientation_patient
            self.image_position_patient = image_position_patient

            self.slice_distance = self._cal_distance()

        def _cal_distance(self):
            normal = [self.image_orientation_patient[1] * self.image_orientation_patient[5] -
                      self.image_orientation_patient[2] * self.image_orientation_patient[4],
                      self.image_orientation_patient[2] * self.image_orientation_patient[3] -
                      self.image_orientation_patient[0] * self.image_orientation_patient[5],
                      self.image_orientation_patient[0] * self.image_orientation_patient[4] -
                      self.image_orientation_patient[1] * self.image_orientation_patient[3]]

            distance = 0
            for i in range(3):
                distance += normal[i] * self.image_position_patient[i]
            return distance

    def is_sop_instance_uid_exist(dcm_info, dcm_infos):
        for item in dcm_infos:
            if dcm_info.sop_instance_uid == item.sop_instance_uid:
                return True
        return False

    def get_dcm_path(dcm_info):
        return dcm_info.dcm_path

    reader = sitk.ImageSeriesReader()
    if sort_by_distance:
        dcm_infos = []

        files = os.listdir(dcm_path)
       
        for file in files:
            file_path = os.path.join(dcm_path, file)

            dcm = dicomio.read_file(file_path, stop_before_pixels=True, force=True)
            _series_instance_uid = dcm.SeriesInstanceUID
            _acquisition_number = "None" #dcm.AcquisitionNumber
            _sop_instance_uid = dcm.SOPInstanceUID
            _instance_number = dcm.InstanceNumber
            _image_orientation_patient = dcm.ImageOrientationPatient
            _image_position_patient = dcm.ImagePositionPatient

            dcm_info = DcmInfo(file_path, _series_instance_uid, _acquisition_number, _sop_instance_uid,
                               _instance_number, _image_orientation_patient, _image_position_patient)

            if is_sop_instance_uid_exist(dcm_info, dcm_infos):
                continue

            dcm_infos.append(dcm_info)

        dcm_infos.sort(key=lambda x: x.slice_distance)
        dcm_series = list(map(get_dcm_path, dcm_infos))
    else:
        dcm_series = reader.GetGDCMSeriesFileNames(dcm_path)

    return dcm_series


def get_bbox_kidney(mask_uid_dir):
    mask = sitk.ReadImage(os.path.join(mask_uid_dir, 'kidney_right.nii.gz'))
    mask_arr = sitk.GetArrayFromImage(mask)
    mask = sitk.ReadImage(os.path.join(mask_uid_dir, 'kidney_left.nii.gz'))
    mask_arr_kindey = mask_arr + sitk.GetArrayFromImage(mask)
    indexes = np.where(mask_arr_kindey == 1)
    z_min, z_max = np.min(indexes[0]), np.max(indexes[0])
    y_min, y_max = np.min(indexes[1]), np.max(indexes[1])
    x_min, x_max = np.min(indexes[2]), np.max(indexes[2])
    return [x_min, x_max, y_min, y_max, z_min, z_max]


def get_bbox_liver(mask_uid_dir):
    mask = sitk.ReadImage(os.path.join(mask_uid_dir, 'liver.nii.gz'))
    mask_arr = sitk.GetArrayFromImage(mask)
    indexes = np.where(mask_arr == 1)
    z_min, z_max = np.min(indexes[0]), np.max(indexes[0])
    y_min, y_max = np.min(indexes[1]), np.max(indexes[1])
    x_min, x_max = np.min(indexes[2]), np.max(indexes[2])
    return [x_min, x_max, y_min, y_max, z_min, z_max]


def get_bbox_spleen(mask_uid_dir):
    mask = sitk.ReadImage(os.path.join(mask_uid_dir, 'spleen.nii.gz'))
    mask_arr = sitk.GetArrayFromImage(mask)
    indexes = np.where(mask_arr == 1)
    z_min, z_max = np.min(indexes[0]), np.max(indexes[0])
    y_min, y_max = np.min(indexes[1]), np.max(indexes[1])
    x_min, x_max = np.min(indexes[2]), np.max(indexes[2])
    return [x_min, x_max, y_min, y_max, z_min, z_max]

def get_instance_number(input_dcm_dir, input_mask_dir, output_csv):
    final_columns = ['patient_id', 'series_id', 'organ', 'instance_number']
    final_records = []
    pids = os.listdir(input_dcm_dir)
    for pid in tqdm(pids):
        pid_dir = os.path.join(input_dcm_dir, pid)
        uids = os.listdir(pid_dir)
        for uid in uids:
            uid_dir = os.path.join(pid_dir, uid)
            dcm_series = load_ct_from_dicom(uid_dir)
            mask_uid_dir = os.path.join(input_mask_dir, uid)
            if not os.path.exists(mask_uid_dir):
                continue
            try:
                kidney_bbox = get_bbox_kidney(mask_uid_dir)
                kidney_slices = dcm_series[kidney_bbox[4]: kidney_bbox[5]]
                for slice in kidney_slices:
                    dcm_num = slice.split('/')[-1]
                    final_records.append([pid, uid, 'kidney', dcm_num])
            except Exception as e:
                print(e)
            try:
                liver_bbox = get_bbox_liver(mask_uid_dir)
                liver_slices = dcm_series[liver_bbox[4]: liver_bbox[5]]
                for slice in liver_slices:
                    dcm_num = slice.split('/')[-1]
                    final_records.append([pid, uid, 'liver', dcm_num])
            except Exception as e:
                print(e)
            try:
                spleen_bbox = get_bbox_spleen(mask_uid_dir)
                spleen_slices = dcm_series[spleen_bbox[4]: spleen_bbox[5]]
                for slice in spleen_slices:
                    dcm_num = slice.split('/')[-1]
                    final_records.append([pid, uid, 'spleen', dcm_num])
            except Exception as e:
                print(e)

    save_csv(final_columns, final_records, output_csv)


get_instance_number(input_dcm_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images', input_mask_dir='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_masks', 
                    output_csv='/data/wangfy/organ_instance.csv')