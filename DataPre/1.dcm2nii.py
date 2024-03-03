import os
import lmdb
import json
import datetime
import numpy as np
import traceback
import SimpleITK as sitk
from pydicom import dicomio
from multiprocessing import Pool, cpu_count


def load_ct_from_dicom(dcm_path, sort_by_distance=True):
    class DcmInfo(object):
        def __init__(self, dcm_path, series_instance_uid, acquisition_number, sop_instance_uid, instance_number,
                     image_orientation_patient, image_position_patient):
            super(DcmInfo, self).__init__()
            self.dcm_path = dcm_path
            self.series_instance_uid = series_instance_uid
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
    dcm_infos = []
    files = os.listdir(dcm_path)
    ''' 1. For each dcm file, get meta info by pydicom library, 
        2. Calculate the distance mainly with the parameters of "ImageOrientationPatient" and "ImagePositionPatient",
        3. Sort dcm files by distance
        4. Get itk image with the sorted dcm files by SimpleITK reader
    '''
    for file in files:
        file_path = os.path.join(dcm_path, file)

        dcm = dicomio.read_file(file_path, force=True)
        _series_instance_uid = dcm.SeriesInstanceUID
        # _acquisition_number = dcm.AcquisitionNumber
        _acquisition_number = None
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

    reader.SetFileNames(dcm_series)
    sitk_image = reader.Execute()
    return sitk_image


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)
        

class Trans_Dicom_2_NII_And_Save_DB(object):
    def __init__(self, in_path, out_path, out_ct_info):
        '''
        :param in_path: path of dicom
        :param out_path: path of nii.gz
        :param out_ct_info: path to save ct info, raw_spacing/raw_origin/raw_size all in x/y/z order.
                            with key: uid,
                                 value: {
                                    "raw_spacing": [0.6, 0.6, 1.0],
                                    "raw_origin": [-20.0, -20.0, 20.0],
                                    "raw_size": [100.0, 100.0, 100.0]}
        '''
        self.in_path = in_path
        self.out_path = out_path
        self.out_ct_info = out_ct_info

        # assert os.path.exists(DICOM_DIR)

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def __call__(self):
        pids = os.listdir(self.in_path)
        pool = Pool(int(cpu_count() * 0.7))

        for pid in pids:
            pid_dir = os.path.join(self.in_path, pid)
            uids = os.listdir(pid_dir)
            for uid in uids:
                if not os.path.exists(os.path.join(self.out_path, uid + '.nii.gz')):
                    try:
                        pool.apply_async(self._single_transform, (pid, uid,))
                    except Exception as err:
                        traceback.print_exc()
                        print('Transform dicom to nii.gz throws exception %s, with series uid %s!' % (err, uid))

        pool.close()
        pool.join()

    def _single_transform(self, pid, uid):
        print('Processing series uid %s' % uid)

        nii_file = os.path.join(self.out_path, uid + '.nii.gz')

        dcm_folder = os.path.join(self.in_path, pid, uid)

        # convert dicom to nii
        try:
            itk_image = load_ct_from_dicom(dcm_folder)
        except Exception as err:
            print('!!!!! Read %s throws exception %s.' % (uid, err))
            return
        try:
            sitk.WriteImage(itk_image, nii_file)
        except Exception as err:
            print('!!!!! Write %s throws exception %s.' % (uid, err))
            return

        # save ct info to db
        env = lmdb.open(self.out_ct_info, map_size=int(1e9))
        txn = env.begin(write=True)

        info_dict = dict()
        info_dict['raw_origin'] = np.array(itk_image.GetOrigin())
        info_dict['raw_spacing'] = np.array(itk_image.GetSpacing())
        info_dict['raw_size'] = np.array(itk_image.GetSize())

        txn.put(key=str(uid).encode(), value=json.dumps(info_dict, cls=MyEncoder).encode())

        txn.commit()
        env.close()


if __name__ == '__main__':
    dcm2nii = Trans_Dicom_2_NII_And_Save_DB(in_path='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images', 
                                            out_path='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_nii', 
                                            out_ct_info='/data/wangfy/rsna-2023-abdominal-trauma-detection/train_images_nii_db_info')
    dcm2nii()