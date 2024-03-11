import torch
import time
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom

# inputs = torch.rand(1, 1, 144, 144, 144).float().cuda()


def resize_bak(image, pad_value=-150.0):
    _, depth, height, width = image.shape
    # input_d = self.input_size[0]
    # input_h = self.input_size[1]
    # input_w = self.input_size[2]
    max_input = 144
    # max_img = max(depth, height, width)
    # scale = max_input / max_img
    scale_z = max_input / depth
    scale_y = max_input / height
    scale_x = max_input / width
    image_resize = zoom(image, [1, scale_z, scale_y, scale_x], order=1, cval=pad_value)
    return image_resize

def resize_pad(image, pad_value=-150.0):
    _, depth, height, width = image.shape
    # input_d = self.input_size[0]
    # input_h = self.input_size[1]
    # input_w = self.input_size[2]
    max_input = 144
    max_img = max(depth, height, width)
    scale = max_input / max_img
    image_resize = zoom(image, [1, scale, scale, scale], order=1, cval=pad_value)

    image_resize_pad = pad(image_resize)
    return image_resize_pad


def pad(image, pad_value=-150.0):
    _, depth, height, width = image.shape
    input_d = 144
    input_h = 144
    input_w = 144
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


def norm(image, hu_min=-150.0, hu_max=250.0):
    image = (np.clip(image.astype(np.float32), hu_min, hu_max) - hu_min) / float(hu_max - hu_min)
    return (image - 0.5) * 2.0

itk_image = sitk.ReadImage("21015_croped.nii.gz")
npy_image = sitk.GetArrayFromImage(itk_image)[np.newaxis, ...]
print(f"npy_image: {npy_image.shape}")
npy_image_resize = resize_pad(npy_image)
npy_image_norm = norm(npy_image_resize)

inputs = torch.from_numpy(npy_image_norm).unsqueeze(0).cuda()
print(f"inputs: {inputs.size()}")


tic = time.time()
model = torch.jit.load("./epoch_82.pt")

output = model(inputs)
toc = time.time()
print(toc - tic)
print(output)