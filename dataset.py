""" train and test dataset

author jundewu
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable,LoadImage
from monai.data import Dataset as monaiDataset
import re
from torch.nn import functional
# prompt
from models.sam.utils.amg import build_all_layer_point_grids

import json
import nibabel as nib

import torchvision

from monai.transforms import Compose#, LoadNifti, EnsureChannelFirst, RescaleIntensity, AddChannel, ScaleIntensity, SpatialPad, SpatialCrop, Orientation, RandCropByPosNegLabel, CropForeground, ToTensor, Lambda
import os
import nibabel as nib
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad

from monai.apps import CrossValidation
from monai.data import CacheDataset, DataLoader, NibabelReader, pad_list_data_collate, partition_dataset, \
    partition_dataset_classes, select_cross_validation_folds
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityD, SpacingD, \
    ToTensorD, ThresholdIntensityD, OrientationD, RandCropByPosNegLabelD, CropForegroundD, apply_transform

from monai.transforms import (
        Activations,
        #AsChannelFirstd,
        AsDiscrete,
        CenterSpatialCropd,
        Compose,
        LoadImaged,
        MapTransform,
        NormalizeIntensityd,
        Orientationd,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropd,
        ResizeWithPadOrCropd,
        Spacingd,
        ToTensord,
        )



class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist() 
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

class PostopPancreas(Dataset):
    def __init__(self, data_path, mode='Training', prompt='click'):
        self.data_path = os.path.join(data_path, "nnUNet_preprocessed", "Dataset001_Postoperative", "nnUNetPlans_3d_fullres")
        self.mode = mode
        self.prompt = prompt
        img_resolution = 128  # Set the new image resolution

        # Load the JSON for the dataset information
        with open(os.path.join(data_path, "nnUNet_preprocessed", "Dataset001_Postoperative", "dataset.json"), 'r') as json_file:
            self.dataset_info = json.load(json_file)
        
        
        all_image_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy') and not f.endswith('_seg.npy')]
        all_image_files.sort()

        if self.mode == 'Training':
            self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) <= 10]
        else:
            self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) > 10]

        self.label_files = [f.replace('.npy', '_seg.npy') for f in self.image_files]
        self.label_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # In your provided dataset, we're ignoring the distinction between 'Training' and 'Testing' for simplicity.
        # Assuming all data is for training. If needed, you can adjust this.
        inout = 1
        point_label = 1

        img_path = os.path.join(self.data_path, self.image_files[index])
        msk_path = os.path.join(self.data_path, self.label_files[index])

        img = np.load(img_path)
        mask = np.load(msk_path)

        # Convert to tensors
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        
        # Pad image and label tensors to a common size

        # Adjust for channel dimension
        # img = img.unsqueeze(0)
        # mask = mask.unsqueeze(0)

        max_depth, max_height, max_width = self.get_max_dimensions()
        img = self.pad_tensor(img, max_depth, max_height, max_width)
        mask = self.pad_tensor(mask, max_depth, max_height, max_width)

       
        img = self.resize_tensor(img)
        mask = self.resize_tensor(mask)

        #select only the odd slices of imagers and masks and make the depth half
        #img = img[:, 1::2, :, :]
        #print(img.shape)
        #mask = mask[:, 1::2, :, :]
        #print(mask.shape)

        #select only the even slices of imagers and masks and make the depth half
        # img = img[:, ::2, :, :]
        # print(img.shape)
        # mask = mask[:, ::2, :, :]
        # print(mask.shape)

        img, mask = self._get_transforms(train_set=True, data_augmentation=True)(
            {
                'image': img,
                'label': mask
            }
        )

        image_meta_dict = {'filename_or_obj': self.image_files[index].replace('.npy', '')}
        return {
            'image': img,
            'label': mask,
            'p_label':1,
            'image_meta_dict': image_meta_dict,
        }
    
    def resize_tensor(self, tensor):
        # Calculate proportional depth
        _, original_depth, original_height, _ = tensor.shape
        new_depth = 148  # The new depth you want

    # Calculate the scaling factors for height and width
        scale_factor_height = 128 / 518
        scale_factor_width = 128 / 518

    # Resize the tensor
        tensor = tensor.unsqueeze(0)
        tensor = functional.interpolate(tensor, size=(new_depth, 128, 128), mode='trilinear', align_corners=True)
        return tensor.squeeze(0)

    

    def get_max_dimensions(self):
        max_depth, max_height, max_width = 0, 0, 0
        for image_filename in self.image_files:  # <- Change made here
            image_filepath = os.path.join(self.data_path, image_filename)
            image_data = np.load(image_filepath)  # Loading .npy file
            _, depth, height, width = image_data.shape
            max_depth = max(max_depth, depth)
            max_height = max(max_height, height)
            max_width = max(max_width, width)
        return max_depth, max_height, max_width

    @staticmethod
    def pad_tensor(tensor, max_depth, max_height, max_width):
        depth, height, width = tensor.shape[-3:]
        pad_depth = max_depth - depth
        pad_height = max_height - height
        pad_width = max_width - width

        # Padding format: (left, right, top, bottom, front, back)
        padding = (0, pad_width, 0, pad_height, 0, pad_depth)
        return pad(tensor, padding, mode='constant', value=0)

    @staticmethod
    def _get_transforms(train_set: bool, data_augmentation: bool, label_spacing: str = 'nearest'):
        keys = ('image', 'label')
        h, w, d = 128, 128, 64

        data_augmentation_transforms = Compose([
            RandCropByPosNegLabelD(keys, label_key=keys[1],
                                spatial_size=(h, w, d),
                                pos=0.5, neg=0.5, num_samples=1),
        ])

        discretize_labels = Compose([
            ThresholdIntensityD(keys[1], threshold=0.5, above=True, cval=0.0),
            ThresholdIntensityD(keys[1], threshold=0.5, above=False, cval=1.0),
        ])

        return Compose([
            # LoadImageD(keys, reader=NibabelReader()),
            EnsureChannelFirstD(keys),
            OrientationD(keys, axcodes='RAS'),
            ThresholdIntensityD(keys[0], threshold=-100, above=True, cval=-100),
            ThresholdIntensityD(keys[0], threshold=240, above=False, cval=240),
            ScaleIntensityD(keys[0]),
            SpacingD(keys, pixdim=(1., 1., 1.), mode=('bilinear', label_spacing)),
            CropForegroundD(
                keys,
                source_key='label',
                select_fn=lambda x: np.full(shape=x.shape, fill_value=np.any(np.any(x > 0, axis=-4), axis=(-2, -3))),
                margin=(0, 0, 10),
                k_divisible=d
            ),
            CropForegroundD(keys, source_key='image', k_divisible=64),
            discretize_labels if label_spacing == 'bilinear' else Compose([]),
            data_augmentation_transforms if (data_augmentation and train_set) else Compose([]),
            ToTensorD(keys),
        ])


class PostopPancreasRaw(Dataset):
    def __init__(self, data_path, mode='Training', prompt='click'):
        self.data_path = os.path.join(data_path, "nnUNet_raw", "Dataset001_Postoperative")
        self.image_path = os.path.join(data_path, "imagesTr")
        self.label_path = os.path.join(data_path, "labelsTr")
        self.mode = mode
        self.prompt = prompt

        all_image_files = [f for f in os.listdir(self.image_path) if f.endswith('.nii.gz')]
        all_image_files.sort()

        if self.mode == 'Training':
            self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) <= 10]
        else:
            self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) > 10]

        self.label_files = [f.replace('_0000.nii.gz', '.nii.gz') for f in self.image_files]
        self.label_files.sort()

        keys = ('image', 'label')
        label_spacing = 'nearest'
        d = 64
        data_augmentation_transforms = Compose([
            # Add your data augmentation transforms here
        ])
        # Define the discretize_labels transform
        discretize_labels = Compose([
            # Add your discretize_labels transforms here
        ])
        self.transform = Compose([
            LoadImageD(keys, reader=NibabelReader()),
            EnsureChannelFirstD(keys),
            OrientationD(keys, axcodes='RAS'),
            ResizeWithPadOrCropd(keys, spatial_size=(128, 128, 100)),
            ThresholdIntensityD(keys[0], threshold=-100, above=True, cval=-100),
            ThresholdIntensityD(keys[0], threshold=240, above=False, cval=240),
            ScaleIntensityD(keys[0]),
            SpacingD(keys, pixdim=(1., 1., 1.), mode=('bilinear', label_spacing)),
            CropForegroundD(
                keys,
                source_key='label',
                select_fn=lambda x: np.full(shape=x.shape, fill_value=np.any(np.any(x > 0, axis=-4), axis=(-2, -3))),
                margin=(0, 0, 10),
                k_divisible=d
            ),
            CropForegroundD(keys, source_key='image', k_divisible=64),
            discretize_labels if label_spacing == 'bilinear' else Compose([]),
            data_augmentation_transforms,
            ResizeWithPadOrCropd(keys, spatial_size=(128, 128, 64)),
            ToTensorD(keys),
        ])
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_filename = self.image_files[index]
        label_filename = self.label_files[index]

        image_filepath = os.path.join(self.image_path, image_filename)
        label_filepath = os.path.join(self.label_path, label_filename)

        transformed = self.transform({
            'image': image_filepath,
            'label': label_filepath
        })

        image_meta_dict = {'filename_or_obj': image_filename}

        return {
            'image': transformed["image"],
            'label': transformed["label"],
            'p_label': 1,  # Just a placeholder
            'image_meta_dict': image_meta_dict
        }

    # def get_max_dimensions(self):
    #     max_depth, max_height, max_width = 0, 0, 0
    #     for image_filename in self.image_files:
    #         image_filepath = os.path.join(self.image_path, image_filename)
    #         image_data = nib.load(image_filepath).get_fdata()
    #         depth, height, width = image_data.shape
    #         max_depth = max(max_depth, depth)
    #         max_height = max(max_height, height)
    #         max_width = max(max_width, width)
    #     return max_depth, max_height, max_width

    # @staticmethod
    # def pad_tensor(tensor, max_depth, max_height, max_width):
    #     depth, height, width = tensor.shape[-3:]
    #     pad_depth = max_depth - depth
    #     pad_height = max_height - height
    #     pad_width = max_width - width

    #     # Padding format: (left, right, top, bottom, front, back)
    #     padding = (0, pad_width, 0, pad_height, 0, pad_depth)
    #     return pad(tensor, padding, mode='constant', value=0)

def get_postoppancreasraw(data_path, mode='Training', prompt='click'):
# class PostopPancreasRaw():
#     def __init__(self, data_path, mode='Training', prompt='click'):
    data_path = os.path.join(data_path, "nnUNet_raw", "Dataset001_Postoperative")
    image_path = os.path.join(data_path, "imagesTr")
    label_path = os.path.join(data_path, "labelsTr")
    mode = mode
    prompt = prompt

    all_image_files = [f for f in os.listdir(image_path) if f.endswith('.nii.gz')]
    all_image_files.sort()

    if mode == 'Training':
        image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) <= 10]
    else:
        image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) > 10]

    label_files = [f.replace('_0000.nii.gz', '.nii.gz') for f in image_files]
    label_files.sort()

    image_files = [os.path.join(image_path, f) for f in image_files]
    label_files = [os.path.join(label_path, f) for f in label_files]
    
    image_meta_dicts = [
        {
            'filename_or_obj': image_files[i]
        } for i in range(len(image_files))
    ]

    image_label_dict = [{'image':image_name, 'label':label_name, 'image_meta_dict': image_meta_dict, 'p_label':1
                         } for image_name, label_name, image_meta_dict in zip(image_files, label_files, image_meta_dicts)
                         ]
    
    keys = ('image', 'label')
    label_spacing = 'nearest'
    d = 64
    data_augmentation_transforms = Compose([
        # Add your data augmentation transforms here
    ])
    # Define the discretize_labels transform
    discretize_labels = Compose([
        # Add your discretize_labels transforms here
    ])
    
    transform = Compose([
        LoadImageD(keys, reader=NibabelReader()),
        EnsureChannelFirstD(keys),
        OrientationD(keys, axcodes='RAS'),
        ResizeWithPadOrCropd(keys, spatial_size=(128, 128, 64)),
        ThresholdIntensityD(keys[0], threshold=-100, above=True, cval=-100),
        ThresholdIntensityD(keys[0], threshold=240, above=False, cval=240),
        ScaleIntensityD(keys[0]),
        SpacingD(keys, pixdim=(1., 1., 1.), mode=('bilinear', label_spacing)),
        CropForegroundD(
            keys,
            source_key='label',
            select_fn=lambda x: np.full(shape=x.shape, fill_value=np.any(np.any(x > 0, axis=-4), axis=(-2, -3))),
            margin=(0, 0, 10),
            k_divisible=d
        ),
        CropForegroundD(keys, source_key='image', k_divisible=64),
        discretize_labels if label_spacing == 'bilinear' else Compose([]),
        data_augmentation_transforms,
        ResizeWithPadOrCropd(keys, spatial_size=(128, 128, 64)),
        ToTensorD(keys),
    ])

    return monaiDataset(data=image_label_dict, transform=transform)

# class PostopPancreas(Dataset):
#     def __init__(self, data_path, mode='Training', prompt='click'):
#         self.data_path = os.path.join(data_path, "nnUNet_preprocessed", "Dataset001_Postoperative", "nnUNetPlans_3d_fullres")
#         self.mode = mode
#         self.prompt = prompt
#         img_resolution = 128  # Set the new image resolution

#         # Load the JSON for the dataset information
#         with open(os.path.join(data_path, "nnUNet_preprocessed", "Dataset001_Postoperative", "dataset.json"), 'r') as json_file:
#             self.dataset_info = json.load(json_file)
        
        
#         all_image_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy') and not f.endswith('_seg.npy')]
#         all_image_files.sort()

#         if self.mode == 'Training':
#             self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) <= 10]
#         else:
#             self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) > 10]

#         self.label_files = [f.replace('.npy', '_seg.npy') for f in self.image_files]
#         self.label_files.sort()

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, index):
#         # In your provided dataset, we're ignoring the distinction between 'Training' and 'Testing' for simplicity.
#         # Assuming all data is for training. If needed, you can adjust this.
#         inout = 1
#         point_label = 1

#         img_path = os.path.join(self.data_path, self.image_files[index])
#         msk_path = os.path.join(self.data_path, self.label_files[index])

#         img = np.load(img_path)
#         mask = np.load(msk_path)



#         # Convert to tensors
#         img = torch.from_numpy(img).float()
#         mask = torch.from_numpy(mask).float()
        
#         # Pad image and label tensors to a common size

#         # Adjust for channel dimension
#         # img = img.unsqueeze(0)
#         # mask = mask.unsqueeze(0)

#         max_depth, max_height, max_width = self.get_max_dimensions()
#         img = self.pad_tensor(img, max_depth, max_height, max_width)
#         mask = self.pad_tensor(mask, max_depth, max_height, max_width)

       
#         img = self.resize_tensor(img)
#         mask = self.resize_tensor(mask)

#         #select only the odd slices of imagers and masks and make the depth half
#         #img = img[:, 1::2, :, :]
#         #print(img.shape)
#         #mask = mask[:, 1::2, :, :]
#         #print(mask.shape)

#         #select only the even slices of imagers and masks and make the depth half
#         # img = img[:, ::2, :, :]
#         # print(img.shape)
#         # mask = mask[:, ::2, :, :]
#         # print(mask.shape)
        

#         image_meta_dict = {'filename_or_obj': self.image_files[index].replace('.npy', '')}
#         return {
#             'image': img,
#             'label': mask,
#             'p_label':1,
#             'image_meta_dict': image_meta_dict,
#         }

#     def resize_tensor(self, tensor):
#         # Calculate proportional depth
#         _, original_depth, original_height, _ = tensor.shape
#         new_depth = 148  # The new depth you want

#     # Calculate the scaling factors for height and width
#         scale_factor_height = 128 / 518
#         scale_factor_width = 128 / 518

#     # Resize the tensor
#         tensor = tensor.unsqueeze(0)
#         tensor = functional.interpolate(tensor, size=(new_depth, 128, 128), mode='trilinear', align_corners=True)
#         return tensor.squeeze(0)

    

#     def get_max_dimensions(self):
#         max_depth, max_height, max_width = 0, 0, 0
#         for image_filename in self.image_files:  # <- Change made here
#             image_filepath = os.path.join(self.data_path, image_filename)
#             image_data = np.load(image_filepath)  # Loading .npy file
#             _, depth, height, width = image_data.shape
#             max_depth = max(max_depth, depth)
#             max_height = max(max_height, height)
#             max_width = max(max_width, width)
#         return max_depth, max_height, max_width

#     @staticmethod
#     def pad_tensor(tensor, max_depth, max_height, max_width):
#         depth, height, width = tensor.shape[-3:]
#         pad_depth = max_depth - depth
#         pad_height = max_height - height
#         pad_width = max_width - width

#         # Padding format: (left, right, top, bottom, front, back)
#         padding = (0, pad_width, 0, pad_height, 0, pad_depth)
#         return pad(tensor, padding, mode='constant', value=0)


# class PancreasRawDataset(Dataset):
#     def __init__(self, data_path, mode='Training', prompt='click'):
#         self.data_path = os.path.join(data_path, "nnUNet_raw", "Dataset001_Postoperative")
#         self.image_path = os.path.join(self.data_path, "imagesTr")
#         self.label_path = os.path.join(self.data_path, "labelsTr")
#         self.mode = mode
#         self.prompt = prompt
        
#         all_image_files = [f for f in os.listdir(self.image_path) if f.endswith('.nii.gz')]
#         all_image_files.sort()

#         if self.mode == 'Training':
#             self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) <= 10]
#         else:
#             self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) > 10]

#         self.label_files = [f.replace('_0000.nii.gz', '.nii.gz') for f in self.image_files]
#         self.label_files.sort()

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, index):
#         # Load image
#         image_filepath = os.path.join(self.image_path, self.image_files[index])
#         image_data = nib.load(image_filepath).get_fdata()
#         image_tensor = torch.tensor(image_data).unsqueeze(0).float()
        
#         # Load label
#         label_filename = self.image_files[index].split('_0000')[0] + '.nii.gz'
#         label_filepath = os.path.join(self.label_path, label_filename)
#         label_data = nib.load(label_filepath).get_fdata()
#         label_tensor = torch.tensor(label_data).unsqueeze(0).float()
        
#         # Pad image and label tensors to a common size
#         max_depth, max_height, max_width = self.get_max_dimensions()
#         image_tensor = self.pad_tensor(image_tensor, max_depth, max_height, max_width)
#         label_tensor = self.pad_tensor(label_tensor, max_depth, max_height, max_width)

#         # Create a dictionary for metadata. For now, I'm just adding the file name.
#         # You can expand on this with more metadata if needed.
#         image_meta_dict = {
#         'filename_or_obj': self.image_files[index]
#         }


#         # Return data
#         return {
#             'image': image_tensor,
#             'label': label_tensor,
#             'p_label': 1,  # Just a placeholder  # Just a placeholder
#             'image_meta_dict': image_meta_dict
#         }

#     def get_max_dimensions(self):
#         max_depth, max_height, max_width = 0, 0, 0
#         for image_filename in self.image_files:
#             image_filepath = os.path.join(self.image_path, image_filename)
#             image_data = nib.load(image_filepath).get_fdata()
#             depth, height, width = image_data.shape
#             max_depth = max(max_depth, depth)
#             max_height = max(max_height, height)
#             max_width = max(max_width, width)
#         return max_depth, max_height, max_width

#     @staticmethod
#     def pad_tensor(tensor, max_depth, max_height, max_width):
#         depth, height, width = tensor.shape[-3:]
#         pad_depth = max_depth - depth
#         pad_height = max_height - height
#         pad_width = max_width - width

#         # Padding format: (left, right, top, bottom, front, back)
#         padding = (0, pad_width, 0, pad_height, 0, pad_depth)
#         return pad(tensor, padding, mode='constant', value=0)

    #def generate_prompt(self, image_data):
        # Your logic to generate the prompt tensor with dimensions [batch_size, number_of_points, depth]
    #    b, h, w, d = image_data.shape
        # Example: Create a random prompt tensor
    #    pt = torch.rand((b, 2, d))
    #    return pt
# import os
# import re
# import json
# import numpy as np
# import torch
# import torchvision.transforms.functional as F
# from torch.utils.data import Dataset
# import nibabel as nib
# from torch.nn.functional import pad



# class PostopPancreas(Dataset):
#     def __init__(self, data_path, mode='Training', prompt='click'):
#         self.data_path = os.path.join(data_path, "nnUNet_preprocessed", "Dataset001_Postoperative", "nnUNetPlans_3d_fullres")
#         self.mode = mode
#         self.prompt = prompt
#         self.img_resolution = (256, 256, 148)  # Set the new image resolution

#         # Load the JSON for the dataset information
#         with open(os.path.join(data_path, "nnUNet_preprocessed", "Dataset001_Postoperative", "dataset.json"), 'r') as json_file:
#             self.dataset_info = json.load(json_file)
        
#         all_image_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy') and not f.endswith('_seg.npy')]
#         all_image_files.sort()

#         if self.mode == 'Training':
#             self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) <= 10]
#         else:
#             self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) > 10]

#         self.label_files = [f.replace('.npy', '_seg.npy') for f in self.image_files]
#         self.label_files.sort()

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, index):
#         inout = 1
#         point_label = 1

#         img_path = os.path.join(self.data_path, self.image_files[index])
#         msk_path = os.path.join(self.data_path, self.label_files[index])

#         img = np.load(img_path)
#         mask = np.load(msk_path)

#         img = torch.from_numpy(img).float()
#         mask = torch.from_numpy(mask).float()

#         # Pad image and label tensors to a common size
#         max_depth, max_height, max_width = self.get_max_dimensions()
#         img = self.pad_tensor(img, max_depth, max_height, max_width)
#         mask = self.pad_tensor(mask, max_depth, max_height, max_width)

#         img = self.resize_tensor(img)
#         mask = self.resize_tensor(mask)

#         image_meta_dict = {'filename_or_obj': self.image_files[index].replace('.npy', '')}
#         return {
#             'image': img,
#             'label': mask,
#             'p_label': 1,
#             'image_meta_dict': image_meta_dict,
#         }

#     def prepare_image(self, image):
#         trans = torchvision.transforms.Compose([torchvision.transforms.Resize(self.img_resolution)])
#         image = torch.as_tensor(image).cuda()
#         return trans(image.permute(2, 0, 1))

#     def resize_tensor(self, tensor):
#         # Calculate new dimensions
#         new_depth, new_height, new_width = self.img_resolution

#         # Resize the tensor
#         tensor = tensor.unsqueeze(0)
#         tensor = F.interpolate(tensor, size=(new_depth, new_height, new_width), mode='trilinear', align_corners=True)
#         return tensor.squeeze(0)

#     def get_max_dimensions(self):
#         max_depth, max_height, max_width = 0, 0, 0
#         for image_filename in self.image_files:
#             image_filepath = os.path.join(self.data_path, image_filename)
#             image_data = np.load(image_filepath)
#             _, depth, height, width = image_data.shape
#             max_depth = max(max_depth, depth)
#             max_height = max(max_height, height)
#             max_width = max(max_width, width)
#         return max_depth, max_height, max_width

#     @staticmethod
#     def pad_tensor(tensor, max_depth, max_height, max_width):
#         depth, height, width = tensor.shape[-3:]
#         pad_depth = max_depth - depth
#         pad_height = max_height - height
#         pad_width = max_width - width

#         # Padding format: (left, right, top, bottom, front, back)
#         padding = (0, pad_width, 0, pad_height, 0, pad_depth)
#         return pad(tensor, padding, mode='constant', value=0)
class PublicPancreasRaw(Dataset):
    def __init__(self, data_path, mode='Training', prompt='click', data_augmentation=False):
        self.data_path = os.path.join(data_path, "nnUNet_raw", "Dataset007_Pancreas")
        self.image_path = os.path.join(data_path, "imagesTr")
        self.label_path = os.path.join(data_path, "labelsTr")
        self.mode = mode
        self.prompt = prompt

        all_image_files = [f for f in os.listdir(self.image_path) if f.endswith('.nii.gz')]
        all_image_files.sort()

        if self.mode == 'Training':
            self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) <= 200]
        else:
            self.image_files = [f for f in all_image_files if int(re.search(r"\d+", f).group(0)) > 200]

        self.label_files = [f.replace('_0000.nii.gz', '.nii.gz') for f in self.image_files]
        self.label_files.sort()

        keys = ('image', 'label')
        label_spacing = 'nearest'
        h, w, d = 96, 96, 100
        data_augmentation_transforms = Compose([
            RandCropByPosNegLabelD(keys, label_key=keys[1],
                                spatial_size=(h, w, d),
                                pos=0.5, neg=0.5, num_samples=1),
        ])

        discretize_labels = Compose([
            ThresholdIntensityD(keys[1], threshold=0.5, above=True, cval=0.0),
            ThresholdIntensityD(keys[1], threshold=0.5, above=False, cval=1.0),
        ])
        self.transform = Compose([
            LoadImageD(keys, reader=NibabelReader()),
            EnsureChannelFirstD(keys),
            OrientationD(keys, axcodes='RAS'),
            ResizeWithPadOrCropd(keys, spatial_size=(96, 96, 100)),
            ThresholdIntensityD(keys[0], threshold=-100, above=True, cval=-100),
            ThresholdIntensityD(keys[0], threshold=240, above=False, cval=240),
            ScaleIntensityD(keys[0]),
            SpacingD(keys, pixdim=(1., 1., 1.), mode=('bilinear', label_spacing)),
            CropForegroundD(
                keys,
                source_key='label',
                select_fn=lambda x: np.full(shape=x.shape, fill_value=np.any(np.any(x > 0, axis=-4), axis=(-2, -3))),
                margin=(0, 0, 10),
                k_divisible=d
            ),
            CropForegroundD(keys, source_key='image', k_divisible=100),
            discretize_labels if label_spacing == 'bilinear' else Compose([]),
            data_augmentation_transforms if (data_augmentation and self.mode == 'Training') else Compose([]),
            ResizeWithPadOrCropd(keys, spatial_size=(96, 96, 100)),
            ToTensorD(keys),
        ])
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_filename = self.image_files[index]
        label_filename = self.label_files[index]

        image_filepath = os.path.join(self.image_path, image_filename)
        label_filepath = os.path.join(self.label_path, label_filename)

        transformed = self.transform({
            'image': image_filepath,
            'label': label_filepath
        })

        if isinstance(transformed, list):
            transformed = transformed[0]

        image_meta_dict = {'filename_or_obj': image_filename}

        return {
            'image': transformed["image"],
            'label': transformed["label"],
            'p_label': 1,  # Just a placeholder
            'image_meta_dict': image_meta_dict
        }