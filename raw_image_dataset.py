import torch
import torch.utils.data as torch_data
import glob
import os
import numpy as np
import rawpy
import cv2


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=0)
    img_shape = im.shape
    H = img_shape[1]
    W = img_shape[2]

    out = np.concatenate((im[:,0:H:2, 0:W:2],
                          im[:,0:H:2, 1:W:2],
                          im[:,1:H:2, 1:W:2],
                          im[:,1:H:2, 0:W:2]), axis=0)
    return out


def input_2_cv(input_tensor):
    input_tensor_numpy=input_tensor.numpy()
    input_tensor_numpy_list=[]
    for tensor in input_tensor_numpy:
        tensor_cv=np.clip(tensor*255.0,0,255)
        tensor_cv=np.uint8(tensor_cv)
        input_tensor_numpy_list.append(tensor_cv)
    return input_tensor_numpy_list
    

def gt_2_cv(gt_tensor):
    gt_tensor_numpy=gt_tensor.numpy()
    gt_tensor_numpy=gt_tensor_numpy.transpose(1,2,0)
    gt_tensor_numpy=np.clip(gt_tensor_numpy*255.0,0,255)
    gt_tensor_numpy = gt_tensor_numpy[:,:,[2,1,0]]
    gt_tensor_numpy=np.uint8(gt_tensor_numpy)
    return gt_tensor_numpy


class ImageDatasetRaw(torch_data.Dataset):
    def __init__(self,input_dir,gt_dir,crop_size=256,phase='train'):
        super(ImageDatasetRaw).__init__()

        self.input_dir=input_dir
        self.gt_dir=gt_dir
        self.phase=phase
        self.crop_size=crop_size

        if self.phase=='train':
            sample_path_list=glob.glob(os.path.join(input_dir,'0*'))
        else:
            sample_path_list=glob.glob(os.path.join(input_dir,'M*'))
        
        self.sample_id_list=sorted([os.path.basename(sample_id) for sample_id in sample_path_list])
       
    def __getitem__(self,index):

        sample_id=self.sample_id_list[index]
        in_path_list=glob.glob(os.path.join(self.input_dir,sample_id,'*.ARW'))
        in_path_1,in_path_2=np.random.choice(in_path_list,2)

        in_raw_1=rawpy.imread(in_path_1)
        in_raw_2=rawpy.imread(in_path_2)
        input_full_size_image_1=pack_raw(in_raw_1)
        input_full_size_image_2=pack_raw(in_raw_2)

        gt_path=glob.glob(os.path.join(self.gt_dir,sample_id,'0001*.ARW'))[0]
        gt_raw=rawpy.imread(gt_path)
        gt_im=gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_im=gt_im.transpose(2,0,1)
        gt_full_size_image=np.float32(gt_im / 65535.0)

        exposure_ratio=np.around(np.mean(gt_full_size_image)/np.mean(input_full_size_image_1))

        input_full_size_image_1=input_full_size_image_1*exposure_ratio
        input_full_size_image_2=input_full_size_image_2*exposure_ratio


        
        
        # crop
        if self.crop_size>0:
            H,W=input_full_size_image_1.shape[1:3]

            if self.phase=='train':
                xx=np.random.randint(0, W-self.crop_size)
                yy=np.random.randint(0, H-self.crop_size)
            else:
                xx=0
                yy=0

            input_patch_1=input_full_size_image_1[:, yy:yy + self.crop_size, xx:xx + self.crop_size]
            input_patch_2=input_full_size_image_2[:, yy:yy + self.crop_size, xx:xx + self.crop_size]
            gt_patch=gt_full_size_image[:, 2*yy:2*yy + 2*self.crop_size, 2*xx:2*xx + 2*self.crop_size]

        else:
            input_patch_1=input_full_size_image_1
            input_patch_2=input_full_size_image_2
            gt_patch=gt_full_size_image

        input_patch_1 = np.minimum(input_patch_1, 1.0)
        input_patch_2 = np.minimum(input_patch_2, 1.0)
        gt_patch = np.minimum(gt_patch, 1.0)

        # random flip and transpose
        if self.phase=='train':
            if np.random.randint(2)==1:
                input_patch_1 = np.flip(input_patch_1, axis=1)
                input_patch_2 = np.flip(input_patch_2, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2)==1:
                input_patch_1 = np.flip(input_patch_1, axis=2)
                input_patch_2 = np.flip(input_patch_2, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
            if np.random.randint(2)==1:
                input_patch_1 = np.transpose(input_patch_1, (0, 2, 1))
                input_patch_2 = np.transpose(input_patch_2, (0, 2, 1))
                gt_patch = np.transpose(gt_patch, (0, 2, 1))
            
        input_patch_1=np.ascontiguousarray(input_patch_1)
        input_patch_2=np.ascontiguousarray(input_patch_2)
        gt_patch=np.ascontiguousarray(gt_patch)
        input_patch_1_torch=torch.from_numpy(input_patch_1)
        input_patch_2_torch=torch.from_numpy(input_patch_2)
        gt_patch_torch=torch.from_numpy(gt_patch)

        return input_patch_1_torch,input_patch_2_torch,gt_patch_torch

    def __len__(self):
        return len(self.sample_id_list)




if __name__=='__main__':

    input_dir = '/media/gtmeng/DataDisk2/Learning-to-See-in-the-Dark/short'
    gt_dir = '/media/gtmeng/DataDisk2/Learning-to-See-in-the-Dark/long'

    dataset=ImageDatasetRaw(input_dir,gt_dir,phase='train')
    dataset[46]
        
    

        
        

