import torch
import torch.utils.data as torch_data
import glob
import os
import numpy as np
import cv2


class ImageDataset(torch_data.Dataset):
    def __init__(self,input_dir,gt_dir,crop_size=256,phase='train'):
        super(ImageDataset).__init__()

        self.input_dir=input_dir
        self.gt_dir=gt_dir
        self.phase=phase
        self.crop_size=crop_size

        if self.phase=='train':
            sample_path_list=glob.glob(os.path.join(gt_dir,'0*'))
        else:
            sample_path_list=glob.glob(os.path.join(gt_dir,'M*'))
        
        self.sample_id_list=sorted([os.path.basename(sample_id) for sample_id in sample_path_list])
       
    def __getitem__(self,index):

        sample_id=self.sample_id_list[index]
        in_path_list=glob.glob(os.path.join(self.input_dir,sample_id,'*.png'))
        in_path_1,in_path_2=np.random.choice(in_path_list,2)
        in_image_1=cv2.imread(in_path_1,cv2.IMREAD_UNCHANGED).transpose(2,0,1)
        in_image_2=cv2.imread(in_path_2,cv2.IMREAD_UNCHANGED).transpose(2,0,1)
        in_image_1=np.float32(in_image_1)/65535.0
        in_image_2=np.float32(in_image_2)/65535.0

        gt_path_list=glob.glob(os.path.join(self.gt_dir,sample_id,'half0001*.png'))
        if len(gt_path_list)>0:
            gt_path=gt_path_list[0]
            self.gt_exists=True
        else:
            gt_path=in_path_1
            self.gt_exists=False
        gt_image=cv2.imread(gt_path,cv2.IMREAD_UNCHANGED).transpose(2,0,1)
        gt_image=np.float32(gt_image)/65535.0
        
        # crop
        if self.crop_size>0:
            H,W=in_image_1.shape[1:3]

            if self.phase=='train':
                xx=np.random.randint(0, W-self.crop_size)
                yy=np.random.randint(0, H-self.crop_size)
            else:
                xx=0
                yy=0

            input_patch_1=in_image_1[:, yy:yy + self.crop_size, xx:xx + self.crop_size]
            input_patch_2=in_image_2[:, yy:yy + self.crop_size, xx:xx + self.crop_size]
            gt_patch=gt_image[:, yy:yy + self.crop_size, xx:xx + self.crop_size]

        else:
            input_patch_1=in_image_1
            input_patch_2=in_image_2
            gt_patch=gt_image

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



class ImageDatasetTest(torch_data.Dataset):
    def __init__(self,input_dir,gt_dir,crop_size=256,phase='train'):
        super(ImageDatasetTest).__init__()

        self.input_dir=input_dir
        self.gt_dir=gt_dir
        self.phase=phase
        self.crop_size=crop_size

        if self.phase=='train':
            sample_path_list=glob.glob(os.path.join(gt_dir,'0*'))
        else:
            sample_path_list=glob.glob(os.path.join(gt_dir,'M*'))
        
        self.sample_id_list=sorted([os.path.basename(sample_id) for sample_id in sample_path_list])
       
    def __getitem__(self,index):

        sample_id=self.sample_id_list[index]
        in_path_list=glob.glob(os.path.join(self.input_dir,sample_id,'*.png'))
        in_path_list=sorted(in_path_list)

        input_patch_torch_list=[]
        for in_path in in_path_list:
            in_image=cv2.imread(in_path,cv2.IMREAD_UNCHANGED).transpose(2,0,1)
            in_image=np.float32(in_image)/65535.0
        
            # crop
            if self.crop_size>0:
                input_patch=in_image[:, 0:self.crop_size, 0:self.crop_size]
            else:
                input_patch=in_image

            input_patch = np.minimum(input_patch, 1.0)
            input_patch=np.ascontiguousarray(input_patch)
            input_patch_torch=torch.from_numpy(input_patch)
            input_patch_torch_list.append(input_patch_torch)


        gt_path_list=glob.glob(os.path.join(self.gt_dir,sample_id,'half0001*.png'))
        if len(gt_path_list)>0:
            gt_path=gt_path_list[0]
        else:
            gt_path=in_path

        gt_image=cv2.imread(gt_path,cv2.IMREAD_UNCHANGED).transpose(2,0,1)
        gt_image=np.float32(gt_image)/65535.0

        if self.crop_size>0:
            gt_patch=gt_image[:, 0:self.crop_size, 0:self.crop_size]
        else:
            gt_patch=gt_image
        
        gt_patch = np.minimum(gt_patch, 1.0)
        gt_patch=np.ascontiguousarray(gt_patch)
        gt_patch_torch=torch.from_numpy(gt_patch)

        return input_patch_torch_list, gt_patch_torch

    def __len__(self):
        return len(self.sample_id_list)




if __name__=='__main__':

    input_dir = '/media/gtmeng/DataDisk2/Learning-to-See-in-the-Dark-pre-processed/VBM4D_rawRGB'
    gt_dir = '/media/gtmeng/DataDisk2/Learning-to-See-in-the-Dark-pre-processed/long'

    dataset=ImageDataset(input_dir,gt_dir,phase='train')
    dataset[26]

        
    

        
        

