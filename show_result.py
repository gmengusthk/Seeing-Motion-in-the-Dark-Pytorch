import torch
import torch.optim as optim
import torch.nn as nn
from network import SeeMotionInDarkNet
from preprocessed_image_dataset import ImageDataset,ImageDatasetTest
# from raw_image_dataset import ImageDatasetRaw as ImageDataset
import cfg_preprocessed as cfg
# import cfg_raw as cfg
import os
import cv2
import argparse
import numpy as np


def array_2_cv(array):
    array_numpy=array.numpy()
    array_numpy=array_numpy.transpose(1,2,0)
    array_numpy=np.clip(array_numpy*255.0,0,255)
    # array_numpy = array_numpy[:,:,[2,1,0]]
    array_numpy=np.uint8(array_numpy)
    return array_numpy


if __name__=='__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epoch', type=int, default=50,
                        help='input the epoch of the pretrained model (default: 50)')
    args = parser.parse_args()



    use_cuda=torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device=torch.device("cuda" if use_cuda else "cpu")
    print('device:',device)

    dataset=ImageDatasetTest(cfg.input_dir,cfg.gt_dir,crop_size=912,phase='test')

    model = SeeMotionInDarkNet(input_channels=3,demosaic=False)
    model=model.to(device)

    snapshot_path=os.path.join(cfg.model_save_path,'model_%05d.pth'%(args.epoch))
    print('test with %s'%(snapshot_path))
    model.load_state_dict(torch.load(snapshot_path))

    image_write_dir='./viz'

    model.eval()

    for data_idx in range(len(dataset)):
        data_list,target=dataset[data_idx]
        
        target_frame_cv=array_2_cv(target)
        target_write_path=os.path.join(image_write_dir,'image_%03d_gt.png'%(data_idx))
        cv2.imwrite(target_write_path,target_frame_cv)
        print(target_write_path)

        for idx in range(len(data_list)):
            data=data_list[idx].unsqueeze(0)
            data=data.to(device)
            
            output=model(data)
            output_frame_cpu=output[0].cpu().data
            output_frame_cv=array_2_cv(output_frame_cpu)
            output_write_path=os.path.join(image_write_dir,'image_%03d_output_%05d.png'%(data_idx,idx))
            cv2.imwrite(output_write_path,output_frame_cv)
            print(output_write_path)
        
        print('-'*50)