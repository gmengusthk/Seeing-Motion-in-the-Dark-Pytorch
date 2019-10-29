import torch
import torch.optim as optim
import torch.nn as nn
from network import SeeMotionInDarkNet
from preprocessed_image_dataset import ImageDataset
import cfg
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

    dataset=ImageDataset(cfg.input_dir,cfg.gt_dir,crop_size=768,phase='test')

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1, shuffle=False,
                                            num_workers=1, pin_memory=True)


    model = SeeMotionInDarkNet(input_channels=3,demosaic=False)
    model=model.to(device)

    snapshot_path='./snapshots/model_%05d.pth'%(args.epoch)
    print('test with %s'%(snapshot_path))
    model.load_state_dict(torch.load(snapshot_path))

    image_write_dir='./viz'

    model.eval()

    for batch_idx, (data_1,data_2,target) in enumerate(data_loader):
        data_1,data_2=data_1.to(device),data_2.to(device)
        if target is not None:
            target=target.to(device)

        output_1=model(data_1)
        output_2=model(data_2)
        batch_size=output_1.size(0)
        for b_idx in range(batch_size):
            output_1_frame_cpu=output_1[b_idx].cpu().data
            output_2_frame_cpu=output_2[b_idx].cpu().data

            output_1_frame_cv=array_2_cv(output_1_frame_cpu)
            output_2_frame_cv=array_2_cv(output_2_frame_cpu)

            output_1_write_path=os.path.join(image_write_dir,'image_%03d_%2d_output_%05d_1.png'%(batch_idx,b_idx,args.epoch))
            output_2_write_path=os.path.join(image_write_dir,'image_%03d_%2d_output_%05d_2.png'%(batch_idx,b_idx,args.epoch))
            cv2.imwrite(output_1_write_path,output_1_frame_cv)
            cv2.imwrite(output_2_write_path,output_2_frame_cv)
            print(output_1_write_path)
            print(output_2_write_path)

            if target is not None:
                target_frame_cpu=target[b_idx].cpu().data
                target_frame_cv=array_2_cv(target_frame_cpu)
                target_write_path=os.path.join(image_write_dir,'image_%03d_%2d_gt.png'%(batch_idx,b_idx))
                cv2.imwrite(target_write_path,target_frame_cv)
                print(target_write_path)
            else:
                print('%d %d No GT!'%(batch_idx,b_idx))
            
            print('-'*50)