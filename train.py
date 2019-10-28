import torch
import torch.optim as optim
import torch.nn as nn
from network import SeeMotionInDarkNet
from preprocessed_image_dataset import ImageDataset
from vgg import VGG19_Extractor
import cfg as cfg
import os

from torch.utils.tensorboard import SummaryWriter

def train(unet_model, vgg_model, device, train_loader, loss_function, optimizer, epoch, tb_writer):
    unet_model.train()

    batch_num=len(train_loader)
    sample_cnt=0.0
    loss_acc=0.0

    for batch_idx, (data_1, data_2, target) in enumerate(train_loader):
        data_1,data_2,target=data_1.to(device),data_2.to(device),target.to(device)

        output_1=unet_model(data_1)
        output_2=unet_model(data_2)
        output_1_vgg_1,output_1_vgg_2,output_1_vgg_3,output_1_vgg_4=vgg_model(output_1)
        output_2_vgg_1,output_2_vgg_2,output_2_vgg_3,output_2_vgg_4=vgg_model(output_2)
        
        loss_gt_1=loss_function(output_1, target)
        loss_gt_2=loss_function(output_2, target)
        loss_c_0=loss_function(output_1,output_2)
        loss_c_1=loss_function(output_1_vgg_1,output_2_vgg_1)
        loss_c_2=loss_function(output_1_vgg_2,output_2_vgg_2)
        loss_c_3=loss_function(output_1_vgg_3,output_2_vgg_3)
        loss_c_4=loss_function(output_1_vgg_4,output_2_vgg_4)
        loss=loss_gt_1+loss_gt_2+loss_c_0+loss_c_1+loss_c_2+loss_c_3+loss_c_4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc+=loss.item()
        sample_cnt+=1

        tb_writer.add_scalar('Loss/loss', loss, (epoch-1)*len(train_loader)+batch_idx)

        if (batch_idx+1) % cfg.log_interval == 0:
            loss_avg=loss_acc/sample_cnt
            loss_acc=0.0
            sample_cnt=0.0
            log_str='Train Epoch:%d %d/%d(%d%%) loss: %.6f smooth loss: %.6f'%(epoch,batch_idx+1,batch_num,(100*float(batch_idx+1)/batch_num),loss.item(),loss_avg)
            print(log_str)
    
    if sample_cnt>0:
        loss_avg=loss_acc/sample_cnt
        log_str='Train Epoch:%d %d/%d(%d%%) loss: %.6f smooth loss: %.6f'%(epoch,batch_idx+1,batch_num,(100*float(batch_idx+1)/batch_num),loss.item(),loss_avg)
        print(log_str)

def test():
    pass

if __name__=='__main__':

    use_cuda=torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device=torch.device("cuda" if use_cuda else "cpu")
    print('device:',device)

    train_dataset=ImageDataset(cfg.input_dir,cfg.gt_dir,crop_size=cfg.train_crop_size,phase='train')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=cfg.train_batch_size, shuffle=True,
                                            num_workers=cfg.data_loader_num_workers, pin_memory=True)


    unet_model=SeeMotionInDarkNet(input_channels=3,demosaic=False)
    vgg_model=VGG19_Extractor(output_layer_list=cfg.vgg_output_layer_list)
    unet_model=unet_model.to(device)
    vgg_model=vgg_model.to(device)

    if cfg.snapshot_path is not None:
        unet_model.load_state_dict(torch.load(cfg.snapshot_path))

    print('model loaded!')

    loss_function=nn.L1Loss()
    loss_function=loss_function.to(device)

    # optimizer=optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum)
    optimizer=optim.Adam(unet_model.parameters(), lr=cfg.base_lr)

    tb_writer = SummaryWriter()

    print('training start!')
    for epoch in range(cfg.start_epoch, cfg.total_epochs + 1):
        print('Epoch %d'%(epoch))
        train(unet_model, vgg_model, device, train_loader, loss_function, optimizer, epoch, tb_writer)

        if epoch in cfg.lr_decay_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*cfg.lr_decay_rate

        if epoch%cfg.model_save_interval==0:
            model_save_path=os.path.join(cfg.model_save_path,'model_%05d.pth'%(epoch))
            torch.save(unet_model.state_dict(),model_save_path)