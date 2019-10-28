import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from net_utils import conv2d, deconv_2d, DepthToSpace
import torch.optim as optim



class ResBlocks(nn.Module):
    def __init__(self,res_block_num,conv_channels,use_bn=False):
        super(ResBlocks,self).__init__()
        self.res_block_num=res_block_num
        for res_block_idx in range(self.res_block_num):
            conv_layer_1=conv2d(conv_channels,conv_channels,kernel_size=3,stride=1,use_bn=use_bn)
            conv_layer_2=conv2d(conv_channels,conv_channels,kernel_size=3,stride=1,activation=False,use_bn=use_bn)
            self.add_module('%d'%(res_block_idx),nn.Sequential(conv_layer_1,conv_layer_2))
    
    def __getitem__(self, index):
        if index < 0 or index >= len(self._modules):
            raise IndexError('index %d is out of range'%(index))

        return(self._modules[str(index)])

    def __len__(self):
        return self.res_block_num    


class SeeMotionInDarkNet(nn.Module):
    def __init__(self,input_channels=4,base_dim=32,res_block_num=16,use_bn=False,demosaic=True):
        super(SeeMotionInDarkNet,self).__init__()

        self.demosaic=demosaic

        self.pool=nn.MaxPool2d(2) 

        self.conv_1_1=conv2d(input_channels,base_dim,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_1_2=conv2d(base_dim,base_dim,kernel_size=3,stride=1,use_bn=use_bn)

        self.conv_2_1=conv2d(base_dim,base_dim*2,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_2_2=conv2d(base_dim*2,base_dim*2,kernel_size=3,stride=1,use_bn=use_bn)

        self.conv_3_1=conv2d(base_dim*2,base_dim*4,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_3_2=conv2d(base_dim*4,base_dim*4,kernel_size=3,stride=1,use_bn=use_bn) 

        self.conv_4_1=conv2d(base_dim*4,base_dim*8,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_4_2=conv2d(base_dim*8,base_dim*8,kernel_size=3,stride=1,use_bn=use_bn) 

        self.conv_5_1=conv2d(base_dim*8,base_dim*16,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_5_2=conv2d(base_dim*16,base_dim*16,kernel_size=3,stride=1,use_bn=use_bn)

        self.res_block_list=ResBlocks(res_block_num,base_dim*16)
        
        self.conv_after_res_block=conv2d(base_dim*16,base_dim*16,kernel_size=3,stride=1,use_bn=use_bn)

        self.deconv_6=deconv_2d(base_dim*16,base_dim*8,use_bn=use_bn)
        self.conv_6_1=conv2d(base_dim*16,base_dim*8,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_6_2=conv2d(base_dim*8,base_dim*8,kernel_size=3,stride=1,use_bn=use_bn)

        self.deconv_7=deconv_2d(base_dim*8,base_dim*4,use_bn=use_bn)
        self.conv_7_1=conv2d(base_dim*8,base_dim*4,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_7_2=conv2d(base_dim*4,base_dim*4,kernel_size=3,stride=1,use_bn=use_bn)

        self.deconv_8=deconv_2d(base_dim*4,base_dim*2,use_bn=use_bn)
        self.conv_8_1=conv2d(base_dim*4,base_dim*2,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_8_2=conv2d(base_dim*2,base_dim*2,kernel_size=3,stride=1,use_bn=use_bn)

        self.deconv_9=deconv_2d(base_dim*2,base_dim,use_bn=use_bn)
        self.conv_9_1=conv2d(base_dim*2,base_dim,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_9_2=conv2d(base_dim,base_dim,kernel_size=3,stride=1,use_bn=use_bn)


        if self.demosaic:
            self.conv_10=conv2d(base_dim,12,kernel_size=1,stride=1,use_bn=use_bn)
            self.depth_to_space=nn.PixelShuffle(2)
            # self.depth_to_space=DepthToSpace(2)
        else:
            self.conv_10=conv2d(base_dim,3,kernel_size=1,stride=1,use_bn=use_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)


    def forward(self, input_tensor):
        conv_1=self.conv_1_1(input_tensor)
        conv_1=self.conv_1_2(conv_1)
        conv_1_pool=self.pool(conv_1)

        conv_2=self.conv_2_1(conv_1_pool)
        conv_2=self.conv_2_2(conv_2)
        conv_2_pool=self.pool(conv_2)

        conv_3=self.conv_3_1(conv_2_pool)
        conv_3=self.conv_3_2(conv_3)
        conv_3_pool=self.pool(conv_3)

        conv_4=self.conv_4_1(conv_3_pool)
        conv_4=self.conv_4_2(conv_4)
        conv_4_pool=self.pool(conv_4)

        conv_5=self.conv_5_1(conv_4_pool)
        conv_5=self.conv_5_2(conv_5)


        #res block
        conv_feature=conv_5
        for res_block_idx in range(len(self.res_block_list)):
            conv_feature=self.res_block_list[res_block_idx](conv_feature)+conv_feature
        
        conv_feature=self.conv_after_res_block(conv_feature)

        conv_feature=conv_feature+conv_5

        conv_6_up=self.deconv_6(conv_feature)
        conv_6=torch.cat((conv_6_up,conv_4),dim=1)
        conv_6=self.conv_6_1(conv_6)
        conv_6=self.conv_6_2(conv_6)

        conv_7_up=self.deconv_7(conv_6)
        conv_7=torch.cat((conv_7_up,conv_3),dim=1)
        conv_7=self.conv_7_1(conv_7)
        conv_7=self.conv_7_2(conv_7)

        conv_8_up=self.deconv_8(conv_7)
        conv_8=torch.cat((conv_8_up,conv_2),dim=1)
        conv_8=self.conv_8_1(conv_8)
        conv_8=self.conv_8_2(conv_8)

        conv_9_up=self.deconv_9(conv_8)
        conv_9=torch.cat((conv_9_up,conv_1),dim=1)
        conv_9=self.conv_9_1(conv_9)
        conv_9=self.conv_9_2(conv_9)

        if self.demosaic:
            conv_10=self.conv_10(conv_9)
            out=self.depth_to_space(conv_10)
        else:
            out=self.conv_10(conv_9)

        return out


if __name__=='__main__':

    
    model=SeeMotionInDarkNet(use_bn=False).cuda()

    print(model)

    '''
    loss_function=nn.L1Loss().cuda()
    optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for idx in range(32):
        input_tensor=torch.zeros((32,4,256,256),dtype=torch.float32).cuda()
        gt_tensor=torch.zeros((32,3,512,512),dtype=torch.float32).cuda()
        output=model(input_tensor)
        print(output.size())
        loss=loss_function(output,gt_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())
    '''