from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Multi_Conv(torch.nn.Module):
    def __init__(self, input_ch):
        super(Multi_Conv, self).__init__()

        self.conv = nn.Conv2d(input_ch, 3, kernel_size=1)
    
    def forward(self, input):
        return self.conv(input)

if __name__ == '__main__':
    batch = 16
    GPU_NAME = "cuda:0"
    dummy = torch.zeros((batch, 5, 512, 512)).to(GPU_NAME) #max_batch=608
    # dummy = torch.zeros((batch, 1, 256, 256)).to("cuda") 
    # net = Single_hourglass_network(input_ch = 1, num_landmarks = 2, input_size = 512, feat_dim_1 = 64, hg_depth = 4, upsample_mode = 'nearest', drop_rate = 0.25).to("cuda") 
    net = Multi_Conv(5).to(GPU_NAME)
    output = net(dummy)
    print(output.shape)