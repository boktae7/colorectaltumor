import torch
import torch.nn as nn
import torch.nn.functional as F

class Single_hourglass_network(torch.nn.Module):
    def __init__(self, input_ch = 1, num_landmarks = 6, input_size = 512, feat_dim_1 = 64, hg_depth = 4, upsample_mode = 'nearest', drop_rate = 0.25):
        
        super(Single_hourglass_network, self).__init__()
        
        self.entry_layer_block = torch.nn.Sequential(
            # torch.nn.Conv2d(input_ch, feat_dim_1, kernel_size=7, stride=2, padding=3), # w 512 -> 256, c 1 -> 64
            torch.nn.Conv2d(input_ch, feat_dim_1, kernel_size=7, stride=1, padding=3), # w 512 -> 512, c 1 -> 64
            torch.nn.BatchNorm2d(feat_dim_1),
            torch.nn.ReLU(),
            Multiscale_residual_block(feat_dim_1, feat_dim_1 * 2), # w 256, c 128       
            Multiscale_residual_block(feat_dim_1 * 2, feat_dim_1 * 2), # added ##################################
            Multiscale_residual_block(feat_dim_1 * 2, feat_dim_1 * 2),
            Multiscale_residual_block(feat_dim_1 * 2, feat_dim_1 * 2),
            Multiscale_residual_block(feat_dim_1 * 2, feat_dim_1 * 4) # w 256, c 128 -> 256
        )

        self.hourglass_block = Hourglass_module_SB(feat_dim_1 * 4, hg_depth=hg_depth, num_landmarks = num_landmarks, upsample_mode=upsample_mode, slice_num = input_ch) # w 256 -> 128 ... -> 64 ... -> 128 -> 512, c 256 -> 512

    def forward(self, input):
        # print("Single_hourglass_network: input", input.shape)
        entry = self.entry_layer_block(input)
        # print("Single_hourglass_network: entry", entry.shape)
        # landmark 갯수 만큼의 heatmap (channel 방향) 나옴 -> argmax -> 2D 좌표를 구함        
        hourglass_return = self.hourglass_block(entry)
        # print("Single_hourglass_network: hourglass_return", hourglass_return.shape)    
        
        return hourglass_return
    
class Multiscale_residual_block(torch.nn.Module): 
    # !! dimension preserved !! 
    def __init__(self, input_ch, output_ch):
        super(Multiscale_residual_block, self).__init__()

        self.skip = torch.nn.Conv2d(input_ch, output_ch, kernel_size=1)
        
        self.conv1 = torch.nn.Conv2d(input_ch, int(output_ch/2), kernel_size=3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(int(output_ch/2), int(output_ch/4), kernel_size=3, padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(int(output_ch/4), int(output_ch/4), kernel_size=3, padding=1, stride=1)

    def forward(self, input):
        skip = self.skip(input) # featdim 1+
        conv1 = self.conv1(input) # featdim 1/2
        conv2 = self.conv2(conv1) # featdim 1/4
        conv3 = self.conv3(conv2) # featdim 1/4

        concat = torch.cat([conv1, conv2, conv3], 1) # featdim 1

        return skip + concat # featdim 1

class Hourglass_module_SB(torch.nn.Module):
    # !! width & height preserved, and channel increased to double !! 

    def __init__(self, input_ch, num_landmarks, hg_depth = 4, upsample_mode = 'nearest', slice_num = 1):
        super(Hourglass_module_SB, self).__init__()

        self.hg_depth = hg_depth

        self.conv_block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), # w 1/2
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch), # added ##################################
            Multiscale_residual_block(input_ch, input_ch)
        )

        self.passing_block = torch.nn.Sequential(
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch), # added ##################################
            Multiscale_residual_block(input_ch, input_ch * 2)
        )

        self.bottom_block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), # w 1/2
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch * 2)
        )

        self.residual_block = Multiscale_residual_block(input_ch, input_ch * 2)
        self.upsample_layer = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv_layer = Multiscale_residual_block(input_ch * 2, input_ch * 2)

        self.conv_final = nn.Conv2d(input_ch * 2, num_landmarks, 1)

        self.slice_num = slice_num
        # self.conv_final = nn.Conv2d(input_ch * 2, slice_num * num_landmarks, 1)
        
    def forward(self, input):
        passing_list = []
        conv_input = input
        residual_block = self.residual_block(conv_input)
        passing_list.append(residual_block)

        # downsample
        for i in range(self.hg_depth-1):            
            conv_block = self.conv_block(conv_input)
            passing_block = self.passing_block(conv_block)    

            passing_list.append(passing_block)

            conv_input = conv_block
            # print("HG down ", i, conv_block.shape)

        # bottom
        bottom_block = self.bottom_block(conv_input)
        conv_input = bottom_block
        # print("bottom block", bottom_block.shape)
        
        # upsample
        for i in range(self.hg_depth-1, -1, -1):
            upsample = self.upsample_layer(conv_input)
            merge = passing_list[i] + upsample

            if i != 0:
                conv_input = self.conv_layer(merge)
            else:
                conv_input = merge
            # print("HG up ", i, conv_input.shape)
        # print(conv_input.shape)
        # final
        tmp = conv_input
        multi_conv = []
        final_output = self.conv_final(conv_input)
        # if self.slice_num == 1:
        #     final_output = self.conv_final(conv_input)
        # else:
        #     multi_conv.append(self.conv_final(conv_input))

        # append.append(conv_input)
        # print("1 번 = ", conv_input.shape)
        # print("1 번 A = ", len(append))
        # print("1 번 A = ", len(append[0]))
        # print("1 번 A = ", append[0].shape)
        # conv_tmp = self.conv_final(tmp)
        # append.append(conv_tmp)
        # print("2 번 = ", conv_tmp.shape)
        # print("2 번 A = ", len(append))
        # print("2 번 A = ", len(append[0]))
        # conv_cat = torch.cat([conv_input, conv_tmp], 1) 
        # print("3 번 = ", conv_cat.shape)
        return final_output # heatmap: w 128 (input 512 기준)

class Hourglass_module(torch.nn.Module):
    # !! width & height preserved, and channel increased to double !! 

    def __init__(self, input_ch, hg_depth = 4, upsample_mode = 'nearest'):
        super(Hourglass_module, self).__init__()

        self.hg_depth = hg_depth

        self.conv_block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), # w 1/2
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch), # added ##################################
            Multiscale_residual_block(input_ch, input_ch)
        )

        self.passing_block = torch.nn.Sequential(
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch), # added ##################################
            Multiscale_residual_block(input_ch, input_ch * 2)
        )

        self.bottom_block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), # w 1/2
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch),
            Multiscale_residual_block(input_ch, input_ch * 2)
        )

        self.upsample_layer = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv_layer = Multiscale_residual_block(input_ch * 2, input_ch * 2)
        
    def forward(self, input):
        
        passing_list = []
        conv_input = input

        # downsample
        for i in range(self.hg_depth-1):            
            conv_block = self.conv_block(conv_input)
            passing_block = self.passing_block(conv_block)    

            passing_list.append(passing_block)

            conv_input = conv_block
            print("HG down ", i, conv_block.shape)

        # bottom
        bottom_block = self.bottom_block(conv_input)
        conv_input = bottom_block
        print("bottom block", bottom_block.shape)
        
        # upsample
        for i in range(self.hg_depth-2, -1, -1):
            upsample = self.upsample_layer(conv_input)
            merge = passing_list[i] + upsample

            if i != 0:
                conv_input = self.conv_layer(merge)
            else:
                conv_input = merge
            print("HG up ", i, conv_input.shape)
        
        return conv_input # heatmap: w 128 (input 512 기준)

if __name__ == '__main__':
    batch = 32
    # dummy = torch.zeros((batch, 1, 512, 512)).to("cuda") #max_batch=608
    dummy = torch.zeros((batch, 1, 128, 128)).to("cuda") 
    dummy2 = torch.zeros((batch, 1, 128, 128)).to("cuda") 
    dummy3 = torch.zeros((batch, 1, 128, 128)).to("cuda") 
    dummy4 = torch.zeros((batch, 1, 128, 128)).to("cuda") 
    dummy5 = torch.zeros((batch, 1, 128, 128)).to("cuda") 
    # dummy = torch.zeros((batch, 1, 256, 256)).to("cuda") 
    # net = Single_hourglass_network(input_ch = 1, num_landmarks = 2, input_size = 512, feat_dim_1 = 64, hg_depth = 4, upsample_mode = 'nearest', drop_rate = 0.25).to("cuda") 
    net = Single_hourglass_network(input_ch = 1, num_landmarks = 2).to("cuda")
    print("Input = ", dummy.shape)
    heat = net(dummy)
    heat_cpu1 = heat.detach().cpu().numpy()
    print("Output = ", heat.shape)
    heat = net(dummy2)
    heat_cpu2 = heat.detach().cpu().numpy()
    heat = net(dummy3)
    heat_cpu3 = heat.detach().cpu().numpy()
    heat = net(dummy4)
    heat_cpu4 = heat.detach().cpu().numpy()
    heat = net(dummy5)
    heat_cpu5 = heat.detach().cpu().numpy()
    
    # print(net.modules)
    # for name, module in net.named_modules():
    #     print(name)

    # last = nn.Sequential(list(net.children())[-1])
    # print(last)
    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    # print(net.state_dict()['hourglass_block.conv_final.weight'].size())

    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))

   
