import torch
from torch import nn
from torch.nn import functional as F


'''
ResBlock
'''
class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        # 通过stride减少参数维度
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(ch_out)
        self.conv2 = nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(ch_out)


        # [b, ch_in, h, w] => [b, ch_out, h, w]
        self.extra = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm3d(ch_out)
        )

    def forward(self, x):
        '''
        :param x: [b, ch, h, w]
        :return:
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out

        return out


'''
ResNet-18
'''
class ResNet18_3d(nn.Module):

    def __init__(self, num_classes, mode='train'):
        super(ResNet18_3d, self).__init__()
        self.mode = mode
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64)
        )

      
        self.blk1 = ResBlk(64, 128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)

        self.bowel_layer = nn.Sequential(
            nn.Linear(512*1*1*1, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.extravasation_layer = nn.Sequential(
            nn.Linear(512*1*1*1, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.kidney_layer = nn.Sequential(
            nn.Linear(512*1*1*1, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

        self.liver_layer = nn.Sequential(
            nn.Linear(512*1*1*1, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

        self.spleen_layer = nn.Sequential(
            nn.Linear(512*1*1*1, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

        # 'kidney', 'liver', 'spleen'
    def get_feat(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool3d(x, [1, 1, 1])
        x_feat = x.view(x.size(0), -1)
        return x_feat

    def forward(self, x, return_feat=False):
        x_feat = self.get_feat(x)
        out_bowel = self.bowel_layer(x_feat)
        out_extravasation = self.extravasation_layer(x_feat)
        out_kindy = self.kidney_layer(x_feat)
        out_liver = self.liver_layer(x_feat)
        out_spleen = self.spleen_layer(x_feat)
        out = torch.cat([out_bowel, out_extravasation, out_kindy, out_liver, out_spleen], dim=1)
        assert out.size(-1) == self.num_classes
        return out
      

class ResNet18_3d_single(nn.Module):

    def __init__(self, num_classes, mode='train'):
        super(ResNet18_3d_single, self).__init__()
        self.mode = mode
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64)
        )

        # follow 4 blocks
        # [b, 64, h, w] => [b, 128, h/2, w/2]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h/2, w/2] => [b, 256, h/4, w/4]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h/4, w/4] => [b, 512, h/8, w/8]
        self.blk3 = ResBlk(256, 512, stride=2)
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
        self.blk4 = ResBlk(512, 512, stride=2)

        self.out_layer = nn.Linear(512*1*1*1, num_classes)

    def get_feat(self, x):
        # [b, 3, h, w] => [b, 64, h, w]
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 512, h/16, w/16]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # [b, 512, h/16, w/16] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool3d(x, [1, 1, 1])

        # [b, 512, 1, 1] => [b, 512]
        x_feat = x.view(x.size(0), -1)
        return x_feat

    
    def forward(self, x, return_feat=False):

        x_feat = self.get_feat(x)
        out = self.out_layer(x_feat)
        assert out.size(-1) == self.num_classes
        return out
      

if __name__ == '__main__':
    # res_net = ResNet18_3d(num_classes=12).cuda()
    # tmp = torch.randn(2, 1, 200, 200, 200).cuda()
    # print(res_net.forward(tmp).shape)

    
    
    from thop import profile
    from copy import deepcopy
    model = ResNet18_3d(num_classes=11)
    tsize = 144
    # model = model.cuda().train()
    img = torch.zeros((1, 1, tsize, tsize, tsize), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    print(info)

