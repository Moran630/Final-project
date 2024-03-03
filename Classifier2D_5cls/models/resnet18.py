from dataclasses import replace
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

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)


        # [b, ch_in, h, w] => [b, ch_out, h, w]
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
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
class ResNet18(nn.Module):

    def __init__(self, num_classes, num_channels, mode='train'):
        super(ResNet18, self).__init__()
        self.mode = mode
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # follow 4 blocks
        # [b, 64, h, w] => [b, 128, h/2, w/2]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h/2, w/2] => [b, 256, h/4, w/4]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h/4, w/4] => [b, 512, h/8, w/8]
        # self.blk3 = ResBlk(256, 512, stride=2)
        # # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
        # self.blk4 = ResBlk(512, 512, stride=2)

        self.blk34_bowel = nn.Sequential(
            ResBlk(256, 512, stride=2),
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
            ResBlk(512, 512, stride=2)
        )
        self.blk34_extravasation = nn.Sequential(
            ResBlk(256, 512, stride=2),
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
            ResBlk(512, 512, stride=2)
        )
        self.blk34_kidney = nn.Sequential(
            ResBlk(256, 512, stride=2),
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
            ResBlk(512, 512, stride=2)
        )
        self.blk34_liver = nn.Sequential(
            ResBlk(256, 512, stride=2),
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
            ResBlk(512, 512, stride=2)
        )
        self.blk34_spleen = nn.Sequential(
            ResBlk(256, 512, stride=2),
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
            ResBlk(512, 512, stride=2)
        )

        self.bowel_layer = nn.Sequential(
            nn.Linear(512*1*1, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        self.extravasation_layer = nn.Sequential(
            nn.Linear(512*1*1, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.kidney_layer = nn.Sequential(
            nn.Linear(512*1*1, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )
        self.liver_layer = nn.Sequential(
            nn.Linear(512*1*1, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )
        self.spleen_layer = nn.Sequential(
            nn.Linear(512*1*1, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

    def get_feat(self, x):
        # [b, 3, h, w] => [b, 64, h, w]
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 512, h/16, w/16]
        x = self.blk1(x)
        x = self.blk2(x)

        x_bowel = self.blk34_bowel(x)
        x_bowel = F.adaptive_avg_pool2d(x_bowel, [1, 1])
        x_feat_bowel = x_bowel.view(x_bowel.size(0), -1)

        x_extravasation = self.blk34_extravasation(x)
        x_extravasation = F.adaptive_avg_pool2d(x_extravasation, [1, 1])
        x_feat_extravasation = x_extravasation.view(x_extravasation.size(0), -1)

        x_kidney = self.blk34_kidney(x)
        x_kidney = F.adaptive_avg_pool2d(x_kidney, [1, 1])
        x_feat_kidney = x_kidney.view(x_kidney.size(0), -1)

        x_liver = self.blk34_liver(x)
        x_liver = F.adaptive_avg_pool2d(x_liver, [1, 1])
        x_feat_liver = x_liver.view(x_liver.size(0), -1)

        x_spleen = self.blk34_spleen(x)
        x_spleen = F.adaptive_avg_pool2d(x_spleen, [1, 1])
        x_feat_spleen = x_spleen.view(x_spleen.size(0), -1)
        
        return x_feat_bowel, x_feat_extravasation, x_feat_kidney, x_feat_liver, x_feat_spleen

    def forward(self, x):

        x_feat_bowel, x_feat_extravasation, x_feat_kidney, x_feat_liver, x_feat_spleen = self.get_feat(x)
        # [b, 512] => [b, 10]
        out_bowel = self.bowel_layer(x_feat_bowel)
        out_extravasation = self.extravasation_layer(x_feat_extravasation)
        out_kidney = self.kidney_layer(x_feat_kidney)
        out_liver = self.liver_layer(x_feat_liver)
        out_spleen = self.spleen_layer(x_feat_spleen)
        out = torch.cat([out_bowel, out_extravasation, out_kidney, out_liver, out_spleen], dim=1)
        assert out.size(1) == self.num_classes
        return out


if __name__ == '__main__':
    res_net = ResNet18(13, 1)
    tmp = torch.randn(1,1,224,224)
    print(res_net.forward(tmp).shape)
