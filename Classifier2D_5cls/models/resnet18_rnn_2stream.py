from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F


def get_model_result(logits):
    logits_bowel = logits[:, :2]
    logits_extravasation = logits[:, 2:4]
    logits_kidney = logits[:, 4:7]
    logits_liver = logits[:, 7:10]
    logits_spleen = logits[:, 10:13]
    probs_bowel = logits_bowel.softmax(dim=1)
    probs_extravasation = logits_extravasation.softmax(dim=1)
    probs_kidney = logits_kidney.softmax(dim=1)
    probs_liver = logits_liver.softmax(dim=1)
    probs_spleen = logits_spleen.softmax(dim=1)

    output_probs = torch.cat([probs_bowel, probs_extravasation, probs_kidney, probs_liver, probs_spleen], dim=1)
    return output_probs

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
class ResNet18_rnn_2stream(nn.Module):

    def __init__(self, num_classes, num_channels, mode='train'):
        super(ResNet18_rnn_2stream, self).__init__()
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
        self.blk3 = ResBlk(256, 512, stride=2)
        # # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
        self.blk4 = ResBlk(512, 512, stride=2)

        self.blk34_extravasation = nn.Sequential(
            ResBlk(256, 512, stride=1), 
            ResBlk(512, 512, stride=2)
        )

        self.bowel_layer = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        self.extravasation_layer = nn.Sequential(
             nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        self.kidney_layer = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        self.liver_layer = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        self.spleen_layer = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        self.dropout= nn.Dropout(0.2)
        self.rnn = nn.LSTM(512, 128, 1)

    def get_feat(self, x):
        # [b, 3, h, w] => [b, 64, h, w]
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 512, h/16, w/16]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])

        x_feat = x.view(x.size(0), -1)
        
        return x_feat

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.get_feat((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.get_feat((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
      
        # [b, 512] => [b, 10]
        out_bowel = self.bowel_layer(out)
        out_extravasation = self.extravasation_layer(out)
        out_kidney = self.kidney_layer(out)
        out_liver = self.liver_layer(out)
        out_spleen = self.spleen_layer(out)
        out = torch.cat([out_bowel, out_extravasation, out_kidney, out_liver, out_spleen], dim=1)
        assert out.size(1) == self.num_classes
        if self.mode == 'deploy':
            out = get_model_result(out)
        return out


if __name__ == '__main__':
    res_net = ResNet18(13, 1)
    tmp = torch.randn(4, 16, 1, 224, 224)
    print(res_net.forward(tmp).shape)
