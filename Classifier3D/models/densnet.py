import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch

# bn_momentum = 0.99
# affine = True
# dropout_rate = 0.2


# def initNetParams(net):
#     '''Init net parameters.'''
#     for m in net.modules():
#         if isinstance(m, (nn.Conv3d, nn.Linear)):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class dense_layer(nn.Module):
    def __init__(self, input_nc, nb_filter, bn_size=4, alpha=0.2, drop_rate=0.2, bn_momentum=0.99):
        super(dense_layer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm3d(input_nc, momentum=bn_momentum),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv3d(input_nc, bn_size * nb_filter, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(bn_size * nb_filter, momentum=bn_momentum),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv3d(bn_size * nb_filter, nb_filter, kernel_size=3, stride=1, padding=1),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], 1)


class dense_block(nn.Module):
    def __init__(self, input_nc, nb_layers, growth_rate, drop_rate=0.2):
        super(dense_block, self).__init__()
        self.nb_layers = nb_layers
        layer = []
        in_nc = input_nc
        for i in range(nb_layers):
            if i == 0:
                in_nc = input_nc
            else:
                in_nc += growth_rate
            layer += [dense_layer(input_nc=in_nc, nb_filter=growth_rate)]
        self.dense_model = nn.Sequential(*layer)

    def forward(self, x):
        # for j in range(self.nb_layers):
        #     x_out = self.layer[j](x)
        #     x = torch.cat((x, x_out), axis=1)
        x = self.dense_model(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate=8, n_channel=32, bn_momentum=0.99, dropout_rate=0.2, mode='train'):
        super(DenseNet, self).__init__()

        self.num_classes = num_classes
        self.mode = mode

        sequence = [nn.Conv3d(1, n_channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(32, momentum=bn_momentum),
                    nn.LeakyReLU(0.2, inplace=True)]
        # depth = self.cfg['crop_size'][-1]
        # num_downsamplings = int(np.log2(depth / 4))
        num_downsamplings = 3
        for i in range(num_downsamplings):
            # nb_layers = self.cfg['n_base_nb_layers'] * (2 ** i)
            # num_filters = self.cfg['n_base_filters'] * (2 ** i)
            nb_layers = 2 * (2 ** i)
            num_filters = 32 * (2 ** i)
            if i == 0:
                input_nc = n_channel
                output_nc = input_nc + nb_layers * growth_rate
            else:
                input_nc = dense_block_out_nc
                output_nc = input_nc + nb_layers * growth_rate
            sequence += [dense_block(input_nc=input_nc, nb_layers=nb_layers, growth_rate=growth_rate)]
            sequence += [nn.BatchNorm3d(output_nc, momentum=bn_momentum),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Conv3d(output_nc, num_filters, kernel_size=3, stride=1, padding=1),
                         nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))]
            dense_block_out_nc = num_filters

        sequence += [nn.AdaptiveAvgPool3d(1)]
       
        self.linear_block = nn.Sequential(nn.Linear(num_filters, 128),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(dropout_rate),
                                              nn.Linear(128, 128),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(dropout_rate),
                                              nn.Linear(128, self.num_classes))

        self.conv_model = nn.Sequential(*sequence)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv3d, nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()


if __name__ == '__main__':
    net = DenseNet(num_classes=3).cuda()
    tmp = torch.randn(2, 1, 128, 128, 128).cuda()
    print(net.forward(tmp).shape)

