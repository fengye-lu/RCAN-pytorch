from model import common

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return RCAN(args)


class CRB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(CRB, self).__init__()
        modules_body1 = []
        for i in range(2):
            modules_body1.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body1.append(nn.BatchNorm2d(n_feat))
            modules_body1.append(act)
        modules_body1.append(CALayer(n_feat, reduction))
        self.body1 = nn.Sequential(*modules_body1)

        modules_body2 = []
        for i in range(2):
            modules_body2.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body2.append(nn.BatchNorm2d(n_feat))
            modules_body2.append(act)
        modules_body2.append(CALayer(n_feat, reduction))
        self.body2 = nn.Sequential(*modules_body2)

        self.res_scale = res_scale

    def forward(self, x):
        res1 = self.body1(x)
        res2 = self.body2(res1+x)
        #res = self.body(x).mul(self.res_scale)
        res = x + res2 + res1
        return res


class MulConv_Block(nn.Module):
    def __init__(self, n_feat, reduction, act=nn.ReLU(True)):
        super(MulConv_Block, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1, stride=1, bias=True),
            act,
            CALayer(n_feat, reduction),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            act,
            CALayer(n_feat, reduction),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            act,
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            act,
            CALayer(n_feat, reduction),
        )
        self.body4 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            act,
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            act,
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            act,
            CALayer(n_feat, reduction),
        )
        self.ConCat = nn.Sequential(
            nn.Conv2d(n_feat*4, n_feat, 1, 1),
            act,
            CALayer(n_feat, reduction)
        )

    def forward(self, x):
        x1 = self.body1(x)
        x2 = self.body2(x+x1)
        x3 = self.body3(x+x2)
        x4 = self.body4(x+x3)
        y = torch.cat([x1+x, x2+x1+x, x3+x2+x, x4+x3+x], dim=1)
        y = self.ConCat(y)
        y = y + x
        return y



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
# class RCAB(nn.Module):
#     def __init__(
#         self, conv, n_feat, kernel_size, reduction,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(RCAB, self).__init__()
#         modules_body = []
#         for i in range(2):
#             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         modules_body.append(CALayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x)
#         #res = self.body(x).mul(self.res_scale)
#         res += x
#         return res

## Residual Group (RG)rcan.py

class MulConvGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(MulConvGroup, self).__init__()
        modules_body = []
        # modules_body = [
        #     RCAB(
        #         conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
        #     for _ in range(n_resblocks)]
        modules_body = [
            MulConv_Block(n_feat, reduction, act=act) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups - 1
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K 1-800
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        # RGB mean for DIVFlickr2K 1-3450
        # rgb_mean = (0.4690, 0.4490, 0.4036)
        if args.data_train == 'DIV2K':
            print('Use DIV2K mean (0.4488, 0.4371, 0.4040)')
            rgb_mean = (0.4488, 0.4371, 0.4040)
        elif args.data_train == 'DIVFlickr2K':
            print('Use DIVFlickr2K mean (0.4690, 0.4490, 0.4036)')
            rgb_mean = (0.4690, 0.4490, 0.4036)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # # define body module
        modules_prebody = []
        modules_prebody.append(
            MulConvGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks)
        )
        modules_body = [
            MulConvGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        # modules_body = [
        #     MulConv_Group(n_feats, reduction, act=act) for _ in range(n_resgroups)
        # ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            CRB(conv, n_feats, kernel_size, reduction),
            ]
        modules_re = [conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.prebody = nn.Sequential(*modules_prebody)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.re = nn.Sequential(*modules_re)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res1 = self.prebody(x)
        res = self.body(res1)
        res += x

        x = self.tail(res+res1)
        x = self.re(x+res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))