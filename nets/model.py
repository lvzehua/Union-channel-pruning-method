from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])

        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # collect encode outputs
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # collect decode outputs
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.cat([x, x2], dim=1))
            decode_outputs.insert(0, x)

        # collect side outputs
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.cat(side_outputs, dim=1))

        return [x] + side_outputs
    
def u2net_full(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 32, 64, False, False],      # En1
                   [6, 64, 32, 128, False, False],    # En2
                   [5, 128, 64, 256, False, False],   # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],   # En5
                   [4, 512, 256, 512, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512, True, True],   # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],    # De3
                   [6, 256, 32, 64, False, True],     # De2
                   [7, 128, 16, 64, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)

# 87.5% pruning ratio 
def u2net_prune_l(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 4, 8, False, False],      # En1
                   [6, 8, 4, 16, False, False],    # En2
                   [5, 16, 8, 32, False, False],   # En3
                   [4, 32, 16, 64, False, False],  # En4
                   [4, 64, 32, 64, True, False],   # En5
                   [4, 64, 32, 64, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 32, 64, True, True],   # De5
                   [4, 128, 16, 32, False, True],  # De4
                   [5, 64, 8, 16, False, True],    # De3
                   [6, 32, 4, 8, False, True],     # De2
                   [7, 16, 2, 8, False, True]]     # De1
    }
    return U2Net(cfg, out_ch)

# 75% pruning ratio 
def u2net_prune_l_s(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 8, 16, False, False],      # En1
                   [6, 16, 8, 32, False, False],    # En2
                   [5, 32, 16, 64, False, False],   # En3
                   [4, 64, 32, 128, False, False],  # En4
                   [4, 128, 64, 128, True, False],   # En5
                   [4, 128, 64, 128, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 256, 64, 128, True, True],   # De5
                   [4, 256, 32, 64, False, True],  # De4
                   [5, 128, 16, 32, False, True],    # De3
                   [6, 64, 8, 16, False, True],     # De2
                   [7, 32, 4, 16, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)

# 62.5% pruning ratio 
def u2net_prune_m(out_ch: int = 1):  
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 12, 24, False, False],      # En1
                   [6, 24, 12, 48, False, False],    # En2
                   [5, 48, 24, 96, False, False],   # En3
                   [4, 96, 48, 192, False, False],  # En4
                   [4, 192, 96, 192, True, False],   # En5
                   [4, 192, 96, 192, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 384, 96, 192, True, True],   # De5
                   [4, 384, 48, 96, False, True],  # De4
                   [5, 192, 24, 48, False, True],    # De3
                   [6, 96, 12, 24, False, True],     # De2
                   [7, 48, 6, 24, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)

# 50% pruning ratio 
def u2net_prune_m_s(out_ch: int = 1):  
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 32, False, False],      # En1
                   [6, 32, 16, 64, False, False],    # En2
                   [5, 64, 32, 128, False, False],   # En3
                   [4, 128, 64, 256, False, False],  # En4
                   [4, 256, 128, 256, True, False],   # En5
                   [4, 256, 128, 256, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 512, 128, 256, True, True],   # De5
                   [4, 512, 64, 128, False, True],  # De4
                   [5, 256, 32, 64, False, True],    # De3
                   [6, 128, 16, 32, False, True],     # De2
                   [7, 64, 8, 32, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)

# 37.5% pruning ratio 
def u2net_prune_mm(out_ch: int = 1):    #剪掉3/8，0.375
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 20, 40, False, False],      # En1
                   [6, 40, 20, 80, False, False],    # En2
                   [5, 80, 40, 160, False, False],   # En3
                   [4, 160, 80, 320, False, False],  # En4
                   [4, 320, 160, 320, True, False],   # En5
                   [4, 320, 160, 320, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 640, 160, 320, True, True],   # De5
                   [4, 640, 80, 160, False, True],  # De4
                   [5, 320, 40, 80, False, True],    # De3
                   [6, 160, 20, 40, False, True],     # De2
                   [7, 80, 10, 40, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)

# 25% pruning ratio 
def u2net_prune_mm_s(out_ch: int = 1):    
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 24, 48, False, False],      # En1
                   [6, 48, 24, 96, False, False],    # En2
                   [5, 96, 48, 192, False, False],   # En3
                   [4, 192, 96, 384, False, False],  # En4
                   [4, 384, 192, 384, True, False],   # En5
                   [4, 384, 192, 384, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 768, 192, 384, True, True],   # De5
                   [4, 768, 96, 192, False, True],  # De4
                   [5, 384, 48, 96, False, True],    # De3
                   [6, 192, 24, 48, False, True],     # De2
                   [7, 96, 12, 48, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)

# 12.5% pruning ratio 
def u2net_prune_s(out_ch: int = 1):    
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 28, 56, False, False],      # En1
                   [6, 56, 28, 112, False, False],    # En2
                   [5, 112, 56, 224, False, False],   # En3
                   [4, 224, 112, 448, False, False],  # En4
                   [4, 448, 224, 448, True, False],   # En5
                   [4, 448, 224, 448, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 896, 224, 448, True, True],   # De5
                   [4, 896, 112, 224, False, True],  # De4
                   [5, 448, 56, 112, False, True],    # De3
                   [6, 224, 28, 56, False, True],     # De2
                   [7, 112, 14, 56, False, True]]     # De1
    }

    return U2Net(cfg, out_ch)

if __name__ == '__main__':
    u2net = u2net_full()
   
