class GELU(nn.Module):
    """高斯误差线性单元（GELU），纯PyTorch张量实现，无外部依赖"""
    def __init__(self):
        super(GELU, self).__init__()
        # 预计算常数（用PyTorch张量存储，适配GPU）
        self.sqrt_2_over_pi = torch.tensor(2.0 / torch.pi).sqrt()  # √(2/π)
        self.coeff = torch.tensor(0.044715)  # 0.044715（GELU公式中的系数）

    def forward(self, x):
        # 确保常数与输入x在同一设备（CPU/GPU）上
        sqrt_2_over_pi = self.sqrt_2_over_pi.to(x.device)
        coeff = self.coeff.to(x.device)
        # GELU公式：0.5 * x * [1 + tanh(√(2/π) * (x + 0.044715 * x³))]
        return 0.5 * x * (1 + torch.tanh(sqrt_2_over_pi * (x + coeff * torch.pow(x, 3))))


class GEConv(nn.Module):
    '''带GELU激活的卷积模块（Conv + BN + GELU）'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2  # 自动计算padding，保证输入输出尺寸对齐
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = GELU()  # 实例化纯PyTorch版GELU

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合BN的前向传播（推理阶段用，加速计算）"""
        return self.act(self.conv(x))


class GSPPF(nn.Module):
    """改进的SPPF模块，纯PyTorch实现，确保通道匹配"""
    def __init__(self, in_channels, out_channels=None, kernel_size=5):
        super().__init__()
        # 关键：未指定out_channels时，默认与in_channels一致（适配YOLO通道逻辑）
        if out_channels is None:
            out_channels = in_channels
        c_ = in_channels // 2  # 中间通道减半，平衡计算量与特征保留
        self.cv1 = GEConv(in_channels, c_, 1, 1)  # 1x1卷积降维：in_channels → c_
        self.cv2 = GEConv(c_ * 4, out_channels, 1, 1)  # 聚合4尺度特征后升维
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)  # 降维（例：1024→512）
        # 多尺度池化（忽略重复池化的警告）
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
        # 通道维度拼接4个尺度特征（x + y1 + y2 + y3）
        concat_feat = torch.cat([x, y1, y2, y3], dim=1)  # 例：512*4=2048
        return self.cv2(concat_feat)  # 升维到目标通道（例：2048→1024）
