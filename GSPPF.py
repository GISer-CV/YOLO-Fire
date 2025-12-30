class GELU(nn.Module):

    def __init__(self):
        super(GELU, self).__init__()
        self.sqrt_2_over_pi = torch.tensor(2.0 / torch.pi).sqrt()  
        self.coeff = torch.tensor(0.044715)  

    def forward(self, x):

        sqrt_2_over_pi = self.sqrt_2_over_pi.to(x.device)
        coeff = self.coeff.to(x.device)
   
        return 0.5 * x * (1 + torch.tanh(sqrt_2_over_pi * (x + coeff * torch.pow(x, 3))))


class GEConv(nn.Module):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2 
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
        self.act = GELU()  

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):

        return self.act(self.conv(x))


class GSPPF(nn.Module):
  
    def __init__(self, in_channels, out_channels=None, kernel_size=5):
        super().__init__()
     
        if out_channels is None:
            out_channels = in_channels
        c_ = in_channels // 2 
        self.cv1 = GEConv(in_channels, c_, 1, 1)  
        self.cv2 = GEConv(c_ * 4, out_channels, 1, 1)  
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)  

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
      
        concat_feat = torch.cat([x, y1, y2, y3], dim=1) 
        return self.cv2(concat_feat) 
