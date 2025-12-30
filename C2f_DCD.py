class DCD(nn.Module):
    """
    DCD (Dual-Context Diffusion) Block
    """
    def __init__(self, c1, c2, ratio=1.0, k=3, s=1):
        super().__init__()
        c_mid = int(c2 * ratio)
        self.dim = c_mid // 2
        

        self.cv1 = Conv(c1, c_mid, 1, 1)
        
        # Context Branch
        self.dw_context = nn.Conv2d(self.dim, self.dim, k, s, autopad(k, d=2), 
                                    dilation=2, groups=self.dim, bias=False)
        self.bn_context = nn.BatchNorm2d(self.dim) 
        # Local Branch
        self.dw_local = nn.Conv2d(self.dim, self.dim, k, s, autopad(k), 
                                  groups=self.dim, bias=False)
        self.bn_local = nn.BatchNorm2d(self.dim)


        self.cv2 = Conv(c_mid, c2, 1, 1)
        self.act = FReLU(c1=c1)
        #nn.SiLU()
        self.add = s == 1 and c1 == c2

    def forward(self, x):
        residual = x if self.add else None

        x = self.cv1(x)
        

        x_local, x_context = torch.split(x, self.dim, dim=1)
        

        out_context = self.bn_context(self.dw_context(x_context))
        

        fused_input = x_local + out_context
        

        out_local = self.bn_local(self.dw_local(fused_input))
        

        x = torch.cat((out_local, out_context), dim=1)
        x = self.act(x)
        

        x = self.cv2(x)
        
        return x + residual if self.add else x

    
class C2f_DCD(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(DCD(self.c, self.c, ratio=1.0) for _ in range(n))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
