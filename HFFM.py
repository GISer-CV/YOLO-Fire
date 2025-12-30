class DAGB(nn.Module):  

    def __init__(self, in_channels, reduction_ratio=8, use_se=True, use_sa=True):
        super().__init__()
        self.use_se = use_se
        self.use_sa = use_sa

      
        se_reduced_channels = max(1, in_channels // reduction_ratio)
        groups = max(1, in_channels // 4)

     
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, se_reduced_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_reduced_channels, in_channels, 1),
                nn.Sigmoid()
            )


        if use_sa:
            self.sa = nn.Sequential(
                nn.Conv2d(in_channels, 1, 3, padding=1),
                nn.Sigmoid()
            )


        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=groups),  
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        out = x


        if self.use_se:
            se_weights = self.se(x)
            out = out * se_weights


        if self.use_sa:
            sa_weights = self.sa(out)
            out = out * sa_weights


        out = self.conv(out)

        return out



    
class PDB(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        mid_channels = max(1, in_channels // 2)

        # Wide-Field Branch
        self.context_branch = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            

            nn.Conv2d(mid_channels, mid_channels, 3, padding=2, dilation=2, 
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )

        # Detail Branch
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, 
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        context = self.context_branch(x)
        detail = self.detail_branch(x)
        
        combined = torch.cat([context, detail], dim=1)
        return self.fusion(combined)    
    
    
    
class HFFM(nn.Module):

    def __init__(self, in_channels, reduction_ratio=8, use_se=True, use_sa=True):
        super().__init__()


        assert in_channels > 0, f"in_channels must be positive, got {in_channels}"


        self.smoke_enhance = DAGB(in_channels, reduction_ratio, use_se, use_sa)


        self.small_target_enhance = PDB(in_channels)  


        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        smoke_feat = self.smoke_enhance(x)
        small_feat = self.small_target_enhance(x)

        if smoke_feat.shape != small_feat.shape:

            target_size = smoke_feat.shape[2:]
            if small_feat.shape[2:] != target_size:
                small_feat = F.interpolate(
                    small_feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=True
                )


        fused = torch.cat([smoke_feat, small_feat], dim=1)
        output = self.fusion_conv(fused)
        output = self.bn(output)
        output = self.activation(output)

        return output + x  
