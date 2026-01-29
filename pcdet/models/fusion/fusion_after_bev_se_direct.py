import torch,cv2
import torch.nn as nn
from torch.nn import functional as F
from .CFMFusion import *
from pcdet.models.backbones_3d.vfe.Vmamba.vmamba import *
from fightingcv_attention.attention.SelfAttention import *


class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.SiLU()
        )
    def forward(self, x):
        return x * self.att(x)

    
# 类似BEVFusion simple 中的融合模块 Sequeeze and Excitation
'''class FusionAfterBEVSEDirect(nn.Module):
    def __init__(self, model_cfg, num_bev_features,
                 image_in_channels,
                 image_out_channels,
                 radar_in_channels,
                 radar_out_channels,
                 **kwargs):
        super().__init__()

        if isinstance(image_in_channels, list):
            image_in_channels = sum(image_in_channels)
        
        self.model_cfg = model_cfg
        self.num_bev_features = num_bev_features
        # self.img_conv = nn.Sequential(
        #     nn.Conv2d(image_in_channels, image_out_channels, [1, 1]),
        #     nn.BatchNorm2d(image_out_channels),
        #     nn.ReLU()
        # )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(image_out_channels + radar_in_channels,
                                    image_out_channels + radar_out_channels,
                                    [3, 3], padding=1),
            nn.BatchNorm2d(image_out_channels + radar_out_channels),
            nn.ReLU()
        )
        self.se_block = SE_Block(image_out_channels + radar_out_channels)
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
    
    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"] # [B, 128, 320, 320]
        radar_features = batch_dict['pillar_features_scattered'] # [B, 128, 160, 160]

        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(image_features, radar_features.shape[-2:], mode='bilinear') # [B, 128, 160, 160]

        fuse_features = torch.concat([image_features, radar_features], dim=1) # [B, 256, 160, 160]
        fuse_features = self.fuse_conv(fuse_features) # [B, 256, 160, 160]

        fuse_features = self.se_block(fuse_features) # [B, 256, 160, 160]

        batch_dict[self.feature_name] = fuse_features
        return batch_dict'''
#output_ms
class FusionAfterBEVSEDirect1(nn.Module):
    def __init__(self, model_cfg, num_bev_features,
                 image_in_channels,
                 image_out_channels,
                 radar_in_channels,
                 radar_out_channels,
                 ms_in_channels=128,
                 ms_out_channels=128,
                 k_group=2,
                 d_model=128):
        super().__init__()
        
        
        self.model_cfg = model_cfg
        self.num_bev_features = num_bev_features
        self.ms_radar = Multi_Scale(ms_in_channels, ms_out_channels)
        self.ms_image = Multi_Scale(ms_in_channels, ms_out_channels)
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(image_out_channels + radar_in_channels,
                                    image_out_channels + radar_out_channels,
                                    [3, 3], padding=1),
            nn.BatchNorm2d(image_out_channels + radar_out_channels),
            nn.ReLU()
        )
       
      
        self.se_block = SE_Block(image_out_channels + radar_out_channels)
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
        '''self.alpha = nn.Parameter(torch.ones((1, ms_in_channels, 1, 1))
        self.beta = nn.Parameter(torch.ones((1, ms_in_channels, 1, 1))'''
        
    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"] # [B, 128, 320, 320]
        radar_features = batch_dict['pillar_features_scattered'] # [B, 128, 160, 160]

        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(image_features, radar_features.shape[-2:], mode='bilinear') # [B, 128, 160, 160]

        cx1_r, cx2_r, cx3_r = self.ms_radar(radar_features)
        radar_features = (cx1_r + cx2_r + cx3_r)/3
        
        cx1_i, cx2_i, cx3_i = self.ms_image(image_features)
        image_features = (cx1_i + cx2_i + cx3_i)/3
        
        fuse_features = torch.concat([image_features, radar_features], dim=1) # [B, 256, 160, 160]
       
        fuse_features = self.fuse_conv(fuse_features)
        fuse_features = self.se_block(fuse_features) # [B, 256, 160, 160]
        #
        #fuse_features2 = self.Mamba(fuse_features)
        

        batch_dict[self.feature_name] = fuse_features# + fuse_features2.permute(0, 3, 1, 2)
        return batch_dict
    
    
#RFIA    
class FusionAfterBEVSEDirect2(nn.Module):
    def __init__(self, model_cfg, num_bev_features,
                 image_in_channels,
                 image_out_channels,
                 radar_in_channels,
                 radar_out_channels,
                 ms_in_channels=128,
                 ms_out_channels=128,
                 k_group=2,
                 d_model=128):
        super().__init__()
        
        
        self.model_cfg = model_cfg
        self.num_bev_features = num_bev_features
        
        self.ms_radar = Multi_Scale(ms_in_channels, ms_out_channels)
        self.ms_image = Multi_Scale(ms_in_channels, ms_out_channels)
        self.p = nn.AdaptiveAvgPool2d(1)
        self.Sa = CrossModalityAttention(dim=d_model, dim_head=d_model)
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(image_out_channels + radar_in_channels,
                                    image_out_channels + radar_out_channels,
                                    [3, 3], padding=1),
            nn.BatchNorm2d(image_out_channels + radar_out_channels),
            nn.ReLU()
        )
       
      
        self.se_block = SE_Block(image_out_channels + radar_out_channels)
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
        '''self.alpha = nn.Parameter(torch.ones((1, ms_in_channels, 1, 1))
        self.beta = nn.Parameter(torch.ones((1, ms_in_channels, 1, 1))'''
        
    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"] # [B, 128, 320, 320]
        radar_features = batch_dict['pillar_features_scattered'] # [B, 128, 160, 160]

        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(image_features, radar_features.shape[-2:], mode='bilinear') # [B, 128, 160, 160]

        cx1_r, cx2_r, cx3_r = self.ms_radar(radar_features)
        radar_features1 = (cx1_r + cx2_r + cx3_r)/3 + radar_features
        cx1_i, cx2_i, cx3_i = self.ms_image(image_features)
        image_features = (cx1_i + cx2_i + cx3_i)/3 + image_features
        
        enh = self.Sa(self.p(radar_features1), self.p(image_features))
        image_features = image_features * enh
        #print(enh.shape)
        
        
        fuse_features = torch.concat([image_features, radar_features], dim=1) # [B, 256, 160, 160]
       
        fuse_features = self.fuse_conv(fuse_features)
        fuse_features = self.se_block(fuse_features) # [B, 256, 160, 160]
        #
        #fuse_features2 = self.Mamba(fuse_features)
        

        batch_dict[self.feature_name] = fuse_features# + fuse_features2.permute(0, 3, 1, 2)
        return batch_dict
    
# RFIA+DMF
class FusionAfterBEVSEDirect(nn.Module):
    def __init__(self, model_cfg, num_bev_features,
                 image_in_channels,
                 image_out_channels,
                 radar_in_channels,
                 radar_out_channels,
                 ms_in_channels=128,
                 ms_out_channels=128,
                 k_group=4,
                 d_model=128):
        super().__init__()
        
        
        self.model_cfg = model_cfg
        self.num_bev_features = num_bev_features
        
        self.ms_radar = Multi_Scale(ms_in_channels, ms_out_channels)
        self.ms_image = Multi_Scale(ms_in_channels, ms_out_channels)
        self.p = nn.AdaptiveAvgPool2d(1)
        self.Sa = ReceptiveFieldInteractiveAttention(dim=d_model, dim_head=d_model)
        
        self.Mamba = Fine_Fusion()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(image_out_channels + radar_in_channels,
                                    image_out_channels + radar_out_channels,
                                    [3, 3], padding=1),
            nn.BatchNorm2d(image_out_channels + radar_out_channels),
            nn.ReLU()
        )
       
      
        self.se_block = SE_Block(image_out_channels + radar_out_channels)
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
        '''self.alpha = nn.Parameter(torch.ones((1, ms_in_channels, 1, 1))
        self.beta = nn.Parameter(torch.ones((1, ms_in_channels, 1, 1))'''
        
    def forward(self, batch_dict):
        print(batch_dict.key())
        image_features = batch_dict["spatial_features"] # [B, 128, 320, 320]
        radar_features = batch_dict['pillar_features_scattered'] # [B, 128, 160, 160]

        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(image_features, radar_features.shape[-2:], mode='bilinear') # [B, 128, 160, 160]

        cx1_r, cx2_r, cx3_r = self.ms_radar(radar_features)
        radar_features1 = (cx1_r + cx2_r + cx3_r)/3 + radar_features
        cx1_i, cx2_i, cx3_i = self.ms_image(image_features)
        image_features = (cx1_i + cx2_i + cx3_i)/3 + image_features
        
        enh = self.Sa(self.p(radar_features1), self.p(image_features))
        image_features = image_features * enh
        #print(enh.shape)
        radar_features, image_features = self.Mamba(radar_features, image_features)
        
        fuse_features = torch.concat([image_features, radar_features], dim=1) # [B, 256, 160, 160]
       
        fuse_features = self.fuse_conv(fuse_features)
        fuse_features = self.se_block(fuse_features) # [B, 256, 160, 160]
        #
        #fuse_features2 = self.Mamba(fuse_features)
        
        
        batch_dict[self.feature_name] = fuse_features# + fuse_features2.permute(0, 3, 1, 2)
        return batch_dict
    

    
#Ablation
class FusionAfterBEVSEDirect3(nn.Module):
    def __init__(self, model_cfg, num_bev_features,
                 image_in_channels,
                 image_out_channels,
                 radar_in_channels,
                 radar_out_channels,
                 ms_in_channels=128,
                 ms_out_channels=128,
                 k_group=4,
                 d_model=128):
        super().__init__()
        
        
        self.model_cfg = model_cfg
        self.num_bev_features = num_bev_features
        
        self.ms_radar = Multi_Scale(ms_in_channels, ms_out_channels)
        self.ms_image = Multi_Scale(ms_in_channels, ms_out_channels)
        self.p = nn.AdaptiveAvgPool2d(1)
        self.Sa1 = ReceptiveFieldInteractiveAttention(dim=d_model, dim_head=d_model)
        self.Sa2 = ReceptiveFieldInteractiveAttention(dim=d_model, dim_head=d_model)
        
        self.Mamba = Fine_Fusion()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(image_out_channels + radar_in_channels,
                                    image_out_channels + radar_out_channels,
                                    [3, 3], padding=1),
            nn.BatchNorm2d(image_out_channels + radar_out_channels),
            nn.ReLU()
        )
       
      
        self.se_block = SE_Block(image_out_channels + radar_out_channels)
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
        '''self.alpha = nn.Parameter(torch.ones((1, ms_in_channels, 1, 1))
        self.beta = nn.Parameter(torch.ones((1, ms_in_channels, 1, 1))'''
        
    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"] # [B, 128, 320, 320]
        radar_features = batch_dict['pillar_features_scattered'] # [B, 128, 160, 160]

        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(image_features, radar_features.shape[-2:], mode='bilinear') # [B, 128, 160, 160]

        cx1_r, cx2_r, cx3_r = self.ms_radar(radar_features)
        radar_features1 = (cx1_r + cx2_r + cx3_r)/3 + radar_features
        cx1_i, cx2_i, cx3_i = self.ms_image(image_features)
        image_features1 = (cx1_i + cx2_i + cx3_i)/3 + image_features
        
        enh = self.Sa1(self.p(image_features1), self.p(radar_features))
        radar_features = radar_features * enh
        #print(enh.shape)
        enh1 = self.Sa2(self.p(radar_features1), self.p(image_features))
        image_features = image_features * enh1
        
        radar_features, image_features = self.Mamba(radar_features, image_features)
        
        fuse_features = torch.concat([image_features, radar_features], dim=1) # [B, 256, 160, 160]
       
        fuse_features = self.fuse_conv(fuse_features)
        fuse_features = self.se_block(fuse_features) # [B, 256, 160, 160]
        #
        #fuse_features2 = self.Mamba(fuse_features)
        

        batch_dict[self.feature_name] = fuse_features# + fuse_features2.permute(0, 3, 1, 2)
        return batch_dict