import torch, math
import torch.nn as nn
from torch.nn import functional as F
from pcdet.models.backbones_3d.vfe.Vmamba.vmamba import mamba_init
from pcdet.models.backbones_3d.vfe.Vmamba.csms6s import SelectiveScanOflex

############################ Difference Scanning and Multi-Scale Scanning ######################################
#   Author: Shucong Li 
#   This part is resposible for fusion stage, for the purposes of light-weight and interactivity improvement #####################

######################### Difference Scanning ###############################################
def cross(seq1: torch.tensor, seq2: torch.tensor):
    stacked = torch.cat([seq1, seq2], dim=0) 
    #print(stacked.shape)
    B, C, L = stacked.shape
    transposed = stacked.reshape(1, C, B * L) 
    return transposed

def recover(seq: torch.tensor):
    B, C, L = seq.shape
    seq = seq.reshape(2, C, L//2)
    seq1, seq2 = seq[0, :, :], seq[1, :, :]
    return seq1[None, ...], seq2[None, ...]

class Difference_Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sub: torch.Tensor, x_vi: torch.Tensor, x_ir: torch.Tensor):
        # B, C, H, W -> B, 2, C, 2 * H * W
        B, C, H, W = x_vi.shape
        ctx.shape = (B, C, H, W)
        x_fus = x_vi.new_empty((B, 4, C, 2 * H * W))
        x_fus[:, 0] = torch.concat([x_vi.flatten(2, 3), x_sub.flatten(2, 3)], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        x_fus[:, 2] = torch.concat([x_ir.flatten(2, 3), x_sub.flatten(2, 3)], dim=2)
        x_fus[:, 3] = torch.flip(x_fus[:, 2], dims=[-1])
        return x_fus

    @staticmethod
    def backward(ctx, x_fus: torch.Tensor):
        # out: (b, 2, d, l)
        B, C, H, W = ctx.shape
        # L = 2 * H * W
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W
        x_fus_2 = x_fus[:, 2] + x_fus[:, 3].flip(dims=[-1])

        # get B, d, H*W
        return (
            ((x_fus_1[:, :, H * W : 2 * H * W] + x_fus_2[:, :, H * W : 2 * H * W]) / 2).view(B, -1, H, W),
            x_fus_1[:, :, 0 : H * W].view(B, -1, H, W),
            x_fus_2[:, :, 0 : H * W].view(B, -1, H, W),
        )


class Difference_Merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_fus: torch.Tensor):
        B, K, D, L = x_fus.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
        x_fus_2 = x_fus[:, 2] + x_fus[:, 3].flip(dims=[-1])
        # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
        return (
            (x_fus_1[:, :, L // 2 : L] + x_fus_2[:, :, L // 2 : L]) / 2,
            x_fus_1[:, :, 0 : L // 2],
            x_fus_2[:, :, 0 : L // 2],
        )

    @staticmethod
    def backward(ctx, x_sub: torch.Tensor, x_vi: torch.Tensor, x_ir: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = x_vi.shape
        x_fus = x_vi.new_empty((B, 4, C, 2 * L))

        x_fus[:, 0] = torch.cat([x_vi, x_sub], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        x_fus[:, 2] = torch.cat([x_ir, x_sub], dim=2)
        x_fus[:, 3] = torch.flip(x_fus[:, 2], dims=[-1])
        x_fus = x_fus.view(B, 4, C, 2 * L)
        return x_fus

######################### Multi-Scale Scanning ###############################################
class Multi_Scale_Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cx1: torch.Tensor, cx2: torch.Tensor, cx3: torch.Tensor):
        # B, C, H, W -> B, 2, C, 2 * H * W
        B, C, H, W = cx1.shape
        ctx.shape = (B, C, H, W)
        x_fus = cx1.new_empty((B, 2, C, 3 * H * W))
        x_fus[:, 0] = torch.cat([cx1.flatten(2, 3), cx2.flatten(2, 3), cx3.flatten(2, 3)], dim = -1)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        return x_fus

    @staticmethod
    def backward(ctx, x_fus: torch.Tensor):
        # out: (b, 2, d, l)
        B, C, H, W = ctx.shape
        # L = 2 * H * W
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W

        return (
            x_fus_1[:, :, 0 : H * W].view(B, -1, H, W),
            x_fus_1[:, :, H * W : 2 * H * W].view(B, -1, H, W),
            x_fus_1[:, :, 2 * H * W : ].view(B, -1, H, W)
        )

class Multi_Scale_Merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_fus: torch.Tensor):
        B, K, D, L = x_fus.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
        sl = int(L//3)
        return (
            x_fus_1[:, :, 0 : sl],
            x_fus_1[:, :, sl : 2 * sl],
            x_fus_1[:, :, 2 * sl : ]
        )

    @staticmethod
    def backward(ctx, cx1: torch.Tensor, cx2: torch.Tensor, cx3: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = cx1.shape
        x_fus = cx1.new_empty((B, 2, C, 3 * L))

        x_fus[:, 0] = torch.cat([cx1, cx2, cx3], dim = -1)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        x_fus = x_fus.view(B, 2, C, 3 * L)
        return x_fus
    
################# The code of Coarse Mamba Fusion ############################
    
class Coarse_Mamba_Fusion(nn.Module, mamba_init):
    def __init__(
        self,
        d_model=128,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        k_group=4,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super(Coarse_Mamba_Fusion, self).__init__()

        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        d_proj = self.d_inner
        self.in_proj_vi = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_ir = nn.Linear(self.d_model, d_proj, bias=bias, **factory_kwargs)
        self.in_proj_sub = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.out_norm_vi = nn.LayerNorm(self.d_inner)
        self.out_norm_ir = nn.LayerNorm(self.d_inner)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d_vi = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_ir = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_sub = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # out proj =======================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # x proj ============================
        #k_group = 4
        self.k_group = k_group
        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for _ in range(self.k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(self.k_group)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.k_group, merge=True)  # (K * D)

    def forward_core(
        self,
        x_sub: torch.Tensor,
        x_vi: torch.Tensor,
        x_ir: torch.Tensor,
        x_proj_bias: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        delta_softplus=True,
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        SelectiveScan=SelectiveScanOflex,
        CrossScan = Difference_Scan,
        CrossMerge = Difference_Merge
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x_vi.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

        xs = CrossScan.apply(x_sub, x_vi, x_ir)  # B, C, H, W -> B, 4, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        xs: torch.Tensor = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, 2 * H * W)

        _, y_vi, y_ir = CrossMerge.apply(xs)

        y_vi = y_vi.view(B, -1, H, W)
        y_vi = y_vi.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_vi = self.out_norm_vi(y_vi).view(B, H, W, -1)

        y_ir = y_ir.view(B, -1, H, W)
        y_ir = y_ir.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_ir = self.out_norm_ir(y_ir).view(B, H, W, -1)

        y_vi = y_vi.to(x_vi.dtype)
        y_ir = y_ir.to(x_ir.dtype)
        return y_vi, y_ir

    def forward(self, x_vi: torch.Tensor, x_ir: torch.Tensor, before=True):
        #x_vi: B,C,H,W
        x_vi = x_vi.permute(0, 3, 2, 1) # B, H, W, C
        x_ir = x_ir.permute(0, 3, 2, 1)
        if before:
            x_sub = x_vi - x_ir
            x_sub = self.in_proj_sub(x_sub)
            x_sub_trans = x_sub.permute(0, 3, 1, 2).contiguous()

        x_vi = self.in_proj_vi(x_vi)
        x_ir = self.in_proj_ir(x_ir)

        x_vi_trans = x_vi.permute(0, 3, 1, 2).contiguous()
        x_ir_trans = x_ir.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x_vi_conv = self.act(self.conv2d_vi(x_vi_trans))
            x_ir_conv = self.act(self.conv2d_ir(x_ir_trans))

            if before:
                x_sub_conv = self.act(self.conv2d_sub(x_sub_trans))  # (b, d, h, w)
            else:
                x_sub_conv = x_vi_conv - x_ir_conv

        y_vi, y_ir = self.forward_core(x_sub=x_sub_conv, 
                                       x_vi=x_vi_conv, 
                                       x_ir=x_ir_conv)  # b, d, h, w -> b, h, w, d
        
        y_vis = y_vi + x_vi
        y_irs = y_ir + x_ir
        
        y_vis = self.dropout(self.out_proj(y_vis))
        y_irs = self.dropout(self.out_proj(y_irs))
        
        return y_vis.permute(0, 3, 1, 2), y_irs.permute(0, 3, 1, 2)   
#########################################################################################

################# The code of Fine Mamba Fusion ############################

class Fine_Mamba_Fusion(Coarse_Mamba_Fusion, mamba_init):
    def __init__(self, k_group_fine):
        # Here, we no longer redefine the constructor. 
        # Instead, we simply inherit from the parent class and perform default initialization.
        Coarse_Mamba_Fusion.__init__(self,
                                    d_model=256,
                                    d_state=4,
                                    ssm_ratio=2,
                                    dt_rank="auto",
                                    d_conv=3,
                                    conv_bias=True,
                                    dropout=0.0,
                                    bias=False,
                                    dt_min=0.001,
                                    dt_max=0.1,
                                    dt_init="random",
                                    dt_scale=1.0,
                                    dt_init_floor=1e-4,
                                    k_group=k_group_fine)
        factory_kwargs = {"device": None, "dtype": None}
        self.out_norm_cx3 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        
    def forward_core(
        self,
        x_cx1_conv: torch.Tensor,
        x_cx2_conv: torch.Tensor,
        x_cx3_conv: torch.Tensor,
        x_proj_bias: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        delta_softplus=True,
        nrows=-1,
        backnrows=-1, 
        ssoflex=True,
        SelectiveScan=SelectiveScanOflex,
        CrossScan = Multi_Scale_Scan,
        CrossMerge = Multi_Scale_Merge
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x_cx1_conv.shape
        D, N = A_logs.shape
        #print(A_logs.shape)
        K, D, R = dt_projs_weight.shape
        #print(dt_projs_weight.shape)
        L = 3 * H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

        xs = CrossScan.apply(x_cx1_conv, x_cx2_conv, x_cx3_conv)  # B, C, H, W -> B, 4, C, 2 * H * W
        #print(xs.shape)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
        xs = xs.view(B, -1, L)
        
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        xs: torch.Tensor = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, 3 * H * W)

        out_cx1, out_cx2, out_cx3 = CrossMerge.apply(xs)

        out_cx1 = out_cx1.view(B, -1, H, W)
        out_cx1 = out_cx1.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous() 
        out_cx1 = self.out_norm_vi(out_cx1).view(B, H, W, -1)

        out_cx2 = out_cx2.view(B, -1, H, W)
        out_cx2 = out_cx2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()
        out_cx2 = self.out_norm_ir(out_cx2).view(B, H, W, -1)
        
        out_cx3 = out_cx3.view(B, -1, H, W)
        out_cx3 = out_cx3.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()
        out_cx3 = self.out_norm_cx3(out_cx3).view(B, H, W, -1)

        out_cx1 = out_cx1.to(out_cx1.dtype)
        out_cx2 = out_cx2.to(out_cx2.dtype)
        out_cx3 = out_cx3.to(out_cx3.dtype)
        
        return out_cx1, out_cx2, out_cx3
    
    def forward(self, cx1: torch.Tensor, cx2: torch.Tensor, cx3: torch.Tensor):
        cx1, cx2, cx3 = cx1.permute(0, 3, 2, 1), cx2.permute(0, 3, 2, 1), cx3.permute(0, 3, 2, 1)
        x_cx3 = self.in_proj_sub(cx3)    
        x_cx2 = self.in_proj_vi(cx2)
        x_cx1 = self.in_proj_ir(cx1)

        x_cx1_trans = x_cx1.permute(0, 3, 1, 2).contiguous()
        x_cx2_trans = x_cx2.permute(0, 3, 1, 2).contiguous()
        x_cx3_trans = x_cx3.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x_cx1_conv = self.act(self.conv2d_vi(x_cx1_trans))
            x_cx2_conv = self.act(self.conv2d_ir(x_cx2_trans))
            x_cx3_conv = self.act(self.conv2d_sub(x_cx3_trans)) 

        out_cx1, out_cx2, out_cx3 = self.forward_core(x_cx1_conv, 
                                       x_cx2_conv, 
                                       x_cx3_conv)  
        y = self.dropout(self.out_proj(out_cx1 + out_cx2 + out_cx3))
        return y 

    
class Multi_Scale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 7, padding=3, bias=False,groups=in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 5, padding=2, bias=False,groups=in_channel)
        self.conv3 = nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False,groups=in_channel)
        '''self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.silu3 = nn.SiLU()'''

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return x1,x2,x3
    
    
class CFMFusion(nn.Module):
    def __init__(self, 
                 model_cfg, 
                 num_bev_features,
                 ms_in_channels,
                 ms_out_channels,
                 d_model=128,
                 d_state=4,
                 ssm_ratio=2,
                 dt_rank="auto",
                 d_conv=3, 
                 conv_bias=True,
                 dropout=0.0,
                 bias=False,
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 disable_z=False,
                 k_group_coarse=4,
                 k_group_fine=2,
                 Coarse = True,
                 Fine = True,
                 **kwargs):
        super(CFMFusion, self).__init__()
        
        self.model_cfg = model_cfg
        self.num_bev_features = num_bev_features
        self.feature_name = self.model_cfg.get('OUTPUT_FEATURE', 'spatial_features_2d')
        self.Coarse, self.Fine = Coarse, Fine
        assert self.Coarse or self.Fine
        
        if self.Coarse:
            self.Coarse_Mamba_Fusion = Coarse_Mamba_Fusion(
                                        d_model=d_model,
                                        d_state=d_state,
                                        ssm_ratio=ssm_ratio,
                                        dt_rank=dt_rank,
                                        d_conv=d_conv, 
                                        conv_bias=conv_bias,
                                        dropout=dropout,
                                        bias=bias,
                                        dt_min=dt_min,
                                        dt_max=dt_max,
                                        dt_init=dt_init,
                                        dt_scale=dt_scale,
                                        dt_init_floor=dt_init_floor,
                                        disable_z=disable_z,
                                        k_group=k_group_coarse,
            )
            
        if self.Fine:
            self.Fine_Mamba_Fusion = Fine_Mamba_Fusion(k_group_fine=k_group_fine)
        
        self.ms = Multi_Scale(ms_in_channels, ms_out_channels)
        
    
    def forward(self, batch_dict):
        image_features = batch_dict["spatial_features"] # [B, 128, 320, 320]
        radar_features = batch_dict['pillar_features_scattered'] # [B, 128, 160, 160]

        if image_features.shape[-2:] != radar_features.shape[-2:]:
            image_features = F.interpolate(image_features, radar_features.shape[-2:], mode='bilinear') # [B, 128, 160, 160]

        coarse_feature = self.Coarse_Mamba_Fusion(radar_features, image_features) #1 320 320 128
        coarse_feature = coarse_feature.permute(0, 3, 1, 2)  # 1 128 320 320
        cx1, cx2, cx3 = self.ms(coarse_feature)
        #print(cx3.shape)
        fuse_features = self.Fine_Mamba_Fusion(cx1, cx2, cx3)#[B, 256, 160, 160]
        #print(fuse_features,shape)
        fuse_features = fuse_features.permute(0, 3, 1, 2)
        batch_dict[self.feature_name] = fuse_features
        return batch_dict
    
    

class Scanning(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor):
        # B, C, H, W -> B, 2, C, 2 * H * W
        B, C, H,W= x1.shape
        ctx.shape = (B, C, H, W)
        x_fus = x1.new_empty((B, 2, C, 2*H*W))
        x_fus[:, 0] = torch.concat([x1.flatten(2, 3), x2.flatten(2, 3)], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        return x_fus

    @staticmethod
    def backward(ctx, x_fus: torch.Tensor):
        # out: (b, 2, d, l)
        #print(x_fus.shape)
        B, C, H, W = ctx.shape
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W
        return (
            x_fus_1[:, :, 0 : H*W].view(B, -1, H,W),
            x_fus_1[:, :, H*W: 2 *H*W].view(B, -1, H,W)
        )
        
class Recover(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_fus: torch.Tensor):
        B, K, D, L = x_fus.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
        return (
            x_fus_1[:, :, 0 : L // 2],
            x_fus_1[:, :, L // 2 : L],
        )

    @staticmethod
    def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = x1.shape
        x_fus = x1.new_empty((B, 2, C, 2 * L))

        x_fus[:, 0] = torch.cat([x1,x2], dim=2)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        return x_fus
    
class Scanning_one(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor):
        # B, C, H, W -> B, 2, C, 2 * H * W
        B, C, H,W= x1.shape
        ctx.shape = (B, C, H, W)
        x_fus = x1.new_empty((B, 2, C, H*W))
        x_fus[:, 0] = x1.flatten(2, 3)
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        return x_fus

    @staticmethod
    def backward(ctx, x_fus: torch.Tensor):
        # out: (b, 2, d, l)
        #print(x_fus.shape)
        B, C, H, W = ctx.shape
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W
        return x_fus_1[:, :, 0 : H*W].view(B, -1, H,W)
    
        
class Recover_one(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_fus: torch.Tensor):
        B, K, D, L = x_fus.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        x_fus_1 = x_fus[:, 0] + x_fus[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
        return x_fus_1[:, :, 0 : ],
        

    @staticmethod
    def backward(ctx, x1: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = x1.shape
        x_fus = x1.new_empty((B, 2, C, L))

        x_fus[:, 0] = x1
        x_fus[:, 1] = torch.flip(x_fus[:, 0], dims=[-1])
        return x_fus

class Mamba_Fusion(Coarse_Mamba_Fusion, mamba_init):
    def __init__(self, k_group, d_model):
        # Here, we no longer redefine the constructor. 
        # Instead, we simply inherit from the parent class and perform default initialization.
        Coarse_Mamba_Fusion.__init__(self,
                                    d_model=d_model,
                                    d_state=4,
                                    ssm_ratio=2,
                                    dt_rank="auto",
                                    d_conv=3,
                                    conv_bias=True,
                                    dropout=0.0,
                                    bias=False,
                                    dt_min=0.001,
                                    dt_max=0.1,
                                    dt_init="random",
                                    dt_scale=1.0,
                                    dt_init_floor=1e-4,
                                    k_group=k_group)
        factory_kwargs = {"device": None, "dtype": None}
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        #self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, stride=1),
            nn.SiLU()
        )
        
    def forward_core(
        self,
        x_vi: torch.Tensor,
        x_sub: torch.Tensor,
        x_proj_bias: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        delta_softplus=True,
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        SelectiveScan=SelectiveScanOflex,
        num = 2,
        CrossScan = Scanning,
        CrossMerge = Recover
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x_vi.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = num * H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

        xs = CrossScan.apply(x_vi, x_sub)  # B, C, H, W -> B, 4, C, 2 * H * W
      
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        xs: torch.Tensor = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, 2 * H * W)

        y_vi, _ = CrossMerge.apply(xs)

        y_vi = y_vi.view(B, -1, H, W)
        y_vi = y_vi.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_vi = self.out_norm_vi(y_vi).view(B, H, W, -1)

        '''y_ir = y_ir.view(B, -1, H, W)
        y_ir = y_ir.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y_ir = self.out_norm_ir(y_ir).view(B, H, W, -1)'''

        y_vi = y_vi.to(x_vi.dtype)
       # y_ir = y_ir.to(x_ir.dtype)
        return y_vi
    
    def forward(self, x_vi: torch.Tensor, x_ir):
        #x_vi: B,C,H,W
        x_vi = x_vi.permute(0, 3, 2, 1) # B, H, W, C
        x_ir = x_ir.permute(0, 3, 2, 1)

        x_vi = self.in_proj_vi(x_vi)
        x_ir = self.in_proj_ir(x_ir)

        x_vi_trans = x_vi.permute(0, 3, 1, 2).contiguous()
        x_ir_trans = x_ir.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x_vi_conv = self.act(self.conv2d_vi(x_vi_trans))
            x_ir_conv = self.act(self.conv2d_ir(x_ir_trans))

        y_vi = self.forward_core(x_vi=x_vi_conv,x_sub = x_ir_conv)  # b, d, h, w -> b, h, w, d
        
        
        y_vi = self.dropout(self.out_proj(y_vi))
        y_vi = y_vi.permute(0, 3, 1, 2)
        
        #print(y_vis.shape)
        
        return y_vi
    
    
