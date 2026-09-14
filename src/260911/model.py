import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34, ResNet34_Weights
import math

def soft_argmax_2d(heatmaps, temp=10.0):
    """
    從 Heatmap 中可微分地提取 2D 座標。
    heatmaps: (B, num_joints, H, W)
    回傳: (B, num_joints, 2) 座標值在 [0, 1] 之間
    """
    B, C, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B, C, -1)
    # 套用 softmax 強化 peak
    probs = F.softmax(heatmaps * temp, dim=-1)
    probs = probs.view(B, C, H, W)
    
    # 建立網格座標 [0, 1]
    y_grid = torch.linspace(0.0, 1.0, H, device=heatmaps.device, dtype=heatmaps.dtype)
    x_grid = torch.linspace(0.0, 1.0, W, device=heatmaps.device, dtype=heatmaps.dtype)
    y_grid = y_grid.view(1, 1, H, 1)
    x_grid = x_grid.view(1, 1, 1, W)
    
    # 期望值 (Expected value)
    y_coords = torch.sum(probs * y_grid, dim=(2, 3))
    x_coords = torch.sum(probs * x_grid, dim=(2, 3))
    
    coords = torch.stack([x_coords, y_coords], dim=-1)  # (B, C, 2)
    return coords

def get_confidence(heatmaps):
    """
    從 Heatmap 中取得每個關節的信心分數 (peak value)
    heatmaps: (B, num_joints, H, W)
    回傳: (B, num_joints)
    """
    # 簡單的做法是取 max
    B, C, H, W = heatmaps.shape
    confidence = heatmaps.view(B, C, -1).max(dim=-1)[0]
    return confidence

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        # 去掉最後的 pooling 和 fc，取得 1/32 解析度的特徵
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
    def forward(self, x):
        # x: (B, 3, 256, 256)
        return self.backbone(x)  # (B, 512, 8, 8)

class HeatmapDecoder(nn.Module):
    def __init__(self, in_channels=512, num_joints=17):
        super().__init__()
        # 3 層 Deconv (8x8 -> 16x16 -> 32x32 -> 64x64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # 輸出 17 個 channel 對應 17 個 joint
        self.head = nn.Conv2d(256, num_joints, kernel_size=1)
        
    def forward(self, x):
        features = self.deconv(x)
        heatmaps = self.head(features) # (B, 17, 64, 64)
        # 不要過 Sigmoid，保持 raw logits，訓練時用 MSE 跟 Gaussian peak 比
        return heatmaps

class EntropyGate(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        # 把 Entropy 轉換成決策機率的小 MLP，保留彈性
        self.mlp = nn.Sequential(
            nn.Linear(num_joints, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, heatmaps):
        """
        heatmaps: (B, 17, 64, 64)
        """
        B, C, H, W = heatmaps.shape
        probs = F.softmax(heatmaps.view(B, C, -1), dim=-1)
        # 計算 Entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1) # (B, 17)
        
        # 將 Entropy 轉換為通訊機率
        # entropy 越高，越看不清楚，越需要通訊
        comm_prob = self.mlp(entropy) # (B, 1)
        return comm_prob

class SoftmaxViewWeighter(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        # 輸入：每個視角的壓縮摘要 (17*3 = 51)
        self.scorer = nn.Sequential(
            nn.Linear(51, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, coords, conf):
        # coords: (B, V, 17, 2), conf: (B, V, 17)
        B, V = coords.shape[0], coords.shape[1]
        summary = torch.cat([coords, conf.unsqueeze(-1)], dim=-1) # (B, V, 17, 3)
        summary = summary.view(B, V, -1) # (B, V, 51)
        scores = self.scorer(summary).squeeze(-1) # (B, V)
        weights = F.softmax(scores, dim=-1) # (B, V)
        return weights

class When2comHeatmapNet(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        self.extractor = FeatureExtractor()
        self.decoder = HeatmapDecoder(num_joints=num_joints)
        
        self.gate = EntropyGate(num_joints=num_joints)
        self.view_weighter = SoftmaxViewWeighter(num_joints=num_joints)
        
    def forward(self, images, force_comm=False, force_no_comm=False, temperature=1.0):
        """
        images: (B, V, 3, 256, 256)
        """
        B, V, C, H, W = images.shape
        
        # 1. 提取所有視角的特徵並解碼成 Heatmap
        images_flat = images.view(B * V, C, H, W)
        feats_flat = self.extractor(images_flat)
        hms_flat = self.decoder(feats_flat)
        
        hms = hms_flat.view(B, V, self.num_joints, 64, 64)
        ego_hm = hms[:, 0]
        
        # 2. 提取壓縮摘要 (座標和信心)
        all_coords = soft_argmax_2d(hms.contiguous().view(-1, self.num_joints, 64, 64)).view(B, V, self.num_joints, 2)
        all_conf = get_confidence(hms.contiguous().view(-1, self.num_joints, 64, 64)).view(B, V, self.num_joints)
        
        # 3. Softmax 視角加權
        weights = self.view_weighter(all_coords, all_conf) # (B, V)
        
        # 加權融合 Heatmap
        fused_hm = (hms * weights.view(B, V, 1, 1, 1)).sum(dim=1) # (B, 17, 64, 64)
        
        # 4. Gate 決策
        comm_prob = self.gate(ego_hm) # (B, 1)
        
        if force_comm:
            gate_mask = torch.ones_like(comm_prob)
        elif force_no_comm:
            gate_mask = torch.zeros_like(comm_prob)
        else:
            # Inference mode: 使用平滑的 Gate 混合
            gate_mask = comm_prob
            
        # 根據 gate_mask 混合 (方案二核心)
        gate_mask_hm = gate_mask.view(B, 1, 1, 1)
        final_hm = gate_mask_hm * fused_hm + (1 - gate_mask_hm) * ego_hm
            
        # 5. 最終預測
        pred_coords = soft_argmax_2d(final_hm)
        pred_conf = get_confidence(final_hm)
        
        return pred_coords, pred_conf, final_hm, comm_prob
