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

class CompressedMatchmaker(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        # 輸入：自己和對方的壓縮摘要 (x, y, confidence) -> 17*3 = 51
        self.scorer = nn.Sequential(
            nn.Linear(51 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, ego_coords, ego_conf, other_coords, other_conf):
        """
        所有維度: ego (B, 17, 2), (B, 17)
                 other (B, V-1, 17, 2), (B, V-1, 17)
        """
        B, V_minus_1 = other_coords.shape[0], other_coords.shape[1]
        
        # 壓平 ego 摘要
        ego_summary = torch.cat([ego_coords, ego_conf.unsqueeze(-1)], dim=-1) # (B, 17, 3)
        ego_summary = ego_summary.view(B, -1).unsqueeze(1).expand(-1, V_minus_1, -1) # (B, V-1, 51)
        
        # 壓平 other 摘要
        other_summary = torch.cat([other_coords, other_conf.unsqueeze(-1)], dim=-1) # (B, V-1, 17, 3)
        other_summary = other_summary.view(B, V_minus_1, -1) # (B, V-1, 51)
        
        pair_features = torch.cat([ego_summary, other_summary], dim=-1) # (B, V-1, 102)
        scores = self.scorer(pair_features).squeeze(-1) # (B, V-1)
        return scores

class HeatmapCrossAttention(nn.Module):
    def __init__(self, channels=17, embed_dim=64):
        super().__init__()
        # 在 heatmap 上做 attention。因為 channel 只有 17，我們用 1x1 conv 升維做投影
        self.query_proj = nn.Conv2d(channels, embed_dim, 1)
        self.key_proj = nn.Conv2d(channels, embed_dim, 1)
        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.scale = math.sqrt(embed_dim)
        
        # 通訊殘差權重，初始化為很小的值
        self.alpha = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, ego_hm, other_hm):
        """
        ego_hm: (B, 17, 64, 64)
        other_hm: (B, 17, 64, 64)
        沒有相機參數時的一般 Cross-Attention (簡化版 Epipolar，讓模型自己學對應關係)
        """
        B, C, H, W = ego_hm.shape
        
        # (B, embed, H*W)
        Q = self.query_proj(ego_hm).view(B, -1, H*W)
        K = self.key_proj(other_hm).view(B, -1, H*W)
        V = self.value_proj(other_hm).view(B, -1, H*W)
        
        # Attention Map: (B, H*W, H*W)
        attn = torch.bmm(Q.transpose(1, 2), K) / self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Output: (B, channels, H*W) -> (B, channels, H, W)
        out = torch.bmm(V, attn.transpose(1, 2))
        out = out.view(B, C, H, W)
        
        # 殘差融合
        fused_hm = ego_hm + self.alpha * out
        return fused_hm

class When2comHeatmapNet(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        self.extractor = FeatureExtractor()
        self.decoder = HeatmapDecoder(num_joints=num_joints)
        
        self.gate = EntropyGate(num_joints=num_joints)
        self.matchmaker = CompressedMatchmaker(num_joints=num_joints)
        self.fusion = HeatmapCrossAttention(channels=num_joints)
        
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
        other_hms = hms[:, 1:]
        
        # 2. 提取壓縮摘要 (座標和信心)
        ego_coords = soft_argmax_2d(ego_hm)
        ego_conf = get_confidence(ego_hm)
        
        other_coords = soft_argmax_2d(other_hms.contiguous().view(-1, self.num_joints, 64, 64)).view(B, V-1, self.num_joints, 2)
        other_conf = get_confidence(other_hms.contiguous().view(-1, self.num_joints, 64, 64)).view(B, V-1, self.num_joints)
        
        # 3. Gate 決策
        comm_prob = self.gate(ego_hm) # (B, 1)
        
        if force_comm:
            gate_mask = torch.ones_like(comm_prob)
        elif force_no_comm:
            gate_mask = torch.zeros_like(comm_prob)
        else:
            # Inference mode: 用 0.5 切割
            gate_mask = (comm_prob > 0.5).float()
            
        # 4. 只有在需要通訊時才做 Matchmaker 和 Fusion
        final_hm = ego_hm.clone()
        
        if gate_mask.sum() > 0:
            # 打分數
            scores = self.matchmaker(ego_coords, ego_conf, other_coords, other_conf) # (B, V-1)
            # Gumbel Softmax 選擇隊友
            weights = F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1) # (B, V-1)
            
            # 選出最佳隊友的 heatmap
            # (B, V-1, 1, 1, 1) * (B, V-1, 17, 64, 64)
            selected_hm = (other_hms * weights.view(B, V-1, 1, 1, 1)).sum(dim=1) 
            
            # 融合
            fused_hm = self.fusion(ego_hm, selected_hm)
            
            # 根據 gate_mask 混合
            gate_mask_hm = gate_mask.view(B, 1, 1, 1)
            final_hm = gate_mask_hm * fused_hm + (1 - gate_mask_hm) * ego_hm
            
        # 5. 最終預測
        pred_coords = soft_argmax_2d(final_hm)
        pred_conf = get_confidence(final_hm)
        
        return pred_coords, pred_conf, final_hm, comm_prob
