import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 載入輕量級的 ResNet-18 預訓練模型
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 移除最後的全連接層與平均池化層，保留空間特徵圖
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # ResNet-18 萃取出的特徵圖通道數為 512

    def forward(self, x):
        # 輸入 x: (B, 3, 256, 256)
        # 輸出: (B, 512, 8, 8)
        return self.backbone(x)

class Matchmaker(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        # 握手訊息生成器 (Handshake Message Generator)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 評分網路，輸入為 Ego 與 Other 的握手訊息串接 (feature_dim * 2)
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # 輸出為單一分數 (Logit)
        )

    def forward(self, ego_feat, other_feats):
        """
        ego_feat: (B, C, 8, 8)
        other_feats: (B, N_others, C, 8, 8)
        返回: 挑選出的分數分布 (B, N_others)
        """
        B, N, C, H, W = other_feats.shape
        
        # 產生握手訊息
        msg_ego = self.gap(ego_feat).reshape(B, -1) 
        msg_other = self.gap(other_feats.reshape(B*N, C, H, W)).reshape(B*N, -1)
        
        # 重複 Ego 的握手訊息去跟每個人配對
        msg_ego_repeated = msg_ego.unsqueeze(1).expand(-1, N, -1).reshape(B*N, -1)
        
        # 串接並評分
        concat_msg = torch.cat([msg_ego_repeated, msg_other], dim=1) # (B*N, C*2)
        scores = self.scorer(concat_msg) # (B*N, 1)
        scores = scores.reshape(B, N) # (B, N_others)
        
        return scores

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # 使用 PyTorch 內建的 MultiheadAttention
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ego_feat, selected_feat):
        """
        ego_feat: (B, C, H, W)
        selected_feat: (B, C, H, W)
        """
        B, C, H, W = ego_feat.shape
        # 將特徵圖拉平成 Sequence (B, H*W, C)
        query = ego_feat.reshape(B, C, -1).permute(0, 2, 1)
        key_val = selected_feat.reshape(B, C, -1).permute(0, 2, 1)
        
        # Cross Attention: 讓 Ego (Query) 去找 Selected (Key/Value) 中有用的資訊
        attn_out, _ = self.attention(query, key_val, key_val)
        
        # Add & Norm
        fused = self.norm(query + attn_out)
        
        # 恢復成特徵圖形狀 (B, C, H, W)
        fused_feat = fused.permute(0, 2, 1).reshape(B, C, H, W)
        return fused_feat

class PoseHead(nn.Module):
    def __init__(self, feature_dim=512, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_joints * 2), # 2D 座標 (X, Y)
            nn.Sigmoid() # 歸一化輸出至 [0, 1] 區間
        )

    def forward(self, x):
        # x: (B, feature_dim, H, W)
        pose_flat = self.head(x) # (B, J*2)
        pose_2d = pose_flat.reshape(x.size(0), self.num_joints, 2)
        return pose_2d, None # 為了與 node.py 相容，第二個回傳值留空

class Who2comPoseNet(nn.Module):
    def __init__(self, num_views=4, num_joints=17, feature_dim=512):
        super().__init__()
        self.num_views = num_views
        self.extractor = FeatureExtractor()
        self.matchmaker = Matchmaker(feature_dim)
        self.fusion = CrossAttentionFusion(embed_dim=feature_dim)
        
        # 簡單的 2D Pose 回歸頭
        self.pose_head = PoseHead(feature_dim, num_joints)
        self.num_joints = num_joints

    def forward(self, views, temperature=1.0):
        """
        views: (B, V, 3, H, W) 其中 V 是視角數，假設 view 0 為 Ego
        temperature: Gumbel-Softmax 的溫度參數
        """
        B, V, C, H, W = views.shape
        
        # 1. 獨立萃取所有視角的特徵圖 (Shared Weights)
        # 將 B 和 V 合併送入 Extractor 以平行運算
        all_feats = self.extractor(views.reshape(B*V, C, H, W))
        all_feats = all_feats.reshape(B, V, -1, all_feats.shape[-2], all_feats.shape[-1]) # (B, V, 512, 8, 8)
        
        ego_feat = all_feats[:, 0]
        other_feats = all_feats[:, 1:] # (B, V-1, 512, 8, 8)
        
        # 2. Matchmaker 評分與 Top-1 挑選
        scores = self.matchmaker(ego_feat, other_feats) # (B, V-1)
        
        # 使用 Gumbel Softmax 進行可微的近似 Argmax 抽樣
        # 在訓練時會有隨機性但可傳遞梯度，推論時(hard=True)會直接轉為 One-hot 向量
        weights = F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1) # (B, V-1)
        
        # 3. 提取被選中的特徵圖
        # weight: (B, V-1, 1, 1, 1) 乘以 other_feats: (B, V-1, 512, 8, 8) 然後加總
        weights_expanded = weights.reshape(B, V-1, 1, 1, 1)
        selected_feat = (other_feats * weights_expanded).sum(dim=1) # (B, 512, 8, 8)
        
        # 4. 特徵融合 (Cross Attention)
        fused_feat = self.fusion(ego_feat, selected_feat) # (B, 512, 8, 8)
        
        # 5. 回歸 2D 關節點 (X, Y)
        pose_2d, _ = self.pose_head(fused_feat) # (B, J, 2)
        
        # 回傳預測的 Pose，以及注意力權重(用來觀測我們挑了哪一個視角)
        return pose_2d, weights

if __name__ == "__main__":
    # 測試網路
    net = Who2comPoseNet(num_views=3)
    dummy_input = torch.rand(2, 3, 3, 256, 256) # Batch=2, Views=3
    pose, weights = net(dummy_input)
    print(f"Output Pose shape: {pose.shape}")
    print(f"Selected View weights:\n{weights}")
