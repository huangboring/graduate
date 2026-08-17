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
            nn.Linear(512, 1)
        )

    def forward(self, ego_feat, other_feats):
        B, N, C, H, W = other_feats.shape
        msg_ego = self.gap(ego_feat).reshape(B, -1)
        msg_other = self.gap(other_feats.reshape(B*N, C, H, W)).reshape(B*N, -1)
        msg_ego_repeated = msg_ego.unsqueeze(1).expand(-1, N, -1).reshape(B*N, -1)
        concat_msg = torch.cat([msg_ego_repeated, msg_other], dim=1)
        scores = self.scorer(concat_msg)
        scores = scores.reshape(B, N)
        return scores

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ego_feat, selected_feat):
        B, C, H, W = ego_feat.shape
        query = ego_feat.reshape(B, C, -1).permute(0, 2, 1)
        key_val = selected_feat.reshape(B, C, -1).permute(0, 2, 1)
        attn_out, _ = self.attention(query, key_val, key_val)
        fused = self.norm(query + attn_out)
        fused_feat = fused.permute(0, 2, 1).reshape(B, C, H, W)
        return fused_feat

class PoseHead(nn.Module):
    """雙分支姿態預測頭：同時預測 2D 座標與關節可見性"""
    def __init__(self, feature_dim=512, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        
        # 共用骨幹：從特徵圖提取高階語意
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 座標分支：預測每個關節的 (X, Y) 座標
        self.coord_head = nn.Sequential(
            nn.Linear(1024, num_joints * 2),
            nn.Sigmoid()  # 約束輸出在 [0, 1] 區間
        )
        
        # 可見性分支：預測每個關節是否可見 (0~1 的機率)
        self.vis_head = nn.Sequential(
            nn.Linear(1024, num_joints),
            nn.Sigmoid()  # 約束輸出在 [0, 1] 區間
        )

    def forward(self, x):
        # x: (B, feature_dim, H, W)
        feat = self.backbone(x)                                     # (B, 1024)
        coords = self.coord_head(feat).reshape(-1, self.num_joints, 2)  # (B, 17, 2)
        vis = self.vis_head(feat)                                    # (B, 17)
        return coords, vis

class Who2comPoseNet(nn.Module):
    def __init__(self, num_views=4, num_joints=17, feature_dim=512):
        super().__init__()
        self.num_views = num_views
        self.extractor = FeatureExtractor()
        self.matchmaker = Matchmaker(feature_dim)
        self.fusion = CrossAttentionFusion(embed_dim=feature_dim)
        self.pose_head = PoseHead(feature_dim, num_joints)
        self.num_joints = num_joints

    def forward(self, views, temperature=1.0):
        B, V, C, H, W = views.shape
        
        # 1. 萃取所有視角的特徵圖
        all_feats = self.extractor(views.reshape(B*V, C, H, W))
        all_feats = all_feats.reshape(B, V, -1, all_feats.shape[-2], all_feats.shape[-1])
        
        ego_feat = all_feats[:, 0]
        other_feats = all_feats[:, 1:]
        
        # 2. Matchmaker 評分與挑選
        scores = self.matchmaker(ego_feat, other_feats)
        weights = F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1)
        
        # 3. 提取被選中的特徵圖
        weights_expanded = weights.reshape(B, V-1, 1, 1, 1)
        selected_feat = (other_feats * weights_expanded).sum(dim=1)
        
        # 4. 特徵融合
        fused_feat = self.fusion(ego_feat, selected_feat)
        
        # 5. 預測 2D 座標與可見性
        pose_2d, visibility = self.pose_head(fused_feat)
        
        return pose_2d, visibility, weights

if __name__ == "__main__":
    net = Who2comPoseNet(num_views=3)
    dummy_input = torch.rand(2, 3, 3, 256, 256)
    pose, vis, weights = net(dummy_input)
    print(f"Output Pose shape: {pose.shape}")        # (2, 17, 2)
    print(f"Output Visibility shape: {vis.shape}")    # (2, 17)
    print(f"Selected View weights:\n{weights}")
