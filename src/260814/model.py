import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class FeatureExtractor(nn.Module):
    """升級為 ResNet-34，比 ResNet-18 多了更多殘差層，特徵表達更豐富。
    輸出維度仍為 512，但中間層更深，能學到更複雜的視覺模式。"""
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # 輸入 x: (B, 3, 256, 256)
        # 輸出: (B, 512, 8, 8)
        return self.backbone(x)

class Matchmaker(nn.Module):
    """升級為三層 MLP，增加非線性表達能力。"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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

class CommunicationGate(nn.Module):
    """When2com 的核心：判斷 Ego 是否需要向外通訊。
    升級為更深的 MLP 以學到更細緻的自我評估能力。"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, ego_feat):
        return self.gate(ego_feat)

class CrossAttentionFusion(nn.Module):
    """升級為雙層 Cross-Attention，讓跨視角的特徵對齊更精準。"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # 第一層注意力
        self.attention1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 第二層注意力 (加深融合深度)
        self.attention2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, ego_feat, selected_feat):
        B, C, H, W = ego_feat.shape
        query = ego_feat.reshape(B, C, -1).permute(0, 2, 1)     # (B, 64, 512)
        key_val = selected_feat.reshape(B, C, -1).permute(0, 2, 1)  # (B, 64, 512)
        
        # 第一層 Cross-Attention + Residual
        attn_out1, _ = self.attention1(query, key_val, key_val)
        x = self.norm1(query + attn_out1)
        
        # 第二層 Self-Attention (讓融合後的特徵自我精煉)
        attn_out2, _ = self.attention2(x, x, x)
        x = self.norm2(x + attn_out2)
        
        # FFN + Residual
        x = self.norm3(x + self.ffn(x))
        
        fused_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        return fused_feat

class PoseHead(nn.Module):
    """升級版雙分支姿態預測頭：更深的骨幹 + Spatial Attention。"""
    def __init__(self, feature_dim=512, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        
        # 共用骨幹：從特徵圖提取高階語意 (更深)
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 座標分支
        self.coord_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints * 2),
            nn.Sigmoid()
        )
        
        # 可見性分支
        self.vis_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)                                        # (B, 512)
        coords = self.coord_head(feat).reshape(-1, self.num_joints, 2) # (B, 17, 2)
        vis = self.vis_head(feat)                                      # (B, 17)
        return coords, vis

class When2comPoseNet(nn.Module):
    """When2com 架構 (升級版)：ResNet-34 骨幹 + 雙層 Cross-Attention + 更深的 PoseHead。"""
    def __init__(self, num_views=4, num_joints=17, feature_dim=512):
        super().__init__()
        self.num_views = num_views
        self.extractor = FeatureExtractor()
        self.matchmaker = Matchmaker(feature_dim)
        self.comm_gate = CommunicationGate(feature_dim)
        self.fusion = CrossAttentionFusion(embed_dim=feature_dim)
        self.pose_head = PoseHead(feature_dim, num_joints)
        self.num_joints = num_joints

    def forward(self, views, temperature=1.0, comm_threshold=0.5,
                force_comm=False, force_no_comm=False):
        """
        Args:
            force_comm:    True = 強制通訊，跳過閘門，直接用融合特徵 (Stage 1 訓練用)
            force_no_comm: True = 強制不通訊，只用 Ego 特徵 (Stage 2 比較用)
        """
        B, V, C, H, W = views.shape
        
        # 1. 萃取所有視角的特徵圖
        all_feats = self.extractor(views.reshape(B*V, C, H, W))
        all_feats = all_feats.reshape(B, V, -1, all_feats.shape[-2], all_feats.shape[-1])
        
        ego_feat = all_feats[:, 0]
        other_feats = all_feats[:, 1:]
        
        # 2. 通訊閘門 (When2com)
        comm_prob = self.comm_gate(ego_feat)
        
        # 3. Matchmaker 評分與挑選 (Who2com)
        scores = self.matchmaker(ego_feat, other_feats)
        weights = F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1)
        
        # 4. 提取被選中的特徵圖
        weights_expanded = weights.reshape(B, V-1, 1, 1, 1)
        selected_feat = (other_feats * weights_expanded).sum(dim=1)
        
        # 5. 特徵融合
        fused_feat = self.fusion(ego_feat, selected_feat)
        
        # 6. 閘門決策
        if force_comm:
            # Stage 1：強制使用融合特徵
            final_feat = fused_feat
        elif force_no_comm:
            # Stage 2 比較用：強制只用 Ego 特徵
            final_feat = ego_feat
        else:
            # 正常推論：由閘門決定
            if self.training:
                comm_decision = (comm_prob > comm_threshold).float() - comm_prob.detach() + comm_prob
            else:
                comm_decision = (comm_prob > comm_threshold).float()
            
            gate = comm_decision.unsqueeze(-1).unsqueeze(-1)
            final_feat = gate * fused_feat + (1 - gate) * ego_feat
        
        # 7. 預測
        pose_2d, visibility = self.pose_head(final_feat)
        
        return pose_2d, visibility, weights, comm_prob


# 向後相容
Who2comPoseNet = When2comPoseNet


if __name__ == "__main__":
    net = When2comPoseNet(num_views=3)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    dummy_input = torch.rand(2, 3, 3, 256, 256)
    pose, vis, weights, comm_prob = net(dummy_input)
    print(f"Output Pose shape: {pose.shape}")
    print(f"Output Visibility shape: {vis.shape}")
    print(f"Selected View weights:\n{weights}")
    print(f"Communication probability: {comm_prob.squeeze().tolist()}")
