import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class FeatureExtractor(nn.Module):
    """ResNet-34 骨幹，輸出 (B, 512, 8, 8) 特徵圖。"""
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.backbone(x)

class Matchmaker(nn.Module):
    """三層 MLP 評分器，決定要跟哪個視角交換特徵。"""
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
    """When2com 閘門：判斷 Ego 是否需要向外通訊。"""
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

class ResidualCrossAttentionFusion(nn.Module):
    """
    殘差式跨視角融合 (Residual Cross-Attention Fusion)
    
    核心改進：Ego 特徵永遠是主角，其他視角的資訊只作為「補充」。
    
    output = ego_feat + alpha * cross_attention_supplement
    
    alpha 是可學習的參數，初始化為 0.1，讓模型一開始幾乎等於單視角，
    隨著訓練慢慢學會「要從其他視角借多少資訊」。
    """
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # Cross-Attention：從其他視角提取補充資訊
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # 可學習的混合權重，初始化為 0.1（接近不融合）
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, ego_feat, selected_feat):
        B, C, H, W = ego_feat.shape
        
        # 展平為序列 (B, 64, 512)
        query = ego_feat.reshape(B, C, -1).permute(0, 2, 1)
        key_val = selected_feat.reshape(B, C, -1).permute(0, 2, 1)
        
        # Cross-Attention：用自己的特徵當問題，別人的特徵當解答
        attn_out, _ = self.cross_attn(query, key_val, key_val)
        supplement = self.norm(attn_out)  # (B, 64, 512)
        
        # 殘差融合：Ego 為主，補充為輔
        alpha_clamped = self.alpha.clamp(0.0, 1.0)
        fused = query + alpha_clamped * supplement
        
        # 還原為特徵圖
        fused_feat = fused.permute(0, 2, 1).reshape(B, C, H, W)
        return fused_feat

class PoseHead(nn.Module):
    """雙分支姿態預測頭：座標 + 可見性。"""
    def __init__(self, feature_dim=512, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        
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
        
        self.coord_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints * 2),
            nn.Sigmoid()
        )
        
        self.vis_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        coords = self.coord_head(feat).reshape(-1, self.num_joints, 2)
        vis = self.vis_head(feat)
        return coords, vis

class When2comPoseNet(nn.Module):
    """When2com + 殘差融合架構。"""
    def __init__(self, num_views=4, num_joints=17, feature_dim=512):
        super().__init__()
        self.num_views = num_views
        self.extractor = FeatureExtractor()
        self.matchmaker = Matchmaker(feature_dim)
        self.comm_gate = CommunicationGate(feature_dim)
        self.fusion = ResidualCrossAttentionFusion(embed_dim=feature_dim)
        self.pose_head = PoseHead(feature_dim, num_joints)
        self.num_joints = num_joints

    def forward(self, views, temperature=1.0, comm_threshold=0.5,
                force_comm=False, force_no_comm=False):
        B, V, C, H, W = views.shape
        
        # 1. 萃取所有視角的特徵圖
        all_feats = self.extractor(views.reshape(B*V, C, H, W))
        all_feats = all_feats.reshape(B, V, -1, all_feats.shape[-2], all_feats.shape[-1])
        
        ego_feat = all_feats[:, 0]
        other_feats = all_feats[:, 1:]
        
        # 2. 通訊閘門
        comm_prob = self.comm_gate(ego_feat)
        
        # 3. Matchmaker 評分與挑選
        scores = self.matchmaker(ego_feat, other_feats)
        weights = F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1)
        
        # 4. 提取被選中的特徵圖
        weights_expanded = weights.reshape(B, V-1, 1, 1, 1)
        selected_feat = (other_feats * weights_expanded).sum(dim=1)
        
        # 5. 殘差融合（Ego 為主，其他視角為輔）
        fused_feat = self.fusion(ego_feat, selected_feat)
        
        # 6. 閘門決策
        if force_comm:
            final_feat = fused_feat
        elif force_no_comm:
            final_feat = ego_feat
        else:
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
    print(f"Total parameters: {total_params:,}")
    print(f"Fusion alpha: {net.fusion.alpha.item():.2f}")
    
    dummy_input = torch.rand(2, 3, 3, 256, 256)
    pose, vis, weights, comm_prob = net(dummy_input)
    print(f"Output Pose shape: {pose.shape}")
    print(f"Output Visibility shape: {vis.shape}")
    print(f"Communication probability: {comm_prob.squeeze().tolist()}")
