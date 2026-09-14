import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34, ResNet34_Weights
import math


# ============================================================
# 基礎工具函數
# ============================================================

def soft_argmax_2d(heatmaps, temp=10.0):
    """
    從 Heatmap 中可微分地提取 2D 座標。
    heatmaps: (B, num_joints, H, W)
    回傳: (B, num_joints, 2) 座標值在 [0, 1] 之間
    """
    B, C, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, C, -1)
    probs = F.softmax(heatmaps_flat * temp, dim=-1)
    probs = probs.view(B, C, H, W)

    y_grid = torch.linspace(0.0, 1.0, H, device=heatmaps.device, dtype=heatmaps.dtype)
    x_grid = torch.linspace(0.0, 1.0, W, device=heatmaps.device, dtype=heatmaps.dtype)
    y_grid = y_grid.view(1, 1, H, 1)
    x_grid = x_grid.view(1, 1, 1, W)

    y_coords = torch.sum(probs * y_grid, dim=(2, 3))
    x_coords = torch.sum(probs * x_grid, dim=(2, 3))

    coords = torch.stack([x_coords, y_coords], dim=-1)  # (B, C, 2)
    return coords


def get_confidence(heatmaps):
    """
    從 Heatmap 中取得每個關節的信心分數 (peak value)。
    heatmaps: (B, num_joints, H, W)
    回傳: (B, num_joints)
    """
    B, C, H, W = heatmaps.shape
    confidence = heatmaps.view(B, C, -1).max(dim=-1)[0]
    return confidence


# ============================================================
# Backbone & Decoder (共用，與 SimpleBaseline 一致)
# ============================================================

class FeatureExtractor(nn.Module):
    """ResNet-34 backbone，輸出 1/32 解析度的特徵圖。"""
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        # 去掉最後的 pooling 和 fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # x: (B, 3, 256, 256) → (B, 512, 8, 8)
        return self.backbone(x)


class HeatmapDecoder(nn.Module):
    """3 層 Deconv 解碼器：8×8 → 16×16 → 32×32 → 64×64"""
    def __init__(self, in_channels=512, num_joints=17):
        super().__init__()
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
        self.head = nn.Conv2d(256, num_joints, kernel_size=1)

    def forward(self, x):
        # x: (B, 512, 8, 8) → (B, 17, 64, 64)
        return self.head(self.deconv(x))


# ============================================================
# When2Com 模組一：非對稱 Query-Key 編碼器
# ============================================================

class QueryKeyEncoder(nn.Module):
    """
    When2Com 核心模組之一：非對稱 Query-Key 編碼器。

    - Query (μ): 壓縮到極小的 d_q，用來廣播給所有 peer（節省頻寬）。
      對應 When2Com Handshake Stage 1: Request。
    - Key (κ): 保持較大的 d_k，留在本地被 peer 的 query 比對。
      對應 When2Com Handshake Stage 2: Match 的輸入。

    非對稱設計 (d_q << d_k) 是 When2Com 的關鍵頻寬優化。
    """
    def __init__(self, feat_channels=512, d_q=32, d_k=128):
        super().__init__()
        self.d_q = d_q
        self.d_k = d_k

        # Query encoder: 空間壓縮到 1×1，再投影到 d_q
        # 極小的 query 可以低成本廣播
        self.query_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feat_channels, d_q)
        )

        # Key encoder: 保留較多空間資訊 (4×4)，投影到 d_k
        # Key 留在本地，不需要傳輸，可以保持較大維度
        self.key_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(feat_channels * 4 * 4, d_k)
        )

    def forward(self, feat):
        """
        feat: (B, 512, 8, 8)
        回傳: query (B, d_q), key (B, d_k)
        """
        query = self.query_encoder(feat)
        key = self.key_encoder(feat)
        return query, key


# ============================================================
# When2Com 模組二：Scaled General Attention
# ============================================================

class ScaledGeneralAttention(nn.Module):
    """
    When2Com 核心模組之二：Scaled General Attention。

    因為 d_q ≠ d_k，不能直接做 dot product。
    用 bilinear projection 把兩者投影到相同的 hidden space：

        Score(i,j) = (W_q · μ_i)^T · (W_k · κ_j) / √d_hidden

    對應 When2Com Handshake Stage 2: Match Score 計算。
    """
    def __init__(self, d_q=32, d_k=128, d_hidden=64):
        super().__init__()
        self.q_proj = nn.Linear(d_q, d_hidden, bias=False)
        self.k_proj = nn.Linear(d_k, d_hidden, bias=False)
        self.scale = 1.0 / math.sqrt(d_hidden)

    def forward(self, queries, keys):
        """
        queries: (B, V, d_q) — 每個 agent 的 query
        keys:    (B, V, d_k) — 每個 agent 的 key
        回傳:    (B, V, V) — 匹配分數矩陣
        """
        q = self.q_proj(queries)   # (B, V, d_hidden)
        k = self.k_proj(keys)      # (B, V, d_hidden)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, V, V)
        return scores


# ============================================================
# When2Com 模組三：通訊閘門 (Self-Attention Gate)
# ============================================================

class When2ComGate(nn.Module):
    """
    When2Com 核心模組之三：通訊閘門。

    不需要獨立的 EntropyGate 網路！When2Com 最巧妙之處：
    gate 是 attention 矩陣的自然副產物。

    機制：
    1. Softmax attention A 包含自注意力 A_{i,i}（對角線）
    2. A_{i,i} ≈ 1 → agent i 自己夠用 → 關閉通訊
    3. A_{i,j} > threshold (j≠i) → agent j 有互補資訊 → 開啟通訊
    4. Activation-based pruning: A_{i,j} < threshold 的連結歸零

    這比獨立的 gate 網路更優雅，因為 gate 決策和視角挑選是
    同一個 attention 矩陣的不同面向，訓練時自然協調。
    """
    def __init__(self, threshold=0.03):
        super().__init__()
        self.threshold = threshold

    def forward(self, raw_scores):
        """
        raw_scores: (B, V, V) — ScaledGeneralAttention 的輸出
        回傳:
            norm_weights: (B, V, V)  — 裁剪並重新歸一化的權重
            comm_decisions: (B, V) — 每個 agent 是否需要通訊 (bool)
        """
        B, V, _ = raw_scores.shape

        # Softmax (包含自注意力，對角線元素)
        attn_weights = F.softmax(raw_scores, dim=-1)  # (B, V, V)

        # 提取自注意力權重 (對角線)
        self_weights = torch.diagonal(attn_weights, dim1=1, dim2=2)  # (B, V)

        # Activation-based pruning
        eye_mask = torch.eye(V, device=raw_scores.device, dtype=torch.bool).unsqueeze(0)  # (1, V, V)

        pruned = attn_weights.clone()
        # 非對角線 & 低於閾值 → 歸零
        off_diag = ~eye_mask.expand(B, -1, -1)
        below_thresh = (pruned < self.threshold) & off_diag
        pruned = pruned.masked_fill(below_thresh, 0.0)

        # 重新歸一化 (保持每列總和為 1)
        row_sums = pruned.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        norm_weights = pruned / row_sums

        # 通訊決策：是否有任何活躍的跨 agent 連結
        cross_weight_sum = (pruned * off_diag.float()).sum(dim=-1)  # (B, V)
        comm_decisions = cross_weight_sum > 0  # (B, V)

        return norm_weights, comm_decisions


# ============================================================
# MVGFormer 模組：Cross-View Feature Fusion
# ============================================================

class CrossViewFusion(nn.Module):
    """
    繼承 MVGFormer 的跨視角特徵融合精神。

    MVGFormer 原作用 MultiScaleDeformableAttention（需自訂 CUDA op），
    我們用 PyTorch 內建的 nn.MultiheadAttention 達到相同概念：

    - Ego 的 feature tokens 作為 Query
    - 所有視角（含 ego）的 feature tokens 作為 Key/Value
    - View weights 從 When2Com 注入，調制 Value 的貢獻
    - key_padding_mask 屏蔽被 When2Com Gate 裁剪掉的 peer

    這樣 Cross-Attention 決定「空間上哪些位置有用」(MVGFormer)，
    View Weights 決定「哪個視角更重要」(When2Com)。
    兩者協同工作。

    8×8 feature map = 64 tokens，3 views = 192 KV tokens，
    計算量完全可接受。
    """
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.d_model = d_model

        # Cross-View Attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-Forward Network (Transformer 標準)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, ego_feat, all_feats, peer_mask, view_weights=None):
        """
        ego_feat:     (B, 512, 8, 8)     — ego 視角的特徵圖
        all_feats:    (B, V, 512, 8, 8)   — 所有視角的特徵圖
        peer_mask:    (B, V) bool          — True = 該視角參與融合
        view_weights: (B, V) float 可選    — When2Com 的視角權重，用來調制 Value

        回傳: fused_feat (B, 512, 8, 8)
        """
        B, C, H, W = ego_feat.shape
        V = all_feats.shape[1]
        HW = H * W

        # ---- Ego tokens 作為 Query ----
        ego_tokens = ego_feat.view(B, C, HW).permute(0, 2, 1)  # (B, HW, C)

        # ---- 所有視角的 tokens 作為 Key/Value ----
        all_tokens = all_feats.view(B, V, C, HW).permute(0, 1, 3, 2)  # (B, V, HW, C)

        # Key tokens (不加權，讓 attention 看到完整的 key 資訊)
        key_tokens = all_tokens.reshape(B, V * HW, C)  # (B, V*HW, C)

        # Value tokens (用 When2Com 視角權重調制)
        if view_weights is not None:
            # view_weights: (B, V) → (B, V, HW, 1) 廣播到每個 spatial token
            w = view_weights.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, HW, C)  # (B, V, HW, C)
            val_tokens = (all_tokens * w).reshape(B, V * HW, C)
        else:
            val_tokens = key_tokens

        # ---- 建立 key_padding_mask ----
        # PyTorch MHA: True = 忽略該位置
        token_ignore = (~peer_mask).unsqueeze(-1).expand(-1, -1, HW)  # (B, V, HW)
        key_padding_mask = token_ignore.reshape(B, V * HW)  # (B, V*HW)

        # ---- Cross-Attention + Residual + LayerNorm ----
        attended, _ = self.cross_attn(
            ego_tokens, key_tokens, val_tokens,
            key_padding_mask=key_padding_mask
        )
        ego_tokens = self.norm1(ego_tokens + attended)

        # ---- FFN + Residual + LayerNorm ----
        ego_tokens = self.norm2(ego_tokens + self.ffn(ego_tokens))

        # ---- Reshape 回 feature map ----
        fused_feat = ego_tokens.permute(0, 2, 1).view(B, C, H, W)
        return fused_feat


# ============================================================
# 主模型：When2ComPoseNet
# ============================================================

class When2ComPoseNet(nn.Module):
    """
    完整的姿態估計模型，同時繼承 MVGFormer 和 When2Com 的核心概念。

    架構流程：
    ┌────────────────────────────────────────────────────┐
    │ 1. 每個視角獨立提取 feature + query + key           │
    │    (FeatureExtractor + QueryKeyEncoder)             │
    ├────────────────────────────────────────────────────┤
    │ 2. When2Com Handshake:                              │
    │    Query-Key matching → Self-attention gate          │
    │    → Activation pruning → 選擇 peer                 │
    │    (ScaledGeneralAttention + When2ComGate)           │
    ├────────────────────────────────────────────────────┤
    │ 3. MVGFormer-style Fusion:                          │
    │    Cross-view attention 融合 selected peer features  │
    │    (CrossViewFusion)                                 │
    ├────────────────────────────────────────────────────┤
    │ 4. 解碼精修預測:                                     │
    │    Fused feature → Heatmap → soft-argmax → 2D       │
    │    (HeatmapDecoder + soft_argmax_2d)                 │
    └────────────────────────────────────────────────────┘
    """
    def __init__(self, num_joints=17, d_q=32, d_k=128, gate_threshold=0.03):
        super().__init__()
        self.num_joints = num_joints

        # 共用模組
        self.extractor = FeatureExtractor()
        self.decoder = HeatmapDecoder(num_joints=num_joints)

        # When2Com 模組
        self.qk_encoder = QueryKeyEncoder(feat_channels=512, d_q=d_q, d_k=d_k)
        self.sga = ScaledGeneralAttention(d_q=d_q, d_k=d_k)
        self.gate = When2ComGate(threshold=gate_threshold)

        # MVGFormer 模組
        self.cross_view_fusion = CrossViewFusion(d_model=512, nhead=8)

    def forward(self, images, force_comm=False, force_no_comm=False):
        """
        images: (B, V, 3, 256, 256)

        回傳:
            pred_coords:    (B, 17, 2)   — ego 視角的預測 2D 座標
            pred_conf:      (B, 17)      — ego 視角的信心分數
            final_hm:       (B, 17, 64, 64) — ego 視角的最終 heatmap
            attn_weights:   (B, V, V)    — When2Com 注意力矩陣 (含 gate 資訊)
            comm_decisions: (B, V)       — 每個 agent 的通訊決策
        """
        B, V, C_img, H_img, W_img = images.shape

        # =============================================
        # 1. 所有視角：提取特徵
        # =============================================
        images_flat = images.view(B * V, C_img, H_img, W_img)
        feats_flat = self.extractor(images_flat)  # (B*V, 512, 8, 8)
        feats = feats_flat.view(B, V, 512, 8, 8)
        ego_feat = feats[:, 0]  # (B, 512, 8, 8)

        # =============================================
        # 2. When2Com: 產生 Query & Key
        # =============================================
        queries_flat, keys_flat = self.qk_encoder(feats_flat)
        queries = queries_flat.view(B, V, -1)  # (B, V, d_q)
        keys = keys_flat.view(B, V, -1)        # (B, V, d_k)

        # =============================================
        # 3. When2Com: Query-Key Matching + Gate
        # =============================================
        raw_scores = self.sga(queries, keys)                   # (B, V, V)
        norm_weights, comm_decisions = self.gate(raw_scores)   # (B, V, V), (B, V)

        # Ego 的視角權重 (用於 cross-view fusion 的 value 調制)
        ego_view_weights = norm_weights[:, 0, :]  # (B, V)

        # =============================================
        # 4. 根據通訊模式決定最終特徵
        # =============================================
        if force_no_comm:
            # Stage 1 訓練：不通訊，直接用 ego 特徵
            final_feat = ego_feat

        elif force_comm:
            # Stage 2 訓練：強制所有視角參與
            all_active = torch.ones(B, V, dtype=torch.bool, device=images.device)
            # 使用 When2Com 的視角權重來調制 Value (讓 QK encoder 得到梯度)
            final_feat = self.cross_view_fusion(
                ego_feat, feats, all_active, view_weights=ego_view_weights
            )

        else:
            # Stage 3 / 推論：When2Com Gate 決定
            peer_active = ego_view_weights > 0  # (B, V)

            fused_feat = self.cross_view_fusion(
                ego_feat, feats, peer_active, view_weights=ego_view_weights
            )

            # 用 ego 的 self-weight 做平滑混合
            # self_w 高 → 不需通訊，用 ego; self_w 低 → 需通訊，用 fused
            ego_self_w = norm_weights[:, 0, 0].view(B, 1, 1, 1)
            final_feat = ego_self_w * ego_feat + (1.0 - ego_self_w) * fused_feat

        # =============================================
        # 5. 解碼最終 Heatmap 並預測座標
        # =============================================
        final_hm = self.decoder(final_feat)       # (B, 17, 64, 64)
        pred_coords = soft_argmax_2d(final_hm)    # (B, 17, 2)
        pred_conf = get_confidence(final_hm)       # (B, 17)

        return pred_coords, pred_conf, final_hm, norm_weights, comm_decisions


# ============================================================
# 快速測試
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = When2ComPoseNet().to(device)

    # 模擬 batch_size=2, 3 個視角, 256x256 影像
    dummy = torch.rand(2, 3, 3, 256, 256, device=device)

    print("=== 測試 force_no_comm (Stage 1) ===")
    coords, conf, hm, attn, comm = model(dummy, force_no_comm=True)
    print(f"  coords: {coords.shape}, conf: {conf.shape}, hm: {hm.shape}")

    print("\n=== 測試 force_comm (Stage 2) ===")
    coords, conf, hm, attn, comm = model(dummy, force_comm=True)
    print(f"  coords: {coords.shape}, conf: {conf.shape}, hm: {hm.shape}")
    print(f"  attn_weights: {attn.shape}, comm_decisions: {comm.shape}")
    print(f"  attn[0]:\n{attn[0].detach().cpu()}")

    print("\n=== 測試 inference mode (Stage 3) ===")
    coords, conf, hm, attn, comm = model(dummy)
    print(f"  coords: {coords.shape}, conf: {conf.shape}")
    print(f"  comm_decisions: {comm}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型參數量: {total_params:,} (可訓練: {trainable_params:,})")
