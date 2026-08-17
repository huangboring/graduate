import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42) # 固定亂數種子，確保每次算出來的數字一樣

print("=== 模擬真實維度 (但縮小為 1個特徵通道, 1x1大小, 1個關節) ===")
B = 1
V = 3
C = 1 # 原本是 512
H = 1 # 原本是 8
W = 1 # 原本是 8

# 假設這是 ResNet-18 萃取出來的特徵 (原本是 512x8x8，我們簡化為 1x1x1 的純數字)
# ego_feat 是主視角，other_feats 是相機 A 和相機 B
ego_feat = torch.tensor([[[[0.5]]]], requires_grad=True) # (1, 1, 1, 1)
other_feats = torch.tensor([[[[[1.0]]]], [[[[2.0]]]]]).permute(1, 0, 2, 3, 4) # (1, 2, 1, 1, 1)
other_feats.requires_grad = True

print(f"1. Ego 特徵值: {ego_feat.item()}")
print(f"   Other 特徵值 (相機 A, B): {other_feats[0, 0].item()}, {other_feats[0, 1].item()}")

# --- Matchmaker (Scorer) 簡化版 ---
# 原本是 Linear(512*2, 512) -> ReLU -> Linear(512, 1)
# 這裡我們用一層簡單的 Linear(2, 1) 來模擬，並手動給定初始權重
scorer = nn.Linear(2, 1, bias=False)
with torch.no_grad():
    scorer.weight.data = torch.tensor([[0.2, 0.4]]) # 初始權重

# 產生握手訊息並串接
msg_ego = ego_feat.reshape(B, -1) # [0.5]
msg_other = other_feats.reshape(B*(V-1), -1) # [[1.0], [2.0]]
msg_ego_repeated = msg_ego.unsqueeze(1).expand(-1, V-1, -1).reshape(B*(V-1), -1) # [[0.5], [0.5]]

concat_msg = torch.cat([msg_ego_repeated, msg_other], dim=1) # [[0.5, 1.0], [0.5, 2.0]]
scores = scorer(concat_msg).reshape(B, V-1)

print(f"\n2. Matchmaker 串接特徵: 相機A={concat_msg[0].tolist()}, 相機B={concat_msg[1].tolist()}")
print(f"   Matchmaker 算出的評分 (Scores): {scores[0].tolist()}")

# --- Gumbel Softmax ---
# 為了看清楚機率，我們先算純 Softmax
probs = F.softmax(scores, dim=-1)
print(f"\n3. 轉換成機率 (Softmax): {probs[0].tolist()}")

# 使用 Gumbel Softmax (hard=True) 抽出特徵
weights = F.gumbel_softmax(scores, tau=1.0, hard=True, dim=-1)
print(f"   Gumbel-Softmax (hard=True) 選擇權重: {weights[0].tolist()}")

# --- 特徵挑選 ---
weights_expanded = weights.reshape(B, V-1, 1, 1, 1)
selected_feat = (other_feats * weights_expanded).sum(dim=1)
print(f"\n4. 成功挑出的特徵值 (selected_feat): {selected_feat.item()}")

# --- Cross Attention Fusion 簡化版 ---
# 簡化：直接將兩者相加 (Query + Value) 代表融合
fused_feat = ego_feat + selected_feat
print(f"\n5. 融合後的特徵 (fused_feat): {fused_feat.item()}")

# --- PoseHead (回歸 2D 座標) 簡化版 ---
# 原本是 Linear -> ReLU -> Linear -> Sigmoid
# 簡化為一層 Linear(1, 1) -> Sigmoid
pose_head = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    pose_head.weight.data = torch.tensor([[0.8]])

pose_flat = pose_head(fused_feat.reshape(1, -1))
pred_pose = torch.sigmoid(pose_flat)
print(f"\n6. PoseHead 預測出來的座標 (0~1): {pred_pose.item():.4f}")

# --- 計算 Loss (Smooth L1 Loss) ---
gt_pose = torch.tensor([[0.9]]) # 假設真實答案是 0.9
criterion = nn.SmoothL1Loss()
loss = criterion(pred_pose, gt_pose)
print(f"\n7. 計算 Smooth L1 Loss (答案 0.9 vs 預測 {pred_pose.item():.4f}): {loss.item():.4f}")

# --- 反向傳播 (Backward Pass) ---
loss.backward()

print("\n=== 反向傳播 (Backpropagation) 梯度檢視 ===")
print(f"8. PoseHead 權重的梯度: {pose_head.weight.grad.item():.4f} (告訴畫筆要怎麼微調)")
print(f"   Matchmaker 權重的梯度: {scorer.weight.grad.tolist()} (告訴評分員要怎麼改分數)")
print(f"   (重點) 相機 A 特徵的梯度: {other_feats.grad[0, 0].item():.4f}")
print(f"   (重點) 相機 B 特徵的梯度: {other_feats.grad[0, 1].item():.4f}")

if other_feats.grad[0, 1].item() != 0 and other_feats.grad[0, 0].item() == 0:
    print("\n[結論] 您可以看到，梯度完美避開了沒被選中的相機 A，精準地流回了被選中的相機 B！這就是程式碼能訓練的原因！")
