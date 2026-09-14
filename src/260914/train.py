import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import When2ComPoseNet, soft_argmax_2d, get_confidence
from dataset import get_dataloader

def plot_loss(losses, stage_num):
    """
    繪製並儲存損失曲線
    """
    plt.figure()
    plt.plot(losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Stage {stage_num} Training Loss')
    plt.legend()
    plt.savefig(f'stage{stage_num}_curve.png')
    plt.close()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 建立模型
    model = When2ComPoseNet(num_joints=17)
    model = model.to(device)
    
    # 取得資料加載器
    dataloader = get_dataloader()

    # 定義損失函數
    mse_loss_fn = nn.MSELoss(reduction='mean')
    smooth_l1_loss_fn = nn.SmoothL1Loss(reduction='mean')

    # ==========================================
    # 第一階段：單視角主幹網路與解碼器訓練
    # ==========================================
    print("開始第一階段：單視角主幹網路與解碼器訓練...")
    # 凍結 qk_encoder, sga, cross_view_fusion
    for param in model.qk_encoder.parameters():
        param.requires_grad = False
    for param in model.sga.parameters():
        param.requires_grad = False
    for param in model.cross_view_fusion.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    stage1_epochs = 60
    stage1_losses = []

    for epoch in range(stage1_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Stage 1 Epoch {epoch+1}/{stage1_epochs}")
        
        for batch in pbar:
            images, gt_heatmaps, gt_coords, visibility, union_visibility, cam_params = batch
            images = images.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            gt_coords = gt_coords.to(device)
            visibility = visibility.to(device)
            
            optimizer.zero_grad()
            
            # 強制不進行通訊 (force_no_comm=True)
            pred_coords, pred_conf, final_hm, attn_weights, comm_decisions = model(images, force_no_comm=True)
            
            # 取得 ego view (view 0) 的標註和可見度
            ego_gt_hm = gt_heatmaps[:, 0]
            ego_gt_coords = gt_coords[:, 0]
            # 整理 mask 形狀，使其可廣播
            ego_vis = visibility[:, 0]                          # (B, 17)
            vis_mask_hm = ego_vis.unsqueeze(-1).unsqueeze(-1)   # (B, 17, 1, 1)
            vis_mask_coord = ego_vis.unsqueeze(-1)              # (B, 17, 1)
            
            # 計算損失 (只計算可見的關節點)
            hm_loss = mse_loss_fn(final_hm * vis_mask_hm, ego_gt_hm * vis_mask_hm) * 1000
            coord_loss = smooth_l1_loss_fn(pred_coords * vis_mask_coord, ego_gt_coords * vis_mask_coord) * 10
            loss = hm_loss + coord_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        stage1_losses.append(avg_loss)
        print(f"Stage 1 - Epoch {epoch+1}/{stage1_epochs}, Average Loss: {avg_loss:.4f}")

    # 儲存第一階段權重並繪製圖表
    torch.save(model.state_dict(), 'when2com_pose_stage1_latest.pth')
    plot_loss(stage1_losses, 1)

    # ==========================================
    # 第二階段：多視角特徵融合訓練
    # ==========================================
    print("開始第二階段：多視角特徵融合訓練...")
    # 讀取第一階段權重
    model.load_state_dict(torch.load('when2com_pose_stage1_latest.pth'))
    
    # 解除凍結，允許所有參數更新
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    stage2_epochs = 40
    stage2_losses = []

    for epoch in range(stage2_epochs):
        model.train()
        epoch_loss = 0.0
        avg_attn_sum = 0.0
        pbar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch+1}/{stage2_epochs}")
        
        for batch in pbar:
            images, gt_heatmaps, gt_coords, visibility, union_visibility, cam_params = batch
            images = images.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            gt_coords = gt_coords.to(device)
            union_visibility = union_visibility.to(device)
            
            optimizer.zero_grad()
            
            # 強制進行通訊 (force_comm=True)
            pred_coords, pred_conf, final_hm, attn_weights, comm_decisions = model(images, force_comm=True)
            
            ego_gt_hm = gt_heatmaps[:, 0]
            ego_gt_coords = gt_coords[:, 0]
            
            # 使用 union_visibility 作為遮罩
            vis_mask_hm = union_visibility.unsqueeze(-1).unsqueeze(-1) # (B, 17, 1, 1)
            vis_mask_coord = union_visibility.unsqueeze(-1)            # (B, 17, 1)
            
            hm_loss = mse_loss_fn(final_hm * vis_mask_hm, ego_gt_hm * vis_mask_hm) * 1000
            coord_loss = smooth_l1_loss_fn(pred_coords * vis_mask_coord, ego_gt_coords * vis_mask_coord) * 10
            loss = hm_loss + coord_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            avg_attn_sum += attn_weights.mean().item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        avg_attn = avg_attn_sum / len(dataloader)
        stage2_losses.append(avg_loss)
        print(f"Stage 2 - Epoch {epoch+1}/{stage2_epochs}, Average Loss: {avg_loss:.4f}, Avg Attn: {avg_attn:.4f}")

    # 儲存第二階段權重並繪製圖表
    torch.save(model.state_dict(), 'when2com_pose_stage2_latest.pth')
    plot_loss(stage2_losses, 2)

    # ==========================================
    # 第三階段：端到端整合 Gate 訓練
    # ==========================================
    print("開始第三階段：端到端整合 Gate 訓練...")
    # 讀取第二階段權重
    model.load_state_dict(torch.load('when2com_pose_stage2_latest.pth'))

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    stage3_epochs = 10
    stage3_losses = []

    for epoch in range(stage3_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Stage 3 Epoch {epoch+1}/{stage3_epochs}")
        
        for batch in pbar:
            images, gt_heatmaps, gt_coords, visibility, union_visibility, cam_params = batch
            images = images.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            gt_coords = gt_coords.to(device)
            union_visibility = union_visibility.to(device)
            
            optimizer.zero_grad()
            
            # 正常模式 (由 Gate 決定是否通訊)
            pred_coords, pred_conf, final_hm, attn_weights, comm_decisions = model(images)
            
            ego_gt_hm = gt_heatmaps[:, 0]
            ego_gt_coords = gt_coords[:, 0]
            
            # 使用 union_visibility 作為遮罩
            vis_mask_hm = union_visibility.unsqueeze(-1).unsqueeze(-1)
            vis_mask_coord = union_visibility.unsqueeze(-1)
            
            hm_loss = mse_loss_fn(final_hm * vis_mask_hm, ego_gt_hm * vis_mask_hm) * 1000
            coord_loss = smooth_l1_loss_fn(pred_coords * vis_mask_coord, ego_gt_coords * vis_mask_coord) * 10
            loss = hm_loss + coord_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        stage3_losses.append(avg_loss)
        print(f"Stage 3 - Epoch {epoch+1}/{stage3_epochs}, Average Loss: {avg_loss:.4f}")

    # 儲存第三階段權重並繪製圖表
    torch.save(model.state_dict(), 'when2com_pose_stage3_latest.pth')
    plot_loss(stage3_losses, 3)
    
    print("訓練完成！")

if __name__ == '__main__':
    train()
