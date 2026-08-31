import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from dataset import get_dataloader
from model import When2comHeatmapNet

def train_stage1(model, loader, optimizer, epochs=60, device='cuda', save_dir='.'):
    """
    Stage 1: 訓練 Heatmap 和 跨視角融合。
    強制進行通訊 (force_comm=True)，不訓練 Gate。
    """
    model.train()
    
    # Heatmap 的損失函數使用 MSE
    criterion_hm = nn.MSELoss()
    # Coordinate 的損失函數使用 SmoothL1
    criterion_coord = nn.SmoothL1Loss()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        total_loss = 0
        total_hm_loss = 0
        total_coord_loss = 0
        
        pbar = tqdm(loader, desc=f"Stage 1 Epoch {epoch+1}/{epochs}")
        for imgs, hms, coords, vis, cams in pbar:
            imgs = imgs.to(device)
            hms = hms.to(device)
            coords = coords.to(device)
            vis = vis.to(device)
            
            optimizer.zero_grad()
            
            # 強制通訊
            pred_coords, pred_conf, pred_hms, _ = model(imgs, force_comm=True)
            
            # 只取 ego view 的 ground truth (因為模型現在輸出的是融合後的 ego)
            gt_ego_hm = hms[:, 0]
            gt_ego_coords = coords[:, 0]
            gt_ego_vis = vis[:, 0]
            
            # Heatmap Loss (只算可見的關節)
            B, C, H, W = pred_hms.shape
            vis_mask = gt_ego_vis.view(B, C, 1, 1)
            hm_loss = criterion_hm(pred_hms * vis_mask, gt_ego_hm * vis_mask)
            
            # Coordinate Loss (輔助)
            vis_mask_coord = gt_ego_vis.unsqueeze(-1)
            coord_loss = criterion_coord(pred_coords * vis_mask_coord, gt_ego_coords * vis_mask_coord)
            
            # 總 Loss (Heatmap 權重較大)
            loss = hm_loss * 1000 + coord_loss * 10
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_hm_loss += hm_loss.item()
            total_coord_loss += coord_loss.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'hm': f"{hm_loss.item()*1000:.4f}",
                'coord': f"{coord_loss.item()*10:.4f}",
                'alpha': f"{model.fusion.alpha.item():.4f}"
            })
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f} | Alpha: {model.fusion.alpha.item():.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, "heatmap_stage1_latest.pth"))

def train_stage2(model, loader, optimizer, epochs=20, device='cuda', save_dir='.'):
    """
    Stage 2: 訓練 EntropyGate。
    凍結其他所有模組，只訓練 Gate 如何把 Entropy 轉成決策機率。
    """
    model.eval() # 其他模組 Eval
    model.gate.train() # Gate Train
    
    criterion_gate = nn.BCELoss()
    criterion_hm = nn.MSELoss(reduction='none') # Per-sample 計算
    
    for epoch in range(epochs):
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        
        pbar = tqdm(loader, desc=f"Stage 2 Epoch {epoch+1}/{epochs}")
        for imgs, hms, coords, vis, cams in pbar:
            imgs = imgs.to(device)
            hms = hms.to(device)
            vis = vis.to(device)
            B = imgs.shape[0]
            
            gt_ego_hm = hms[:, 0]
            gt_ego_vis = vis[:, 0]
            vis_mask = gt_ego_vis.view(B, 17, 1, 1)
            
            # ==========================================
            # 1. 產生 Ground Truth 標籤 (不計算梯度)
            # ==========================================
            with torch.no_grad():
                # 為了節省計算，我們先把特徵抽出來
                B, V, C_img, H, W = imgs.shape
                feats_flat = model.extractor(imgs.view(B * V, C_img, H, W))
                hms_flat = model.decoder(feats_flat)
                all_hms = hms_flat.view(B, V, 17, 64, 64)
                
                ego_hm = all_hms[:, 0]
                other_hms = all_hms[:, 1:]
                
                # 計算通訊與不通訊的 Heatmap 結果
                # 不通訊
                pred_hm_no_comm = ego_hm
                loss_no_comm = criterion_hm(pred_hm_no_comm * vis_mask, gt_ego_hm * vis_mask).view(B, -1).mean(dim=1)
                
                # 通訊 (最佳隊友)
                ego_coords = model.decoder.soft_argmax_2d(ego_hm)
                ego_conf = model.decoder.get_confidence(ego_hm)
                # ... 省略 matchmaker 呼叫，為簡化直接取平均隊友當成通訊結果
                selected_hm = other_hms.mean(dim=1) 
                pred_hm_comm = model.fusion(ego_hm, selected_hm)
                loss_comm = criterion_hm(pred_hm_comm * vis_mask, gt_ego_hm * vis_mask).view(B, -1).mean(dim=1)
                
                # 產生 per-sample 的 GT 標籤 (通訊的 loss 比較小就標 1)
                gate_gt = (loss_comm < loss_no_comm).float().unsqueeze(-1) # (B, 1)
            
            # ==========================================
            # 2. 訓練 Gate (只有這裡有梯度)
            # ==========================================
            optimizer.zero_grad()
            
            # Gate 的輸入是 Ego Heatmap
            comm_prob = model.gate(ego_hm.detach()) # (B, 1)
            
            loss = criterion_gate(comm_prob, gate_gt)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = (comm_prob > 0.5).float()
            correct_preds += (preds == gate_gt).sum().item()
            total_samples += B
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_preds/total_samples:.4f}"
            })
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f} | Acc: {correct_preds/total_samples:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, "heatmap_stage2_latest.pth"))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 建立 DataLoader 和 Model
    loader = get_dataloader(batch_size=4)
    model = When2comHeatmapNet().to(device)
    
    # =============== Stage 1 ===============
    print("\n--- 啟動 Stage 1 (Heatmap & Fusion) ---")
    # 凍結 Gate
    for param in model.gate.parameters():
        param.requires_grad = False
        
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # 測試執行可以把 epoch 設 1
    train_stage1(model, loader, optimizer1, epochs=60, device=device)
    
    # =============== Stage 2 ===============
    print("\n--- 啟動 Stage 2 (Entropy Gate) ---")
    # 載入 Stage 1 權重
    model.load_state_dict(torch.load("heatmap_stage1_latest.pth"))
    
    # 凍結所有，只開 Gate
    for param in model.parameters():
        param.requires_grad = False
    for param in model.gate.parameters():
        param.requires_grad = True
        
    optimizer2 = optim.Adam(model.gate.parameters(), lr=1e-3)
    
    # 測試執行可以把 epoch 設 1
    train_stage2(model, loader, optimizer2, epochs=20, device=device)
