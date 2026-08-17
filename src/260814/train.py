import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import get_dataset
from model import Who2comPoseNet

def train():
    # 1. 參數設定
    batch_size = 4
    num_epochs = 50
    num_views = 3
    learning_rate = 5e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用裝置: {device}")

    # 2. 準備 Dataset 並切分訓練集 / 驗證集 (80% / 20%)
    full_dataset = get_dataset(num_views=num_views)
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"訓練集: {train_size} 幀, 驗證集: {val_size} 幀")

    # 3. 建立模型與優化器
    model = Who2comPoseNet(num_views=num_views).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 4. 訓練迴圈
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # === 訓練階段 ===
        model.train()
        total_train_loss = 0.0
        temp = max(0.1, 1.0 * (0.85 ** epoch))
        
        for batch_idx, (images, gt_poses, gt_vis) in enumerate(train_loader):
            images = images.to(device)
            gt_poses = gt_poses.to(device)
            gt_vis = gt_vis.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            pred_poses, pred_vis, selection_weights = model(images, temperature=temp)
            
            # Visibility-Aware Loss:
            # 1. 座標損失：只計算「看得到」的關節
            vis_mask = gt_vis.unsqueeze(-1)  # (B, 17, 1) 用來遮蔽座標
            coord_loss = F.smooth_l1_loss(
                pred_poses * vis_mask,
                gt_poses * vis_mask
            )
            
            # 2. 可見性損失：BCE 讓模型學會判斷哪些關節看得到
            vis_loss = F.binary_cross_entropy(pred_vis, gt_vis)
            
            # 3. 總損失
            loss = coord_loss + 0.5 * vis_loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} (Coord: {coord_loss.item():.4f}, "
                      f"Vis: {vis_loss.item():.4f}) | Temp: {temp:.2f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # === 驗證階段 ===
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for images, gt_poses, gt_vis in val_loader:
                images = images.to(device)
                gt_poses = gt_poses.to(device)
                gt_vis = gt_vis.to(device)
                
                pred_poses, pred_vis, _ = model(images, temperature=0.1)
                
                vis_mask = gt_vis.unsqueeze(-1)
                coord_loss = F.smooth_l1_loss(
                    pred_poses * vis_mask,
                    gt_poses * vis_mask
                )
                vis_loss = F.binary_cross_entropy(pred_vis, gt_vis)
                loss = coord_loss + 0.5 * vis_loss
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"==> Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 儲存最佳模型 (以 Validation Loss 為準)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'who2com_pose_best.pth')
            print(f"    ★ 新的最佳模型！Val Loss: {best_val_loss:.4f}，已儲存 who2com_pose_best.pth")
    
    # 最後也儲存一份最終模型
    torch.save(model.state_dict(), 'who2com_pose_final.pth')
    print("\n訓練完成！")
    print(f"  最佳驗證 Loss: {best_val_loss:.4f}")
    print(f"  最終模型: who2com_pose_final.pth")
    print(f"  最佳模型: who2com_pose_best.pth")

    # 5. 繪製 Loss 曲線圖 (雙曲線：Train + Val)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', color='b', label='Train Loss', markersize=3)
    plt.plot(range(1, num_epochs+1), val_losses, marker='s', color='r', label='Val Loss', markersize=3)
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Coord + 0.5 * Vis)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=150)
    plt.close()
    print("Loss 曲線已儲存為 loss_curve.png！")

if __name__ == "__main__":
    train()
