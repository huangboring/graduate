import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import get_dataset
from model import When2comPoseNet

def train():
    # 1. 參數設定
    batch_size = 4
    num_epochs = 80            # 升級版網路需要更多 Epoch 來收斂
    num_views = 3
    learning_rate = 5e-4
    comm_threshold = 0.5
    comm_loss_weight = 0.1   # 通訊正則化權重
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
    model = When2comPoseNet(num_views=num_views).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 4. 訓練迴圈
    train_losses = []
    val_losses = []
    train_mpjpes = []
    val_mpjpes = []
    train_comm_rates = []
    val_comm_rates = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # === 訓練階段 ===
        model.train()
        total_train_loss = 0.0
        total_train_mpjpe = 0.0
        total_train_comm = 0.0
        num_train_batches = 0
        temp = max(0.1, 1.0 * (0.85 ** epoch))
        
        for batch_idx, (images, gt_poses, gt_vis) in enumerate(train_loader):
            images = images.to(device)
            gt_poses = gt_poses.to(device)
            gt_vis = gt_vis.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass (新增 comm_prob 回傳值)
            pred_poses, pred_vis, selection_weights, comm_prob = model(
                images, temperature=temp, comm_threshold=comm_threshold
            )
            
            # Visibility-Aware Coordinate Loss
            vis_mask = gt_vis.unsqueeze(-1)  # (B, 17, 1)
            coord_loss = F.smooth_l1_loss(
                pred_poses * vis_mask,
                gt_poses * vis_mask
            )
            
            # Visibility BCE Loss
            vis_loss = F.binary_cross_entropy(pred_vis, gt_vis)
            
            # Communication Regularization Loss (鼓勵節省頻寬)
            comm_loss = comm_prob.mean()
            
            # 總損失
            loss = coord_loss + 0.5 * vis_loss + comm_loss_weight * comm_loss
            
            loss.backward()
            optimizer.step()
            
            # 統計
            total_train_loss += loss.item()
            total_train_comm += (comm_prob > comm_threshold).float().mean().item()
            num_train_batches += 1
            
            # MPJPE (不計入梯度)
            with torch.no_grad():
                pixel_dist = torch.norm((pred_poses - gt_poses) * 256.0, dim=-1)  # (B, 17)
                total_train_mpjpe += pixel_dist.mean().item()
            
            if batch_idx % 10 == 0:
                comm_rate = (comm_prob > comm_threshold).float().mean().item()
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} (Coord: {coord_loss.item():.4f}, "
                      f"Vis: {vis_loss.item():.4f}, Comm: {comm_loss.item():.4f}) | "
                      f"CommRate: {comm_rate:.0%} | Temp: {temp:.2f}")
        
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_mpjpe = total_train_mpjpe / num_train_batches
        avg_train_comm = total_train_comm / num_train_batches
        train_losses.append(avg_train_loss)
        train_mpjpes.append(avg_train_mpjpe)
        train_comm_rates.append(avg_train_comm)
        
        # === 驗證階段 ===
        model.eval()
        total_val_loss = 0.0
        total_val_mpjpe = 0.0
        total_val_comm = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for images, gt_poses, gt_vis in val_loader:
                images = images.to(device)
                gt_poses = gt_poses.to(device)
                gt_vis = gt_vis.to(device)
                
                pred_poses, pred_vis, _, comm_prob = model(
                    images, temperature=0.1, comm_threshold=comm_threshold
                )
                
                vis_mask = gt_vis.unsqueeze(-1)
                coord_loss = F.smooth_l1_loss(
                    pred_poses * vis_mask,
                    gt_poses * vis_mask
                )
                vis_loss = F.binary_cross_entropy(pred_vis, gt_vis)
                comm_loss = comm_prob.mean()
                loss = coord_loss + 0.5 * vis_loss + comm_loss_weight * comm_loss
                
                total_val_loss += loss.item()
                total_val_comm += (comm_prob > comm_threshold).float().mean().item()
                num_val_batches += 1
                
                pixel_dist = torch.norm((pred_poses - gt_poses) * 256.0, dim=-1)
                total_val_mpjpe += pixel_dist.mean().item()
        
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_mpjpe = total_val_mpjpe / num_val_batches
        avg_val_comm = total_val_comm / num_val_batches
        val_losses.append(avg_val_loss)
        val_mpjpes.append(avg_val_mpjpe)
        val_comm_rates.append(avg_val_comm)
        
        print(f"==> Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train MPJPE: {avg_train_mpjpe:.1f}px | Val MPJPE: {avg_val_mpjpe:.1f}px | "
              f"Train Comm: {avg_train_comm:.0%} | Val Comm: {avg_val_comm:.0%}")
        
        # 儲存最佳模型 (以 Validation Loss 為準)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'when2com_pose_best.pth')
            print(f"    ★ 新的最佳模型！Val Loss: {best_val_loss:.4f}，已儲存 when2com_pose_best.pth")
    
    # 最後也儲存一份最終模型
    torch.save(model.state_dict(), 'when2com_pose_final.pth')
    print(f"\n訓練完成！")
    print(f"  最佳驗證 Loss: {best_val_loss:.4f}")
    print(f"  最終模型: when2com_pose_final.pth")
    print(f"  最佳模型: when2com_pose_best.pth")

    # 5. 繪製三合一圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, num_epochs + 1)
    
    # 子圖 1: Loss 曲線
    axes[0].plot(epochs_range, train_losses, 'b-o', label='Train Loss', markersize=3)
    axes[0].plot(epochs_range, val_losses, 'r-s', label='Val Loss', markersize=3)
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Coord + 0.5*Vis + 0.1*Comm)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 子圖 2: MPJPE 曲線
    axes[1].plot(epochs_range, train_mpjpes, 'b-o', label='Train MPJPE', markersize=3)
    axes[1].plot(epochs_range, val_mpjpes, 'r-s', label='Val MPJPE', markersize=3)
    axes[1].set_title('MPJPE (Mean Per Joint Position Error)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pixel Error')
    axes[1].legend()
    axes[1].grid(True)
    
    # 子圖 3: 通訊率曲線
    axes[2].plot(epochs_range, train_comm_rates, 'b-o', label='Train Comm Rate', markersize=3)
    axes[2].plot(epochs_range, val_comm_rates, 'r-s', label='Val Comm Rate', markersize=3)
    axes[2].set_title('Communication Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Rate')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    plt.close()
    print("三合一圖表已儲存為 loss_curve.png！")

if __name__ == "__main__":
    train()
