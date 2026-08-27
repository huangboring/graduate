import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
from dataset import get_dataset
from model import When2comPoseNet


def compute_loss(pred_poses, pred_vis, gt_poses, gt_vis):
    """計算 Visibility-Aware Loss (座標 + 可見性)"""
    vis_mask = gt_vis.unsqueeze(-1)  # (B, 17, 1)
    coord_loss = F.smooth_l1_loss(pred_poses * vis_mask, gt_poses * vis_mask)
    vis_loss = F.binary_cross_entropy(pred_vis, gt_vis)
    return coord_loss, vis_loss


def compute_mpjpe(pred_poses, gt_poses, gt_vis):
    """計算 MPJPE (只算可見關節)"""
    pixel_dist = torch.norm((pred_poses - gt_poses) * 256.0, dim=-1)  # (B, 17)
    vis_mask = gt_vis > 0.5
    if vis_mask.sum() > 0:
        return pixel_dist[vis_mask].mean().item()
    return pixel_dist.mean().item()


def train_stage1(model, train_loader, val_loader, device, num_epochs=80):
    """
    第一階段：強制通訊模式
    - 跳過 Communication Gate，永遠使用融合後的特徵
    - 訓練 FeatureExtractor, Matchmaker, CrossAttentionFusion, PoseHead
    - comm_gate 不參與訓練（但參數會被保留）
    """
    print("=" * 60)
    print("  第一階段：強制通訊模式（訓練融合能力）")
    print("=" * 60)
    
    # 凍結 comm_gate（第一階段不需要訓練它）
    for param in model.comm_gate.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4, weight_decay=1e-4
    )
    
    train_losses, val_losses = [], []
    train_mpjpes, val_mpjpes = [], []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # === 訓練 ===
        model.train()
        total_loss, total_mpjpe, num_batches = 0, 0, 0
        temp = max(0.1, 1.0 * (0.85 ** epoch))
        
        for batch_idx, (images, gt_poses, gt_vis) in enumerate(train_loader):
            images = images.to(device)
            gt_poses = gt_poses.to(device)
            gt_vis = gt_vis.to(device)
            
            optimizer.zero_grad()
            
            # 強制通訊：force_comm=True
            pred_poses, pred_vis, weights, _ = model(
                images, temperature=temp, force_comm=True
            )
            
            coord_loss, vis_loss = compute_loss(pred_poses, pred_vis, gt_poses, gt_vis)
            loss = coord_loss + 0.5 * vis_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            with torch.no_grad():
                total_mpjpe += compute_mpjpe(pred_poses, gt_poses, gt_vis)
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  [S1] Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} (Coord: {coord_loss.item():.4f}, "
                      f"Vis: {vis_loss.item():.4f}) | Temp: {temp:.2f}")
        
        avg_train_loss = total_loss / num_batches
        avg_train_mpjpe = total_mpjpe / num_batches
        train_losses.append(avg_train_loss)
        train_mpjpes.append(avg_train_mpjpe)
        
        # === 驗證 ===
        model.eval()
        total_loss, total_mpjpe, num_batches = 0, 0, 0
        
        with torch.no_grad():
            for images, gt_poses, gt_vis in val_loader:
                images = images.to(device)
                gt_poses = gt_poses.to(device)
                gt_vis = gt_vis.to(device)
                
                pred_poses, pred_vis, _, _ = model(
                    images, temperature=0.1, force_comm=True
                )
                
                coord_loss, vis_loss = compute_loss(pred_poses, pred_vis, gt_poses, gt_vis)
                loss = coord_loss + 0.5 * vis_loss
                
                total_loss += loss.item()
                total_mpjpe += compute_mpjpe(pred_poses, gt_poses, gt_vis)
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        avg_val_mpjpe = total_mpjpe / num_batches
        val_losses.append(avg_val_loss)
        val_mpjpes.append(avg_val_mpjpe)
        
        print(f"==> [S1] Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train MPJPE: {avg_train_mpjpe:.1f}px | Val MPJPE: {avg_val_mpjpe:.1f}px")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'stage1_best.pth')
            print(f"    ★ 新的最佳模型！Val Loss: {best_val_loss:.4f}")
    
    # 解凍 comm_gate（為第二階段做準備）
    for param in model.comm_gate.parameters():
        param.requires_grad = True
    
    # 繪製第一階段圖表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, num_epochs + 1)
    
    axes[0].plot(epochs_range, train_losses, 'b-o', label='Train', markersize=2)
    axes[0].plot(epochs_range, val_losses, 'r-s', label='Val', markersize=2)
    axes[0].set_title('Stage 1: Loss (Forced Communication)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs_range, train_mpjpes, 'b-o', label='Train', markersize=2)
    axes[1].plot(epochs_range, val_mpjpes, 'r-s', label='Val', markersize=2)
    axes[1].set_title('Stage 1: MPJPE (Visible Joints Only)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pixel Error')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('stage1_curve.png', dpi=150)
    plt.close()
    print(f"\n第一階段完成！最佳 Val Loss: {best_val_loss:.4f}")
    print(f"圖表已儲存為 stage1_curve.png")
    
    return best_val_loss


def train_stage2(model, train_loader, val_loader, device, num_epochs=20):
    """
    第二階段：訓練 Communication Gate
    - 凍結所有模組，只訓練 comm_gate
    - 動態產生 GT 標籤：比較「通訊 vs 不通訊」的 Loss 差異
    """
    print("\n" + "=" * 60)
    print("  第二階段：訓練通訊閘門（學習何時通訊）")
    print("=" * 60)
    
    # 載入第一階段最佳權重
    ckpt = torch.load('stage1_best.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print("  已載入 stage1_best.pth")
    
    # 凍結所有模組，只保留 comm_gate 可訓練
    for name, param in model.named_parameters():
        if 'comm_gate' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可訓練參數: {trainable:,} / {total:,} ({trainable/total:.1%})")
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3  # 閘門參數少，可以用較大的學習率
    )
    
    train_losses, val_losses = [], []
    train_comm_rates, val_comm_rates = [], []
    train_gate_accs, val_gate_accs = [], []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # === 訓練 ===
        model.train()
        total_gate_loss, total_comm, total_gate_correct, total_gate_total = 0, 0, 0, 0
        num_batches = 0
        
        for batch_idx, (images, gt_poses, gt_vis) in enumerate(train_loader):
            images = images.to(device)
            gt_poses = gt_poses.to(device)
            gt_vis = gt_vis.to(device)
            
            # Step 1: 用「強制通訊」跑一次，算出通訊時的 Loss
            with torch.no_grad():
                pred_comm, vis_comm, _, _ = model(images, force_comm=True)
                coord_loss_comm, vis_loss_comm = compute_loss(
                    pred_comm, vis_comm, gt_poses, gt_vis
                )
                loss_with_comm = coord_loss_comm + 0.5 * vis_loss_comm
            
            # Step 2: 用「強制不通訊」跑一次，算出不通訊時的 Loss
            with torch.no_grad():
                pred_no, vis_no, _, _ = model(images, force_no_comm=True)
                coord_loss_no, vis_loss_no = compute_loss(
                    pred_no, vis_no, gt_poses, gt_vis
                )
                loss_without_comm = coord_loss_no + 0.5 * vis_loss_no
            
            # Step 3: 產生動態 GT 標籤
            # 通訊有幫助 (Loss 降低) → 應該通訊 (GT=1)
            # 通訊沒幫助 → 不該通訊 (GT=0)
            gate_gt = (loss_with_comm < loss_without_comm).float().unsqueeze(0)  # (1,1)
            gate_gt = gate_gt.expand(images.shape[0], -1)  # (B, 1)
            
            # Step 4: 正常跑一次前向傳播，取得 comm_prob
            optimizer.zero_grad()
            _, _, _, comm_prob = model(images, force_comm=True)  # force_comm 確保梯度流過
            
            # Step 5: BCE Loss 訓練閘門
            gate_loss = F.binary_cross_entropy(comm_prob, gate_gt)
            gate_loss.backward()
            optimizer.step()
            
            # 統計
            total_gate_loss += gate_loss.item()
            total_comm += (comm_prob > 0.5).float().mean().item()
            pred_gate = (comm_prob > 0.5).float()
            total_gate_correct += (pred_gate == gate_gt).float().sum().item()
            total_gate_total += gate_gt.numel()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                gt_label = "通訊" if gate_gt[0, 0] > 0.5 else "不通訊"
                print(f"  [S2] Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} | "
                      f"Gate Loss: {gate_loss.item():.4f} | "
                      f"CommProb: {comm_prob[0, 0].item():.3f} | "
                      f"GT: {gt_label} | "
                      f"LossΔ: {(loss_without_comm - loss_with_comm).item():.4f}")
        
        avg_gate_loss = total_gate_loss / num_batches
        avg_comm_rate = total_comm / num_batches
        avg_gate_acc = total_gate_correct / total_gate_total
        train_losses.append(avg_gate_loss)
        train_comm_rates.append(avg_comm_rate)
        train_gate_accs.append(avg_gate_acc)
        
        # === 驗證 ===
        model.eval()
        total_gate_loss, total_comm, total_gate_correct, total_gate_total = 0, 0, 0, 0
        num_batches = 0
        
        with torch.no_grad():
            for images, gt_poses, gt_vis in val_loader:
                images = images.to(device)
                gt_poses = gt_poses.to(device)
                gt_vis = gt_vis.to(device)
                
                # 通訊 vs 不通訊比較
                pred_comm, vis_comm, _, _ = model(images, force_comm=True)
                c1, v1 = compute_loss(pred_comm, vis_comm, gt_poses, gt_vis)
                loss_comm = c1 + 0.5 * v1
                
                pred_no, vis_no, _, _ = model(images, force_no_comm=True)
                c2, v2 = compute_loss(pred_no, vis_no, gt_poses, gt_vis)
                loss_no = c2 + 0.5 * v2
                
                gate_gt = (loss_comm < loss_no).float().unsqueeze(0).expand(images.shape[0], -1)
                
                # 取得 comm_prob
                _, _, _, comm_prob = model(images)
                gate_loss = F.binary_cross_entropy(comm_prob, gate_gt)
                
                total_gate_loss += gate_loss.item()
                total_comm += (comm_prob > 0.5).float().mean().item()
                pred_gate = (comm_prob > 0.5).float()
                total_gate_correct += (pred_gate == gate_gt).float().sum().item()
                total_gate_total += gate_gt.numel()
                num_batches += 1
        
        avg_val_loss = total_gate_loss / num_batches
        avg_val_comm = total_comm / num_batches
        avg_val_acc = total_gate_correct / total_gate_total
        val_losses.append(avg_val_loss)
        val_comm_rates.append(avg_val_comm)
        val_gate_accs.append(avg_val_acc)
        
        print(f"==> [S2] Epoch {epoch+1} | "
              f"Train Gate Loss: {avg_gate_loss:.4f} | Val Gate Loss: {avg_val_loss:.4f} | "
              f"Train Comm: {avg_comm_rate:.0%} | Val Comm: {avg_val_comm:.0%} | "
              f"Train Acc: {avg_gate_acc:.1%} | Val Acc: {avg_val_acc:.1%}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'when2com_pose_best.pth')
            print(f"    ★ 新的最佳模型！Val Gate Loss: {best_val_loss:.4f}")
    
    # 儲存最終模型
    torch.save(model.state_dict(), 'when2com_pose_final.pth')
    
    # 繪製第二階段圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, num_epochs + 1)
    
    axes[0].plot(epochs_range, train_losses, 'b-o', label='Train', markersize=3)
    axes[0].plot(epochs_range, val_losses, 'r-s', label='Val', markersize=3)
    axes[0].set_title('Stage 2: Gate Loss (BCE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs_range, train_comm_rates, 'b-o', label='Train', markersize=3)
    axes[1].plot(epochs_range, val_comm_rates, 'r-s', label='Val', markersize=3)
    axes[1].set_title('Stage 2: Communication Rate')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(epochs_range, train_gate_accs, 'b-o', label='Train', markersize=3)
    axes[2].plot(epochs_range, val_gate_accs, 'r-s', label='Val', markersize=3)
    axes[2].set_title('Stage 2: Gate Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('stage2_curve.png', dpi=150)
    plt.close()
    print(f"\n第二階段完成！最佳 Val Gate Loss: {best_val_loss:.4f}")
    print(f"圖表已儲存為 stage2_curve.png")


def train():
    # 參數設定
    batch_size = 4
    num_views = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # 準備 Dataset
    full_dataset = get_dataset(num_views=num_views)
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"訓練集: {train_size} 幀, 驗證集: {val_size} 幀")
    
    # 建立模型
    model = When2comPoseNet(num_views=num_views).to(device)
    
    # 如果 stage1_best.pth 已經存在，跳過第一階段
    if os.path.exists('stage1_best.pth'):
        print("\n偵測到 stage1_best.pth 已存在，跳過第一階段！")
        print("如果要重新訓練第一階段，請先刪除 stage1_best.pth")
    else:
        train_stage1(model, train_loader, val_loader, device, num_epochs=80)
    
    # 第二階段
    train_stage2(model, train_loader, val_loader, device, num_epochs=20)
    
    print("\n" + "=" * 60)
    print("  兩階段訓練全部完成！")
    print("  stage1_best.pth  → 融合模組的最佳權重")
    print("  when2com_pose_best.pth → 完整模型（含閘門）的最佳權重")
    print("=" * 60)


if __name__ == "__main__":
    train()
