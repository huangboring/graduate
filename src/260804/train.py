import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloader
from model import Who2comPoseNet

def train():
    # 1. 參數設定 (快速驗證版)
    batch_size = 4
    num_epochs = 50        # 微型資料集跑很快，提高 Epoch 讓它收斂
    num_views = 3
    learning_rate = 5e-4   # 稍微提高學習率，加速收斂
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用裝置: {device}")

    # 2. 準備 DataLoader 與 模型
    train_loader = get_dataloader(batch_size=batch_size, num_views=num_views)
    model = Who2comPoseNet(num_views=num_views).to(device)
    
    # 3. 設定優化器與損失函數
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss() # 使用 SmoothL1Loss (Huber Loss) 回歸關節座標

    # 4. 訓練迴圈
    model.train()
    epoch_losses = [] # 用來記錄每個 Epoch 的平均 Loss 供最後畫圖

    for epoch in range(num_epochs):
        total_loss = 0.0
        # 隨著 Epoch 遞減 Gumbel-Softmax 的 Temperature
        # 讓挑選行為從「較為平滑的機率分布」慢慢變成「接近絕對的 One-hot 挑選」
        temp = max(0.1, 1.0 * (0.85 ** epoch)) 
        
        for batch_idx, (images, gt_poses) in enumerate(train_loader):
            images = images.to(device)
            gt_poses = gt_poses.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            pred_poses, selection_weights = model(images, temperature=temp)
            
            # 計算 Loss
            loss = criterion(pred_poses, gt_poses)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f} | Temp: {temp:.2f} | "
                      f"Ego 選擇了 View 索引: {selection_weights[0].argmax().item() + 1}")
                
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"==> Epoch {epoch+1} 結束, 平均 Loss: {avg_loss:.4f}\n")

    print("訓練完成！儲存權重檔中...")
    torch.save(model.state_dict(), 'who2com_pose_fast.pth')
    
    # 5. 繪製並儲存 Loss 曲線圖
    print("繪製 Loss 曲線圖中...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o', color='b', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Smooth L1 Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    print("Loss 曲線已儲存為 loss_curve.png！")

if __name__ == "__main__":
    train()
