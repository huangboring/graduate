import torch
import os
from dataset import get_dataloader
from model import Who2comPoseNet

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = "who2com_pose_fast.pth"
    num_views = 3
    
    print(f"[*] 準備進行模型評估 (裝置: {device})")
    
    # 1. 載入模型與權重
    model = Who2comPoseNet(num_views=num_views).to(device)
    if os.path.exists(weight_path):
        print(f"[*] 載入訓練好的權重: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print(f"[!] 找不到權重檔 {weight_path}，評估無法進行！")
        return
        
    model.eval()
    
    # 2. 載入測試資料集 (我們使用原本的 data_dir，設定 shuffle=False)
    print("[*] 正在載入資料集...")
    test_loader = get_dataloader(batch_size=4, num_views=num_views, shuffle=False)
    
    # 3. 開始評估
    total_mpjpe = 0.0
    total_samples = 0
    
    print("[*] 開始計算 MPJPE (Mean Per Joint Position Error)...")
    with torch.no_grad():
        for batch_idx, (images, gt_poses) in enumerate(test_loader):
            images = images.to(device)
            gt_poses = gt_poses.to(device)
            
            # 推論 (推論時我們把溫度設到非常小，讓 Gumbel Softmax 非常接近 Argmax)
            pred_poses, _ = model(images, temperature=0.01)
            
            # 計算這個 Batch 的 MPJPE
            # 將 0~1 的座標乘上影像大小 256 換算回像素
            pixel_dist = torch.norm((pred_poses - gt_poses) * 256.0, dim=-1) # (B, 17)
            
            # 加總這個 Batch 裡所有圖片的平均誤差
            batch_mpjpe = pixel_dist.mean(dim=-1).sum().item()
            
            total_mpjpe += batch_mpjpe
            total_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                print(f"  處理進度: Batch {batch_idx}/{len(test_loader)}")
                
    final_mpjpe = total_mpjpe / total_samples
    print("\n" + "="*50)
    print(f"🎉 評估完成！")
    print(f"👉 您訓練出來的模型最終 MPJPE 為: {final_mpjpe:.2f} 像素 (Pixels)")
    print("="*50)

if __name__ == "__main__":
    evaluate()
