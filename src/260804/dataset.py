import torch
from torch.utils.data import Dataset, DataLoader

class DummyMultiViewPoseDataset(Dataset):
    def __init__(self, num_samples=100, num_views=4, image_size=256, num_joints=17):
        """
        模擬多視角 3D 關節點的 Dataset。
        :param num_samples: 總資料筆數
        :param num_views: 攝影機視角數量 (包含 Ego 視角)
        :param image_size: 影像的長寬
        :param num_joints: 3D 關節點的數量 (通常為 17)
        """
        self.num_samples = num_samples
        self.num_views = num_views
        self.image_size = image_size
        self.num_joints = num_joints

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模擬產生多個視角的影像 Tensor (V, C, H, W)
        # 在現實中，這裡應該是讀取多張照片並做 Transform
        images = torch.rand(self.num_views, 3, self.image_size, self.image_size)
        
        # 模擬產生真實的 3D 關節點 (J, 3) 
        # (通常相對於 Ego 視角的 Root Joint，數值範圍依需求而定，這裡假設在 -1 到 1 之間)
        gt_3d_pose = torch.rand(self.num_joints, 3) * 2 - 1.0
        
        return images, gt_3d_pose

def get_dataloader(batch_size=8, num_samples=100, num_views=4, num_joints=17, shuffle=True):
    dataset = DummyMultiViewPoseDataset(
        num_samples=num_samples, 
        num_views=num_views, 
        num_joints=num_joints
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader

if __name__ == "__main__":
    # 測試 DataLoader
    dataloader = get_dataloader(batch_size=2, num_views=3) # 模擬 3 個 Webcam
    for batch_idx, (images, poses) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f" - Images shape: {images.shape} (B, V, C, H, W)")
        print(f" - Poses shape:  {poses.shape} (B, J, 3)")
        break
