import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class CMUMultiViewPoseDataset(Dataset):
    def __init__(self, root_dir, seq_name="160422_ultimatum1", num_views=3, image_size=256):
        """
        讀取真實 CMU Panoptic Dataset 的 DataLoader。
        資料集結構預期：
        root_dir/
           ├── calibration_{seq_name}.json
           ├── hdPose3d_stage1_coco19/ (解壓縮後的資料夾)
           └── hdVideos/
                 ├── hd_00_00/
                 ├── hd_00_01/
                 └── hd_00_02/
        """
        self.root_dir = root_dir
        self.seq_name = seq_name
        self.num_views = num_views
        self.image_size = image_size
        self.cams = ["00_00", "00_01", "00_02"][:num_views]
        
        # 載入相機參數 (Calibration)
        calib_path = os.path.join(root_dir, f"calibration_{seq_name}.json")
        self.cameras = {}
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
            for cam in calib_data['cameras']:
                if cam['type'] == 'hd' and cam['name'] in self.cams:
                    self.cameras[cam['name']] = {
                        'K': np.array(cam['K']),
                        'R': np.array(cam['R']),
                        't': np.array(cam['t'])
                    }
        else:
            print(f"[警告] 找不到相機校正檔: {calib_path}")

        # 掃描 3D 關節點 JSON (找出有幾幀畫面)
        self.pose3d_dir = os.path.join(root_dir, "hdPose3d_stage1_coco19")
        self.frames = []
        if os.path.exists(self.pose3d_dir):
            all_files = sorted(os.listdir(self.pose3d_dir))
            for f in all_files:
                if f.endswith('.json'):
                    self.frames.append(f)
        else:
            print(f"[警告] 找不到 3D 關節點資料夾: {self.pose3d_dir}。請確認已解壓縮 .tar 檔！")
            
        # 如果找不到真實資料，給個假的長度避免 Crash
        if len(self.frames) == 0:
            print("[警告] 資料集為空！將退回測試模式。")
            self.frames = ["dummy" for _ in range(10)]
            self.is_dummy = True
        else:
            self.is_dummy = False
            print(f"[*] 成功載入 CMU Dataset: 找到 {len(self.frames)} 幀標註資料。")

    def __len__(self):
        return len(self.frames)

    def project_3d_to_2d(self, pt3d, K, R, t):
        # pt3d: (3,)
        pt3d_cam = R @ pt3d + t
        pt2d_homo = K @ pt3d_cam
        # 避免除以零
        z = pt2d_homo[2] if pt2d_homo[2] != 0 else 1e-5
        x, y = pt2d_homo[0] / z, pt2d_homo[1] / z
        return x, y

    def __getitem__(self, idx):
        if self.is_dummy:
            # 退回隨機測試資料 (避免沒載好資料的人 Crash)
            images = torch.rand(self.num_views, 3, self.image_size, self.image_size)
            gt_2d_pose = torch.rand(17, 2)
            return images, gt_2d_pose

        frame_file = self.frames[idx]
        
        # 1. 讀取 3D 關節點 (Ground Truth)
        pose_path = os.path.join(self.pose3d_dir, frame_file)
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
            
        # 簡單起見，我們只抓第一個人 (bodies[0])
        if len(pose_data['bodies']) > 0:
            joints_19 = np.array(pose_data['bodies'][0]['joints19']).reshape(-1, 4)
            # 取前 17 個點的 X, Y, Z (第四個數字是 confidence)
            joints_3d = joints_19[:17, :3] 
        else:
            joints_3d = np.zeros((17, 3))

        # 2. 讀取多視角影像並計算 2D 投影
        images = []
        # 我們假設主視角 (Ego) 是第一台相機 (00_00)，所以投影座標以它為準
        ego_cam_name = self.cams[0]
        gt_2d_pose = np.zeros((17, 2))
        
        # 假設原始 CMU 影片解析度是 1920x1080 (HD)
        orig_w, orig_h = 1920, 1080
        
        for v_idx, cam_name in enumerate(self.cams):
            # 讀取圖片 (CMU 的檔名通常是 00000000.json 對應到 00000000.jpg 類似的格式)
            frame_num = frame_file.replace('body3DScene_', '').replace('.json', '')
            img_path = os.path.join(self.root_dir, "hdVideos", f"hd_{cam_name}", f"{frame_num}.jpg")
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # 找不到圖就給黑畫面
                img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                
            img = cv2.resize(img, (self.image_size, self.image_size))
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            images.append(img_tensor)
            
            # 投影第一台相機的 2D 關節點做為訓練答案 (0~1 之間)
            if v_idx == 0 and ego_cam_name in self.cameras:
                cam = self.cameras[ego_cam_name]
                for j in range(17):
                    px, py = self.project_3d_to_2d(joints_3d[j], cam['K'], cam['R'], cam['t'])
                    # 轉成 0~1 的正規化座標
                    gt_2d_pose[j, 0] = px / orig_w
                    gt_2d_pose[j, 1] = py / orig_h

        images_tensor = torch.stack(images) # (V, 3, 256, 256)
        gt_2d_tensor = torch.from_numpy(gt_2d_pose).float() # (17, 2)
        
        return images_tensor, gt_2d_tensor

def get_dataloader(data_dir="./data/160422_ultimatum1", batch_size=4, num_views=3, num_joints=17, shuffle=True):
    dataset = CMUMultiViewPoseDataset(
        root_dir=data_dir, 
        num_views=num_views
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader

if __name__ == "__main__":
    # 測試
    loader = get_dataloader("./data/160422_ultimatum1")
    for imgs, poses in loader:
        print(imgs.shape)
        print(poses.shape)
        break
