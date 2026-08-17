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

        # 掃描 3D 關節點 JSON
        self.pose3d_dir = os.path.join(root_dir, "hdPose3d_stage1_coco19")
        self.frames = []
        if os.path.exists(self.pose3d_dir):
            for root, dirs, files in os.walk(self.pose3d_dir):
                for f in sorted(files):
                    if f.endswith('.json'):
                        full_path = os.path.join(root, f)
                        try:
                            with open(full_path, 'r') as fp:
                                p_data = json.load(fp)
                                if len(p_data.get('bodies', [])) > 0:
                                    self.frames.append(full_path)
                        except Exception:
                            pass
        else:
            print(f"[警告] 找不到 3D 關節點資料夾: {self.pose3d_dir}")
            
        if len(self.frames) == 0:
            print("[警告] 資料集為空！將退回測試模式。")
            self.frames = ["dummy" for _ in range(10)]
            self.is_dummy = True
        else:
            self.is_dummy = False
            print(f"[*] 成功載入 CMU Dataset: 找到 {len(self.frames)} 幀有效標註資料。")

    def __len__(self):
        return len(self.frames)

    def project_3d_to_2d(self, pt3d, K, R, t):
        pt3d = pt3d.reshape(3, 1)
        t = t.reshape(3, 1)
        pt3d_cam = R @ pt3d + t
        pt2d_homo = K @ pt3d_cam
        z = float(pt2d_homo[2, 0]) if float(pt2d_homo[2, 0]) != 0 else 1e-5
        x = float(pt2d_homo[0, 0]) / z
        y = float(pt2d_homo[1, 0]) / z
        return x, y

    def __getitem__(self, idx):
        if self.is_dummy:
            images = torch.rand(self.num_views, 3, self.image_size, self.image_size)
            gt_2d_pose = torch.rand(17, 2)
            visibility = torch.ones(17)  # dummy visibility
            return images, gt_2d_pose, visibility

        pose_path = self.frames[idx]
        frame_filename = os.path.basename(pose_path)
        
        # 1. 讀取 3D 關節點
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
            
        if len(pose_data['bodies']) > 0:
            joints_19 = np.array(pose_data['bodies'][0]['joints19']).reshape(-1, 4)
            joints_3d = joints_19[:17, :3]  # (17, 3)
            confidence = joints_19[:17, 3]   # (17,) 信心分數
            # 將 confidence 轉成二元可見性標記
            # CMU 資料中 -1 代表完全無效，>0 代表有偵測到
            visibility = (confidence > 0).astype(np.float32)
        else:
            joints_3d = np.zeros((17, 3))
            visibility = np.zeros(17, dtype=np.float32)

        # 2. 讀取多視角影像並計算 2D 投影
        images = []
        ego_cam_name = self.cams[0]
        gt_2d_pose = np.zeros((17, 2))
        orig_w, orig_h = 1920, 1080
        
        for v_idx, cam_name in enumerate(self.cams):
            frame_num = frame_filename.replace('body3DScene_', '').replace('.json', '')
            img_path = os.path.join(self.root_dir, "hdVideos", f"hd_{cam_name}", f"{frame_num}.jpg")
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                
            img = cv2.resize(img, (self.image_size, self.image_size))
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            images.append(img_tensor)
            
            if v_idx == 0 and ego_cam_name in self.cameras:
                cam = self.cameras[ego_cam_name]
                for j in range(17):
                    px, py = self.project_3d_to_2d(joints_3d[j], cam['K'], cam['R'], cam['t'])
                    gt_2d_pose[j, 0] = px / orig_w
                    gt_2d_pose[j, 1] = py / orig_h

        # 3. Check if projected 2D coordinates are within frame bounds
        # If a joint projects outside the image, mark it as not visible
        in_frame = (gt_2d_pose[:, 0] >= 0) & (gt_2d_pose[:, 0] <= 1) & \
                   (gt_2d_pose[:, 1] >= 0) & (gt_2d_pose[:, 1] <= 1)
        visibility = visibility * in_frame.astype(np.float32)

        images_tensor = torch.stack(images)
        gt_2d_tensor = torch.from_numpy(gt_2d_pose).float()
        vis_tensor = torch.from_numpy(visibility).float()
        
        return images_tensor, gt_2d_tensor, vis_tensor

def get_dataset(data_dir=None, num_views=3):
    """回傳 Dataset 物件，方便外部做 train/val split"""
    if data_dir is None or not os.path.exists(data_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 往上兩層找共用 data 資料夾
        project_root = os.path.dirname(os.path.dirname(base_dir))
        default_dir = os.path.join(project_root, "data", "160422_ultimatum1")
        if os.path.exists(default_dir):
            data_dir = default_dir
        else:
            # 嘗試舊版路徑
            old_dir = os.path.join(os.path.dirname(base_dir), "260804", "data", "160422_ultimatum1")
            if os.path.exists(old_dir):
                data_dir = old_dir
            else:
                data_dir = "./data/160422_ultimatum1"
    
    return CMUMultiViewPoseDataset(root_dir=data_dir, num_views=num_views)

def get_dataloader(data_dir=None, batch_size=4, num_views=3, shuffle=True):
    """回傳 DataLoader (向後相容)"""
    dataset = get_dataset(data_dir=data_dir, num_views=num_views)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

if __name__ == "__main__":
    loader = get_dataloader()
    for imgs, poses, vis in loader:
        print(f"Images: {imgs.shape}")
        print(f"Poses: {poses.shape}")
        print(f"Visibility: {vis.shape}")
        print(f"Visible joints in first sample: {vis[0].sum().item():.0f}/17")
        break
