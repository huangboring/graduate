import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

def generate_heatmap(joints_2d, visibility, heatmap_size=64, sigma=2.0):
    """
    joints_2d: (17, 2) 座標值在 [0, 1] 之間
    visibility: (17,) 0 或 1
    回傳: (17, 64, 64) 的 heatmap
    """
    num_joints = joints_2d.shape[0]
    heatmaps = np.zeros((num_joints, heatmap_size, heatmap_size), dtype=np.float32)
    
    for i in range(num_joints):
        if visibility[i] > 0:
            # 轉換到 heatmap 座標系
            x = int(joints_2d[i, 0] * heatmap_size)
            y = int(joints_2d[i, 1] * heatmap_size)
            
            # 確保在邊界內
            if x < 0 or x >= heatmap_size or y < 0 or y >= heatmap_size:
                continue
                
            # 生成 2D 高斯分佈
            grid_y, grid_x = np.mgrid[0:heatmap_size, 0:heatmap_size]
            dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
            heatmaps[i] = np.exp(-dist / (2.0 * sigma ** 2))
            
    return torch.from_numpy(heatmaps)

class CMUMultiViewPoseDataset(Dataset):
    def __init__(self, root_dir, seq_name="160422_ultimatum1", num_views=3, image_size=256, heatmap_size=64):
        """
        讀取真實 CMU Panoptic Dataset 的 DataLoader。
        """
        self.root_dir = root_dir
        self.seq_name = seq_name
        self.num_views = num_views
        self.image_size = image_size
        self.heatmap_size = heatmap_size
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
            gt_heatmaps = torch.zeros((self.num_views, 17, self.heatmap_size, self.heatmap_size))
            gt_2d_coords = torch.rand(self.num_views, 17, 2)
            visibility = torch.ones(self.num_views, 17)
            # Dummy 相機矩陣
            cam_params = {
                'K': torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1),
                'R': torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1),
                't': torch.zeros(3).unsqueeze(0).repeat(self.num_views, 1)
            }
            return images, gt_heatmaps, gt_2d_coords, visibility, cam_params

        pose_path = self.frames[idx]
        frame_filename = os.path.basename(pose_path)
        
        # 1. 讀取 3D 關節點
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
            
        if len(pose_data['bodies']) > 0:
            joints_19 = np.array(pose_data['bodies'][0]['joints19']).reshape(-1, 4)
            joints_3d = joints_19[:17, :3]  # (17, 3)
            base_confidence = joints_19[:17, 3]   # (17,) 信心分數
            base_visibility = (base_confidence > 0).astype(np.float32)
        else:
            joints_3d = np.zeros((17, 3))
            base_visibility = np.zeros(17, dtype=np.float32)

        # 2. 讀取多視角影像並計算 2D 投影
        images = []
        gt_2d_coords_list = []
        gt_heatmaps_list = []
        visibility_list = []
        
        orig_w, orig_h = 1920, 1080
        
        cam_K = []
        cam_R = []
        cam_t = []
        
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
            
            gt_2d_pose = np.zeros((17, 2))
            vis = base_visibility.copy()
            
            if cam_name in self.cameras:
                cam = self.cameras[cam_name]
                cam_K.append(torch.from_numpy(cam['K']).float())
                cam_R.append(torch.from_numpy(cam['R']).float())
                cam_t.append(torch.from_numpy(cam['t']).float().view(3))
                
                for j in range(17):
                    px, py = self.project_3d_to_2d(joints_3d[j], cam['K'], cam['R'], cam['t'])
                    gt_2d_pose[j, 0] = px / orig_w
                    gt_2d_pose[j, 1] = py / orig_h
            else:
                cam_K.append(torch.eye(3).float())
                cam_R.append(torch.eye(3).float())
                cam_t.append(torch.zeros(3).float())

            # Check bounds
            in_frame = (gt_2d_pose[:, 0] >= 0) & (gt_2d_pose[:, 0] <= 1) & \
                       (gt_2d_pose[:, 1] >= 0) & (gt_2d_pose[:, 1] <= 1)
            vis = vis * in_frame.astype(np.float32)
            
            # 產生 Heatmap
            hm = generate_heatmap(gt_2d_pose, vis, self.heatmap_size)
            
            gt_2d_coords_list.append(torch.from_numpy(gt_2d_pose).float())
            visibility_list.append(torch.from_numpy(vis).float())
            gt_heatmaps_list.append(hm)

        images_tensor = torch.stack(images)
        gt_2d_tensor = torch.stack(gt_2d_coords_list)
        vis_tensor = torch.stack(visibility_list)
        gt_hms_tensor = torch.stack(gt_heatmaps_list)
        
        cam_params = {
            'K': torch.stack(cam_K),
            'R': torch.stack(cam_R),
            't': torch.stack(cam_t)
        }
        
        return images_tensor, gt_hms_tensor, gt_2d_tensor, vis_tensor, cam_params

def get_dataset(data_dir=None, num_views=3):
    if data_dir is None or not os.path.exists(data_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(base_dir)
        project_root = os.path.dirname(src_dir)
        thesis_root = os.path.dirname(project_root)
        
        candidates = [
            os.path.join(src_dir, "data", "160422_ultimatum1"),
            os.path.join(project_root, "data", "160422_ultimatum1"),
            os.path.join(thesis_root, "data", "panoptic-toolbox-master", "data", "160422_ultimatum1"),
            os.path.join(src_dir, "260804", "data", "160422_ultimatum1"),
            os.path.join(".", "data", "160422_ultimatum1"),
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                data_dir = candidate
                print(f"[*] 資料集路徑: {os.path.abspath(data_dir)}")
                break
        else:
            data_dir = candidates[0]
            print(f"[!] 找不到資料集，預設路徑: {os.path.abspath(data_dir)}")
    
    return CMUMultiViewPoseDataset(root_dir=data_dir, num_views=num_views)

def get_dataloader(data_dir=None, batch_size=4, num_views=3, shuffle=True):
    dataset = get_dataset(data_dir=data_dir, num_views=num_views)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

if __name__ == "__main__":
    loader = get_dataloader()
    for imgs, hms, coords, vis, cams in loader:
        print(f"Images: {imgs.shape}")
        print(f"Heatmaps: {hms.shape}")
        print(f"Coords: {coords.shape}")
        print(f"Visibility: {vis.shape}")
        print(f"Cameras K: {cams['K'].shape}")
        break
