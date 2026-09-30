import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

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
            x = int(joints_2d[i, 0] * heatmap_size)
            y = int(joints_2d[i, 1] * heatmap_size)
            
            if x < 0 or x >= heatmap_size or y < 0 or y >= heatmap_size:
                continue
                
            grid_y, grid_x = np.mgrid[0:heatmap_size, 0:heatmap_size]
            dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
            heatmaps[i] = np.exp(-dist / (2.0 * sigma ** 2))
            
    return torch.from_numpy(heatmaps)

class CMUMultiViewPoseDataset(Dataset):
    def __init__(self, root_dir, seq_name="160422_ultimatum1", num_views=3, image_size=256, heatmap_size=64):
        self.root_dir = root_dir
        self.seq_name = seq_name
        self.num_views = num_views
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.cams = ["00_00", "00_01", "00_02"][:num_views]
        
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        
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
                        't': np.array(cam['t']),
                        'distCoef': np.array(cam['distCoef'])
                    }
        else:
            print(f"[警告] 找不到相機校正檔: {calib_path}")

        # 掃描 3D 關節點 JSON
        self.pose3d_dir = os.path.join(root_dir, "hdPose3d_stage1_coco19")
        # 嘗試尋找子目錄
        inner_dir = os.path.join(self.pose3d_dir, "hdPose3d_stage1_coco19")
        if os.path.exists(inner_dir):
            self.pose3d_dir = inner_dir
            
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

    def project_3d_to_2d(self, pt3d, K, R, t, distCoef):
        # 使用 OpenCV 的 projectPoints 處理畸變
        rvec, _ = cv2.Rodrigues(R)
        imgpt, _ = cv2.projectPoints(pt3d.reshape(1, 3).astype(np.float32), rvec, t, K, distCoef)
        return imgpt[0, 0, 0], imgpt[0, 0, 1]

    def __getitem__(self, idx):
        if self.is_dummy:
            images = torch.rand(self.num_views, 3, self.image_size, self.image_size)
            images = torch.stack([self.normalize(img) for img in images])
            
            gt_heatmaps = torch.zeros((self.num_views, 17, self.heatmap_size, self.heatmap_size))
            gt_2d_coords = torch.rand(self.num_views, 17, 2)
            visibility = torch.ones(self.num_views, 17)
            union_vis = torch.ones(17)
            cam_params = {
                'K': torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1),
                'R': torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1),
                't': torch.zeros(3).unsqueeze(0).repeat(self.num_views, 1)
            }
            return images, gt_heatmaps, gt_2d_coords, visibility, union_vis, cam_params

        pose_path = self.frames[idx]
        frame_filename = os.path.basename(pose_path)
        
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
                
            gt_2d_pose = np.zeros((17, 2))
            vis = base_visibility.copy()
            
            if cam_name in self.cameras:
                cam = self.cameras[cam_name]
                cam_K.append(torch.from_numpy(cam['K']).float())
                cam_R.append(torch.from_numpy(cam['R']).float())
                cam_t.append(torch.from_numpy(cam['t']).float().view(3))
                
                for j in range(17):
                    px, py = self.project_3d_to_2d(joints_3d[j], cam['K'], cam['R'], cam['t'], cam['distCoef'])
                    gt_2d_pose[j, 0] = px
                    gt_2d_pose[j, 1] = py
            else:
                cam_K.append(torch.eye(3).float())
                cam_R.append(torch.eye(3).float())
                cam_t.append(torch.zeros(3).float())

            # 判斷可見度：若投影點在影像邊界外，設為不可見
            in_frame = (gt_2d_pose[:, 0] >= 0) & (gt_2d_pose[:, 0] < orig_w) & \
                       (gt_2d_pose[:, 1] >= 0) & (gt_2d_pose[:, 1] < orig_h)
            vis = vis * in_frame.astype(np.float32)
            
            # 決定 Bounding Box
            valid_pts = gt_2d_pose[vis > 0]
            if len(valid_pts) > 0:
                xmin, ymin = np.min(valid_pts[:, 0]), np.min(valid_pts[:, 1])
                xmax, ymax = np.max(valid_pts[:, 0]), np.max(valid_pts[:, 1])
                
                center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=np.float32)
                width = xmax - xmin
                height = ymax - ymin
                
                # BBox margin (20% enlarged)
                scale = max(width, height) * 1.2
            else:
                center = np.array([orig_w / 2, orig_h / 2], dtype=np.float32)
                scale = orig_h # fallback scale
                
            # 轉換矩陣
            trans = get_affine_transform(center, scale, 0, [self.image_size, self.image_size])
            
            # 對影像進行 crop & pad 變形
            img_cropped = cv2.warpAffine(
                img, 
                trans, 
                (self.image_size, self.image_size), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            # 對關節點進行相同的變形
            for j in range(17):
                if vis[j] > 0:
                    pt = np.array([gt_2d_pose[j, 0], gt_2d_pose[j, 1], 1.0]).reshape(3, 1)
                    pt_trans = trans @ pt
                    gt_2d_pose[j, 0] = pt_trans[0, 0] / self.image_size
                    gt_2d_pose[j, 1] = pt_trans[1, 0] / self.image_size
                else:
                    gt_2d_pose[j, 0] = 0.0
                    gt_2d_pose[j, 1] = 0.0

            # 變形後的點也可能超出 256x256，再次檢查
            in_crop = (gt_2d_pose[:, 0] >= 0) & (gt_2d_pose[:, 0] <= 1) & \
                      (gt_2d_pose[:, 1] >= 0) & (gt_2d_pose[:, 1] <= 1)
            vis = vis * in_crop.astype(np.float32)

            # 正規化影像
            img_tensor = torch.from_numpy(img_cropped).float().permute(2, 0, 1) / 255.0
            img_tensor = self.normalize(img_tensor)
            images.append(img_tensor)
            
            # 產生 Heatmap (現在已經對齊 256x256 了)
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
        
        union_vis_tensor = (vis_tensor.sum(dim=0) > 0).float()
        
        return images_tensor, gt_hms_tensor, gt_2d_tensor, vis_tensor, union_vis_tensor, cam_params

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
    loader = get_dataloader(batch_size=1, shuffle=True)
    for batch in loader:
        imgs, hms, coords, vis, union_vis, cams = batch
        
        # 畫出 view 0 的 GT
        img0 = imgs[0, 0].permute(1, 2, 0).cpu().numpy()
        # 復原 ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img0 = std * img0 + mean
        img0 = np.clip(img0, 0, 1)
        img0 = (img0 * 255).astype(np.uint8)
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        
        gt_coord = coords[0, 0].cpu().numpy()
        v = vis[0, 0].cpu().numpy()
        
        for i in range(17):
            if v[i] > 0:
                x = int(gt_coord[i, 0] * 256)
                y = int(gt_coord[i, 1] * 256)
                cv2.circle(img0, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(img0, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
        cv2.imwrite('test_gt_new.jpg', img0)
        print('Saved test_gt_new.jpg')
        break
