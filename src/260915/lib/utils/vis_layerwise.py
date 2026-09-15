import numpy as np
import cv2
import os
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
           [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

PERSON_COLORS_3D = ['blue', 'green', 'cyan', 'magenta', 'yellow',
                    'orange', 'purple', 'lime', 'pink', 'teal']

PERSON_COLORS_2D = [
    (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (128, 0, 255), (0, 255, 128),
    (255, 128, 0), (128, 255, 0)
]


def project_3d_to_2d(joints_3d, K, R, t):
    if isinstance(joints_3d, torch.Tensor):
        joints_3d = joints_3d.cpu().numpy()
    pts_cam = (R @ joints_3d.T + t.reshape(3, 1))
    pts_2d_homo = K @ pts_cam
    pts_2d = pts_2d_homo[:2] / (pts_2d_homo[2:3] + 1e-8)
    return pts_2d.T


def draw_skeleton_2d(img, joints_2d, color=(0, 255, 0), thickness=2, radius=3):
    for limb in LIMBS15:
        j1, j2 = limb
        if j1 >= len(joints_2d) or j2 >= len(joints_2d):
            continue
        pt1 = (int(joints_2d[j1, 0]), int(joints_2d[j1, 1]))
        pt2 = (int(joints_2d[j2, 0]), int(joints_2d[j2, 1]))
        h, w = img.shape[:2]
        if any(p < -200 or p > max(h, w) + 200 for p in [pt1[0], pt1[1], pt2[0], pt2[1]]):
            continue
        cv2.line(img, pt1, pt2, color, thickness)
    for j in range(len(joints_2d)):
        pt = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
        h, w = img.shape[:2]
        if pt[0] < -200 or pt[0] > w + 200 or pt[1] < -200 or pt[1] > h + 200:
            continue
        cv2.circle(img, pt, radius, color, -1)
    return img


def create_figure2_visualization(
    images_per_view,          # list of (H, W, 3) images, one per view to show
    camera_params_per_view,   # list of {'K', 'R', 't'} per view
    init_3d_queries,          # (N_queries, num_joints, 3) initial 3D positions
    layer_3d_poses,           # list of (num_filtered, num_joints, 3), one per layer
    layer_scores,             # list of (num_filtered,) confidence, one per layer
    gt_3d_poses,              # (num_gt, num_joints, 3) ground truth
    output_path,
    num_views_to_show=2,
    score_threshold=0.5,
    num_joints=15
):
    """
    Create Figure-2 style visualization:
    - Row 1: View 1 with 2D projections per layer
    - Row 2: View 2 with 2D projections per layer
    - Row 3: 3D skeleton per layer
    - Column 0: Initialization
    - Columns 1-4: Layer 1-4
    """
    num_layers = len(layer_3d_poses)
    num_cols = num_layers + 1  # +1 for initialization
    num_rows = num_views_to_show + 1  # views + 3D
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    
    views_to_show = list(range(min(num_views_to_show, len(images_per_view))))
    
    for col_idx in range(num_cols):
        if col_idx == 0:
            # Initialization column
            title = 'Initialization'
            poses_3d = init_3d_queries  # all queries
            show_all_queries = True
        else:
            layer_idx = col_idx - 1
            title = f'Layer {col_idx}'
            poses_3d = layer_3d_poses[layer_idx]
            show_all_queries = False
        
        # Rows for each view
        for row_idx, v_idx in enumerate(views_to_show):
            ax = axes[row_idx, col_idx]
            img = images_per_view[v_idx].copy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            K = camera_params_per_view[v_idx]['K']
            R = camera_params_per_view[v_idx]['R']
            t = camera_params_per_view[v_idx]['t']
            
            if show_all_queries:
                # Draw a subset of initial queries as blue dots
                step = max(1, len(poses_3d) // 50)  # show ~50 queries
                for q_idx in range(0, len(poses_3d), step):
                    joints = poses_3d[q_idx]
                    pts_2d = project_3d_to_2d(joints, K, R, t)
                    for j in range(len(pts_2d)):
                        pt = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
                        h, w = img.shape[:2]
                        if 0 <= pt[0] < w and 0 <= pt[1] < h:
                            cv2.circle(img, pt, 2, (255, 200, 0), -1)
            else:
                # Draw filtered predictions
                scores = layer_scores[col_idx - 1]
                for p_idx in range(len(poses_3d)):
                    if scores is not None and p_idx < len(scores) and scores[p_idx] < score_threshold:
                        continue
                    color = PERSON_COLORS_2D[p_idx % len(PERSON_COLORS_2D)]
                    joints = poses_3d[p_idx]
                    pts_2d = project_3d_to_2d(joints, K, R, t)
                    img = draw_skeleton_2d(img, pts_2d, color=color, thickness=2, radius=3)
            
            # Draw GT in dashed (approximate with thin red)
            if gt_3d_poses is not None:
                for g_idx in range(len(gt_3d_poses)):
                    pts_2d = project_3d_to_2d(gt_3d_poses[g_idx], K, R, t)
                    img = draw_skeleton_2d(img, pts_2d, color=(0, 0, 255), thickness=1, radius=2)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            if col_idx == 0:
                ax.set_ylabel(f'View {v_idx + 1}', fontsize=13, fontweight='bold')
            if row_idx == 0:
                ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')
        
        # 3D skeleton row
        ax_3d = axes[num_rows - 1, col_idx]
        ax_3d.remove()
        ax_3d = fig.add_subplot(num_rows, num_cols, (num_rows - 1) * num_cols + col_idx + 1,
                               projection='3d')
        
        # Draw GT
        if gt_3d_poses is not None:
            for g_idx in range(len(gt_3d_poses)):
                joints = gt_3d_poses[g_idx]
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                for limb in LIMBS15:
                    j1, j2 = limb
                    if j1 < len(joints) and j2 < len(joints):
                        ax_3d.plot([joints[j1, 0], joints[j2, 0]],
                                  [joints[j1, 1], joints[j2, 1]],
                                  [joints[j1, 2], joints[j2, 2]],
                                  'r-', linewidth=1.5, marker='o', markersize=2,
                                  markerfacecolor='red')
        
        # Draw predictions
        if not show_all_queries:
            scores = layer_scores[col_idx - 1]
            for p_idx in range(len(poses_3d)):
                if scores is not None and p_idx < len(scores) and scores[p_idx] < score_threshold:
                    continue
                joints = poses_3d[p_idx]
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                color = PERSON_COLORS_3D[p_idx % len(PERSON_COLORS_3D)]
                for limb in LIMBS15:
                    j1, j2 = limb
                    if j1 < len(joints) and j2 < len(joints):
                        ax_3d.plot([joints[j1, 0], joints[j2, 0]],
                                  [joints[j1, 1], joints[j2, 1]],
                                  [joints[j1, 2], joints[j2, 2]],
                                  color=color, linewidth=1.5, marker='o', markersize=2,
                                  markerfacecolor='white')
        else:
            # Draw initial queries as scatter
            step = max(1, len(poses_3d) // 30)
            for q_idx in range(0, len(poses_3d), step):
                joints = poses_3d[q_idx]
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                ax_3d.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                            c='blue', s=1, alpha=0.3)
        
        if col_idx == 0:
            ax_3d.set_ylabel('3D', fontsize=13, fontweight='bold')
        if row_idx == 0:
            ax_3d.set_title(title, fontsize=13, fontweight='bold') if col_idx > 0 else None
        ax_3d.set_xlabel('')
        ax_3d.set_xticklabels([])
        ax_3d.set_yticklabels([])
        ax_3d.set_zticklabels([])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 2 visualization saved to {output_path}')
