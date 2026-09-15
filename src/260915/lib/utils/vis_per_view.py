import numpy as np
import cv2
import os
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Panoptic 15-joint skeleton
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
           [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

JOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255)
]

PERSON_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (255, 0, 128),
    (128, 255, 0), (0, 128, 255)
]


def project_3d_to_2d(joints_3d, camera_params):
    """
    Project 3D joints to 2D image coordinates.
    joints_3d: (num_joints, 3) in world coordinates
    camera_params: dict with 'K' (3x3), 'R' (3x3), 't' (3x1)
    Returns: (num_joints, 2) pixel coordinates
    """
    K = camera_params['K']
    R = camera_params['R']
    t = camera_params['t']
    
    # World to camera
    if isinstance(joints_3d, torch.Tensor):
        joints_3d = joints_3d.cpu().numpy()
    
    pts_cam = (R @ joints_3d.T + t.reshape(3, 1))  # (3, N)
    
    # Camera to pixel
    pts_2d_homo = K @ pts_cam  # (3, N)
    pts_2d = pts_2d_homo[:2] / (pts_2d_homo[2:3] + 1e-8)  # (2, N)
    
    return pts_2d.T  # (N, 2)


def draw_skeleton_on_image(image, joints_2d, confidence=None, person_color=(0, 255, 0),
                            limbs=LIMBS15, thickness=2, radius=4):
    """
    Draw skeleton on an image.
    image: (H, W, 3) BGR
    joints_2d: (num_joints, 2)
    confidence: (num_joints,) optional confidence scores
    """
    img = image.copy()
    num_joints = joints_2d.shape[0]
    
    # Draw limbs
    for limb in limbs:
        j1, j2 = limb
        if j1 >= num_joints or j2 >= num_joints:
            continue
        pt1 = (int(joints_2d[j1, 0]), int(joints_2d[j1, 1]))
        pt2 = (int(joints_2d[j2, 0]), int(joints_2d[j2, 1]))
        
        # Skip if out of image bounds
        if any(p < -100 or p > max(img.shape[:2]) + 100 for p in [pt1[0], pt1[1], pt2[0], pt2[1]]):
            continue
        
        cv2.line(img, pt1, pt2, person_color, thickness)
    
    # Draw joints
    for j in range(num_joints):
        pt = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
        if pt[0] < -100 or pt[0] > img.shape[1] + 100 or pt[1] < -100 or pt[1] > img.shape[0] + 100:
            continue
        
        color = JOINT_COLORS[j % len(JOINT_COLORS)]
        cv2.circle(img, pt, radius, color, -1)
        
        if confidence is not None and j < len(confidence):
            conf_text = f'{confidence[j]:.2f}'
            cv2.putText(img, conf_text, (pt[0] + 5, pt[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    return img


def visualize_multiview_2d(images, pred_3d_poses, camera_params_list, output_path,
                           scores=None, gt_3d_poses=None):
    """
    Visualize 2D projections on each camera view.
    images: list of (H, W, 3) BGR images, one per view
    pred_3d_poses: (num_people, num_joints, 3) predicted 3D poses
    camera_params_list: list of camera param dicts, one per view
    output_path: where to save the visualization
    scores: (num_people,) optional confidence scores
    gt_3d_poses: (num_people, num_joints, 3) optional ground truth
    """
    num_views = len(images)
    num_people = pred_3d_poses.shape[0] if pred_3d_poses is not None else 0
    
    fig, axes = plt.subplots(1, num_views, figsize=(6 * num_views, 5))
    if num_views == 1:
        axes = [axes]
    
    for v_idx in range(num_views):
        img = images[v_idx].copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Draw GT in red
        if gt_3d_poses is not None:
            for p_idx in range(gt_3d_poses.shape[0]):
                joints_2d = project_3d_to_2d(gt_3d_poses[p_idx], camera_params_list[v_idx])
                img = draw_skeleton_on_image(img, joints_2d, person_color=(0, 0, 255),
                                           thickness=1, radius=3)
        
        # Draw predictions in different colors per person
        if pred_3d_poses is not None:
            for p_idx in range(num_people):
                color = PERSON_COLORS[p_idx % len(PERSON_COLORS)]
                joints_2d = project_3d_to_2d(pred_3d_poses[p_idx], camera_params_list[v_idx])
                img = draw_skeleton_on_image(img, joints_2d, person_color=color,
                                           thickness=2, radius=4)
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[v_idx].imshow(img_rgb)
        axes[v_idx].set_title(f'View {v_idx + 1}', fontsize=14)
        axes[v_idx].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_3d_skeleton(pred_3d_poses, output_path, gt_3d_poses=None):
    """
    Visualize 3D skeletons in a 3D plot.
    pred_3d_poses: (num_people, num_joints, 3)
    gt_3d_poses: (num_people, num_joints, 3) optional
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw GT in red
    if gt_3d_poses is not None:
        for p_idx in range(gt_3d_poses.shape[0]):
            joints = gt_3d_poses[p_idx]
            for limb in LIMBS15:
                j1, j2 = limb
                if j1 < len(joints) and j2 < len(joints):
                    ax.plot([joints[j1, 0], joints[j2, 0]],
                           [joints[j1, 1], joints[j2, 1]],
                           [joints[j1, 2], joints[j2, 2]],
                           'r-', linewidth=2, marker='o', markersize=3,
                           markerfacecolor='red')
    
    # Draw predictions
    colors = ['blue', 'green', 'cyan', 'magenta', 'yellow',
              'orange', 'purple', 'lime', 'pink', 'teal']
    if pred_3d_poses is not None:
        for p_idx in range(pred_3d_poses.shape[0]):
            joints = pred_3d_poses[p_idx]
            color = colors[p_idx % len(colors)]
            for limb in LIMBS15:
                j1, j2 = limb
                if j1 < len(joints) and j2 < len(joints):
                    ax.plot([joints[j1, 0], joints[j2, 0]],
                           [joints[j1, 1], joints[j2, 1]],
                           [joints[j1, 2], joints[j2, 2]],
                           color=color, linewidth=2, marker='o', markersize=3,
                           markerfacecolor='white')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Estimation', fontsize=14)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
