import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import Who2comPoseNet

def get_3d_plot_image(pose_3d):
    """
    將 3D 關節點 (17, 3) 繪製成 Matplotlib 圖片並轉為 OpenCV 格式
    """
    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    
    # 取出 X, Y, Z 座標
    xs = pose_3d[:, 0]
    ys = pose_3d[:, 1]
    zs = pose_3d[:, 2]
    
    # 畫點 (若是真實資料，這裡可以加入骨架連線)
    ax.scatter(xs, ys, zs, c='red', marker='o', s=40)
    
    # 設定視角與標籤
    ax.set_title("Predicted 3D Pose", fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 將 Matplotlib 圖片轉換為 numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # 將 RGB 轉回 OpenCV 的 BGR
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def deploy_webcams(model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"部署環境使用裝置: {device}")
    
    num_views = 3 # 0: Ego, 1: View1, 2: View2
    caps = []
    for i in range(num_views):
        cap = cv2.VideoCapture(i)
        caps.append(cap)
        if not cap.isOpened():
            print(f"警告: 無法開啟 Webcam {i}，將以黑畫面代替。")

    # 載入模型
    model = Who2comPoseNet(num_views=num_views).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    print("開始展示 (Demo Mode)。請在跳出的視窗按下 'q' 鍵退出。")
    
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
                
        # 準備送入模型的 Tensor
        input_tensors = []
        for f in frames:
            rgb_f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            input_tensors.append(transform(rgb_f))
        batch_input = torch.stack(input_tensors).unsqueeze(0).to(device)
        
        # 進行推論
        with torch.no_grad():
            pred_pose, weights = model(batch_input, temperature=0.01)
            
        selected_idx = weights[0].argmax().item() + 1
        
        # === 視覺化處理 (Demo UI) ===
        display_frames = []
        for i, f in enumerate(frames):
            f_resized = cv2.resize(f, (320, 240))
            
            # 標示 Ego 視角
            if i == 0:
                cv2.putText(f_resized, "Ego View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(f_resized, f"Agent {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 如果是被挑選到的視角，畫上醒目的綠色外框與文字
            if i == selected_idx:
                cv2.rectangle(f_resized, (5, 5), (315, 235), (0, 255, 0), 4)
                cv2.putText(f_resized, "SELECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            display_frames.append(f_resized)
            
        # 拼接三個 Webcam 畫面 (水平)
        webcam_concat = np.hstack(display_frames)
        
        # 繪製預測出來的 3D 關節點 (取得 numpy 圖片)
        pose_numpy = pred_pose[0].cpu().numpy()
        plot_img = get_3d_plot_image(pose_numpy)
        plot_img_resized = cv2.resize(plot_img, (320, 320))
        
        # 將 Webcam 畫面與 3D 骨架畫面拼裝展示
        cv2.imshow("Who2com Multi-view Demo", webcam_concat)
        cv2.imshow("3D Pose Output", plot_img_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    deploy_webcams()
