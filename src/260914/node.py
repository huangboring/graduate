import argparse
import socket
import threading
import time
import json
import struct
import pickle
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model import When2ComPoseNet, soft_argmax_2d, get_confidence
from net_utils import (
    send_tensor, recv_tensor, send_query, recv_query, 
    send_score, recv_score, send_feature_map, recv_feature_map
)

# 影像正規化常數 (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# COCO 17 關節點定義
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
    (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6)
]

# 存放當前節點特徵和 key 的全域變數
global_feat = None
global_key = None
feat_lock = threading.Lock()

def server_thread(my_ip, my_port):
    """
    背景伺服器執行緒：監聽來自其他節點的請求。
    處理三種請求類型：
    - 'Q': 查詢請求，回傳本節點的 key 向量。
    - 'S': 分數請求，接收對方的 query，計算匹配分數並回傳。
    - 'F': 特徵請求，回傳本節點的 feature map。
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((my_ip, my_port))
    server.listen(5)
    print(f"[伺服器] 開始監聽 {my_ip}:{my_port}")

    while True:
        try:
            client, addr = server.accept()
            # 讀取請求類型，假設佔用 1 byte
            req_type = client.recv(1).decode('utf-8')
            
            with feat_lock:
                feat = global_feat.clone() if global_feat is not None else None
                key = global_key.clone() if global_key is not None else None
                
            if req_type == 'Q':
                if key is not None:
                    # 將 key 回傳給對方
                    send_tensor(client, key)
                else:
                    # 如果還沒有準備好，傳送空 tensor
                    send_tensor(client, torch.zeros((1, 1)))
            elif req_type == 'S':
                # 在實際網路中可能由伺服器處理或直接回傳 key
                pass
            elif req_type == 'F':
                if feat is not None:
                    # 回傳特徵圖
                    send_feature_map(client, feat)
                else:
                    send_feature_map(client, torch.zeros((1, 1)))
            
            client.close()
        except Exception as e:
            print(f"[伺服器] 發生錯誤: {e}")

def main():
    parser = argparse.ArgumentParser(description='Real-time Multi-camera Pose Estimation Node')
    parser.add_argument('--my-ip', type=str, default='127.0.0.1', help='本機 IP')
    parser.add_argument('--my-port', type=int, default=5000, help='本機 Port')
    parser.add_argument('--peers', type=str, default='', help='其他節點 (格式: ip:port,ip:port)')
    parser.add_argument('--cam', type=int, default=0, help='攝影機索引')
    parser.add_argument('--model-path', type=str, required=True, help='模型權重路徑')
    parser.add_argument('--conf-threshold', type=float, default=0.2, help='信心度閾值')
    parser.add_argument('--comm-threshold', type=float, default=0.5, help='通訊決策閾值')
    args = parser.parse_args()

    # 解析 peers
    peer_list = []
    if args.peers:
        for p in args.peers.split(','):
            ip, port = p.split(':')
            peer_list.append((ip, int(port)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[節點] 使用設備: {device}")

    # 載入模型
    model = When2ComPoseNet()
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("[節點] 成功載入模型權重")
    except Exception as e:
        print(f"[節點] 警告: 載入模型權重失敗，將使用初始權重 ({e})")
    
    model.to(device)
    model.eval()

    # 啟動背景伺服器
    server_t = threading.Thread(target=server_thread, args=(args.my_ip, args.my_port), daemon=True)
    server_t.start()

    # 開啟攝影機
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[節點] 錯誤: 無法開啟攝影機 {args.cam}")
        return

    # 影像前處理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    print("[節點] 開始進行推論...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        orig_h, orig_w = frame.shape[:2]

        # 影像前處理
        img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        with torch.no_grad():
            # 1. 特徵擷取
            feat = model.extractor(img_tensor) # (1, 512, 8, 8)
            
            # 2. 生成 query 和 key
            query, key = model.qk_encoder(feat)
            
            # 更新全域變數，供背景伺服器使用
            global global_feat, global_key
            with feat_lock:
                global_feat = feat.cpu()
                global_key = key.cpu()

            final_feat = feat
            
            # 當有其他節點時，執行 When2Com 3-stage Handshake
            if peer_list:
                peer_keys = []
                # Stage 1: Request keys from peers
                for p_ip, p_port in peer_list:
                    try:
                        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client.connect((p_ip, p_port))
                        client.send(b'Q')
                        p_key = recv_tensor(client).to(device)
                        client.close()
                        if p_key.numel() > 1:
                            peer_keys.append(p_key)
                    except Exception:
                        pass
                
                # Stage 2: Match — 計算 Query-Key 匹配分數
                if peer_keys:
                    all_keys = [key] + peer_keys  # ego key + peer keys
                    V = len(all_keys)

                    # 組裝成 (1, V, d_q) 和 (1, V, d_k)
                    # query 會重複 V 次（每個 agent 都有自己的 query，這裡簡化為 ego query 重複）
                    queries_tensor = query.unsqueeze(0).expand(1, V, -1).clone()  # (1, V, d_q)
                    queries_tensor[:, 0, :] = query  # ego 的 query
                    keys_tensor = torch.stack(all_keys, dim=0).unsqueeze(0)  # (1, V, d_k)

                    # Scaled General Attention 匹配
                    raw_scores = model.sga(queries_tensor, keys_tensor)  # (1, V, V)
                    norm_weights, comm_decisions = model.gate(raw_scores)  # (1, V, V), (1, V)

                    ego_needs_comm = comm_decisions[0, 0].item()  # ego 是否需要通訊
                    ego_view_weights = norm_weights[0, 0, :]  # (V,) ego 的視角權重

                    # Stage 3: Connect — 從有價值的 peer 請求 feature map
                    if ego_needs_comm:
                        peer_feats = []
                        for idx, (p_ip, p_port) in enumerate(peer_list):
                            peer_idx = idx + 1  # peer 在 all_keys 中的 index
                            if ego_view_weights[peer_idx].item() > 0:  # 只找被 gate 保留的 peer
                                try:
                                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    client.settimeout(3.0)
                                    client.connect((p_ip, p_port))
                                    client.send(b'F')
                                    p_feat = recv_feature_map(client).to(device)
                                    client.close()
                                    if p_feat.numel() > 1:
                                        peer_feats.append(p_feat.unsqueeze(0))  # (1, 512, 8, 8)
                                except Exception:
                                    pass

                        if peer_feats:
                            # 組裝 all_feats: (1, V_active, 512, 8, 8)
                            all_feats = torch.cat([feat.unsqueeze(1)] + [pf.unsqueeze(1) for pf in peer_feats], dim=1)
                            V_active = all_feats.shape[1]
                            peer_mask = torch.ones(1, V_active, dtype=torch.bool, device=device)
                            # 取對應的 view_weights (ego + active peers)
                            active_weights = ego_view_weights[:V_active].unsqueeze(0)  # (1, V_active)
                            final_feat = model.cross_view_fusion(
                                feat, all_feats, peer_mask, view_weights=active_weights
                            )
            
            # 3. 產生熱力圖並解析關節點
            hm = model.decoder(final_feat)   # (1, 17, 64, 64)
            coords = soft_argmax_2d(hm)      # (1, 17, 2) 座標在 [0, 1]
            conf = get_confidence(hm)         # (1, 17)

            coords = coords[0].cpu().numpy()  # (17, 2)
            conf = conf[0].cpu().numpy()      # (17,)

        # 繪製關節點 (coords 在 [0, 1]，乘以原始解析度)
        visible_joints = {}
        for i in range(17):
            if conf[i] > args.conf_threshold:
                px = int(coords[i, 0] * orig_w)
                py = int(coords[i, 1] * orig_h)
                visible_joints[i] = (px, py)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"{conf[i]:.2f}", (px + 8, py - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # 繪製骨架
        for (idx1, idx2) in SKELETON:
            if idx1 in visible_joints and idx2 in visible_joints:
                cv2.line(frame, visible_joints[idx1], visible_joints[idx2], (0, 255, 255), 2)

        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
