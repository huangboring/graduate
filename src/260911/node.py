import argparse
import socket
import threading
import time
import os
import numpy as np
import torch
import cv2

from model import When2comHeatmapNet, soft_argmax_2d, get_confidence
from net_utils import send_tensor, recv_tensor, send_compressed_heatmap, recv_compressed_heatmap

# 全域變數，用來讓 Ego (主執行緒) 算完特徵後，分享給 Server (背景執行緒) 傳送給別人
latest_heatmap = None
latest_coords = None
latest_conf = None
data_lock = threading.Lock()

def server_thread(ip, port):
    """背景伺服器執行緒：負責接聽其他節點的請求。"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(5)
    print(f"[*] 伺服器啟動於 {ip}:{port}，隨時準備幫助其他攝影機...")

    while True:
        client_sock, addr = server.accept()
        try:
            req_type = client_sock.recv(1).decode('utf-8')
            with data_lock:
                hm = latest_heatmap
                coords = latest_coords
                conf = latest_conf
            
            if req_type == 'H':  # Handshake Request → 傳送壓縮摘要
                if coords is not None and conf is not None:
                    send_compressed_heatmap(client_sock, coords, conf)
                else:
                    send_compressed_heatmap(client_sock, torch.zeros(17, 2), torch.zeros(17))
            elif req_type == 'F':  # Feature Request → 傳送完整 Heatmap
                if hm is not None:
                    send_tensor(client_sock, hm)
                else:
                    send_tensor(client_sock, torch.zeros(17, 64, 64))
        except Exception as e:
            print(f"[!] 伺服器處理錯誤: {e}")
        finally:
            client_sock.close()

def request_handshake(target_ip, target_port):
    """向其他節點請求壓縮 Heatmap 摘要"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((target_ip, target_port))
        sock.sendall(b'H')
        coords, conf = recv_compressed_heatmap(sock)
        sock.close()
        return coords, conf
    except Exception as e:
        return None, None

def request_heatmap(target_ip, target_port):
    """向其他節點請求完整 Heatmap"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect((target_ip, target_port))
        sock.sendall(b'F')
        hm = recv_tensor(sock)
        sock.close()
        return hm
    except Exception as e:
        return None

# COCO 17 關節點的骨架連接定義
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 頭部
    (5, 6),                                 # 肩膀
    (5, 7), (7, 9), (6, 8), (8, 10),       # 手臂
    (5, 11), (6, 12),                       # 軀幹
    (11, 12),                               # 臀部
    (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
]

def main():
    parser = argparse.ArgumentParser(description="When2com Heatmap P2P 分散式節點")
    parser.add_argument("--my-ip", type=str, default="0.0.0.0", help="本機的 IP")
    parser.add_argument("--my-port", type=int, default=5000, help="本機的連接埠")
    parser.add_argument("--peers", type=str, default="", help="其他節點，用逗號分隔 (格式: ip:port)")
    parser.add_argument("--cam", type=int, default=0, help="本機要使用的 Webcam ID")
    parser.add_argument("--model-path", type=str, default="", help="模型權重檔路徑")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="信心門檻 (低於此值的關節不顯示)")
    parser.add_argument("--comm-threshold", type=float, default=0.5, help="通訊閘門門檻")
    args = parser.parse_args()

    # 解析 Peers
    peer_list = []
    if args.peers and args.peers.lower() != "none":
        for p in args.peers.split(','):
            parts = p.strip().split(':')
            if len(parts) == 2:
                peer_list.append((parts[0], int(parts[1])))

    # 1. 啟動背景 Server
    t = threading.Thread(target=server_thread, args=(args.my_ip, args.my_port), daemon=True)
    t.start()

    # 2. 載入模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] 正在載入神經網路... (device: {device})")
    model = When2comHeatmapNet().to(device)
    
    # 自動搜尋權重檔
    weight_path = None
    if args.model_path and os.path.exists(args.model_path):
        weight_path = args.model_path
    else:
        for candidate in ["heatmap_stage2_latest.pth", "heatmap_stage1_latest.pth"]:
            if os.path.exists(candidate):
                weight_path = candidate
                break
    
    if weight_path:
        print(f"[*] 成功找到並載入訓練好的權重: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    else:
        print("[!] 找不到權重檔，目前將使用隨機權重進行 Demo！")

    model.eval()

    # 開啟攝影機
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[!] 無法開啟攝影機 {args.cam}")
        return
    
    print(f"[*] 開始執行 When2com Heatmap 分散式協同推論迴圈！")
    print(f"    信心門檻: {args.conf_threshold} | 通訊閘門門檻: {args.comm_threshold}")
    print(f"    按 'q' 離開")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 影像前處理
            img = cv2.resize(frame, (256, 256))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # 1. 提取特徵 + 生成 Heatmap
            my_feature = model.extractor(img_tensor)
            my_heatmap = model.decoder(my_feature)  # (1, 17, 64, 64)
            
            my_coords = soft_argmax_2d(my_heatmap)  # (1, 17, 2)
            my_conf = get_confidence(my_heatmap)     # (1, 17)

            # 更新全域變數供其他節點請求
            global latest_heatmap, latest_coords, latest_conf
            with data_lock:
                latest_heatmap = my_heatmap.squeeze(0).cpu()
                latest_coords = my_coords.squeeze(0).cpu()
                latest_conf = my_conf.squeeze(0).cpu()

            # 2. When2com 核心：先問 EntropyGate「我需不需要幫忙？」
            comm_prob = model.gate(my_heatmap).item()
            need_comm = comm_prob > args.comm_threshold
            
            final_heatmap = my_heatmap
            comm_status = ""
            
            if not need_comm:
                comm_status = f"Gate: {comm_prob:.2f} < {args.comm_threshold} | SKIP"
            elif len(peer_list) == 0:
                comm_status = f"Gate: {comm_prob:.2f} | No peers"
            else:
                # 閘門說「需要」→ 啟動 Handshake + Matchmaker 流程
                peer_handshakes = []
                valid_peers = []
                for ip, port in peer_list:
                    p_coords, p_conf = request_handshake(ip, port)
                    if p_coords is not None:
                        peer_handshakes.append((p_coords, p_conf))
                        valid_peers.append((ip, port))

                if len(peer_handshakes) > 0:
                    # 用 Matchmaker 評分
                    best_score = float('-inf')
                    best_peer_idx = -1
                    
                    ego_coords_gpu = my_coords.to(device)  # (1, 17, 2)
                    ego_conf_gpu = my_conf.to(device)       # (1, 17)
                    
                    for i, (p_coords, p_conf) in enumerate(peer_handshakes):
                        p_coords_gpu = p_coords.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 17, 2)
                        p_conf_gpu = p_conf.unsqueeze(0).unsqueeze(0).to(device)      # (1, 1, 17)
                        
                        score = model.matchmaker(ego_coords_gpu, ego_conf_gpu, p_coords_gpu, p_conf_gpu).item()
                        
                        if score > best_score:
                            best_score = score
                            best_peer_idx = i
                    
                    if best_peer_idx >= 0:
                        best_ip, best_port = valid_peers[best_peer_idx]
                        comm_status = f"Gate: {comm_prob:.2f} | Fusing with {best_ip}:{best_port}"
                        
                        # 請求完整 Heatmap 並融合
                        peer_hm = request_heatmap(best_ip, best_port)
                        if peer_hm is not None:
                            peer_hm = peer_hm.unsqueeze(0).to(device)  # (1, 17, 64, 64)
                            final_heatmap = model.fusion(my_heatmap, peer_hm)
                        else:
                            comm_status = f"Gate: {comm_prob:.2f} | Peer unreachable for feature"
                    else:
                        comm_status = f"Gate: {comm_prob:.2f} | Matchmaker: no good peer"
                else:
                    comm_status = f"Gate: {comm_prob:.2f} | Peers unreachable"

            # 3. 最終預測
            pred_coords = soft_argmax_2d(final_heatmap)  # (1, 17, 2)
            pred_conf = get_confidence(final_heatmap)     # (1, 17)
            
            joints_2d = pred_coords[0].cpu().numpy()   # (17, 2)
            confidence = pred_conf[0].cpu().numpy()     # (17,)
            
            h, w, _ = frame.shape
            
            # 只繪製信心高於門檻的關節
            visible_joints = {}
            for i in range(17):
                if confidence[i] > args.conf_threshold:
                    x = int(joints_2d[i, 0] * w)
                    y = int(joints_2d[i, 1] * h)
                    visible_joints[i] = (x, y)
                    # 畫圓點
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    # 顯示信心分數
                    cv2.putText(frame, f"{confidence[i]:.2f}", (x + 8, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            # 繪製骨架連線
            for (a, b) in SKELETON:
                if a in visible_joints and b in visible_joints:
                    cv2.line(frame, visible_joints[a], visible_joints[b], (0, 255, 255), 2)
            
            # 在左上角顯示狀態資訊
            num_visible = len(visible_joints)
            cv2.putText(frame, f"Visible: {num_visible}/17", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, comm_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Device: {device}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                
            cv2.imshow(f"When2com Heatmap AR View (Port {args.my_port})", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
