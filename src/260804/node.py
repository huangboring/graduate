import argparse
import socket
import threading
import time
import numpy as np
import torch
import cv2

import os
from model import FeatureExtractor, Matchmaker, CrossAttentionFusion, PoseHead, Who2comPoseNet
from net_utils import send_tensor, recv_tensor

# 全域變數，用來讓 Ego (主執行緒) 算完特徵後，分享給 Server (背景執行緒) 傳送給別人
latest_feature_map = None
latest_handshake = None
data_lock = threading.Lock()

def server_thread(ip, port):
    """
    背景伺服器執行緒：負責接聽其他節點的請求。
    當別人發送 "REQ_HANDSHAKE" 時，回傳本機最新的 handshake。
    當別人發送 "REQ_FEATURE" 時，回傳本機最新的 feature_map。
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(5)
    print(f"[*] 伺服器啟動於 {ip}:{port}，隨時準備幫助其他攝影機...")

    while True:
        client_sock, addr = server.accept()
        try:
            # 讀取請求指令 (長度 13)
            req = client_sock.recv(13).decode('utf-8')
            with data_lock:
                feat = latest_feature_map
                hand = latest_handshake
            
            if req == "REQ_HANDSHAKE":
                if hand is not None:
                    send_tensor(client_sock, hand)
                else:
                    # (512 維度對應 ResNet-18)
                    send_tensor(client_sock, torch.zeros(1, 512, 1, 1))
            elif req == "REQ_FEATURE":
                if feat is not None:
                    send_tensor(client_sock, feat)
                else:
                    send_tensor(client_sock, torch.zeros(1, 512, 8, 8))
        except Exception as e:
            print(f"[!] 伺服器處理錯誤: {e}")
        finally:
            client_sock.close()

def request_data(target_ip, target_port, req_type):
    """
    作為 Client 去向別的節點要資料。
    req_type 必須是 "REQ_HANDSHAKE" 或 "REQ_FEATURE"
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((target_ip, target_port))
        sock.sendall(req_type.encode('utf-8'))
        tensor = recv_tensor(sock)
        sock.close()
        return tensor
    except Exception as e:
        # 連線失敗或 Timeout，回傳 None
        return None

def main():
    parser = argparse.ArgumentParser(description="Who2com P2P 分散式節點")
    parser.add_argument("--my-ip", type=str, default="127.0.0.1", help="本機的 IP")
    parser.add_argument("--my-port", type=int, default=5000, help="本機的連接埠")
    parser.add_argument("--peers", type=str, default="", help="其他節點，用逗號分隔 (留空代表獨立運作)")
    parser.add_argument("--cam", type=int, default=0, help="本機要使用的 Webcam ID")
    args = parser.parse_args()

    # 解析 Peers
    peer_list = []
    if args.peers and args.peers.lower() != "none":
        for p in args.peers.split(','):
            ip, port = p.split(':')
            peer_list.append((ip, int(port)))

    # 1. 啟動背景 Server
    t = threading.Thread(target=server_thread, args=(args.my_ip, args.my_port), daemon=True)
    t.start()

    # 2. 載入模型組件
    print("[*] 正在載入神經網路...")
    
    # 建立完整模型以載入我們在 train.py 訓練並儲存的權重檔
    net = Who2comPoseNet(num_views=3)
    weight_path = "who2com_pose_fast.pth"
    if os.path.exists(weight_path):
        print(f"[*] 成功找到並載入訓練好的權重: {weight_path}")
        net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    else:
        print(f"[!] 找不到權重檔 {weight_path}，目前將使用隨機權重進行 Demo！")

    # 提取子模組供分散式推論使用
    extractor = net.extractor
    matchmaker = net.matchmaker
    fusion = net.fusion
    pose_head = net.pose_head
    
    # 全部設定為 eval 模式 (Demo 不做訓練)
    extractor.eval()
    matchmaker.eval()
    fusion.eval()
    pose_head.eval()

    # 開啟攝影機
    cap = cv2.VideoCapture(args.cam)
    
    # (移除 3D 繪圖視窗)
    
    print("[*] 開始執行分散式協同推論迴圈！")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 影像前處理
            img = cv2.resize(frame, (256, 256))
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            # 1. 自己看畫面 (Ego)
            my_feature = extractor(img_tensor)
            my_handshake = matchmaker.gap(my_feature) # 只需要生成壓縮版的名片 (1, 512, 1, 1)

            # 更新全域變數，讓背景 Server 可以把這些資料發給別人
            global latest_feature_map, latest_handshake
            with data_lock:
                latest_feature_map = my_feature
                latest_handshake = my_handshake

            # 2. 向所有 Peers 索取他們的名片 (Handshake)
            peer_handshakes = []
            valid_peers = []
            for ip, port in peer_list:
                hs = request_data(ip, port, "REQ_HANDSHAKE")
                if hs is not None:
                    # 確保維度是 (1, 512, 1, 1)
                    if hs.shape[1] == 512:
                        peer_handshakes.append(hs)
                        valid_peers.append((ip, port))

            # 3. 挑選最適合的夥伴 (Matchmaking)
            selected_feature = None
            if len(peer_handshakes) > 0:
                # 您的神切入點：Ego 必須先評估「自己」的狀況！
                # 把自己的名片也加入評分網路 (Scorer) 中
                msg_ego_flat = my_handshake.reshape(1, -1)
                concat_self = torch.cat([msg_ego_flat, msg_ego_flat], dim=1)
                best_score = matchmaker.scorer(concat_self).item()
                best_peer_idx = -1 # -1 代表選自己
                
                for i, peer_hs in enumerate(peer_handshakes):
                    peer_hs_flat = peer_hs.reshape(1, -1)
                    concat_peer = torch.cat([msg_ego_flat, peer_hs_flat], dim=1)
                    score = matchmaker.scorer(concat_peer).item()
                    
                    if score > best_score:
                        best_score = score
                        best_peer_idx = i
                
                if best_peer_idx == -1:
                    print(f"[Ego] 我自己看得很清楚 (Score: {best_score:.2f})，不需要麻煩別人，節省頻寬！")
                else:
                    best_ip, best_port = valid_peers[best_peer_idx]
                    print(f"[Ego] 覺得我被遮蔽了，而且 {best_ip}:{best_port} 最有幫助 (Score: {best_score:.2f})，去要特徵圖！")
                    selected_feature = request_data(best_ip, best_port, "REQ_FEATURE")

            # 4. 特徵融合 (Cross-Attention Fusion)
            if selected_feature is not None and selected_feature.shape[1] == 512:
                final_feature = fusion(my_feature, selected_feature)
            else:
                # 如果選了自己，或者大家都斷線，就只用自己的特徵
                final_feature = my_feature

            # 5. 預測 2D 關節點並畫在畫面上 (AR 疊加)
            # 這裡回傳的 pose_2d 會包含 (17, 2) 的座標
            # 為了簡單起見，我們假設網路輸出的是相對於 256x256 的正規化座標 (0~1) 
            # 或是直接相對於原圖的座標。這裡我們將其縮放回原圖解析度。
            pred_pose, _ = pose_head(final_feature) 
            joints_2d = pred_pose[0].cpu().numpy() # (17, 2)
            
            h, w, _ = frame.shape
            
            # 在原圖上畫出關節點
            for i in range(17):
                x = int(joints_2d[i, 0] * w) # 如果輸出不是 0~1，這行可能需要調整
                y = int(joints_2d[i, 1] * h)
                
                # 畫紅色圓點
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                
            # 秀出擴增實境 (AR) 疊加畫面
            cv2.imshow(f"AR Camera View {args.my_port}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
