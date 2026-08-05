import argparse
import socket
import threading
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from model import FeatureExtractor, Matchmaker, CrossAttentionFusion, PoseHead
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
                    # 如果還沒準備好，傳送一個全為零的 Dummy Tensor
                    send_tensor(client_sock, torch.zeros(1, 2048, 1, 1))
            elif req == "REQ_FEATURE":
                if feat is not None:
                    send_tensor(client_sock, feat)
                else:
                    send_tensor(client_sock, torch.zeros(1, 2048, 8, 8))
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
    parser.add_argument("--peers", type=str, default="127.0.0.1:5001", help="其他節點，用逗號分隔")
    parser.add_argument("--cam", type=int, default=0, help="本機要使用的 Webcam ID")
    args = parser.parse_args()

    # 解析 Peers
    peer_list = []
    if args.peers:
        for p in args.peers.split(','):
            ip, port = p.split(':')
            peer_list.append((ip, int(port)))

    # 1. 啟動背景 Server
    t = threading.Thread(target=server_thread, args=(args.my_ip, args.my_port), daemon=True)
    t.start()

    # 2. 載入模型組件
    print("[*] 正在載入神經網路...")
    extractor = FeatureExtractor()
    matchmaker = Matchmaker(feature_dim=512)
    fusion = CrossAttentionFusion(feature_dim=512)
    pose_head = PoseHead(feature_dim=512, num_joints=17)
    
    # 全部設定為 eval 模式 (Demo 不做訓練)
    extractor.eval()
    matchmaker.eval()
    fusion.eval()
    pose_head.eval()

    # 開啟攝影機
    cap = cv2.VideoCapture(args.cam)
    
    # 準備 3D 繪圖視窗
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
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
            my_handshake = matchmaker(my_feature) # (1, 512, 1, 1)

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
                # 這裡為了簡單示範，我們隨機挑選或直接用內積算分數
                # 論文中是用 Attention，為了 Demo 穩定，我們用 cosine similarity
                best_score = -999
                best_peer_idx = -1
                for i, peer_hs in enumerate(peer_handshakes):
                    score = torch.sum(my_handshake * peer_hs).item()
                    if score > best_score:
                        best_score = score
                        best_peer_idx = i
                
                # 去向最高分的夥伴索取完整特徵圖
                best_ip, best_port = valid_peers[best_peer_idx]
                print(f"[Ego] 覺得 {best_ip}:{best_port} 最有幫助 (Score: {best_score:.2f})，去要特徵圖！")
                selected_feature = request_data(best_ip, best_port, "REQ_FEATURE")

            # 4. 特徵融合 (Cross-Attention Fusion)
            if selected_feature is not None and selected_feature.shape[1] == 512:
                final_feature = fusion(my_feature, selected_feature)
            else:
                # 如果大家都斷線，或者沒有人有用的資訊，只好自己硬算
                final_feature = my_feature
                print("[Ego] 沒有取得其他人的特徵，只能靠自己單打獨鬥！")

            # 5. 預測 3D 關節點並畫圖
            pred_pose = pose_head(final_feature)
            joints_3d = pred_pose[0].numpy()
            
            # 畫骨架
            ax.clear()
            ax.scatter(joints_3d[:, 0], joints_3d[:, 2], joints_3d[:, 1], c='red', s=50)
            ax.set_title(f"Node {args.my_port} - 3D Pose")
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 2)
            ax.set_zlim(-1, 1)
            plt.draw()
            plt.pause(0.01)

            # 秀出攝影機畫面
            cv2.imshow(f"Camera View {args.my_port}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
