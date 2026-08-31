import argparse
import socket
import threading
import time
import os
import numpy as np
import torch
import cv2

from model import FeatureExtractor, Matchmaker, CommunicationGate, ResidualCrossAttentionFusion, PoseHead, When2comPoseNet
from net_utils import send_tensor, recv_tensor

# 全域變數，用來讓 Ego (主執行緒) 算完特徵後，分享給 Server (背景執行緒) 傳送給別人
latest_feature_map = None
latest_handshake = None
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
            req = client_sock.recv(13).decode('utf-8')
            with data_lock:
                feat = latest_feature_map
                hand = latest_handshake
            
            if req == "REQ_HANDSHAKE":
                if hand is not None:
                    send_tensor(client_sock, hand)
                else:
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
    """作為 Client 去向別的節點要資料。"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((target_ip, target_port))
        sock.sendall(req_type.encode('utf-8'))
        tensor = recv_tensor(sock)
        sock.close()
        return tensor
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
    parser = argparse.ArgumentParser(description="When2com P2P 分散式節點 (Visibility-Aware + Communication Gate)")
    parser.add_argument("--my-ip", type=str, default="127.0.0.1", help="本機的 IP")
    parser.add_argument("--my-port", type=int, default=5000, help="本機的連接埠")
    parser.add_argument("--peers", type=str, default="", help="其他節點，用逗號分隔")
    parser.add_argument("--cam", type=int, default=0, help="本機要使用的 Webcam ID")
    parser.add_argument("--vis-threshold", type=float, default=0.5, help="可見性門檻 (低於此值的關節不顯示)")
    parser.add_argument("--comm-threshold", type=float, default=0.5, help="通訊閘門門檻 (低於此值則不發送 Handshake)")
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

    # 2. 載入模型組件 (自動載入訓練好的權重)
    print("[*] 正在載入神經網路...")
    net = When2comPoseNet(num_views=3)
    
    # 依序嘗試載入最佳模型或最終模型
    weight_path = None
    for candidate in ["when2com_pose_best.pth", "when2com_pose_final.pth",
                       "who2com_pose_best.pth", "who2com_pose_final.pth"]:
        if os.path.exists(candidate):
            weight_path = candidate
            break
    
    if weight_path:
        print(f"[*] 成功找到並載入訓練好的權重: {weight_path}")
        net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    else:
        print("[!] 找不到權重檔，目前將使用隨機權重進行 Demo！")

    # 提取子模組
    extractor = net.extractor
    matchmaker = net.matchmaker
    comm_gate = net.comm_gate
    fusion = net.fusion
    pose_head = net.pose_head
    
    extractor.eval()
    matchmaker.eval()
    comm_gate.eval()
    fusion.eval()
    pose_head.eval()

    # 開啟攝影機
    cap = cv2.VideoCapture(args.cam)
    
    print(f"[*] 開始執行 When2com 分散式協同推論迴圈！")
    print(f"    可見性門檻: {args.vis_threshold} | 通訊閘門門檻: {args.comm_threshold}")
    
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
            my_handshake = matchmaker.gap(my_feature)

            # 更新全域變數
            global latest_feature_map, latest_handshake
            with data_lock:
                latest_feature_map = my_feature
                latest_handshake = my_handshake

            # 2. When2com 核心：先問閘門「我需不需要幫忙？」
            comm_prob = comm_gate(my_feature)
            need_comm = comm_prob.item() > args.comm_threshold
            
            selected_feature = None
            comm_status = ""
            
            if not need_comm:
                # 閘門說「不需要」→ 完全跳過 Handshake，節省頻寬
                comm_status = f"Gate: {comm_prob.item():.2f} < {args.comm_threshold} | SKIP"
            elif len(peer_list) == 0:
                # 需要通訊但沒有隊友
                comm_status = f"Gate: {comm_prob.item():.2f} | No peers"
            else:
                # 閘門說「需要」→ 才啟動 Handshake + Matchmaker 流程
                peer_handshakes = []
                valid_peers = []
                for ip, port in peer_list:
                    hs = request_data(ip, port, "REQ_HANDSHAKE")
                    if hs is not None:
                        if hs.shape[1] == 512:
                            peer_handshakes.append(hs)
                            valid_peers.append((ip, port))

                if len(peer_handshakes) > 0:
                    msg_ego_flat = my_handshake.reshape(1, -1)
                    concat_self = torch.cat([msg_ego_flat, msg_ego_flat], dim=1)
                    best_score = matchmaker.scorer(concat_self).item()
                    best_peer_idx = -1
                    
                    for i, peer_hs in enumerate(peer_handshakes):
                        peer_hs_flat = peer_hs.reshape(1, -1)
                        concat_peer = torch.cat([msg_ego_flat, peer_hs_flat], dim=1)
                        score = matchmaker.scorer(concat_peer).item()
                        
                        if score > best_score:
                            best_score = score
                            best_peer_idx = i
                    
                    if best_peer_idx == -1:
                        comm_status = f"Gate: {comm_prob.item():.2f} | Matchmaker: self best"
                    else:
                        best_ip, best_port = valid_peers[best_peer_idx]
                        comm_status = f"Gate: {comm_prob.item():.2f} | Fusing with {best_ip}:{best_port}"
                        selected_feature = request_data(best_ip, best_port, "REQ_FEATURE")
                else:
                    comm_status = f"Gate: {comm_prob.item():.2f} | Peers unreachable"

            # 4. 特徵融合 (只有在真正拿到外部特徵時才融合)
            if selected_feature is not None and selected_feature.shape[1] == 512:
                final_feature = fusion(my_feature, selected_feature)
            else:
                final_feature = my_feature

            # 5. 預測 2D 關節點與可見性
            pred_coords, pred_vis = pose_head(final_feature)
            joints_2d = pred_coords[0].cpu().numpy()   # (17, 2)
            visibility = pred_vis[0].cpu().numpy()       # (17,)
            
            h, w, _ = frame.shape
            
            # 只繪製可見性高於門檻的關節
            visible_joints = {}
            for i in range(17):
                if visibility[i] > args.vis_threshold:
                    x = int(joints_2d[i, 0] * w)
                    y = int(joints_2d[i, 1] * h)
                    visible_joints[i] = (x, y)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
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
                
            cv2.imshow(f"When2com AR View {args.my_port}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
