import socket
import threading
import torch
import numpy as np
from model import When2comHeatmapNet
from net_utils import send_tensor, recv_tensor, send_compressed_heatmap, recv_compressed_heatmap
import time

class PeerNode:
    def __init__(self, node_id, port, peers, model_path=None, device='cuda'):
        """
        分散式推論節點
        peers: [(peer_id, ip, port), ...]
        """
        self.node_id = node_id
        self.port = port
        self.peers = peers
        self.device = device
        
        self.model = When2comHeatmapNet().to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(('0.0.0.0', port))
        self.server_sock.listen(5)
        
        self.is_running = True
        
    def start_server(self):
        """
        啟動背景 Server 監聽來自其他節點的請求
        """
        def listen_loop():
            while self.is_running:
                try:
                    conn, addr = self.server_sock.accept()
                    # 每收到一個連線，開一個 thread 去處理
                    client_thread = threading.Thread(target=self.handle_client, args=(conn,))
                    client_thread.start()
                except:
                    if not self.is_running:
                        break
        
        self.server_thread = threading.Thread(target=listen_loop)
        self.server_thread.start()
        print(f"[Node {self.node_id}] Server started on port {self.port}")
        
    def handle_client(self, conn):
        """
        處理其他節點的請求。
        1. 收到 Handshake 請求 -> 傳送自己的壓縮 Heatmap
        2. 收到特徵請求 -> 傳送完整的 Heatmap
        """
        try:
            req_type = conn.recv(1).decode('utf-8')
            if req_type == 'H': # Handshake Request
                # 假設 current_frame_data 存在
                # 這裡需要一個機制知道當前在推論哪一幀，為簡化，假設回傳快取
                if hasattr(self, 'current_coords'):
                    send_compressed_heatmap(conn, self.current_coords, self.current_conf)
                else:
                    # dummy
                    send_compressed_heatmap(conn, torch.zeros(17, 2), torch.zeros(17))
                    
            elif req_type == 'F': # Feature Request
                if hasattr(self, 'current_hm'):
                    send_tensor(conn, self.current_hm)
                else:
                    send_tensor(conn, torch.zeros(17, 64, 64))
        except Exception as e:
            print(f"[Node {self.node_id}] Client handle error: {e}")
        finally:
            conn.close()

    def process_frame(self, image_tensor):
        """
        處理單幀影像 (Ego View)
        image_tensor: (1, 3, 256, 256)
        """
        with torch.no_grad():
            img = image_tensor.to(self.device)
            
            # 1. Feature Extraction & Heatmap Decoding
            feat = self.model.extractor(img)
            ego_hm = self.model.decoder(feat) # (1, 17, 64, 64)
            
            # 快取供別人請求
            self.current_hm = ego_hm.squeeze(0).cpu()
            self.current_coords = self.model.decoder.soft_argmax_2d(ego_hm).squeeze(0).cpu()
            self.current_conf = self.model.decoder.get_confidence(ego_hm).squeeze(0).cpu()
            
            # 2. Gate Decision
            comm_prob = self.model.gate(ego_hm).item()
            
            if comm_prob > 0.5:
                print(f"[Node {self.node_id}] Decision: COMMUNICATE (Prob: {comm_prob:.2f})")
                
                # 3. Handshake phase
                best_peer = None
                best_score = float('-inf')
                
                for peer_id, ip, port in self.peers:
                    try:
                        # 向 Peer 請求壓縮摘要
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect((ip, port))
                        sock.sendall(b'H')
                        peer_coords, peer_conf = recv_compressed_heatmap(sock)
                        sock.close()
                        
                        # Matchmaker 評分
                        # 注意 Matchmaker 預期 batch format
                        peer_coords = peer_coords.unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, 17, 2)
                        peer_conf = peer_conf.unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, 17)
                        
                        ego_coords_gpu = self.current_coords.unsqueeze(0).to(self.device)
                        ego_conf_gpu = self.current_conf.unsqueeze(0).to(self.device)
                        
                        score = self.model.matchmaker(ego_coords_gpu, ego_conf_gpu, peer_coords, peer_conf).item()
                        print(f"  -> Peer {peer_id} score: {score:.2f}")
                        
                        if score > best_score:
                            best_score = score
                            best_peer = (peer_id, ip, port)
                    except Exception as e:
                        print(f"  -> Failed to handshake with peer {peer_id}: {e}")
                
                # 4. Request full heatmap and fuse
                if best_peer:
                    print(f"[Node {self.node_id}] Selected Peer {best_peer[0]} for fusion.")
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect((best_peer[1], best_peer[2]))
                        sock.sendall(b'F')
                        peer_hm = recv_tensor(sock).unsqueeze(0).to(self.device) # (1, 17, 64, 64)
                        sock.close()
                        
                        # Fusion
                        fused_hm = self.model.fusion(ego_hm, peer_hm)
                        ego_hm = fused_hm
                    except Exception as e:
                        print(f"[Node {self.node_id}] Failed to get feature from peer {best_peer[0]}: {e}")
            else:
                print(f"[Node {self.node_id}] Decision: NO COMMUNICATE (Prob: {comm_prob:.2f})")
                
            # 5. Final Prediction
            final_coords = self.model.decoder.soft_argmax_2d(ego_hm)
            return final_coords

    def stop(self):
        self.is_running = False
        # Connect to self to unblock accept
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', self.port))
            sock.close()
        except:
            pass
        self.server_thread.join()
        self.server_sock.close()
        print(f"[Node {self.node_id}] Server stopped.")

if __name__ == "__main__":
    # Test script: Launch 2 nodes
    node1 = PeerNode(1, 5001, peers=[(2, '127.0.0.1', 5002)])
    node2 = PeerNode(2, 5002, peers=[(1, '127.0.0.1', 5001)])
    
    node1.start_server()
    node2.start_server()
    
    time.sleep(1) # Wait for servers to start
    
    # Create dummy images
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    
    # Process asynchronously to simulate real-world
    def run_node1():
        coords = node1.process_frame(img1)
        print(f"Node 1 finished. Coords shape: {coords.shape}")
        
    def run_node2():
        coords = node2.process_frame(img2)
        print(f"Node 2 finished. Coords shape: {coords.shape}")
        
    t1 = threading.Thread(target=run_node1)
    t2 = threading.Thread(target=run_node2)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    node1.stop()
    node2.stop()
