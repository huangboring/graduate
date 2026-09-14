import socket
import struct
import io
import torch


def send_tensor(sock, tensor):
    """
    透過 socket 傳送 PyTorch Tensor。
    """
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    buffer.seek(0)
    data = buffer.read()

    # 送出資料大小 (4 bytes)
    size_bytes = struct.pack("!I", len(data))
    sock.sendall(size_bytes)
    # 送出資料
    sock.sendall(data)


def recv_tensor(sock):
    """
    透過 socket 接收 PyTorch Tensor。
    """
    # 接收資料大小 (4 bytes)
    size_bytes = b""
    while len(size_bytes) < 4:
        chunk = sock.recv(4 - len(size_bytes))
        if not chunk:
            raise ConnectionError("Socket 在接收大小之前關閉。")
        size_bytes += chunk

    size = struct.unpack("!I", size_bytes)[0]

    # 接收資料
    data = b""
    while len(data) < size:
        chunk = sock.recv(min(4096, size - len(data)))
        if not chunk:
            raise ConnectionError("Socket 在接收完整資料前關閉。")
        data += chunk

    # 解開 Tensor
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, map_location='cpu', weights_only=True)
    return tensor


# ============================================================
# When2Com Handshake Stage 1: Query 傳輸（極低頻寬）
# ============================================================

def send_query(sock, query_tensor):
    """
    傳送壓縮的 Query 向量 (d_q ≈ 32 個浮點數)。
    對應 When2Com Handshake Stage 1: Request。
    頻寬成本：極低 (~128 bytes)
    """
    send_tensor(sock, query_tensor)


def recv_query(sock):
    """
    接收壓縮的 Query 向量。
    """
    return recv_tensor(sock)


# ============================================================
# When2Com Handshake Stage 2: Matching Score 傳輸（幾乎為零）
# ============================================================

def send_score(sock, score_float):
    """
    傳送標量 matching score (1 個浮點數)。
    對應 When2Com Handshake Stage 2: Match Score。
    頻寬成本：4 bytes，幾乎為零。
    """
    sock.sendall(struct.pack("!f", score_float))


def recv_score(sock):
    """
    接收標量 matching score。
    回傳: float
    """
    data = b""
    while len(data) < 4:
        chunk = sock.recv(4 - len(data))
        if not chunk:
            raise ConnectionError("Socket 在接收 score 前關閉。")
        data += chunk
    return struct.unpack("!f", data)[0]


# ============================================================
# When2Com Handshake Stage 2: Key 傳輸
# ============================================================

def send_key(sock, key_tensor):
    """
    傳送 Key 向量 (d_k ≈ 128 個浮點數)。
    Key 通常留在本地，只在被請求時傳送。
    """
    send_tensor(sock, key_tensor)


def recv_key(sock):
    """
    接收 Key 向量。
    """
    return recv_tensor(sock)


# ============================================================
# When2Com Handshake Stage 3: Feature Map 傳輸（較高頻寬）
# ============================================================

def send_feature_map(sock, feat_tensor):
    """
    傳送完整 feature map (512, 8, 8)。
    對應 When2Com Handshake Stage 3: Connect。
    頻寬成本：較高 (~131 KB)，但只在 Gate 開啟且配對成功時才傳送。
    """
    send_tensor(sock, feat_tensor)


def recv_feature_map(sock):
    """
    接收完整 feature map。
    """
    return recv_tensor(sock)


# ============================================================
# 向下相容：壓縮 Heatmap 傳輸（保留給舊版 node.py）
# ============================================================

def send_compressed_heatmap(sock, coords, confidence):
    """
    優化頻寬: 傳送壓縮後的 2D 座標和信心分數 (共 17 * 3 = 51 個浮點數)。
    """
    compressed = torch.cat([coords, confidence.unsqueeze(-1)], dim=-1)  # (17, 3)
    send_tensor(sock, compressed)


def recv_compressed_heatmap(sock):
    """
    接收壓縮後的 2D 座標和信心分數。
    回傳 coords (17, 2), confidence (17,)
    """
    compressed = recv_tensor(sock)  # (17, 3)
    coords = compressed[:, :2]
    confidence = compressed[:, 2]
    return coords, confidence
