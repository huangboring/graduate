import socket
import struct
import io
import torch

# 避免安全漏洞，加上 weights_only=True，但為了向下相容或者處理非張量資料，
# 對於已知安全的張量傳輸，最好改用 safetensors 或確認只載入 Tensor
def send_tensor(sock, tensor):
    """
    透過 socket 傳送 PyTorch Tensor
    """
    buffer = io.BytesIO()
    # torch.save 預設是 pickle，有安全風險。這是一個研究用 prototype，
    # 但加上 weights_only=True 可以降低風險。
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
    透過 socket 接收 PyTorch Tensor
    """
    # 接收資料大小 (4 bytes)
    size_bytes = b""
    while len(size_bytes) < 4:
        chunk = sock.recv(4 - len(size_bytes))
        if not chunk:
            raise ConnectionError("Socket closed before receiving size.")
        size_bytes += chunk
        
    size = struct.unpack("!I", size_bytes)[0]
    
    # 接收資料
    data = b""
    while len(data) < size:
        chunk = sock.recv(min(4096, size - len(data)))
        if not chunk:
            raise ConnectionError("Socket closed before receiving all data.")
        data += chunk
        
    # 解開 Tensor
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, map_location='cpu', weights_only=True)
    return tensor

def send_compressed_heatmap(sock, coords, confidence):
    """
    優化頻寬: 傳送壓縮後的 2D 座標和信心分數 (共 17 * 3 = 51 個浮點數)
    """
    compressed = torch.cat([coords, confidence.unsqueeze(-1)], dim=-1) # (17, 3)
    send_tensor(sock, compressed)

def recv_compressed_heatmap(sock):
    """
    接收壓縮後的 2D 座標和信心分數
    回傳 coords (17, 2), confidence (17,)
    """
    compressed = recv_tensor(sock) # (17, 3)
    coords = compressed[:, :2]
    confidence = compressed[:, 2]
    return coords, confidence
