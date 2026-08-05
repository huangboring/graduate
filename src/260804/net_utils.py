import struct
import io
import socket
import torch

def send_tensor(sock: socket.socket, tensor: torch.Tensor):
    """
    將 PyTorch Tensor 序列化並透過 Socket 傳送。
    協定格式：[4 bytes 長度 (Network Byte Order)] + [序列化後的位元組資料]
    """
    # 確保資料在 CPU 上，避免 GPU 之間傳輸的記憶體指標錯誤
    tensor_cpu = tensor.cpu().detach()
    
    # 將 Tensor 轉換成 Bytes
    buffer = io.BytesIO()
    torch.save(tensor_cpu, buffer)
    data_bytes = buffer.getvalue()
    
    # 計算長度並打包成 4 bytes 的標頭
    length = len(data_bytes)
    header = struct.pack('!I', length)
    
    # 傳送標頭與資料
    sock.sendall(header + data_bytes)

def recv_exact(sock: socket.socket, n: int) -> bytes:
    """從 Socket 中準確讀取 n 個 bytes"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("網路連線意外中斷")
        data.extend(packet)
    return bytes(data)

def recv_tensor(sock: socket.socket) -> torch.Tensor:
    """
    從 Socket 接收資料並反序列化為 PyTorch Tensor。
    """
    # 讀取 4 bytes 標頭
    header = recv_exact(sock, 4)
    length = struct.unpack('!I', header)[0]
    
    # 讀取資料本體
    data_bytes = recv_exact(sock, length)
    
    # 將 Bytes 還原成 Tensor
    buffer = io.BytesIO(data_bytes)
    # weights_only=False 確保我們可以載入標準的 Tensor 物件
    tensor = torch.load(buffer, map_location='cpu', weights_only=False)
    return tensor
