import cv2
import os
import glob

def extract_frames(data_dir="./data/160422_ultimatum1", target_size=(256, 256)):
    video_dir = os.path.join(data_dir, "hdVideos")
    mp4_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    if len(mp4_files) == 0:
        print(f"在 {video_dir} 找不到任何 MP4 影片！")
        return

    for video_path in mp4_files:
        cam_name = os.path.basename(video_path).replace('.mp4', '')
        output_dir = os.path.join(video_dir, cam_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"開始處理影片: {cam_name}")
        cap = cv2.VideoCapture(video_path)
        
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # CMU 的 json 檔名通常是 8 位數補零，例如 00000000
            frame_str = str(frame_idx).zfill(8)
            out_path = os.path.join(output_dir, f"{frame_str}.jpg")
            
            # 為了節省硬碟空間並加速後續神經網路讀取，我們直接在這裡把它縮小成 256x256
            small_frame = cv2.resize(frame, target_size)
            
            cv2.imwrite(out_path, small_frame)
            
            frame_idx += 1
            saved_count += 1
            
            if frame_idx % 500 == 0:
                print(f"  已抽取 {frame_idx} 幀...")
                
        cap.release()
        print(f"✅ {cam_name} 處理完成！共儲存了 {saved_count} 張圖片。")

if __name__ == "__main__":
    print("=== CMU Panoptic 影片抽圖與壓縮腳本 ===")
    print("此腳本會將 MP4 影片的每一幀抽取出來，並直接壓縮成 256x256 的小圖。")
    extract_frames()
