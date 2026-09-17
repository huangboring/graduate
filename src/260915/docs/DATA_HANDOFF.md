# 舊資料處理與搬機核對（2026-09-17）

## 1. 從舊程式確認的事實

查閱工作區 `graduate/src/260911`、`260914` 的 dataset.py、train.py、model.py、node.py，及 `data/panoptic-toolbox-master/scripts`。

- 預設序列是 `160422_ultimatum1`。優先尋找 `graduate/src/data/160422_ultimatum1`，其次有 `graduate/data` 及 toolbox/data 等候選路徑。
- 相機清單寫死為 `00_00`、`00_01`、`00_02`；只把 num_views 改大不會自動加入更多相機。
- loader 掃描已解壓的 `hdPose3d_stage1_coco19`，包含遞迴子資料夾；只收 bodies 非空的 JSON。
- 由 `body3DScene_00001234.json` 對應到 `hdVideos/hd_00_00/00001234.jpg`；它不直接讀 MP4，也不讀官方 hdImgs 路徑。
- 每幀只取 `bodies[0]`，從 joints19 直接切前 17 點，以 K/R/t 投影成 2D。沒有使用鏡頭畸變係數。
- 1920×1080 影像被 resize 成 256×256，產生 64×64 的 17 通道 heatmap；260914 加上 ImageNet 正規化。
- 260914 train.py 使用同一個 dataloader 跑 60/40/10 epochs，主要是 ego view 的 2D heatmap 與座標 loss；cam_params 沒有送進這個模型的 training forward，沒有獨立驗證 loader。
- 舊 node.py 使用標準 COCO17 skeleton，但資料只是 COCO19 前 17 點，沒有做對應轉換；不能假設關節語意已對齊。
- 影像路徑不存在時補黑圖；找不到任何標註時改用 dummy 隨機資料；node.py 權重載入失敗時仍繼續用初始權重。

這些是「程式會怎麼做」，不是證明另一台電腦當時真的走到了缺檔 fallback。
舊權重檔存在、loss 曾下降，都不足以單獨證明影像與標註已正確對齊。
舊模型是自訂 ResNet-34 / 17 點 2D 模型，不能拿它的 stage1/2/3 權重當作新的 MVGFormer 完整權重。

## 2. 推測另一台資料結構

若舊 loader 讀到的是真實影像，另一台很可能有：

```text
<資料父目錄>/
  160422_ultimatum1/
    calibration_160422_ultimatum1.json
    hdPose3d_stage1_coco19/
      body3DScene_00001234.json
      ...
    hdVideos/
      hd_00_00.mp4                    # 可能仍保留
      hd_00_01.mp4
      hd_00_02.mp4
      hd_00_00/
        00001234.jpg
        ...
      hd_00_01/
        00001234.jpg
      hd_00_02/
        00001234.jpg
```

這份工作區沒有找到產生上述自訂 JPG 路徑的抽幀腳本，因此抽幀命令、是否 start_number=0、是否跳幀及 resize 仍未知。
附帶的官方 hdImgsExtractor.sh 會輸出 `hdImgs/00_00/00_00_00000000.jpg` 且 start_number=0，與舊 loader 路徑不同。
附帶 getData.sh 的 HD 順序是 0,1,2,...，能解釋「指定下載 3 台時得到 00_00/01/02」；不能證明當時實際下了哪條命令。

## 3. 本機實際資料，與另一台分開看

已實際掃描 `C:/碩論/data/panoptic-toolbox-master/data/160422_ultimatum1`：

| 項目 | 本機觀察 |
| --- | --- |
| HD MP4 | 只有 hd_00_00.mp4，約 533.5 MB |
| 影片 | 可開啟，1920×1080，約 29.97 FPS，27170 幀，約 15.1 分鐘 |
| 校正檔 | 有 31 台 HD 校正；不代表有 31 台影片 |
| 3D 標註 | TAR 存在，尚未解壓；26794 個 JSON |
| 標註影格 | 173–26966 |
| 有人體的標註影格 | 24181 個；第一個在 266 |
| 首個符合 root confidence > 0.1 | 268 |
| 每幀最多標註人體 | 7 |
| JPG | 此路徑未找到 |

詳細機器可讀結果見 `local_data_audit.json`。本機只有一台影片，無法做三視角訓練；不能因此推論另一台缺資料。
影格範圍相容也不等於視覺對齊已驗證。這次沒有宣稱確認另一台 JPG 的影格偏移。

## 4. 本輪相容性修正

- 新版 source 同時支援 MP4、舊版 JPG 目錄與官方 hdImgs。
- 保留檔名中的原 HD 影格 ID；不把 JPG 排序位置當作 GT index，也不自動補償 0/1 起始偏移。
- 支援標註解壓後多一層同名子資料夾；重複影格 ID 拒絕讀取。
- 缺圖、不同相機影格不一致、解析度不符校正、缺相機均明確報錯；不補黑圖或隨機資料。
- 短版 demo 測試從驗證樣本中有人體標註的位置開始，避免只測片頭空場。
- 新版沿用 MVGFormer 15 點、多人物與相機畸變處理；不沿用舊的 17 點 heatmap 標籤或黑圖 fallback。

## 5. 另一台先執行資料稽核

整個 `260915` 資料夾搬過去。下列指令在該資料夾執行；data-root 可以填一段序列，或包含多段序列的父資料夾。

```powershell
python run/audit_data.py --data-root "D:/你的資料目錄" --output "data_audit.json"
```

這是唯讀檢查，不會解壓、刪除或改名原資料。只需 Python；安裝 OpenCV 後能多檢查影片與影像解析度，不需要 CUDA 或 torch。
若只有 TAR，先在序列目錄解壓：

```powershell
tar -xf hdPose3d_stage1_coco19.tar
```

如果保留完整 MP4，先指定 `input_format: videos`，可以避開舊抽幀命名與縮放的不確定性。
只有 JPG 時，先核對實際命名與解析度，再指定 `legacy_images` 或 `hdimgs`。
只改檔名不能證明同步；應拿幾幀 GT 投影疊回原圖，檢查有無固定一幀偏移、錯相機或變形。

## 6. 該如何訓練

提供 `configs/train_ultimatum_pilot.json` 作為符合舊預設序列的排錯設定：

- data_root 預設 `../data`，對應舊版優先路徑 `graduate/src/data`；若另一台不同，改為實際資料父目錄，**不要填到單一序列資料夾**。
- train_sequences 已填 `160422_ultimatum1`。
- val_sequences 刻意留空。請填另一個實際存在的獨立序列；不會把同段影片偷偷隨機切成驗證。
- start_frame=300、frame_stride=30、每序列最多 300 個時間點、20 epochs、K=2。
- cameras=null，自動依實際資料找相機；原始選擇訓練至少需要 3 台。
- pose_checkpoint 要放完整官方 MVGFormer checkpoint，不是 `when2com_pose_stage3_latest.pth`。

填妥後：

```powershell
python run/train_mvp.py --config configs/train_ultimatum_pilot.json --check
python run/train_mvp.py --config configs/train_ultimatum_pilot.json
```

若另一台也只有 ultimatum1，一段序列在資料格式上可產生訓練樣本，但目前缺獨立驗證序列，不能完成這份研究訓練配置。
需補至少一個獨立序列；時間區塊切分只能另作單場景排錯實驗，不能代替跨序列泛化評估。

本段有 26794 標註影格，但約 15 分鐘資料以每秒一筆抽樣只有約 900 個同步時間點；不能把它當作 26794 個互相獨立的場景。
先以 300 點確認教師品質與完整流程，再增加到整段及更多序列。MVP 的 5000–20000 點仍應來自多段活動與視角配置。

## 7. 搬機驗收邊界

本機已做合成影片、JPG、巢狀 GT 路徑與模型介面測試；沒有在另一台執行真實多視角 GPU 訓練。
正式訓練前仍需確認：實際相機數、是否有完整 MP4／原解析度 JPG、獨立驗證序列、GPU/CUDA 與完整權重。
程式可攜帶，但在 data_audit 與 train --check 通過前，不宣稱另一台已準備好開訓。
