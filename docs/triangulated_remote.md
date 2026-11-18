# 遠端訓練與多視角動畫

以 Windows 遠端主機既有 CUDA 虛擬環境執行。在 PowerShell 更新程式後執行：

```powershell
Set-Location D:\Ray\graduate
git pull --ff-only
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_triangulated_remote.ps1 -Appearance cnn -Epochs 16
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_triangulated_remote.ps1 -Appearance none -Epochs 16
```

預設為 EgoHumans `01_tagging/001_tagging`，目標 aria01，本機 aria02，候選 aria03、aria04 與 cam01、cam02、cam03。可用 `-Sources` 調整候選名單。使用已確認備妥的其他資料時，傳入 `-Prepared`、`-Sequence`、`-SkipPrepare` 與新的 `-Run`。新資料需要已下載的原始影像、校正、SMPL 標註以及既有的官方三點 checkpoint、SMPL-H、ViTPose 和 DINOv2 權重。Python 預設與先前的遠端流程相同。預設從 `processed_data/poses2d/<相機>/rgb` 按 `human_name=aria01` 讀取官方逐幀人物框，依各相機的校正把框轉到整流影像，再由 ViTPose 和 DINOv2 抽特徵。官方框缺失時保留缺失。這是使用官方框和人物 ID 的離線實驗，報表的 experiment.json 記為 `egohumans_annotation`。需要自行追蹤時，可指定 `-TrackBoxes` 及不同的 `-Prepared`、`-Run` 資料夾。

同一指令會先檢查 CUDA 與依賴，再進行前處理、一個視窗的短訓練及正式訓練。短訓練另用同名的 `_smoke` 資料夾。重新執行同一指令會從最近完整 epoch 繼續，前處理也會重用已完成的視角。原始資料或來源 roster 有變更時，用新資料夾。更改門檻後的新預設訓練目錄保留既有 Top-2 結果。

使用 `-TrackBoxes` 時，在 `Target box:` 可輸入 `f 8`，直接到原始第 8 幀產生框號預覽，再輸入框號。已知框號時輸入 `8:2`，直接用第 8 幀的 2 號框初始化。幀號須晚於當前提示。輸入 `q` 保存後可續跑，跳過的幀維持缺失。

CNN 和無外觀分別輸出到 `runs/tagging001_triangulated_vitpose_cnn` 與 `runs/tagging001_triangulated_vitpose_none`。兩者共用既有前處理。CNN 是凍結的 torchvision MobileNetV3-Small ImageNet 權重，首次會載入或下載權重，特徵按視角快取。無外觀版使用關節、歷史與相機幾何。兩個版本的三角化都直接使用 ViTPose 信心，ScoreNet 留在對照用途。正式訓練預設 16 個 epoch，增加 `-Epochs` 可沿用同一份 checkpoint 繼續。報表另含同路數與重訪的固定兩路診斷。

主要輸出：

- `multiview_selection.mp4` 左上本機、右上斜角 3D 骨架、下方全部候選。選中變亮，未選變暗，初始化幀全部亮。本機與選中來源會疊有效的 2D 關節，並以校正反映射到原始影像。相機畫面來自原始錄製，由動畫匯出端在重建完成後另外讀取。
- `figures/overview.png` 與 PDF 包含 MPJPE 時序、誤差與取用量、來源選擇時間圖和訓練曲線。
- `frames.csv`、`selection.json`、`metrics.json` 與 `predictions.npz` 可核對每幀名單、誤差、三角化關節及 3D 骨架。
- `checkpoint.pt` 含評分器、Exo 殘差投影與 optimizer。

每幀共同使用本機與實際取回的所有視角，逐關節以可用的射線做加權最小平方。可靠射線不足、視差不足或重投影誤差過大時回退到單視角。每幀 39 維幾何加 1 維觀測狀態送入原版投影，另以初始化為零的殘差投影適配三角化特徵。EgoExoMoCap 權重固定。評分器每幀取分數高於本機的所有候選，並加上排到的重訪來源。訓練以來源與本機的分數差抽樣，保留每路取用成本。隨機基準逐幀對齊路數與重訪來源。固定相機位姿由 COLMAP 重複估計整合。移動來源的決策幾何來自上次取得的位姿。

這是單一錄製序列上的訓練與比較。三點追蹤與地面正規化來自資料集標註，並清楚記錄在 experiment.json。第一個原始幀做初始化，從第二幀起每 96 幀重建一段，末尾不足一段的數量列在 metrics.json。
