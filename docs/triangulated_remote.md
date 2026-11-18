# 遠端訓練與多視角動畫

以 Windows 遠端主機既有 CUDA 虛擬環境執行。在 PowerShell 更新程式後執行：

```powershell
Set-Location D:\Ray\graduate
git pull --ff-only
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_triangulated_remote.ps1
```

預設為 EgoHumans `01_tagging/001_tagging`，目標 aria01，本機 aria02，候選 aria03、aria04 與 cam01、cam02、cam03。可用 `-Sources` 調整候選名單。使用已確認備妥的其他資料時，傳入 `-Prepared`、`-Sequence`、`-SkipPrepare` 與新的 `-Run`。新資料需要已下載的原始影像、校正、SMPL 標註以及既有的官方三點 checkpoint、SMPL-H、ViTPose 和 DINOv2 權重。Python 預設與先前的遠端流程相同。第一次前處理會顯示人物框索引，請確認每個視角選到同一位 aria01。這個人工選框用來初始化追蹤。可以用 `-PreparedOnly` 先處理資料並檢查 `tracking_preview.mp4`。

同一指令會先檢查 CUDA 與依賴，再進行前處理、一個視窗的短訓練及正式訓練。短訓練另用同名的 `_smoke` 資料夾。重新執行同一指令會從最近完整 epoch 繼續，前處理也會重用已完成的視角。原始資料、來源 roster 或取用預算有變更時，用新資料夾。

`runs/tagging001_triangulated_mixed` 主要輸出：

- `multiview_selection.mp4` 左上本機、右上骨架、下方全部候選。選中變亮，未選變暗，初始化幀全部亮。相機畫面來自原始錄製，由動畫匯出端在重建完成後另外讀取。
- `figures/overview.png` 與 PDF 包含 MPJPE 時序、誤差與取用量、來源選擇時間圖和訓練曲線。
- `frames.csv`、`selection.json`、`metrics.json` 與 `predictions.npz` 可核對每幀名單、誤差、三角化關節及 3D 骨架。
- `checkpoint.pt` 含評分器、Exo 殘差投影與 optimizer。

每幀共同使用本機與實際取回的所有視角，逐關節以可用的射線做加權最小平方。可靠射線不足、視差不足或重投影誤差過大時回退到單視角。每幀 39 維幾何加 1 維觀測狀態送入原版投影，另以初始化為零的殘差投影適配三角化特徵。EgoExoMoCap 權重固定。評分器使用 0 到 K 路的取用規則，訓練時以 STOP 動作和每路取用成本學習，每幀發出一張名單。固定相機位姿由 COLMAP 重複估計整合。移動來源的決策幾何來自上次取得的位姿。

這是單一錄製序列上的訓練與比較。三點追蹤與地面正規化來自資料集標註，並清楚記錄在 experiment.json。第一個原始幀做初始化，從第二幀起每 96 幀重建一段，末尾不足一段的數量列在 metrics.json。
