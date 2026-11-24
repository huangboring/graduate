# 依本機需求選擇外部視角

目前唯一訓練主線為 **current_e_shared_head_v2**。本機 O₀ 處理目標 T 的影像，依本機 E／Q 與逐來源記憶選擇外部視角 O′。這是離線 EgoHumans 回放實驗；網路傳輸應用程式是情境假設，尚未實作通訊系統。

## 開始訓練

在遠端 PowerShell 執行：

```powershell
Set-Location D:\Ray\graduate
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git 更新失敗' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\train_current_remote.ps1 -Epochs 100
```

腳本依序訓練幾何 Exo 適配層 16 epochs、MLP 評分器 100 epochs、Attention 對照 100 epochs；共用同一凍結適配層及子集標籤。以 cam01/02 訓練，cam03/04 做固定權重相機替換；aria03/04 共同保留。若先只訓練，加入 `-SkipTransfer`。需要動畫時加入 `-Video`。可用 `-Python` 指定 Python、`-Root` 指定新的輸出位置，`-Epochs 50` 改跑 50 epochs。

預設輸出 `runs/tagging001_current_e_v2`，不覆寫舊結果。中斷後重跑相同命令會續訓；架構、資料及重要設定不相符時會停止。舊選擇器不能直接續訓。`-From 舊適配層路徑` 僅明確重用通過資料、相機集合及 ViTPose 權重檢查的適配層，不會載入舊評分器。

前提：既有準備資料 `data/prepared/tagging001_triangulated_annotated`、EgoHumans 原始資料、`external/EgoExoMoCap` 的指定權重、授權 SMPL-H 和 GPU 環境已就緒。上游版本記錄於 `configs/upstream.lock.json`。若缺 cam03/04，使用 `-SkipTransfer` 或先準備相應來源。

## 看結果

- `comparison.png`、`comparison.csv`：MLP／Attention 學習曲線及相機替換比較。
- `mlp/metrics.json`、`attention/metrics.json`：Learned、Local only、All sources、Random matched count，使用同一適配層。
- `mlp/round_trace.json`：逐幀評分、選擇、重訪、評分前資料年齡與實際更新來源。
- `mlp/predictions.npz`：預測、GT、原始幀號、關節幾何狀態。
- `mlp/epoch_0050`、`epoch_0100`：固定 epoch 權重與評估；`transfer34/camera_transfer.json` 為固定權重相機替換。
- 加 `-Video` 會輸出 `mlp/multiview_selection.mp4` 和 `attention/multiview_selection.mp4`：多視角選中變亮、未選變暗，並列預測與 GT。

## 目前架構

1. 初始化取得全部來源一次，不計入逐幀平均請求數。
2. 本機特徵 71 維 → MLP → E64；最近最多 4 筆本機觀測經 GRU → Q64。本機 GRU 每輪重新計算此視窗，來源 GRU 則保存各來源狀態。
3. 從狀態字典讀 H64、最後已取得相機資訊及拍攝時間。G9 與 `log1p(age)` 和 H 串接，Linear74→64＋LayerNorm 得 Z。
4. 同一評分頭處理 `[Q,LN(E)]` 與 `[Q,Z]`，得到本機及來源分數。Attention 是明確選擇的對照版本，在相同 `[E,Z…]` 內容上加 self-attention，仍共用 Q 和評分頭。
5. 取得所有分數高於本機的來源，再加必要重訪。無 Top-k 上限；超過 1 秒的來源依最早到期順序每輪強制取 1 個，不重複入列。
6. 選擇鎖定後才讀當前來源封包。姿態特徵 41 維 → MLP → 來源 GRU，僅更新選中的 H、拍攝時間和相機資訊。本輪更新供下一輪評分；未取得新觀測者保持原值。
7. 當前本機與實際取得的來源做信心加權幾何三角化；不足兩條有效射線時使用單視角近似回退。Exo embedding 產生單一 512 維 Exo token，接既有 Ego token 及凍結的空間／時間 Transformer。

目前選擇器沒有 CNN／DINO 外觀輸入，也不使用額外 DINO ScoreNet 產生三角化權重。ViTPose 的關節信心直接作為幾何權重。DINO 仍可能用於資料準備的身份追蹤，以及明確標示的原版融合比較；不進入目前選擇器。

## 訓練與比較的解讀

第一階段只學 52→512 幾何殘差適配層，原版40→512投影及重建模型凍結。第二階段凍結適配層，以離線子集誤差＋每次請求成本＋熵項訓練評分器；不是新增一個加權信心網路。子集標籤每隔8幀計算，在固定隨機雙來源背景中只替換被測幀，並非整個96幀視窗所有組合的最優解。

結果是同序列內評估；換相機不等於換人物或場景。目標的頭／腕追蹤由資料集 SMPL FK 取得，地板高度來自參考姿態，因此不能直接和論文50 mm測試成績比較。請求數減少也不能直接宣稱端到端 FLOPs 或時間下降，需另行量測。

需要原版融合對照時，在已訓練的主線命令加入 `--resume --compare-original`，保留同一 `--adapter-from`、來源集合及其他設定。該對照沿用 DINOv2 描述子，輸出明確標示替換，不能視為官方完整重現。

## 檔案與驗證

`selector.py` 是網路；`state_bank.py` 是來源字典；`selection.py` 是選擇／重訪規則；`rollout.py` 是回合時序；`fit_triangulated.py` 是訓練入口；`subset_training.py` 是離線子集標籤與損失；`triangulation.py` 和 `single_exo.py` 是幾何及重建串接。資料準備與原始影像追蹤工具保留，早期 local-aux、獨立評分頭、CNN 選擇器及 Top-k 訓練分支已移除，歷史可由 Git 取得。舊準備檔中的 `local_aux_*` 字串是相容格式標籤，不會啟用已刪除的舊後端。

```powershell
python -m unittest discover -s tests -q
python -m observer_selection --help
python -m observer_selection.fit_triangulated --help
```

本機驗證涵蓋因果時序、按ID更新、整批失敗不部分寫入、E/Q共用評分、MLP／Attention排列等變、訓練／續訓／相機替換、原版融合介面與動畫。CPU 測試通過不等於已完成遠端 GPU 訓練，也不保證特定 MPJPE。觀測與封包契約見 [目前流程](docs/current_method.md)，原始資料準備見 [資料準備](docs/egohumans_preparation.md)。
