# MLP / attention、50 / 100 epoch 與固定權重相機測試

在遠端 `D:\Ray\graduate` 執行：

```powershell
Set-Location D:\Ray\graduate
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git 更新失敗' }
.\train_attention_transfer_remote.ps1 -Epochs 100
```

可以先跑 `-Epochs 50`，之後同一指令改 `100`，會接續 checkpoint，不必從頭訓練。中斷後也可重跑同一指令。不要在同一個 Root 更改 Cost、輸入資料或模型架構；變更實驗設定時指定新的 `-Root`。

## 實驗的確切差異

- 目標 aria01、本機 aria02；aria03、aria04 一直保留為候選。
- 本次訓練候選只有 aria03、aria04、cam01、cam02，cam03、cam04 的封包會從訓練 replay 移除。
- 先在此集合重訓殘差 adapter，預設 16 epoch，再凍結，同時供兩個評分器使用。舊 adapter 用過 cam03，這裡不能拿來宣稱 cam03 未參與本次訓練。
- 接著 MLP、attention **各自**訓練 100 epoch，沿用既有 subset expected-cost 目標、成本 0.002 m/來源、ViTPose 權重、無外觀特徵。兩者共享經驗證相容的離線 subset 成本快取，避免重算。
- 這個序列每 epoch 是 6 次 optimizer 更新，100 epoch 共 600 次。不是增加新的訓練樣本；不保證持續改善。
- Attention 只取代評分部分：需求 q 是 64 維，各來源是 `[歷史 h64, 幾何9, log(age), seen]` 共 75 維，線性投影成 64 維。加上需求 token 後，4 個候選形成 `[1,5,64]`，使用 4 heads 的 self-attention、殘差與 LayerNorm、共享線性頭輸出本機與各來源分數。沒有相機 ID 或順序位置 embedding。歷史 GRU、需求 GRU 維持既有做法。MLP 路徑保持原計算。
- Attention 可以讓來源間互相比較，但推論仍然是逐來源比較分數：`s_i > s_local` 全取，另加強制重訪，沒有 Top-K 限制。固定取 2 只是一個對照測驗。
- 不會提前讀取尚未請求的當前來源觀測。第一幀所有候選初始化一次，各方法相同。

## 相機轉移

自動另建測試資料 `data/prepared/tagging001_transfer_34`（官方人物標註框），候選換為 aria03、aria04、cam03、cam04。這可能需要額外的 GPU 特徵抽取時間；模型下載可重用 ModelCache。

50、100 epoch 各載入對應 checkpoint，在測試相機上只推論，不建立 optimizer、不更新 selector 或 adapter。`evaluation_protocol.json` 記錄權重 SHA256、epoch、訓練／測試來源、共享與未見來源。

這是**同序列內固定相機替換測試**：人物、動作、時間幀相同，動態候選 aria03/04 共用。可以檢查換視角是否退化，不能當成跨人物／跨場景泛化。官方預訓練模型持續沿用，其預訓練資料並未在本實驗重新劃分，因此也不宣稱測試相機從未出現在官方預訓練中。

## 看結果

`runs/tagging001_camera_ablation/`：

- `comparison.png`：兩條訓練曲線、平均請求數、50/100 epoch 換相機結果。
- `comparison.csv`：逐 epoch 檢查點、評分器、來源集合及六種策略的結果。
- `mlp/epoch_0050/metrics.json`、`attention/epoch_0100/metrics.json`：同相機對照。
- 各檢查點內的 `transfer34/camera_transfer.json`：換相機對照。
- `learning_curve.json`：各訓練 epoch 的 learned 誤差、請求數、loss。

除了 learned，也評估 local only、all sources、matched-count random、rank fixed 2、random fixed 2。要一起看誤差與請求數：如果 100 epoch 的誤差變小但請求更多，不能把改善全歸因於評分能力；固定 2 的對照有助判讀。隨機基準目前固定 seed 2026，屬單次探索實驗。

不要依 transfer34 成績挑 checkpoint 再把同一成績當未見測試。50/100 是預先指定的比較點；正式研究仍需獨立驗證／測試序列與多個 seed。

腳本預設不產影片，以減少這輪比較的重複工作；既有影片功能保留。若只先做評分器與 epoch 比較，加 `-SkipTransfer`，後續移除此參數即可補跑相機轉移。

本機測試驗證程式與合成小型流程，不代表已完成遠端 GPU 訓練或已證實 attention 更好。
