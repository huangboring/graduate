# 新版精簡編碼與 MLP／Attention 比較

遠端在 `D:\Ray\graduate` 執行：

```powershell
Set-Location D:\Ray\graduate
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git 更新失敗' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\train_compact_comparison_remote.ps1 -Epochs 100
```

可先指定 `-Epochs 50`，之後改 `100` 接續。同一個實驗輸出資料夾不能變更成本、資料或架構。新版輸出預設是 `runs/tagging001_compact_ablation`，不會續寫舊選擇器。

## 本次實際修改

| 欄位 | 舊版 legacy | 新版 compact |
|---|---:|---:|
| 本機每關節 | xy2、座標旗標1、信心1、信心旗標1、prior殘差2＋旗標1、變化率2＋旗標1，共11 | xy2、信心1、變化率2，共5 |
| 本機全域 | 中心／尺度2、log尺度1、log前幀間隔1、prior年齡1 | 中心／尺度2、log尺度1、log前幀間隔1、觀測可用1、前後幀可比較1 |
| 本機總維度 | 148 | **71** |
| 來源每關節 | xy2、座標旗標1、信心1、信心旗標1，共5 | xy2、信心1，共3 |
| 來源全域 | log取得間隔1 | 觀測可用1、log取得間隔1 |
| 來源總維度 | 66 | **41** |
| 本機／來源 GRU hidden | 64／64 | 64／64 |
| 固定為1的 has_history 輸入 | 保留 | **移除** |
| 本機與來源評分頭 | MLP版分開 | **兩個新版都使用共享評分頭** |

本機與來源資料仍由人物框的可用性表示整筆觀測是否存在，不表示逐關節遮擋。低信心與無觀測分開處理。新版輸入假設有效遮罩在同一視角的13關節一致、可用關節都有有限信心；若輸入是混合逐關節缺失遮罩，會明確拒絕，避免移除旗標後失去語意。既有 EgoHumans prepared 格式符合此假設。

本機範例：框 `[100,50,300,450]`、目前左膝 `(240,330)`、前幀 `(232,326)`、間隔0.05秒、信心0.72。

- 新版左膝5維：`[0.1,0.2,0.72,0.4,0.2]`。
- 新版本機6個全域值：`[0.5,0.625,ln401,ln1.05,1,1]`。
- 來源左膝3維：`[0.1,0.2,0.72]`；全域為 `[觀測可用,ln(1+距前次取得間隔)]`。
- 第一幀、前一幀缺失或當幀缺失時，變化率填零，`前後幀可比較=0`。沒有觀測時仍能經 GRU 與評分網路輸出。

## 新版兩個模型的確切比較

兩者共用同一種輸入、短窗GRU、來源GRU、token投影、正規化和評分MLP。固定seed時，共用部分的**初始權重逐項相同**。

來源 token 原始輸入：`h64 + geometry9 + log(age)1 = 74`，線性投影成64維。本機 token 是64維需求 q。所有 token 都使用相同的 LayerNorm。

- **MLP**：每個 token 直接與 q 串接，以共享 `128→64→32→1` MLP 計算分數；本機也使用同一評分頭。移除舊本機頭額外讀取當幀 e 的做法。
- **Attention**：先將本機與所有來源 token 一起送進4-head self-attention，經殘差與 LayerNorm，再與 q 串接，使用相同結構的共享 `128→64→32→1` MLP。

| 無外觀選擇器 | 參數量 |
|---|---:|
| 舊 MLP | 132,674 |
| 舊 Attention | 134,977 |
| **新 MLP** | **96,833** |
| **新 Attention** | **113,601** |

參數量包含需求／來源編碼器與GRU，不含固定ViTPose、EgoExoMoCap與重建adapter。舊版含未使用的預留記憶模組，因此參數量下降不能直接當成同等比例的實際運算量下降。Attention多出的16,768個參數來自attention與其殘差正規化；這不是等參數量比較。

新版 MLP／Attention 的主要差別是attention block。**新舊版本之間同時改了輸入編碼、token表示及評分頭，不能把新舊結果差異全歸因於移除有效性旗標。**

## 保留的部分

- 無CNN／DINO外觀，三角化繼續使用ViTPose信心。
- 幾何仍是既有9維：2個啟用值、2個保留值、4個旗標、1個幾何年齡。沒有偷偷移除它們。
- 扇形5公尺、約90度Aria整流視野與固定相機各自內參不變。
- 分數高於本機者全取，聯集每幀最多1個到期重訪；到期門檻1秒。
- 不新增持續性的FIFO物件，仍每幀依到期時間排序，取得後更新時間，未輪到者繼續等待；語意與先前討論一致。
- 三角化、回退、Exo token、96幀重建與既有 subset expected-cost 目標不變。
- 推論前只讀本機與來源歷史，不偷看未取得的當幀來源封包。

## 訓練與資料重用

1. 如果 `runs/tagging001_camera_ablation/adapter_train12` 已有完成的 checkpoint／experiment／metrics，預設沿用它的凍結重建adapter。否則在新Root中先用原本legacy方式準備adapter，預設16 epoch。
2. MLP與Attention各自訓練100 epoch；每25 epoch及最後一輪保留checkpoint與對照，包含50／100兩個預定比較點。
3. 如果有相同凍結模型與資料的舊subset快取，新MLP可重用；新Attention與新MLP共用成本快取。標籤由固定重建模型的離線集合測試產生，不依選擇器編碼而改變。
4. 既有 prepared 影像、ViTPose輸出、相機校正直接重用，不需為71／41維重跑特徵抽取。若換相機測試資料尚未準備，cam03／04那份仍需準備一次。
5. 訓練候選 aria03、aria04、cam01、cam02；換相機測試 aria03、aria04、cam03、cam04，固定權重。

只先跑訓練相機的模型比較，可加 `-SkipTransfer`；之後移除此參數重跑即可補測。

若要指定某個兼容的凍結adapter：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\train_compact_comparison_remote.ps1 -Epochs 100 -From 'runs/tagging001_camera_ablation/adapter_train12'
```

新版與舊版 selector checkpoint 不相容，會以 experiment 合約拒絕混用；舊重建adapter則可獨立沿用，因為它不接收71／41維特徵。

## 結果位置

`runs/tagging001_compact_ablation/`：

- `comparison.png`、`comparison.csv`：新版MLP與Attention的誤差、請求數、相機轉移與六種策略對照。
- `model_comparison.csv`：兩模型的輸入維度、評分頭與參數量。
- `mlp/model_architecture.json`、`attention/model_architecture.json`：可核對的模型配置。
- 各自 `epoch_0050/`、`epoch_0100/`：檢查點與對照結果；`transfer34/`包含相機轉移結果。
- `baseline_comparison_status.json`：舊實驗是否具備可比較條件。
- `old_vs_new.csv`、`old_vs_new.png`：有同條件、同epoch舊成績時才產生。不同資料、來源、成本或凍結重建模型時不硬湊比較；舊相機測試若缺測試replay指紋，也不當成嚴格匹配的轉移對照。

每epoch約6次optimizer更新，100epoch約600次。仍是同序列、單seed探索，換相機不等於跨人物／場景泛化；attention與精簡特徵是否改善必須以實際GPU結果判斷。
