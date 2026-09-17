# MVP 訓練、資料與 demo

## 範圍

本版本固定完整預訓練 MVGFormer（backbone + decoder），訓練前置視角組合選擇器。
「一條指令」涵蓋標籤產生、選擇器訓練、驗證、打包與短版 demo 檢查；不包含自動下載大型資料、安裝 CUDA 或從零訓練姿態網路。
之前的 `run/train_3d.py`、`validate_3d.py` 與其他資料集 loader 已移除；原始參考程式仍在工作區的 `code/MVGFormer`。

## 1. 環境與資料

使用有 NVIDIA GPU 的 MVGFormer 相容 CUDA/PyTorch 環境。原作者使用 Python 3.10。
先依 GPU 選擇相容的 torch/torchvision，再安裝本目錄 requirements，最後编譯 extension：

```powershell
python -m pip install -r requirements.txt
python -m pip install ./lib/models/ops --no-build-isolation
python run/train_mvp.py --check
```

CUDA Toolkit、編譯器與 torch CUDA 版本需要相容；沿用已能跑原 MVGFormer 的環境最容易排除問題。
本次移除主流程對 mmcv、wandb、SMPL、Human3.6M 等套件的依賴。
上游 CUDA 擴充套件不保證在所有新 GPU/torch 組合直接編譯成功；`--check` 只檢查可找到模組與 CUDA 可用，真正載入與 ABI 相容由後續推論檢查。

每個序列需要：

```text
data/panoptic/<sequence>/
  hdVideos/hd_00_00.mp4
  hdVideos/hd_00_01.mp4
  hdVideos/hd_00_02.mp4
  ...
  calibration_<sequence>.json
  hdPose3d_stage1_coco19/body3DScene_00000000.json
  ...
```

支援 `hdVideos` 與 `hdvideos`。相機 ID 自動偵測，不要求下載最前面的台數。
訓練與推論支援 MP4、舊版 `hdVideos/hd_00_XX/00000000.jpg` 及官方 `hdImgs/00_XX/00_XX_00000000.jpg`。`input_format` 可選 `videos`、`legacy_images`、`hdimgs` 或 `auto`；auto 選擇涵蓋最多相機的格式，同數時優先 MP4，不混用格式。只需把 3D 標註 tar 解壓；若保留 MP4，不必抽取所有 JPG。圖片資料各相機的影格 ID 必須一致；缺圖會報錯，不補黑圖。
不同序列可以有不同候選相機數。至少 3 台才能形成多種 K>=2 的選擇方案。
影片必須是同一序列的同步 HD 原影片；相同 FPS 不等於任意來源影片已同步。
影片解析度若不符校正檔會拒絕執行，避免錯誤投影。

`pose_checkpoint` 必須是完整 MVGFormer 權重，不能只放 PoseResNet backbone，或舊 When2com 的 ResNet-34 權重。

## 2. 一條指令

先編輯 `configs/train_mvp.json`。所有相對路徑以 `260915` 為根，不以終端目前位置或設定檔位置為根。
訓練／驗證序列名稱範例需要換成實際資料；程式不會默默把測試資料當訓練資料。

```powershell
python run/train_mvp.py
```

自動流程：

1. 檢查資料、GT、完整權重、相機數與 CUDA；禁止訓練／驗證序列重疊。
2. 每隔 `frame_stride` 個影格讀一組同步影像，每序列最多 `frames_per_sequence` 組。
3. 訓練集隨機選 3 到 `max_candidate_views` 個候選相機，從 `k_values` 選擇小於候選數的 K。
4. 每組影格先抽取全部候選特徵一次，再測試 K 視角組合與全候選基準。
5. 對照 GT，產生各組合的姿態品質成本，快取輕量描述向量、幾何與標籤。
6. 訓練共享集合編碼器，以獨立驗證序列的成本選擇最佳 epoch，支援 early stopping。
7. 匯出姿態權重、選擇器權重、設定、hash 與預設 demo 策略到 `bundle/`。
8. 另開程序使用同一個 `demo_mvp.py` 和模型包，從驗證樣本中有有效人體標註的影格開始跑 `smoke_frames` 幀。未達幀數或完全沒偵測到人會報錯，不產生 `READY.json`。

快取以程式、模型 SHA256、設定及資料檔案大小／修改時間作為一致性檢查。

```powershell
python run/train_mvp.py --resume
```

resume 重用教師快取後重新訓練小選擇器；不是 optimizer 精確續訓。設定或資料變更時使用新的 `output` 目錄。
輸出資料夾已存在時預設拒絕覆寫。標籤產生通常遠比小選擇器訓練耗時。

## 3. 多台相機的訓練方式

每個視角的 32 維描述向量加上 6 個幾何值（世界座標相機中心 / 10000 mm、觀察方向），由同一組網路權重編碼。
組合評分使用所選視角的編碼均值、變異量、全候選集合均值與選取數資訊。
不使用相機 ID embedding，也不把固定數量的視角直接串成固定長度向量。
因此相機順序與候選數不綁死；三／四台訓練後在程式上可以輸入更多台。

但幾何、場景、遮擋與相機數的泛化仍需真實資料驗證，不能只靠介面支援宣稱有效。
候選很多時不枚舉所有組合：最多評估 `subset_limit` 組，固定 seed 並納入前 K 台基準。
小候選池可枚舉完整組合；大候選池的最佳結果只是「抽樣候選中的最佳」，不是全域最佳。
目前仍由使用者設定每幀 K，不自動學習要用幾台；混合 K 的訓練是讓同一選擇器支援多種預算。

## 4. 需要多少資料？

以下是本專案的實驗起始建議，不是論文保證或統計上已證明足夠的門檻。
一筆資料是「同一時間點的所有候選視角 + 對應 3D 標註」，不是把每張相機照片各算一筆獨立樣本。

| 階段 | 訓練同步時間點 | 資料安排 | 目的 |
| --- | ---: | --- | --- |
| 排錯 | 300–1000 | 至少 1 個訓練序列 + 1 個獨立驗證序列 | 驗證教師輸出、標籤、訓練、demo |
| 初版 MVP | 5000–20000 | 約 5–10 個不同訓練序列；另外 2 個驗證序列，約 1000–3000 點 | 看學習式選擇是否優於固定 K |
| 泛化研究 | 按學習曲線增加 | 更多視角位置、人物、遮擋、活動；另留未用過的測試序列 | 評估新相機配置與場景 |

預設設定是每序列最多 1000 點、兩個訓練序列加一個驗證序列，適合先建立試跑結果；短影片可能不足上限。
建議先以每秒約 1–2 個同步時間點取樣，而非把相鄰的 30 幀視為 30 個高度獨立的例子。
預設 `frame_stride: 30` 對約 30 FPS 的 HD 影片約為每秒一筆。
對快速動作可提高取樣率。不要把同一段影片隨機逐幀切成訓練與驗證，會造成時間洩漏。
只有單一序列時，應補獨立序列；本入口刻意不自動做隨機影格切分。

若每筆測 16 種組合加全視角，20000 筆就要約 340000 次 decoder 推論。
先用數百筆測實際耗時，確認教師能穩定偵測人物，再放大資料量。
資料擴充應以驗證曲線判斷：新增資料後成本是否下降、固定 K 差距是否改善，而不是只看 training loss。
原版 MVGFormer 訓練序列清單可參考 https://github.com/XunshanMan/MVGFormer/blob/master/docs/CMU_sequences.md 。

## 5. 目標與評估

教師對每個組合輸出人體姿態，以 Hungarian 一對一匹配 GT。
對可見關節計算距離，超過 `match_mm` 的配對視為未匹配；對漏檢及誤檢都加罰則。
這個 `cost_mm` 是自訂的偵測感知成本，**不是官方 MPJPE 或 AP**。
沒有偵測到人不會被算成零誤差；所有組合成本完全一樣時拒絕訓練，提示先檢查教師模型。

訓練使用組合成本產生 soft targets，做組合排序的交叉熵；不經過 hard top-K 反向傳播，也不是端到端聯合訓練。
驗證報告比較 learned、固定前 K、抽樣組合隨機選取的期望成本、抽樣 oracle 與全視角。
validation 是用來選 checkpoint 的資料，不能同時當作未見測試集。正式報告需額外獨立測試集與官方姿態指標。

若選擇器在 demo K 上未優於固定 K，模型包預設使用固定 K，避免把未證明有效的策略當成推薦結果。
可以明確加 `--policy learned` 觀察差異。這個 fallback 只表示比較結果，不保證固定 K 已有良好姿態品質。

## 6. Demo 與輸出

```powershell
python run/demo_mvp.py --bundle output/mvp_training/bundle/bundle.json --sequence "D:/data/panoptic/160906_pizza1"
python run/demo_mvp.py --bundle output/mvp_training/bundle/bundle.json --sequence "D:/data/panoptic/160906_pizza1" --policy learned --k 2
```

`bundle.json` 攜帶同一份前處理設定、姿態模型、選擇器、K、組合上限、閾值與 NMS。
啟動時驗證檔案 hash；避免拿錯 selector 配不同 pose checkpoint。執行仍需本專案原始碼與 CUDA 環境。
`--cameras 00_00 00_01 00_02` 可指定相機；省略時使用該序列所有影片。
全視角與學習式策略比較時，請固定候選相機清單、影格範圍與 K。

訓練輸出：`cache/`、`selector.pt`、`training_history.json`、`validation.json`、`bundle/`、`smoke/`、成功後的 `READY.json`。
Demo 輸出：`manifest.json`、`frames.jsonl`、`summary.json`。
3D 座標為 MVGFormer 世界座標（mm）。畫面為骨架等角投影，不是假裝疊在某一相機上的 2D 投影。
顯示所有相機與被選取視角；subset selector 的個別相機顯示分數是包含該相機的抽樣組合平均分，不是關節可見性。

Demo 逐幀盡快處理，不主動丟幀或依影片速度睡眠。pipeline latency 含讀幀、特徵、選擇、decoder 與後處理，排除顯示／寫檔；wall FPS 包含全部。
`READY.json` 代表模型包實際推論成功並至少有一幀偵測到人，**不代表姿態已經準確、所有人都偵測到或達到即時**。

## 7. 頻寬與 webcam

單機模擬：所有相機抽特徵 → 每台傳 32 維 float32 摘要 → 回傳每台 1 byte 選取 mask → 只有選中視角的完整特徵進 decoder。
成本是 tensor 的 numel × element_size；固定與全視角預先共享設定，不加每幀協商成本。
相機幾何視為已共享的靜態校正資訊，不計每幀傳送；移動相機需要另外計入更新成本。
不包含網路封包頭、重傳、初次校正、完整 RGB 顯示串流。不是 USB/網卡實測，也不是和壓縮 H.264 影片的比較。

目前的真實輸入來源是 Panoptic HD 影片或由它抽出的原解析度影像。多 webcam 仍需另加擷取、時間對齊與校正流程，不能把幾台未校正 webcam 直接接上就保證 3D 正確。
先完成影片 demo 的全視角基準与選擇比較，再用固定相機、單人、慢動作驗證 webcam，最後才做移動相機。

## 8. 清理與驗證

刪除範圍僅 `graduate/src/260915`；保留根 LICENSE、上游程式授權與必要 tpose。
刪除前完整備份位於工作區 `mvp_backups/`，逐檔 SHA256 驗證。`cleanup_manifest.json` 記錄 72 個刪除檔與備份位置。
`prune_audit.json` 記錄清理時的 Python 依賴分析。沒有修改其他日期的實驗或兩份原始 clone。

```powershell
python -m unittest discover -s tests -v
```

測試使用合成影片／標註／小型 tensor，檢查相機數變動、集合順序不變性、訓練與模型包契約、座標與成本。
本機沒有實際資料、完整姿態權重與 CUDA，因此尚未完成真實資料的訓練、demo 視覺品質與效能驗證。

## 舊資料相容性與搬機

詳見 [DATA_HANDOFF.md](DATA_HANDOFF.md)。`run/audit_data.py` 可以不依賴 torch/CUDA 檢查 MP4、兩種 JPG 路徑、標註 TAR 與影格範圍。另提供 `configs/train_ultimatum_pilot.json`；驗證序列刻意留空，必須填入實際獨立序列才可訓練。
