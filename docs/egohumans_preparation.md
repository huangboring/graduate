# EgoHumans：因果資料準備與重建介面

這份實作支援把已準備的 EgoHumans wearer/observer pairs 轉為選擇器回放資料，不代表已完成下載、真實資料訓練或全部原始影像處理。原始碼目前未附資料、模型權重或 SMPL-H，實際準確度必須補齊資產後驗證。

## 為什麼不能直接沿用官方所有輸出

查閱本地 `EgoExoMoCap/data_prep/egohumans/extract_egohumans_pairs_stage1.py` 可見：

- `wearer_ego.npz` 也來自 GT SMPL FK，不能因檔名含 ego 就宣稱感測器量測。
- `bbox.npz` 是從 GT 關節投影產生。
- `data_prep/shared/extract_egonet_bboxes.py` 透過原資料載入器，以完整離線視窗產生所有時間位置的框；名稱為 `bbox_egonet` 並不證明它沒有使用未來資料。
- 原載入器的 floor 正規化使用 GT 腳部高度；本轉換器不調用該載入器。
- 世界座標為 `aria01_local`。不能不檢查就宣稱是模型需要的 z-up 座標。

因此本轉換器**不接受直接把 `vitpose[_egonet].npz` 改名當作本機觀測**。它要求由獨立影像偵測/追蹤得到的目標框重新提取特徵，且留下時間與檔案雜湊紀錄。

## 1. 每個 observer pair 需要哪些原始準備檔

沿用官方 stage1/stage2-video 輸出：

```text
pair/
  meta.json
  timestamps.csv
  observer.npz
  observer_full.mp4
  wearer_smplh_fk.npz  # 僅讀 joints_pos 作姿態參考
```

`meta.json` 必須有 `source_dataset=egohumans`, `source_scene`, `source_subseq`, `wearer_aria`, `observer_aria`, `world_frame`, `n_frames`, `t_start_ns`, `dt_ns`。

`timestamps.csv` 是 `frame_idx,t_ns`。所有 observer pairs 與 target tracker 必須完全相同，不猜幀、不最近鄰配對、不取最短共同長度。這是資料轉換的前提；若原始資料未同步，必須先建立有紀錄的同步流程。

## 2. 準備獨立的因果目標框

這版不擅自指定新的偵測器。使用你採用的本機人物偵測/追蹤方法，針對同一已知 target identity 輸出：

```text
detector_boxes.npz:
  timestamps_ns      [T] int64
  last_input_t_ns    [T] int64：該輸出實際用到的最新影像時間
  bbox              [T,4] xyxy pixels
  bbox_valid        [T] bool
```

每列 `last_input_t_ns <= timestamps_ns`。不可使用雙向追蹤、未來平滑或 GT 身體姿態產生框後，僅修改時間標記。

偵測程式應同時產生 `detector_receipt.json`：

```json
{
  "crop_source": "local_image_detector",
  "reference_pose_used": false,
  "output_sha256": "detector_boxes.npz 的真實 SHA256",
  "detector_checkpoint": "實際模型版本或權重 SHA256",
  "identity_source": "local_tracking"
}
```

`identity_source` 可為 `local_tracking` 或 `manual_initial_identity`；後者只表示初始指定目標身分，不表示逐幀用 GT 身體位置選框。收據是可稽核的來源契約，雜湊保證內容一致，**不可能單靠文字收據證明上游演算法誠實**；需保存真實提取程式與版本，不能手工替既有有洩漏的檔案簽收據。

## 3. 使用既有 ViTPose、DINO helper 逐幀重新提取

API 已實作：

```python
from observer_selection.egohumans import prepare_local_features

prepare_local_features(
    pair_dir,
    boxes_path,
    boxes_receipt_path,
    vit,   # 官方 ViTPosePredictor 實例，vit.batch(images, boxes)
    dino,  # 官方 DinoV3Encoder 實例，dino.batch_cls(crops)
    feature_model_ids={"vitpose": "實際checkpoint ID/digest", "dinov3": "實際checkpoint ID/digest"},
)
```

兩個 predictor 的來源是官方 `data_prep/shared/vitpose_predictor.py` 與 `dinov3_encoder.py`。需自行備齊其已知相依套件/權重並初始化；本工具不繞過權重授權，也不假裝原始圖像已提取好。

本 API 逐幀讀取 `observer_full.mp4`，用提供的當幀框呼叫既有 predictor。資料少幀、多幀、時間不合，均報錯。它輸出：

- `selector_observations.npz`：時間、最新輸入時間、2D關節/信心、框/有效性、DINO特徵。
- `selector_preparation.json`：來源檔與特徵模型記錄、輸出SHA256、提取器版本SHA256。

無效框不刪掉該幀、不把來源移出候選名單。它保留全false關節有效標記，使用已知影像範圍作正規化的無訊息佔位框，射線/DINO為0並遮罩。未選來源的當幀有效性不會提前傳入評分模型。

## 4. 設定一次明確的 target/requester/candidate 實驗

建立 export manifest；檔案路徑相對此 manifest 解析。以下是結構示例，不能照抄假雜湊或身分：

```json
{
  "schema_version": 1,
  "source_dataset": "egohumans",
  "scene_id": "04_basketball",
  "sequence_id": "實際source_subseq",
  "target_id": "aria01",
  "requester_id": "aria02",
  "source_ids": ["aria03", "aria04"],
  "pairs": {
    "aria02": "對應aria01目標與aria02觀測者pair目錄",
    "aria03": "對應aria01目標與aria03觀測者pair目錄",
    "aria04": "對應aria01目標與aria04觀測者pair目錄"
  },
  "target_tracker": {
    "path": "target_tracker_with_timestamps.npz",
    "provenance": "measured",
    "sha256": "真實SHA256"
  },
  "floor": {"height_m": 0.0, "provenance": "calibrated"},
  "world_to_zup": {
    "rotation": [[1,0,0],[0,1,0],[0,0,1]],
    "provenance": "calibrated"
  },
  "num_input": 1
}
```

必須至少有兩位可選的**外部**候選者，requester 和 target 不可在候選名單內。不是每個 EgoHumans 片段都滿足這個條件；不滿足就不能在該設定下做有意義的選人訓練。不得為了湊候選者把 target 或本機重複算進去。

target tracker 是 B 的資訊，不是 A 的資訊。tracker NPZ 包含 `timestamps_ns[T]`、scalar `world_frame`、`wearer_head_pos[T,3]`、`wearer_head_R[T,3,3]`。三點版本再包含左右腕同名pos/R。來源可以是 `measured`、`estimated`；若真的採官方 GT-FK 基準，必須標 `benchmark_gt` 並在轉換與訓練都明確啟用 benchmark 選項，不能宣稱實際部署可直接取得。

`world_to_zup.rotation` 將官方世界軸轉成重建器所需的 z-up；**只有已確認世界軸一致，才可填單位矩陣**。它必須來自校正或估計，不能利用完整 GT 姿態反推。本工具同時轉換相機、tracker與姿態參考，再將明確提供的 floor 高度從 tracker z 減去，從不使用 GT feet。

## 5. 轉換與檢查

```python
from observer_selection.egohumans import export_prepared_egohumans
export_prepared_egohumans("manifest.json", "replay/sequence.npz")
```

輸出符合 replay schema，包含 scene/sequence/requester/source/target、13點偵測觀測、22點SMPL-H參考、tracker、射線與DINO。本工具丟棄原始第一幀，原因是 tracker finite difference 需要前一幀；所有欄位一起對齊 `[1:]`，並記入 `.provenance.json`。不丟棄其他幀，不靠目標GT可見性篩資料。

GT關節轉為 joint0-root-relative，與目前後端評估定義一致；不做逐幀Procrustes對齊。人體形狀不從GT偷偷傳入重建網路；後端預設中性形狀或明確供給的獨立形狀，可能與原論文指標不同，須在實驗報告聲明。

不允許覆寫已有輸出。每次輸出記錄所有輸入檔SHA256、校正、來源與明確的benchmark標記。

## 6. 資料切分與已驗證範圍

轉換器不自動替你挑 train/val/test，也不隨機切相鄰幀。應由明確 protocol 把完整 scene 分組；同一段影片的其他requester/target permutations仍必須留在同一分區。依需求另檢查同人物/同場景泛化；不能把scene holdout直接宣稱subject holdout。

已以小型測試fixture驗證轉換數值、時間、source roster、GT隔離與preprocessing來源拒絕；這些fixture不是真實EgoHumans。仍需真實 detector輸出、特徵與權重跑一段完整資料，才能聲稱真實pipeline成功。缺少任何資產或因果前處理步驟時應報錯，不以官方離線框或GT欄位補洞。

## v3多來源取用的實驗條件

轉換器保持全部合法候選供離線成本試算；每幀只取得選中集合。候選數必須大於取用數k才有固定大小子集選擇可學。例如外部候選只有aria03/aria04時，k=2即全取，不能把它寫成選人效果；可用k=1或選擇實際具有更多候選的場景，不得複製來源湊數。重訪門檻和名額由訓練CLI設定並保存於v3 checkpoint，不能用test挑門檻。
