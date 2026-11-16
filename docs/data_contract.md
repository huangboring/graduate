# 資料契約與指令

一個NPZ是一個target/requester的連續session，allow_pickle=False。T為幀數，N為候選數，J為偵測關節數，K為重建關節數。EgoExoMoCap使用J=13、K=22。

> 本文件下方的 tracker/head-local rays 適用原版 `--backend egoexo`。v4 預設 local-aux 使用相機內外參，禁止 tracker/舊射線欄位；見 [新資料與訓練介面](local_aux_v4.md)。

| 必填key | shape或定義 |
|---|---|
| timestamps | [T]秒，有限、嚴格遞增 |
| source_ids / target_id | [N]唯一字串 / scalar字串 |
| local_xy / local_valid / local_bbox | [T,J,2]像素 / [T,J] / [T,4] xyxy |
| packet_xy / packet_valid / packet_bbox | [T,N,J,2] / [T,N,J] / [T,N,4] |
| gt_joints / gt_valid | [T,K,3]米制 / [T,K] |
| coordinate_frame / units | root_relative（joint0原點） / metres |
| input_provenance | measured、estimated、synthetic_demo、benchmark_gt_derived |

真實資料另要求scene_id、sequence_id、requester_id、joint_layout=smplh22、crop_provenance=local_detector、temporal_provenance=causal、preparation_manifest_sha256與對應.provenance.json。至少兩位外部候選，不能包含target或requester。receipt必須和NPZ雜湊相符；它記錄來源，不能單靠文字保證上游沒有洩漏。

可選confidence：[T,J]／[T,N,J]，值0..1，NaN為缺失；packet_timestamps[T,N]須等於當幀。可選prior_xy[T,J,2]、prior_valid[T,J]、prior_timestamp[T]，來源嚴格早於本幀，且已投影至A當前視角。第一個實驗可省略整組先驗；禁止使用整段離線重建偷帶未來。

可選geometry_values[T,N,4]、geometry_valid[T,N,4]、geometry_timestamp[T,N]、geometry_provenance。四值為交疊、目標涵蓋、sin/cos yaw。provenance僅local_estimate/history，history時間嚴格早於當幀；不能把當幀外部方向改標籤當本機資訊。

## 重建輸入

需要sparse_input[T,18或54]（B tracker）、packet_exo_rays[T,N,39]、packet_dino_cls[T,N,768]、packet_bbox_valid[T,N]。本機local_exo_rays[T,39]、local_dino_cls[T,768]、local_bbox_valid[T]若有須一起提供。必須聲明tracker_provenance、coordinate_convention=egoexomocap_headlocal_rays_zup、floor_provenance、floor_height_m。

Observation.backend_context只供重建，Packet.payload只在選後取用；需求與評分網路不讀這兩組欄位。離線磁碟可存所有候選封包，但不會直接進選擇網路。部署入口SelectionEngine.step不需要GT。

合成後端用packet_joints3d[T,N,K,3]、可選packet_joint_weights[T,N,K]及local_joints3d[T,K,3]；它不是真實重建替代品。

## 真實資料流程

先按[EgoHumans準備](egohumans_preparation.md)產生因果features及export manifest。以下路徑是待替換佔位值，不是已存在的訓練資料：

```powershell
python -m observer_selection export-egohumans --manifest data/export_manifest.json --output data/replay/session.npz
python -m observer_selection check-data --protocol data/protocol.json --output data/protocol_audit.json
python -m observer_selection train --protocol data/protocol.json --output runs/real_v3 --backend egoexo --repo external/EgoExoMoCap --config options/test_egohumans_syn_tracker_1pt.yaml --checkpoint model_zoo/egohumans_syn_tracker_1pt/model.pth --body-model path/to/neutral/model.npz --epochs 8 --max-sources 2 --refresh-after-s 1 --refresh-slots 1
python -m observer_selection evaluate --protocol data/protocol.json --model runs/real_v3/best.pt --output runs/real_test_v3 --backend egoexo --repo external/EgoExoMoCap --config options/test_egohumans_syn_tracker_1pt.yaml --checkpoint model_zoo/egohumans_syn_tracker_1pt/model.pth --body-model path/to/neutral/model.npz
```

config/checkpoint/body-model路徑相對--repo解析，或使用絕對路徑。真實CLI必須提供protocol。protocol僅含train/validation/test三個非空明確NPZ路徑陣列，路徑相對protocol檔案，參考configs/protocol.example.json。整個scene及其target/requester排列必須在同一分區。程式拒絕跨集合scene/相同雜湊、合成與真實混用。

若明確研究GT-tracker基準，轉換、訓練及評估都需--allow-benchmark-inputs並揭露限制。這不豁免GT框、未來資訊或GT-floor。scene切分也不保證凍結重建器預訓練資料與test不重疊，須另外查核官方checkpoint來源。

## 輸出與版本

manifest包含資料、參數、後端資產雜湊、目標定義；training.json記錄每epoch。selector.pt為最後epoch，best.pt只看validation scene macro MPJPE。checkpoint版本3、objective=expected_subset_mpjpe，拒絕v1誤差回歸與v2單來源權重；config.acquisition保存預算、重訪與精確集合上限。

evaluate驗證test與checkpoint的train/validation不重疊、後端資產相同；輸出evaluation.json及離線viewer，含source_logits、legal_subsets、subset_probabilities、selected_ids、required_refresh_ids、等待時間、需求向量、重建/GT及指標。viewer是電腦端點雲檢視，不是頭盔渲染。oracle只是各策略自身歷史下的單步診斷。

check-data主要檢查metadata、來源receipt與切分；完整數值shape、模型維度、時間與座標檢查在loader/backend。通過metadata檢查不等於已執行真實模型。


## v3批次協議

NPZ候選資料欄位不變，仍保存供離線使用的全部候選。每幀完整selected_ids先固定，再呼叫provider.acquire_many一次，回傳tuple[Packet,...]。AcquiredFrame保存packets集合。commit要求與選中集合完全一致，無重複、額外、缺件或時間錯誤；全部通過才更新任何來源。單人Packet只作單來源相容便利，不能用它代表多來源決策。

train/demo提供--max-sources（預設2）、--refresh-after-s（預設1秒）、--refresh-slots（預設1）、--max-exact-subsets（預設4096）。--no-refresh關閉重訪，應標為獨立消融。evaluate嚴格沿用checkpoint的設定。若某幀候選少於k，取全體；若所有訓練序列都如此則沒有選擇可學，訓練拒絕。EgoHumans若只有2位外部候選，固定取2位只能作全取對照；要做選擇需用k=1，或使用真實存在更多候選的片段。

後端slot0為本機，其他slot是視窗內實際取用來源的ID聯集，各幀未取用者遮罩。時間視窗中集合可改變，禁止回填從未取得的歷史。選擇、排程、取用、後端各層分開，可更換集合選法而不改資料取得界線。
