# EgoHumans 資料準備

## 已完成的準備資料

目前訓練直接讀取已完成資料夾的 `manifest.json`、`replay.npz`、各視角 `meta.json`／`observer.npz`。資料雜湊和完成標記會先檢查，無須因本次評分器整理而重跑已完成的 ViTPose 或身份標註。舊資料內 `local_aux_camera_rays_v1`、`local_aux_export_v1` 為保留的格式識別字串，不代表使用舊重建網路。

## 準備新的來源集合

```powershell
$py = 'D:\Ray\venvs\graduate-research\Scripts\python.exe'
& $py -m observer_selection prepare-raw `
  --sequence 'data/EgoHumans/01_tagging/001_tagging' `
  --output 'data/prepared/tagging001_transfer_34' `
  --target aria01 --requester aria02 `
  --sources aria03 aria04 cam03 cam04 `
  --annotation-boxes --device cuda `
  --model-cache 'data/prepared/tagging001_v2/model_cache'
```

`--annotation-boxes` 使用資料集身份標註決定目標人物框，屬於有標註輔助的實驗；框內關節和信心仍由姿態模型預測，不把GT關節當偵測結果。原始畫面編號是當幀 detection index，不能當成跨幀人物ID。可用 `audit-identity --sequence ... --output ... --frame 95` 輸出對照圖。

不加該旗標時會啟用本機 detector／DINO 身份追蹤，要求初始化目標。提示接受 `FRAME:BOX` 或 `f FRAME` 檢視；`n` 仍表示跳20幀。`--review-tracking` 啟用額外人工恢復檢查，不可和全自動追蹤混稱。

資料準備會使用影像整流、相機標定、逐幀姿態及特徵快取。此離線步驟可預先處理所有畫面以便比較策略；評分回放時只有選中來源的當前觀測能進入狀態字典。DINO 描述子供手動身份追蹤與原版融合對照；目前71／41維選擇器不讀這些描述子。

## 中斷與版本

相同設定與相同準備版本中斷，可加 `--resume` 重用已驗證的完整視角與逐幀快取；`--reset-view ID` 搭配resume可重做指定視角身份。程式碼／原始資料或關鍵設定改變時，準備契約會停止而不混用結果；這種未完成的舊版本需換新輸出目錄，必要時明確用 `--reuse-candidates 舊資料夾` 重用通過檢查的偵測候選快取。不要刪除原始資料或模型快取來嘗試解決契約不符。

完成但尚未導出回放，可用 `finalize-raw --prepared 準備資料夾 --output 新匯出資料夾`；此步只驗證與匯出，不重跑GPU特徵。新資料夾會包含replay、manifest及對既有視角檔案的參照。

本資料準備使用受信任的官方 EgoHumans `.npy` 標註及 SMPL 檔案，含必要的pickle讀取；訓練回放NPZ本身以 `allow_pickle=False` 讀取。不要把來源不明的pickle檔案放入資料集。
