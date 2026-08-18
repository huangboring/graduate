# Multi-View Pose Estimation (When2com)

本專案實作了基於 When2com 概念的跨視角 3D/2D 姿態估計系統 (Visibility-Aware Architecture)，具備通訊閘門與特徵融合機制。

## 環境設定

請確保使用 Python 虛擬環境並安裝必要的套件 (支援 CUDA)：
```powershell
# 啟動虛擬環境 (Windows)
venv\Scripts\activate
```

## 下載模型權重與資料

因為模型權重 (ResNet-34 骨幹) 超過 100MB，無法直接放在 GitHub 上。
請從以下雲端硬碟下載最新的最佳權重檔，並放置於 `src/260814/` 資料夾下：

* **下載連結**：`[請在此貼上你的 Google Drive 連結]`
* **檔案名稱**：`when2com_pose_best.pth`

同樣地，龐大的 CMU Panoptic Dataset 影像資料請統一放置於 `src/data/160422_ultimatum1/` 底下。

## 如何訓練 (Train)

```powershell
cd src/260814
python train.py
```
訓練預設執行 80 個 Epoch，並且具備 **Communication Loss Warm-up** 機制，前 10 個 Epoch 不會懲罰通訊，確保模型能先學會交換特徵。

## 如何推論與測試 (Demo)

```powershell
cd src/260814
python node.py
```
執行後將自動讀取 `when2com_pose_best.pth`，並視覺化輸出預測的 2D 骨架與關節點可見性。
