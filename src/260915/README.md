# 多視角 3D 姿態估計 MVP（260915）

以預訓練 MVGFormer 為姿態模型，訓練 decoder 前的輕量視角選擇器。
訓練支援 Panoptic 同步 MP4、舊版 hdVideos/hd_00_XX/*.jpg 與官方 hdImgs；影像需保留原始 HD 影格編號。

搬到另一台電腦前請先看 [舊資料處理與搬機核對](docs/DATA_HANDOFF.md)，並執行 `run/audit_data.py`。

## 先修改設定，再一條指令訓練

編輯 `configs/train_mvp.json` 的資料根目錄、完整姿態權重、訓練／驗證序列。
相機欄位 `cameras: null` 會自動找到各序列現有視角。

```powershell
python run/train_mvp.py
```

流程：環境／資料檢查 → 固定 MVGFormer 產生組合品質標籤 → 訓練選擇器 → 驗證 → 匯出模型包 → 實際執行短版 demo。
這不是從零訓練 MVGFormer，也不是兩個模型端到端聯合訓練；姿態模型保持預訓練權重。

```powershell
python run/train_mvp.py --check
python run/train_mvp.py --resume
```

`--resume` 重用通過資料、程式、設定、權重比對的教師標籤快取，重新訓練小型選擇器；不是還原 optimizer 的精確中斷續訓。

## 使用訓練輸出做 demo

```powershell
python run/demo_mvp.py --bundle output/mvp_training/bundle/bundle.json --sequence "D:/data/panoptic/160906_pizza1"
```

若驗證結果無法支持學習式選擇优於固定 K，模型包預設採固定 K，並在報告說明。
加上 `--policy learned` 可明確比較學習式策略，`--policy all` 可比較全部視角。
`--headless` 關閉視窗；`--k` 可指定保留視角數（至少 2）。

## 驗證狀態

本機缺少 CUDA 推論環境與實際 Panoptic 訓練資料、完整權重。
已執行 CPU 單元／整合測試；不能因此宣稱已完成真實資料訓練、達到某精度或即時速度。

```powershell
python -m unittest discover -s tests -v
```

詳細環境、資料量建議、成本定義、評估與 webcam 後續步驟：[MVP.md](docs/MVP.md)。

## 精簡後目錄

- `run/train_mvp.py`：一條指令訓練與匯出。
- `run/demo_mvp.py`：影片串流推論與顯示。
- `lib/mvp_demo/`：選擇、訓練、校正、影片與通訊成本。
- `lib/models/`、`lib/core/`、`lib/mvn/`、`lib/structural/`、`lib/utils/`：MVGFormer 實際依賴；部分上游類別仍需保留以載入原 checkpoint。
- `configs/`：一份姿態模型設定、一份訓練設定。
- `tests/`：CPU 測試。
- `tpose.pt`：初始化必要的小型姿態範本，已取消 Git 忽略。
- `docs/cleanup_manifest.json`：刪除檔案清單與可還原備份位置。

## 來源與授權

姿態模型基於 [MVGFormer 官方程式](https://github.com/XunshanMan/MVGFormer)。
通訊設計受 [When2com](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_When2com_Multi-Agent_Perception_via_Communication_Graph_Grouping_CVPR_2020_paper.pdf) 啟發，並非原協定的直接重現。
集合編碼方式參考 [Deep Sets](https://arxiv.org/abs/1703.06114) 的共享編碼與不依順序的聚合概念。
保留上游 [LICENSE](LICENSE) 與原始碼授權標頭。
