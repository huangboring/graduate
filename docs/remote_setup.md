# 在另一台主機部署

本 Git 儲存庫只包含選擇器、local-aux 改版後端、資料介面、測試與說明。資料集、SMPL-H、預訓練權重、影片、環境及 runs 不提交。

## 先確認版本能力

- local-aux 不讀目標 B 的 tracker，沿用上游空間／時序融合元件與 SMPL-H 輸出。
- **目前 local-aux 從頭初始化，尚未實作官方 checkpoint 的部分權重移植。**下載原版 model.pth 後不會自動用於 local-aux；不能將它改名成 best.pt。
- 原版 `--backend egoexo` 可以載入官方 checkpoint，但需要 B tracker，資料契約不同。
- 目前沒有真實 EgoHumans 訓練成果；單元測試包含明確的生成資料與 fake FK，只證明程式串接。
- 原始影片需要先產生獨立影像偵測框與 ViTPose/DINO 特徵；目前不是 raw dataset 一鍵訓練流程。

## 取得程式

以下在 Linux GPU 主機示範，使用既有 graduate 儲存庫。

```bash
mkdir -p ~/research
cd ~/research
git clone --recurse-submodules https://github.com/huangboring/graduate.git
cd graduate
```

EgoExoMoCap 放在 `external/EgoExoMoCap`，由 Git submodule 固定在 `99576ed5805d3b81b59e60ec26b809210364bafa`；預設 `--repo external/EgoExoMoCap`。固定版本另記錄於 `configs/upstream.lock.json`。若修改上游，必須記錄差異；local-aux 訓練 checkpoint 亦會驗證關鍵來源檔及人體模型雜湊。

如果另一台主機已經 clone 過 graduate，先確認沒有要保留的未提交變更，再執行：

```bash
git pull --ff-only
git submodule update --init --recursive
```

## 環境與驗證

使用獨立環境，Python >=3.10。先依主機 NVIDIA driver/GPU 安裝相容的 CUDA PyTorch，再安裝本專案。不要將本機 CPU-only 環境直接複製過去。

```bash
python -m pip install -e '.[data]'
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

本專案的基本依賴不涵蓋整個影像前處理與人體模型環境。正式重建另需依上游說明安裝 `human_body_prior`；影像特徵需要 `transformers`、DINOv3 程式及其依賴。

```bash
python -c "from human_body_prior.body_model.body_model import BodyModel; print('BodyModel import OK')"
```

測試通過不代表 SMPL-H 已下載、CUDA 可用或資料格式已準備完成。

## 建議磁碟位置

```text
~/research/
├── graduate/                     # 本儲存庫，程式直接在根目錄
│   ├── data/                    # 私有 manifest、protocol、prepared replay（忽略）
│   ├── runs/                    # 訓練輸出（忽略）
│   └── external/EgoExoMoCap/     # 固定版本的官方程式
│       ├── model_zoo/           # 官方 checkpoint
│       └── third_party/
│           ├── body_models/smplh/neutral/model.npz
│           ├── vitpose_hf/vitpose-base-simple/
│           └── dinov3_repo/checkpoints/
└── datasets/EgoHumans/           # 原始資料，完全不進本儲存庫
```

manifest 中的資料路徑需改成新主機實際位置；已生成的前處理 provenance receipts 記錄了輸入絕對路徑，不能直接把 Windows 路徑搬到 Linux 後假設仍可驗證。建議在新主機執行前處理與匯出，保留真實來源與雜湊。不要修改 receipt 來跳過輸入驗證。

下載入口：

- [EgoHumans 官方資料](https://drive.google.com/drive/folders/1JD963urzuzV_R_6FOVOtlx8UupwUuknR)
- [EgoExoMoCap 官方 model_zoo](https://drive.google.com/drive/folders/1nID3fKYi7HudVeT5kY2mlA9imxTAeBJG)
- [SMPL-H 註冊與下載](https://mano.is.tue.mpg.de/)
- [ViTPose HF 檔案](https://huggingface.co/usyd-community/vitpose-base-simple/tree/main)
- [DINOv3 程式及模型取得說明](https://github.com/facebookresearch/dinov3)

## 訓練前的入口

```bash
python -m observer_selection train-reconstructor --help
python -m observer_selection export-local-egohumans --help
python -m observer_selection check-data --protocol data/protocol.json
```

完整資料欄位與兩階段訓練指令見 [local_aux_v4.md](local_aux_v4.md)。GPU 訓練需在 train-reconstructor 與 train 指令都加 `--device cuda`；預設是 cpu。若要先使用官方預訓練權重作適配初始化，請先完成該移植功能，不要誤把目前的從頭訓練當成預訓練微調。

兩個遠端候選使用 k=1；要一次取 k=2，需至少三個遠端候選才能學習選擇。正式 train/validation/test 必須用不同場景。
