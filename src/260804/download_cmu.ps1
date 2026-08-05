# 下載 CMU Panoptic Dataset 微型測試集 (160422_ultimatum1) 的 PowerShell 腳本
# 執行方式：在 PowerShell 視窗中輸入 .\download_cmu.ps1

$dataset = "160422_ultimatum1"
$baseDir = ".\data\$dataset"

Write-Host "正在建立資料夾結構: $baseDir\hdVideos"
New-Item -ItemType Directory -Force -Path "$baseDir\hdVideos" | Out-Null
Set-Location -Path $baseDir

$endpoint = "http://domedb.perception.cs.cmu.edu/webdata/dataset/$dataset"

Write-Host "開始下載相機空間校正參數 (Calibration)..."
Invoke-WebRequest -Uri "$endpoint/calibration_$dataset.json" -OutFile "calibration_$dataset.json"

Write-Host "開始下載 3D 關節點解答 (Ground Truth)..."
Invoke-WebRequest -Uri "$endpoint/hdPose3d_stage1_coco19.tar" -OutFile "hdPose3d_stage1_coco19.tar"

# 這裡我們只下載 Camera 00, 01, 02 來節省空間與時間
$cams = @("00", "01", "02")
foreach ($c in $cams) {
    $fname = "hd_00_$c.mp4"
    Write-Host "開始下載高畫質影片 $fname ..."
    Invoke-WebRequest -Uri "$endpoint/videos/hd_shared_crf20/$fname" -OutFile "hdVideos\$fname"
}

Write-Host "下載完成！"
Write-Host "請使用您的 convert_mp4_to_image 腳本將 hdVideos 資料夾內的 MP4 抽成圖片。"
Write-Host "強烈建議設定 each_x_frame=30 (每秒抽1張) 來保護您的硬碟空間！"
