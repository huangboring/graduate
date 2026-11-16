# v3：整批選人與避免來源長期不被取用

> 2026-11-17：選擇器與重訪設計仍沿用本文件；原版 EgoExoMoCap 後端段落只適用 `--backend egoexo`。目前預設改為不需 B tracker 的 local-aux，請見 [v4 架構與訓練](local_aux_v4.md)。

A是本機請求者，B是重建目標，C/D/E等為外部候選。選前輸入只有本機當幀/歷史觀測、真正取用過的來源記憶、合法本機幾何及時間。來源ID只索引記憶。當幀外部關節、方向、confidence、摘要與DINO不進選擇器。

## 保留的網路

需求特徵為11J+5：每關節正規化xy2、valid1、confidence與可用mask2、合法歷史投影殘差與mask3、本機2D變化率與mask3；另五維框與歷史時間資訊。J=13時148維，MLP148→128（LayerNorm/GELU）→64→GELU，再將最多四幀token送進64維GRU取得q。無歷史時數值為0、對應mask關閉，不能當作靜止或沒有遮蔽。

來源封包特徵5J+1：xy2J、valid J、confidence J、confidence mask J與上次接收間隔。J=13時66→128→64。GRUCell更新該來源64維記憶；attention替代版保留32次實際取得token，加位置編碼，以現在q查詢。未取用者不更新。沒有ID embedding或當時需求輸入。

評分器輸入q64＋來源特徵64＋幾何9＋age/seen2，共139維，MLP139→64→32→1。幾何為交疊/涵蓋/sin yaw/cos yaw四值＋四mask＋年齡；可省略。輸出為無單位logit，不是MPJPE、遮蔽機率或校準過的不確定性。

需求分支不讀backend_context或Packet.payload。prior_xy介面仍可接合法過去姿態在本機視角的投影，但本版未新增自動3D投影回饋、扇形計算或遮蔽分類器。收到不可用觀測時valid為false，不能把confidence當GT。

## 每幀一次決策與原子更新

1. 計算當前q、各來源歷史分數；完全不呼叫provider。
2. 用首次出現時間與上次實際取得時間決定本幀強制重訪集合F。
3. 在預算k內先包含F，其餘依分數Top-k。完整selected_ids固定後不再修改。
4. 只呼叫一次provider.acquire_many(selected_ids,target,timestamp)。
5. 驗證來源集合、時間、目標與所有特徵；固定後端接收整組封包。
6. 整批成功才更新所有選中者的GRU/attention及歷史；未選者記憶不動。

Decision為frozen dataclass。收到部分、重複、額外、跨目標或過期封包會失敗，任何來源都不更新；仍保留pending決策可重試/取消。同步延遲模型不支援繞過失敗直接跳下一幀。無效bbox但格式合法的回傳，是一次成功取得的「當幀不可用」觀測，會刷新時間及availability記憶。

SelectionEngine.step是無梯度推論入口，避免長期累積autograd記憶。訓練使用score/sample與commit，保留記憶梯度，於TBPTT邊界detach。

## 重訪排程

設定max_sources=k、refresh_after_s=τ、refresh_slots=r，其中1≤r≤k。預設k=2、τ=1秒、r=1，是可調工程預設，須用validation決定實驗參數。

每來源到期時間d_i：
- 未取得過：首次出現在候選名單的時間。
- 已取得過：上次成功取得時間＋τ。

當前仍在候選名單且d_i≤現在的來源，依(d_i,來源ID)穩定排序，取最前min(r,k,到期數)位作F。ID只解決相同到期時間的平手，不是語意特徵。新來源按自己的首次出現時間排隊，不會永久插到更早過期的來源前面。自然被模型選中，也會刷新同一時計。

其餘名額可包含其他到期來源；F是必須包含的集合，不是最多只能更新r位。若沒有到期者，所有名額都由模型選擇。--no-refresh令F為空，作為可能長期忽略來源的消融。

**可保證的範圍**：固定有限N位來源、一直在候選名單、每個決策都成功取用、r≥1時，一位來源到期後，最遲在接下來ceil(N/r)次服務決策內被取用。理由：排在它前面的來源每被服務一次，到期時間就移到它後面，不能無限插隊；正常Top-k服務只會加速。

若相鄰決策間隔最多Δ，保守未更新時間界可寫τ＋ceil(N/r)Δ，初次取得另以首次出現起算。τ不是無條件上限。候選持續增減、來源離線、失敗或停止決策時，不套用這個界；不透過外部當幀可見性刪掉候選來美化結果。

## 集合分布與訓練

現在是獨立評分的批次Top-k，尚未建模來源互補。k_eff=min(k,N)，合法集合為：

A_t = { S ⊆ candidates : |S|=k_eff 且 F⊆S }

對合法集合，z(S)=Σ_i∈S z_i，Pθ(S)=softmax(z(S))。共同強制項Σ_i∈F z_i在實作先移除，得到完全相同的機率，減少數值相消並令強制來源分數的直接梯度為0。這是無序固定大小集合分布，不是獨立Bernoulli，也不是逐次Plackett–Luce抽樣。

```text
e(S) = MPJPE(R(O_t, H_t, {P_i,t : i∈S}), GT_t)
L_t = Σ_{S∈A_t} Pθ(S) stop_gradient(e(S))
∂L/∂z_i = Σ_S Pθ(S) [e(S)−L] 1{i∈S}  （非強制來源）
```

e(S)必須真的聯合重建，不能相加單人誤差。SubsetCostEvaluator只在離線支線取得完整候選封包與GT；每個集合試算共用同一份實際歷史，試算不更新記憶。模型先抽取整組動作，再算成本，動作不看GT。推論取合法集合的MAP，等價於F＋剩餘來源Top-k。

凍結重建器與人體模型，梯度只到選擇網路。AdamW、clip norm1、每16幀截斷BPTT。整組被強制或全取而只剩一種合法集合時，當幀沒有選擇梯度，不單獨觸發optimizer/weight decay；收到的記憶仍可接受後續幀損失。若整段資料沒有可訓練選擇，明確報錯。

依然是給定實際歷史的當幀期望MPJPE，不包含早先離散動作的長期策略梯度，不宣稱學會了最佳探索時機。重訪是明確演算法約束。

精確計算需要C(N−|F|,k_eff−|F|)次候選重建；超過max_exact_subsets時直接報錯。線上MAP不列舉子集。未來採樣估計器需另外定義，不能靜默截斷再稱精確目標。

## 多視角後端與時序

直接呼叫官方AvatarNet與SMPL-H FK，不修改上游或用GT wrapper。B tracker為[1,W,18或54]；射線[1,W,V,39]、DINO[1,W,V,768]、mask[1,W,V]。V=1＋這個因果視窗中真正取得過資料的來源ID聯集大小。

slot0是本機A，其餘依來源ID固定排列；只在當時實際取得且有效的幀填值，其他格子全部遮罩。不是取得C這一幀就補上過去W幀的C。空間融合後再時間融合，視窗只到現在，取最後一幀；不足W幀只累積真實取得資料。資產/座標錯誤不當暖機略過。

輸出中性或獨立形狀的SMPL-H前22點、米制、z-up、joint0-root-relative；MPJPE包含joint0的零誤差、平均22點，沒有Procrustes對齊，不衡量世界位置。B tracker、floor與世界軸校正仍須合法取得。

## 未來修改接點

selection.py的SubsetPolicy提供map與distribution，兩者都能讀PreAcquisitionState（q、來源歷史、合法幾何與時間）。未來可加入集合互補評分，而不改provider、封包原子提交或重訪排程。替代策略須同時定義一致的推論MAP/訓練分布，並包含F；目前CLI與checkpoint只使用已實作的AdditiveSubsetPolicy。

若日後改成「先收到C，再決定E」，資訊時序與動作空間不同，需要新版本的協議與實驗，不能混入本版當作相同設定。

## 評估與實驗誠實性

learned/random/round_robin/last/geometry使用同預算和重訪規則，各自累積歷史；all全取單列，明示成本不同。另用--no-refresh訓練獨立消融。記錄MPJPE、來源請求數、整批呼叫數、強制重訪名額、最長未更新時間、超過門檻時間、每來源次數與從未取得者。最長未更新包含未選來源截至序列終點的等待；有限序列結束不保證所有新加入者已服務。

one-step oracle是相同歷史、相同預算、且包含F的最小集合誤差；不是取消重訪限制後的oracle，也不是長期最優。selector_expected_mpjpe_mm在其他baseline下仍是選擇網路分布期望，不是baseline策略本身期望。

沿用scene分離protocol，best.pt只看validation的scene macro MPJPE（session平均→scene平均→scene等權）；test獨立。候選數等於k時就是全取，不能聲稱學會選擇。預訓練資料重疊、真實精度、延遲與實際成本仍待資產備齊驗證。
