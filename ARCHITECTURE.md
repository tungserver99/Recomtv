# Kiến trúc

Cách hệ thống gợi ý không tuần tự (non-sequential) của MyTV được xây dựng. Yêu cầu
và hợp đồng với client nằm trong `PRD.md`; đặc tả chi tiết từng phần (feature,
hyperparameter, công thức) trong `SPEC.md`; định nghĩa chỉ số và kết quả đo trong
`METRICS.md`; vận hành production trong `OPERATIONS.md`.

## Luồng xử lý

```
dữ liệu thô  ->  canonical data  ->  temporal label  ->  candidate retrieval
             ->  reranking       ->  dựng trang      ->  file client đã kiểm tra
```

Mỗi mũi tên tạo ra một artifact riêng với hash riêng và có thể tái tạo độc lập.
Bốn entrypoint theo domain (`all`, `clip`, `movie`, `music`) là lớp bọc mỏng
quanh `recsys_vnpt/`. Các bản triển khai cũ vẫn import được để phục vụ audit và
đọc checkpoint cũ, nhưng train/eval/infer đều chuyển vào pipeline dùng chung.

## Module

| Module | Trách nhiệm |
|---|---|
| `canonical.py` | dữ liệu thô → artifact canonical, giới hạn metadata theo thời điểm (as-of) |
| `engagement.py` | target có phân cấp `engagement_v1` và hợp đồng label |
| `temporal.py` | cửa sổ cutoff, truth theo thời gian, lọc nội dung mới và nội dung đã xem |
| `retrieval.py` | affinity theo rule, các kênh candidate theo rule, ngân sách candidate |
| `rule_scorer.py` | thứ tự rule học được (gradient boosting, kết hợp RRF) |
| `content_scorer.py` | thứ tự nội dung trong hàng học được (gradient boosting, kết hợp RRF) |
| `models.py`, `model_runner.py` | các họ mô hình neural và vòng lặp chấm điểm |
| `losses.py` | hàm mục tiêu huấn luyện, gồm dạng listwise theo nhóm và dạng pairwise |
| `retrieval_metrics.py` | candidate recall, gain recall và phân bổ theo nguồn |
| `rule_assembly.py` | dựng trang: chọn hàng, phân bổ theo hàng, khử trùng lặp |
| `evaluation.py`, `metrics.py` | chỉ số xếp hạng, độ phủ và sức khỏe output |
| `metrics_store.py` | bundle chỉ số schema-v2, manifest, gate |
| `export.py` | tuần tự hóa file client, generation bản giao và `CURRENT.json` |
| `prepare.py` | nguồn thô → artifact canonical |
| `evaluation_runs.py` | đánh giá offline cho cả luồng popularity lẫn luồng neural |
| `serving.py` | serving context và dựng trang cho bất kỳ tập profile nào; `live-test` |
| `scorer_bundle.py` | scorer bundle đã lưu và con trỏ promote |
| `batch_contracts.py` | cấu hình batch, giao thức `_READY.json`, mã lỗi |
| `batch_state.py` | tài liệu trạng thái, khóa job, log có cấu trúc |
| `batch_runner.py` | các job inference, retrain và qualification; chia shard và gộp |
| `batch_cli.py` | dòng lệnh production |
| `smoke_fixture.py` | batch đầu vào giả lập cho test và diễn tập |
| `runtime.py` | helper dùng chung giữa các bước |
| `pipeline_v2.py` | dựng dữ liệu temporal, huấn luyện, và facade re-export mọi bước |
| `pipeline_cli.py`, `domain_entrypoint.py` | dòng lệnh nghiên cứu |

## Hợp đồng dữ liệu và thời gian

- Manifest nguồn, hash schema, file bị bỏ qua, xung đột định danh, tombstone nội
  dung và bản trùng canonical đều là artifact có thể audit.
- Nạp toàn bộ dữ liệu duration lưu 14,643,639 dòng thành 30 phần parquet trên 91
  phân vùng theo ngày sự kiện. Mô hình hóa temporal lọc lười (lazy) tập con khớp
  catalog thay vì nạp dữ liệu thô vào RAM.
- Các phiên bản nội dung giữ timestamp gốc; feature temporal chọn phiên bản mới
  nhất tại cutoff. Việc điền ngược từ nguồn tương lai chỉ áp dụng cho trường bất
  biến và được đánh dấu `metadata_backfill`.
- Metadata được giới hạn theo ba nhóm, không phải một allowlist duy nhất: trường
  bất biến được giữ; trường mô tả **thay đổi chậm** (`tag_names`,
  `content_category_name`, `content_country`, `content_director`,
  `content_actor`, `vod_type`) được giữ; trường **biến động** (`total_episode`,
  `content_hot`, `content_status`, `locked_level`) bị loại vì chúng mang thông tin
  sau cutoff ngược về quá khứ.
- Label được dựng trước khi áp giới hạn smoke, nên một nhóm smoke không thể xóa
  mất positive của tập test do lấy mẫu id profile và id nội dung độc lập nhau.
- Các cặp đã xem được loại khỏi dữ liệu huấn luyện và khỏi truth khi phục vụ.

## Label: `engagement_v1`

Playback quan sát được tạo ra hai target có phân cấp thay vì một label click nhị
phân:

| Target | Định nghĩa |
|---|---|
| `completion_gain` | thời lượng xem ở tập được xem lâu nhất ÷ thời lượng của một đơn vị nội dung, cắt về [0, 1] |
| `watch_time_gain` | tổng thời lượng xem, chuẩn hóa log, giới hạn trên bằng p95 chỉ tính trên dữ liệu huấn luyện |

Phiên xem được tách khi có khoảng nghỉ 30 phút; thời lượng xem là **hợp** các
khoảng thời gian, nên xem lại một đoạn không bị tính hai lần.

Các cặp chưa quan sát được lấy mẫu là **mẫu tương phản (contrastive), không phải
negative đã xác minh**: hệ thống không biết profile đã từng thấy nội dung đó hay
chưa. Vì vậy AUC, precision, F1 và log-loss chỉ mang tính tham khảo lịch sử và
không bao giờ là bằng chứng để promote. Negative quan sát được với thời lượng xem
thấp giữ trọng số 1.0; negative chưa thấy được lấy mẫu dùng 0.25, mỗi positive đi
kèm hai hard negative cùng rule, một negative khớp độ phổ biến và một negative
ngẫu nhiên đều.

Thời lượng, số lần xem và phần trăm xem của chính cặp target bị loại khỏi
feature: chúng chính là label viết ngược lại.

## Feature

Schema feature là tất định. Lúc huấn luyện, hệ thống ghi lại thứ tự các cột
feature; lúc inference, hệ thống nạp schema đó, điền `0.0` cho cột thiếu, bỏ cột
thừa và sinh ra giá trị số thực.

Các mô hình embedding dùng artifact `FeatureContract` chặt chẽ hơn, ghi lại
domain, phiên bản schema, chính sách label, họ mô hình, hash cấu hình, thứ tự và
giá trị mặc định của feature số, cùng từ vựng categorical và đa giá trị.
Inference phải nạp contract trước khi chấm điểm, ánh xạ category chưa gặp thành
`__UNK__` và giá trị thiếu thành `__MISSING__`, bỏ qua các cột chỉ có ở
inference, và dừng lỗi trước khi chấm điểm nếu họ mô hình hoặc hash cấu hình không
khớp.

## Retrieval

Chấm điểm toàn bộ catalog 48,406 nội dung cho từng profile là quá tốn kém, nên
retrieval thu hẹp xuống tối đa 1,000 nội dung mỗi profile.

1. **Affinity theo rule** chấm điểm nhanh mọi rule thường: tổng độ phổ biến của
   các nội dung trong rule (chuẩn hóa về [0, 1]) cộng log1p của số sự kiện
   profile–rule trong quá khứ.
2. Chọn `candidate_rules` rule đứng đầu (30 khi phục vụ), giữ chỗ
   `reserved_per_rule` nội dung trong mỗi rule (33). Với cùng một ngân sách, lấy
   ít rule hơn nhưng sâu hơn tìm được nhiều nội dung mà profile sẽ xem hơn so với
   40 × 10.
3. Gộp các kênh theo rule: độ phổ biến, co-watch từ các profile có lịch sử chồng
   lấn, và kênh recency giữ chỗ cho 8 nội dung mới nhất của mỗi rule (cửa sổ 30
   ngày). Kênh tag-affinity có sẵn nhưng đang tắt: kênh này chiếm chỗ của các slot
   phổ biến sâu hơn và làm giảm recall.
4. Hợp các kênh được lấp đầy tới `candidate_budget` (1,000) nội dung khác nhau.

Ràng buộc `candidate_rules × reserved_per_rule ≤ candidate_budget` giới hạn phần
giữ chỗ. Mỗi tham số trên đều là cấu hình trong CLI và batch config, và được ghi
vào run manifest. Phương án dự phòng ở tầng sâu vẫn là độ phổ biến, không phải
two-tower học được, nên chỉ số ghi `learned_retriever=false`. Co-watch dùng sparse
cosine chính xác là tùy chọn chẩn đoán, mặc định tắt.

## Xếp hạng

Thứ tự nội dung trong một hàng kết hợp hai nguồn theo thứ hạng, thay vì dùng
nguồn này thay cho nguồn kia:

```
final = rule_score + rank(incumbent formula) + w · rank(learned scorer)
```

Công thức incumbent là `normalised popularity + tag affinity + co-watch`; khi
phục vụ, kênh tag tắt nên thành phần tag affinity bằng 0. Scorer
học được gồm hai head `HistGradientBoosting` trên các gain có phân cấp, huấn luyện
trên các cửa sổ hoàn toàn trước cutoff, và dùng những feature mà incumbent chưa
từng có: metadata nội dung đọc trực tiếp từ catalog canonical, cộng các feature
chéo user×item (tag affinity có idf, affinity theo category và theo family).

Hai điểm thiết kế quan trọng:

- **RRF được tính trong từng khối (profile, rule).** Sắp thứ tự nội dung là bài
  toán trong phạm vi một hàng; nếu xếp hạng trên toàn bộ profile, phần đuôi của
  một hàng mạnh sẽ vượt lên phần đầu của một hàng yếu.
- **Sampler lấy negative từ nhóm phổ biến nhất trong chính rule của positive**
  (`rule_head`). Huấn luyện trên đúng phân phối mà mô hình sẽ xếp hạng tốt hơn hẳn
  so với lấy mẫu hard negative: hard negative khớp theo dải phổ biến đo được thấp
  hơn 31% so với chỉ dùng độ phổ biến.

Thứ tự rule dùng cùng dạng kết hợp ở cấp hàng. Vì thành phần điểm rule được giữ
nguyên, các cải tiến phía nội dung không làm thay đổi thứ tự hàng — đã xác minh
bằng khoảng tin cậy paired bootstrap đúng bằng `[0, 0]`.

## Dựng trang

`rule_assembly.py` biến các candidate đã chấm điểm thành trang được phục vụ.

**Giai đoạn sàn.** Mỗi hàng được đảm bảo `contents_per_rule` (10) nội dung. Với
`balanced_head` và `balanced_regret`, giai đoạn này chạy theo chiều sâu — mọi hàng
lấy nội dung tốt nhất còn lại trước khi bất kỳ hàng nào lấy nội dung thứ hai — nên
thiệt hại do khử trùng lặp được dàn đều thay vì dồn hết vào các hàng cuối.
`balanced_regret` còn sắp thứ tự mỗi lượt theo chênh lệch điểm giữa hai candidate
kế tiếp của từng hàng.

**Giai đoạn phần dư.** Ngân sách còn lại (600 − 20×10) được chia từng slot một
thông qua một heap chứa candidate kế tiếp của mỗi hàng, nên điểm cắt của từng hàng
hình thành từ sự cạnh tranh giữa các hàng chứ không phải một ngưỡng cố định. Độ
phức tạp là `O(budget · log rows)`.

Các chính sách khử trùng lặp nằm trong `PAGE_DEDUP_POLICIES`, nguồn duy nhất dùng
chung cho cả hai CLI và test parity: `none`, `greedy_rule_rank`, `relative_fit`,
`balanced_head`, `balanced_regret`, `tail_only`. Ở mọi chính sách, mức sàn cứng 10
nội dung được ưu tiên hơn khử trùng lặp.

## Đánh giá và promote

- Bằng chứng trên tập test dùng cùng retrieval và đúng bước dựng trang như
  inference, so với truth theo thời gian cố định.
- Chỉ số chính là `rule_ndcg@20` và `macro_content_ndcg@10` lấy trung bình macro
  theo profile. Recall/NDCG@200, nhóm OOV, phương sai điểm, đóng góp của fallback,
  thời gian, RAM/VRAM và sức khỏe output là chỉ số chẩn đoán hoặc ràng buộc cứng.
- Hash của cặp candidate và hash của truth đánh giá độc lập với nhau. So sánh mô
  hình yêu cầu cả hai khớp; so sánh retrieval được phép đổi hash candidate nhưng
  phải giữ nguyên hash truth.
- Mọi kết luận phải đúng trên **cả hai** split, với cận dưới paired bootstrap 95%
  lớn hơn 0.
- Một challenger vẫn chỉ là challenger cho đến khi qua gate; các lần thất bại được
  giữ lại làm artifact benchmark và không bao giờ di chuyển con trỏ promote.

Cả hai luồng đánh giá — popularity baseline và neural — dựng trang từ cùng bộ cờ
CLI dùng chung (`_add_scoring_arguments`) và ghi cùng cấu hình chấm điểm vào
manifest, nên số liệu của hai luồng so sánh được với nhau ngay từ thiết kế.

## Họ mô hình

| Họ | Ghi chú |
|---|---|
| `dense_repaired_baseline` | baseline dense, an toàn khi inference |
| `embedding_mlp_baseline` | embedding + MLP (trước đây là `embedding_fcn`) |
| `legacy_embedding_dcnv3` | bản xấp xỉ dùng cổng sigmoid trước đây |
| `fcn_official` | FCN bám sát bản gốc: embedding căn theo field, bố cục multi-head, Self-Mask, công thức đệ quy ECN/LCN đúng như bản gốc, dropout/norm theo nhánh, bias chéo đồng nhất, Tri-BCE |
| `ecn_official` | ablation chỉ gồm ECN |
| `hierarchical_fcn` | backbone chính thức với head riêng cho rule và cho nội dung |

Các mô hình cross chính thức dùng BF16 autocast khi phần cứng hỗ trợ vì FP16 bị
tràn số ở phép đệ quy hàm mũ; không chèn clamp hay chuẩn hóa L2 vào các phương
trình gốc. Checkpoint mới dùng schema phiên bản 3 và bị từ chối khi họ mô hình,
cấu hình hoặc feature contract không khớp.

Luồng neural **không** phải luồng phục vụ. Scorer production là độ phổ biến theo
rule cộng hai scorer gradient boosting. Lý do đã được đo: bộ dựng feature
production không tạo ra cột nào thay đổi giữa các candidate của cùng một profile,
nên mô hình neural không thể xếp hạng trong phạm vi một profile, bất kể kiến trúc.

## Hệ thống batch production

Hai job dùng chung một batch đầu vào và một cấu hình (`OPERATIONS.md`).

**Fit và apply tách rời.** Mỗi scorer học được có một bước fit trả về mô hình (hai
head gradient boosting cùng chỉ mục rule hoặc bảng ánh xạ mức category dùng khi
huấn luyện), một serving scope tính một lần cho mỗi cutoff, và một bước apply cho
bất kỳ tập profile nào. Retrain thì fit và lưu; inference chỉ apply. Rule mà mô
hình đã lưu chưa từng thấy vẫn giữ mã category thiếu thay vì bị rơi khỏi trang.
Các lệnh nghiên cứu vẫn fit tại chỗ qua cùng các hàm này, nên trang của chúng
giống từng byte so với trước khi tách.

**Bundle chỉ được promote qua gate.** Retrain ghi một bundle bất biến (cả hai
scorer, SHA-256 của payload, phiên bản thư viện), đánh giá trên cửa sổ test theo
thời gian bằng đúng code đánh giá đã tạo ra bằng chứng offline, và chỉ di chuyển
`promoted_model.json` nếu bundle thắng thứ tự heuristic (hand-built) và nằm trong
ngưỡng cho phép so với bundle đang được promote. Con trỏ trước đó được giữ lại để
rollback.

**Các profile độc lập nhau, nên tập profile được chia shard.** Mọi thứ mà một trang
phụ thuộc, ngoại trừ chính profile — dữ liệu canonical, chỉ mục retrieval (thứ
hạng catalog và bảng tra co-watch), độ phổ biến của rule, scope của các scorer —
được nạp một lần cho mỗi process dưới dạng `ServingContext`. Khi đó, chi phí của
một shard chỉ tỷ lệ với số profile của chính nó. Shard được gán theo hash của
`profile_id`, được kiểm tra đầy đủ, và được ghi với bản ghi thành công ở cuối
cùng, nên lần chạy lại chỉ làm lại những shard chưa xác minh được. Đã xác minh
trên toàn bộ dữ liệu: các trang dựng qua ba shard xen kẽ có hash giống hệt một
lượt chạy duy nhất.

**Mỗi output có một điểm commit.** Các shard được gộp bằng cách stream theo thứ tự
profile; bước này đồng thời từ chối profile bị thiếu, bị lặp hoặc lệch hàng, và
làm cho file giống từng byte với mọi số shard hay số worker. File đã gộp được đặt
vào thư mục stage, trở thành một generation bất biến, được kiểm tra lại, và chỉ
khi đó `CURRENT.json` mới được di chuyển. Phía sau không nhìn thấy gì trước thời
điểm này.

## Artifact chỉ số

Mỗi lượt chạy ghi một bundle schema-v2 trong
`data/data/metrics/<run_id>/<domain>/`: manifest, data, training, evaluation,
inference, output, gate và một báo cáo Markdown. Schema checkpoint (v3) và schema
chỉ số (v2) độc lập với nhau.

```powershell
python -m recsys_vnpt.metrics_cli show --run latest --domain all
python -m recsys_vnpt.metrics_cli compare --baseline <run> --candidate <run> --domain all
python -m recsys_vnpt.metrics_cli gate --run <run> --baseline <run> --domain all --phase <phase>
python -m recsys_vnpt.metrics_cli report --run <run> --domain all --output reports/METRICS_LATEST.md
```
