# Vận hành

Cách cài đặt, tích hợp, vận hành và khôi phục hệ thống gợi ý trang chủ MyTV trên
production. Đầu ra của hệ thống được đặc tả trong `PRD.md`; cách hệ thống hoạt
động nằm trong `ARCHITECTURE.md`.

## 1. Các job

| Job | Lệnh | Tần suất | Kết quả |
|---|---|---|---|
| Retrain | `batch_cli run-retrain` | 48 giờ một lần | một scorer bundle mới, chỉ được promote nếu qua gate |
| Inference | `batch_cli run-inference` | hai lần mỗi ngày | một generation bản giao gồm ba file MyTV |
| Qualification | `batch_cli qualify-hardware` | mỗi khi đổi phần cứng | báo cáo đạt/không đạt so với SLA của chu kỳ |

Cả ba job đọc một batch đầu vào từ filesystem dùng chung và một file cấu hình JSON
cho mỗi môi trường triển khai. Inference không bao giờ huấn luyện: job này phục
vụ bundle đang được promote, hoặc thất bại với `MODEL_UNAVAILABLE`. Vì vậy, một
môi trường triển khai mới phải chạy retrain trước.

```bash
python -m recsys_vnpt.batch_cli preflight        --config /etc/recsys/batch.json --batch-id 20260923T0000
python -m recsys_vnpt.batch_cli run-retrain      --config /etc/recsys/batch.json --batch-id 20260923T0000
python -m recsys_vnpt.batch_cli run-inference    --config /etc/recsys/batch.json --batch-id 20260923T0000
python -m recsys_vnpt.batch_cli qualify-hardware --config /etc/recsys/batch.json --batch-id 20260923T0000
```

## 2. Cài đặt

```bash
git clone https://github.com/Itvc0110/RecSys-VNPT.git && cd RecSys-VNPT
python -m pip install -r requirements.txt
```

Yêu cầu Python 3.10 trở lên. Luồng batch chạy trên CPU; các wheel CUDA trong
`requirements-cuda.txt` chỉ cần cho các mô hình neural phục vụ nghiên cứu.
`scikit-learn` được ghim phiên bản vì bundle huấn luyện bằng phiên bản khác sẽ bị
từ chối thay vì được tin dùng; máy chủ retrain và máy chủ inference phải cài cùng
một bộ requirements. Khi triển khai không kèm `.git`, đặt
`RECSYS_VNPT_CODE_VERSION` bằng tag phát hành để mọi báo cáo ghi lại code nào đã
tạo ra nó.

## 3. Cấu trúc thư mục

```text
<incoming_root>/<batch_id>/          batch đầu vào, do upstream ghi (hệ thống chỉ đọc)
    duration/ mytv_vmp_content/ tags/ content_tags/ month_mytv_info.parquet
    [behavior/] [rule_info.parquet]
    _READY.json                        ghi sau cùng; điểm commit của batch
<work_root>/
    artifacts/<input_fingerprint>/     dữ liệu canonical và temporal cho mỗi đầu vào khác nhau
    batches/<batch_id>/<kind>/         status.json, log.jsonl, report.json, shards/
    models/<bundle_id>/                các scorer bundle bất biến
    locks/                             mỗi job đang chạy một thư mục
<model_registry>/
    promoted_model.json                bundle mà inference đang phục vụ
    previous_model.json                bundle trước đó, dùng để rollback
<delivery_root>/
    generations/<generation>/          ba file f_*, validation_report.json, manifest.json
    CURRENT.json                       generation đã commit; điểm commit của bản giao
```

Mỗi thư mục gốc có thể là một mount khác nhau, nhưng `<delivery_root>` phải nằm
trên một filesystem duy nhất: việc xuất bản dựa vào tính atomic của `rename` bên
trong filesystem đó. Cần xác minh điều này trên filesystem dùng chung thực tế
trong giai đoạn staging.

## 4. Giao thức đầu vào

Upstream sao chép một snapshot đầy đủ vào `<incoming_root>/<batch_id>/` và ghi
`_READY.json` **sau cùng**. Pipeline chờ file này tối đa `input_wait_seconds` và
không bao giờ suy đoán batch đã đầy đủ dựa trên timestamp.

```json
{
  "schema_version": 1,
  "batch_id": "20260923T0000",
  "data_cutoff": "2026-09-22T23:59:59+07:00",
  "created_at": "2026-09-23T00:30:00+07:00",
  "files": [
    {"path": "duration/part-00001.parquet", "size_bytes": 123456, "sha256": "<64 lowercase hex>"}
  ]
}
```

Các quy tắc, quy tắc nào cũng được kiểm tra (nếu vi phạm: `INPUT_INTEGRITY`):

- mọi file trong thư mục batch đều được liệt kê, với đúng kích thước và SHA-256;
  file không được liệt kê sẽ bị từ chối, vì các bộ đọc glob toàn bộ thư mục;
- đường dẫn theo chuẩn POSIX, tương đối, không chứa `..`, và không được thoát ra
  ngoài thư mục qua symlink;
- `data_cutoff` có múi giờ và, với inference, phải muộn hơn cutoff của bản giao
  đang được xuất bản;
- phải có `duration/`, `mytv_vmp_content/`, `tags/`, `content_tags/` và
  `month_mytv_info.parquet`. `behavior/` chỉ được đọc khi `include_behavior` là
  true. Nếu batch không kèm `rule_info.parquet`, hệ thống dùng bản trong
  repository.

Batch id có thể dùng chữ cái, chữ số và `. _ + -`. Bản triển khai tham chiếu của
bên cung cấp:

```bash
python -m recsys_vnpt.batch_cli make-ready --batch-dir /srv/recsys/incoming/20260923T0000 \
    --data-cutoff 2026-09-22T23:59:59+07:00
```

Snapshot nội dung phải có các cột `content_id`, `type_id`, `content_duration`,
`content_status`, `content_publish_year`, `total_episode`,
`content_category_name`, `content_country`, `createdate` và `publish_date`;
learned content scorer đọc năm cột cuối. Tên cột và định dạng theo các snapshot
đã giao đến nay.

## 5. Cấu hình

Một file JSON; key không xác định sẽ bị từ chối. `configs/batch.example.json` là
mẫu đầy đủ.

| Key | Mặc định | Ý nghĩa |
|---|---|---|
| `incoming_root`, `work_root`, `delivery_root`, `model_registry` | bắt buộc | bốn thư mục gốc nêu trên |
| `domain` | `all` | `all`, `clip`, `movie` hoặc `music` |
| `input_wait_seconds`, `poll_seconds` | 3600, 30 | thời gian chờ `_READY.json` |
| `min_free_bytes` | 10 GiB | từ chối khởi chạy khi dung lượng trống dưới `work_root` thấp hơn mức này |
| `include_behavior` | false | nạp thêm log hành vi ngoài playback |
| `num_shards` | 16 | số shard profile; quyết định bộ nhớ cho mỗi shard, xem §9 |
| `workers` | 1 | số shard được tính song song trên các process |
| `shard_retries` | 1 | số lần tự động chạy lại shard lỗi vì nguyên nhân tạm thời (worker chết, lỗi I/O); lỗi hợp đồng không bao giờ được chạy lại |
| `candidate_budget` | 1000 | số nội dung retrieval lấy cho mỗi profile |
| `candidate_rules`, `reserved_per_rule`, `source_candidates_per_rule` | 30, 33, 33 | số rule retrieval lấy cho mỗi profile, số nội dung giữ chỗ trong mỗi rule, và độ sâu nguồn của mỗi rule; `candidate_rules × reserved_per_rule` phải nằm trong ngân sách |
| `recency_candidates_per_rule`, `recency_reserved_per_rule`, `recency_max_age_days` | 8, 8, 30 | kênh retrieval recency: các nội dung mới nhất của mỗi rule, được giữ chỗ trong pool |
| `rule_ordering`, `content_ordering` | `learned_rrf` | `count`/`hand_sum` phục vụ thứ tự heuristic và không cần bundle |
| `rule_scorer_fusion_weight`, `content_scorer_fusion_weight` | 1.25, 5.0 | trọng số RRF của từng scorer học được |
| `gbdt_backend`, `gbdt_device` | `sklearn`, `auto` | thư viện của hai scorer (`sklearn` chỉ chạy CPU; `xgboost` chạy GPU CUDA) và thiết bị (`auto`, `cuda`, `cpu`); `auto` dùng GPU nếu có |
| `contents_per_rule_max`, `page_content_budget` | 30, 600 | trần mỗi hàng và ngân sách toàn trang |
| `page_dedup` | `balanced_regret` | chính sách khử trùng lặp toàn trang |
| `gid` | 3 | nhóm A/B ghi vào file grouping cho đến khi VNPT định nghĩa |
| `gate_max_regression` | 0.02 | mức giảm tương đối tối đa của bundle mới so với bundle đang promote |
| `gate_evaluation_profiles` | 5000 | số profile lấy mẫu cho gate retrain; mỗi lượt đánh giá chạy trong process riêng và bộ nhớ tăng theo giá trị này |
| `sla_seconds` | 21600 | ngân sách thời gian mỗi chu kỳ, được inference và qualification báo cáo |
| `qualification_workloads`, `qualification_repetitions` | 10k…2M, 2 | các mức workload tăng dần khi qualification |

Thay đổi bất kỳ cấu hình nào ảnh hưởng tới trang sẽ làm thay đổi fingerprint của
trang, nên các shard tính theo cấu hình cũ sẽ được tính lại thay vì tái sử dụng.

## 6. Chi tiết từng job

**Retrain** kiểm tra đầu vào, chuẩn bị dữ liệu canonical và các cửa sổ temporal
tương đối, fit cả hai scorer học được, rồi ghi một bundle. Sau đó job đánh giá
trên cửa sổ test theo thời gian, bằng đúng code đã tạo ra bằng chứng để promote:
thứ tự heuristic (hand-built), bundle mới, và bundle đang promote nếu có. Gate
(`work/batches/<id>/retrain/gate.json`) chỉ đạt khi cửa sổ có positive mới, bundle
mới tốt ít nhất bằng thứ tự heuristic trên `rule_ndcg@20` và
`macro_content_ndcg@10`, và không thấp hơn bundle đang promote quá
`gate_max_regression`. Đạt: con trỏ được di chuyển. Không đạt: trạng thái
`REJECTED`, exit 22, con trỏ giữ nguyên.

**Inference** kiểm tra đầu vào, chuẩn bị dữ liệu canonical (dùng chung với lượt
retrain trên cùng đầu vào), nạp bundle đang promote một lần, và chia toàn bộ
profile thành `num_shards` shard theo hash của `profile_id`. Mỗi shard được kiểm
tra theo toàn bộ hợp đồng output và được ghi với bản ghi thành công ở cuối cùng.
Sau đó các shard được gộp theo thứ tự profile thành ba file, kiểm tra thêm một lần
nữa để phát hiện profile bị thiếu, bị lặp hoặc lệch hàng, rồi xuất bản thành một
generation mới. Chạy lại một batch đã xuất bản thì không làm gì.

**Qualification** chạy các chu kỳ inference hoàn chỉnh, không xuất bản, ở mỗi mức
workload đã cấu hình, mỗi mức hai lần, và báo cáo thời gian thực, throughput, bộ
nhớ đỉnh và việc byte output có lặp lại giống nhau hay không. Kết luận chỉ là
`pass` khi workload lớn nhất thực sự được đo, nằm trong `sla_seconds` kể cả thời
gian chuẩn bị dữ liệu, cho kết quả tất định, bộ nhớ đỉnh dưới 80% RAM, và còn lại
ít nhất `min_free_bytes` dung lượng đĩa. Workload lớn hơn dữ liệu đầu vào được báo
là `not_measurable`, không bao giờ ngoại suy; exit code là 31 nếu không `pass`.

Lịch đề xuất: inference lúc 02:00 và 14:00 trên batch mới nhất; retrain hai ngày
một lần, sau lượt inference 02:00, trên cùng batch đó.

## 7. Đọc bản giao

Bên tiêu thụ phải đi qua `CURRENT.json`, không bao giờ tự liệt kê `generations/`:

1. đọc `<delivery_root>/CURRENT.json` → `generation`, `manifest_sha256`;
2. đọc `generations/<generation>/manifest.json` và kiểm tra SHA-256 của file này;
3. đọc ba file `f_*` và kiểm tra SHA-256 của từng file theo manifest.

`recsys_vnpt.export.read_current_delivery` làm đúng các bước trên. Một generation
không bao giờ bị sửa sau khi xuất bản; generation mới thay thế `CURRENT.json` một
cách atomic. Định dạng file nằm ở `PRD.md` §2. `<date>` trong tên file là ngày
`data_cutoff` của batch.

## 8. Giám sát và khôi phục

Mỗi job in ra một kết quả JSON trên stdout và thoát với mã 0, hoặc in
`{"error_code", "message"}` ra stderr và thoát với:

| Exit | Mã | Ý nghĩa | Việc cần làm |
|---:|---|---|---|
| 10 | `INPUT_TIMEOUT` | `_READY.json` không xuất hiện kịp | kiểm tra việc truyền dữ liệu từ upstream; chạy lại khi sẵn sàng |
| 11 | `INPUT_INTEGRITY` | manifest, hash, đường dẫn hoặc cutoff sai | sửa hoặc giao lại batch; không bao giờ sửa tay `_READY.json` để vượt qua |
| 12 | `SCHEMA_DRIFT` | chuẩn bị dữ liệu thô thất bại | so sánh schema của snapshot với batch trước |
| 13 | `BATCH_LOCKED` | cùng job cho batch này đang chạy | nếu chủ sở hữu khóa (`locks/<name>/owner.json`) không còn, xóa thư mục khóa đó |
| 14 | `CONFIG_INVALID` | file cấu hình hoặc batch id không hợp lệ | sửa cấu hình |
| 20 | `MODEL_UNAVAILABLE` | không có bundle promote nào dùng được | chạy retrain, hoặc rollback con trỏ |
| 21 | `RETRAIN_FAILED` | fit hoặc đánh giá scorer thất bại | xem `log.jsonl`; bundle đang promote vẫn tiếp tục phục vụ |
| 22 | `QUALITY_GATE_FAILED` | bundle mới không qua gate | xem `gate.json`; bundle đang promote vẫn tiếp tục phục vụ |
| 30 | `INFERENCE_FAILED` | một shard vẫn lỗi sau `shard_retries` lần tự động chạy lại | chạy lại; các shard đã xác minh được tái sử dụng |
| 31 | `CAPACITY` | dung lượng đĩa dưới `min_free_bytes`, hết bộ nhớ, hoặc qualification không đạt | giải phóng dung lượng, tăng `num_shards`, hoặc bổ sung bộ nhớ |
| 40 | `OUTPUT_INVALID` | một trang vi phạm hợp đồng output | không được bỏ qua; báo cáo kèm shard id |
| 50 | `PUBLISH_FAILED` | xuất bản thất bại hoặc không đọc được bản giao hiện tại | kiểm tra filesystem của bản giao |

Không có gì trước lần thay `CURRENT.json` cuối cùng hiển thị ra phía sau, nên mọi
lỗi đều giữ bản giao trước đó tiếp tục được phục vụ.

Nơi cần xem: `work/batches/<id>/<kind>/status.json` (giai đoạn, lần thử, mã lỗi và
giai đoạn bị lỗi), `log.jsonl` (mỗi dòng một sự kiện JSON, đồng thời in ra
stderr), `report.json` (thời gian, kết quả kiểm tra, `sla_met`). Cảnh báo khi exit
code khác 0, khi `sla_met: false`, và khi một lượt inference chưa xuất bản kể từ
khung giờ trước.

Rollback:

- **mô hình** — thay `promoted_model.json` bằng `previous_model.json`
  (`mv previous_model.json promoted_model.json` là atomic); lượt inference kế tiếp
  sẽ phục vụ bundle đó.
- **bản giao** — ghi một `CURRENT.json` mới trỏ tới generation cũ hơn, kèm SHA-256
  của `manifest.json` thuộc generation đó.

Việc dọn dẹp được làm thủ công: có thể xóa các `generations/` cũ, `work/artifacts/`,
`work/batches/*/*/shards/` và các `work/models/` không được promote; không bao giờ
xóa generation mà `CURRENT.json` đang trỏ tới, hay các bundle mà hai con trỏ đang
trỏ tới.

## 9. Mở rộng quy mô

Trang của mỗi profile độc lập với mọi profile khác, nên tập profile được chia
thành các shard và các byte được giao giống hệt nhau với mọi giá trị `num_shards`
và `workers`; bộ test kiểm tra tính chất này.

Bộ nhớ của mỗi worker gồm serving context (catalog, lịch sử trước cutoff, bảng tra
co-watch, cả hai scorer) cộng với candidate của một shard, khoảng 1,000 dòng cho
mỗi profile. Tăng `num_shards` cho đến khi một shard vừa bộ nhớ; tăng `workers`
cho đến khi hết core hoặc hết bộ nhớ: mỗi worker giữ context riêng, nên bộ nhớ
tăng theo `workers`. Ở quy mô dữ liệu hiện tại, mỗi worker cần khoảng 4.5 GB RAM.
Bước qualification trên máy chủ production sẽ chọn cả hai giá trị.

Giới hạn cần theo dõi là context: context giữ toàn bộ lịch sử trước cutoff và bảng
tra co-watch, nên tăng theo quy mô tập profile chứ không theo shard. Nếu
qualification ở 2 triệu profile thất bại vì bộ nhớ, hãy stream phần lịch sử vào
chỉ mục retrieval (dùng id có kiểu thay cho chuỗi, rồi xử lý out-of-core) trước
khi bổ sung phần cứng.

Đĩa: một bản giao chiếm khoảng 4.9 KB mỗi profile (khoảng 10 GB ở 2 triệu
profile), các shard chiếm thêm chừng ấy cho đến khi được dọn, và mỗi đầu vào khác
nhau tạo thêm artifact canonical trong `work/artifacts/` (1.8 GB với đầu vào hiện
tại 6.2 GB, không kèm log hành vi).

## 10. Checklist staging và go-live

1. Cài đặt trên máy chủ staging; `python -m pytest -q` phải đạt.
2. Diễn tập end-to-end với dữ liệu giả lập và cấu hình staging (thư mục gốc không
   được trùng với production, nếu không bản diễn tập sẽ trở thành bản giao hiện
   hành):
   ```bash
   python -m recsys_vnpt.smoke_fixture --incoming-root <staging incoming_root> --batch-id smoke-1
   python -m recsys_vnpt.batch_cli run-retrain   --config batch.json --batch-id smoke-1
   python -m recsys_vnpt.batch_cli run-inference --config batch.json --batch-id smoke-1
   ```
3. Xác nhận `rename` là atomic trên filesystem của bản giao, và người đọc chỉ thấy
   `CURRENT.json` cũ hoặc mới, không bao giờ thấy file dở dang.
4. Batch thật đầu tiên: `preflight`, `run-retrain` (đọc `gate.json`),
   `run-inference`; cho phía MyTV nạp generation qua `CURRENT.json`.
5. Chạy `qualify-hardware` ở các mức workload production; lưu lại báo cáo.
6. Giai đoạn shadow: chạy đúng lịch trong khi bên tiêu thụ chưa chuyển sang; kiểm
   tra mọi lượt đều xuất bản và `sla_met` luôn đạt.
7. Go-live: bên tiêu thụ đọc `CURRENT.json` sau mỗi khung giờ đã lên lịch.
8. Thống nhất với VNPT về `gid` và thiết kế A/B test online (`PRD.md` §9).

## 11. Lệnh nghiên cứu và đánh giá

Pipeline nghiên cứu (`pipeline_cli prepare`, `build-temporal`,
`evaluate-baseline`, `live-test`, `metrics_cli`) được mô tả trong `README.md` và
`METRICS.md`. `live-test` fit các scorer tại chỗ và quay về thứ tự heuristic nếu
không huấn luyện được, nên phù hợp cho diễn tập, không dùng cho production. Test
GPU trên Kaggle được mô tả trong `docs/KAGGLE_TESTING.md`.
