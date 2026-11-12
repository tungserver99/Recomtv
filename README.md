# RecSys-VNPT

Hệ thống gợi ý cá nhân hóa trang chủ VNPT MyTV. Với mỗi profile, hệ thống quyết
định **hiển thị 20 hàng nào** và **đặt nội dung gì vào từng hàng**, sau đó xuất ra
ba file mà client MyTV đọc.

## Cách hoạt động

```
batch đầu vào ─► chuẩn bị ─► retrieval ─► xếp hạng ─► dựng trang ─► kiểm tra ─► xuất bản
dữ liệu thô +    làm sạch,   1,000        hàng và     20 hàng,      kiểm tra    3 file sau
_READY.json      nối bảng,   candidate    nội dung    10–30/hàng,   từng trang  CURRENT.json
                 gắn label   mỗi profile              không lặp
```

- **Retrieval**: chọn 30 hàng mà profile có nhiều khả năng muốn xem nhất, mỗi hàng
  33 candidate, cộng thêm các nội dung mới nhất của từng hàng.
- **Xếp hạng**: mặc định là hai mô hình gradient boosting (XGBoost), huấn luyện
  trên lịch sử xem, xếp thứ tự các hàng và các nội dung trong từng hàng, rồi kết
  hợp với một điểm số dựa trên độ phổ biến và mức độ quan tâm của profile. Có thể
  đổi sang FCN hoặc heuristic bằng cấu hình (mục "Chạy production batch theo từng
  tổ hợp mô hình").
- **Dựng trang**: mỗi hàng có ít nhất 10 nội dung; phần còn lại của trang 600 slot
  được chia dần cho hàng đang có nội dung kế tiếp tốt nhất, và không nội dung nào
  xuất hiện ở hai hàng.
- **Xuất bản**: chỉ giao file khi mọi trang đều đạt hợp đồng output; lượt chạy
  thất bại thì bản giao trước đó vẫn được giữ nguyên.

Hệ thống vận hành bằng hai job định kỳ: **inference** chạy hai lần mỗi ngày để
tạo file; **retrain** chạy 48 giờ một lần để làm mới mô hình, và chỉ thay mô hình
đang dùng khi mô hình mới không kém mô hình đó quá 2%.

## Đầu ra

| File | Nội dung |
|---|---|
| `f_personalized_homepage_rule_ms_<domain>_0_700000_DL_MyTV_<date>.json` | 20 hàng của mỗi profile theo đúng thứ tự, kèm 4 hàng đặc biệt cố định |
| `f_personalized_rule_content_ms_0_700000_DL_MyTV_<date>.csv` | các nội dung trong từng hàng, theo đúng thứ tự |
| `f_profile_grouping_ms_DL_MyTV_0_<date>.json` | id của profile, của thuê bao (member) và nhóm A/B |

Định dạng chi tiết nằm ở `PRD.md` §2; file mẫu nằm trong `rs_output/`.

## Chạy thử trong năm phút

Chỉ dùng dữ liệu giả lập, không đụng đến dữ liệu khách hàng. Yêu cầu Python 3.10+.

```bash
pip install -r requirements.txt

# batch.json: bốn thư mục làm việc
echo '{"incoming_root": "/tmp/recsys/incoming", "work_root": "/tmp/recsys/work",
       "delivery_root": "/tmp/recsys/delivery", "model_registry": "/tmp/recsys/registry",
       "min_free_bytes": 0, "input_wait_seconds": 0}' > batch.json

python -m recsys_vnpt.smoke_fixture --incoming-root /tmp/recsys/incoming --batch-id smoke-1
python -m recsys_vnpt.batch_cli run-retrain   --config batch.json --batch-id smoke-1
python -m recsys_vnpt.batch_cli run-inference --config batch.json --batch-id smoke-1
```

Ba file sẽ xuất hiện trong `/tmp/recsys/delivery/generations/`, và
`/tmp/recsys/delivery/CURRENT.json` trỏ tới chúng. Môi trường production dùng
đúng hai lệnh này với cấu hình trong `configs/batch.example.json`.

## Chạy production batch theo từng tổ hợp mô hình

Trang có hai khâu xếp hạng: **thứ tự hàng** và **thứ tự nội dung trong hàng**.
Mỗi khâu dùng được một mô hình riêng, và cặp lựa chọn đó gọi là một tổ hợp. Mọi
tổ hợp đã đo đều chọn được bằng cấu hình, không cần sửa code.

### Bước 1. Chọn tổ hợp trong `batch.json`

```json
{
  "incoming_root": "/srv/recsys/incoming",
  "work_root": "/srv/recsys/GxEp/work",
  "delivery_root": "/srv/recsys/GxEp/delivery",
  "model_registry": "/srv/recsys/GxEp/registry",
  "combination": "GxEp"
}
```

Không ghi `combination` thì hệ thống chạy tổ hợp mặc định `GxGp`. Key nào không
ghi sẽ lấy giá trị mặc định (bảng cấu hình ở `OPERATIONS.md` §5). Trên production,
nên sao chép `configs/batch.example.json` rồi thêm key `combination`.

Tên tổ hợp gồm ký hiệu thứ tự hàng, chữ `x`, rồi ký hiệu thứ tự nội dung:

| Ký hiệu | Mô hình |
|---|---|
| `H` | heuristic: hàng theo số lượt xem trong lịch sử, nội dung theo độ phổ biến trong rule |
| `G` | GBDT trên XGBoost, chạy GPU nếu máy có |
| `Gs` | GBDT trên scikit-learn, chỉ chạy CPU |
| `F` | FCN `fcn_official`, chấm từng cặp (profile, nội dung) |
| `Fh` | FCN `hierarchical_fcn`, có thêm head chấm rule nên tự xếp được hàng |
| `Hseg` | heuristic theo nhóm thuê bao (gói cước, tỉnh, nhóm tuổi) cho profile chưa có lịch sử VOD |

Hậu tố của thứ tự nội dung:
- không có hậu tố: kết hợp RRF với heuristic, thứ hạng tính trong từng khối
  (profile, rule); riêng `F` và `Fh` dùng thẳng điểm FCN;
- `p`: như trên, nhưng thứ hạng tính trên toàn bộ candidate của profile;
- `rrf`: FCN kết hợp RRF với heuristic, thứ hạng trong khối;
- `Ep`: heuristic + GBDT + FCN, thứ hạng trên profile;
- `Rp`, `Rseg`: profile có embedding trong FCN dùng `Fp`; profile còn lại dùng
  `Gp` (`Rp`) hoặc heuristic theo nhóm (`Rseg`).

| Tổ hợp | Thứ tự hàng | Thứ tự nội dung | FCN mà `run-retrain` train thêm |
|---|---|---|---|
| `HxH` | heuristic | heuristic | không cần retrain |
| `HxG` | heuristic | GBDT, RRF trong khối | — |
| `HxF` | heuristic | điểm `fcn_official` | `fcn_official` |
| `HxFp` | heuristic | `fcn_official` + heuristic, RRF trên profile | `fcn_official` |
| `GxH` | GBDT | heuristic | — |
| `GxG` | GBDT | GBDT, RRF trong khối | — |
| **`GxGp`** (mặc định) | GBDT | GBDT, RRF trên profile | — |
| `GxF` | GBDT | điểm `fcn_official` | `fcn_official` |
| `GxFrrf` | GBDT | `fcn_official` + heuristic, RRF trong khối | `fcn_official` |
| `GxFp` | GBDT | `fcn_official` + heuristic, RRF trên profile | `fcn_official` |
| `GxEp` | GBDT | heuristic + GBDT + `fcn_official`, RRF trên profile | `fcn_official` |
| `GxRp` | GBDT | định tuyến `Fp` / `Gp` | `fcn_official` |
| `GsxGs` | GBDT scikit-learn | GBDT scikit-learn, RRF trong khối | — |
| `Gx_cpuxGx_cpu` | GBDT XGBoost trên CPU | như `GxG`, trên CPU | — |
| `FhxH` | `hierarchical_fcn` | heuristic | `hierarchical_fcn` |
| `FhxG` | `hierarchical_fcn` | GBDT, RRF trong khối | `hierarchical_fcn` |
| `FhxF` | `hierarchical_fcn` | điểm `fcn_official` | cả hai |
| `FhxFh` | `hierarchical_fcn` | điểm `hierarchical_fcn` | `hierarchical_fcn` |
| `FhxFp` | `hierarchical_fcn` | `fcn_official` + heuristic, RRF trên profile | cả hai |
| `HsegxHseg` | heuristic theo nhóm | heuristic theo nhóm | không cần retrain |
| `HsegxFp` | heuristic theo nhóm | `fcn_official` + heuristic, RRF trên profile | `fcn_official` |
| `HsegxRseg` | heuristic theo nhóm | định tuyến `Fp` / heuristic theo nhóm | `fcn_official` |

### Bước 2. Tạo batch đầu vào từ dữ liệu thật

Job batch chỉ đọc dữ liệu từ `incoming_root/<batch_id>/`. Lệnh `smoke_fixture` ở
mục "Chạy thử trong năm phút" tạo batch này từ dữ liệu giả lập (30 profile). Để
chạy trên toàn bộ dữ liệu thật, sao chép snapshot dữ liệu thô vào đó rồi ghi
`_READY.json`:

```bash
BATCH=/srv/recsys/incoming/20250221T0000
mkdir -p $BATCH
cp -r data/data/duration data/data/mytv_vmp_content data/data/tags \
      data/data/content_tags data/data/month_mytv_info.parquet $BATCH/
python -m recsys_vnpt.batch_cli make-ready --batch-dir $BATCH \
    --data-cutoff 2025-02-20T23:59:59+07:00
```

- `--data-cutoff` là thời điểm cuối của snapshot, kèm múi giờ; mỗi batch sau phải
  có cutoff muộn hơn batch trước.
- Thêm `behavior/` nếu đặt `include_behavior` là true.
- Trên Linux, `cp -al` tạo hard link thay vì sao chép, nên không tốn thêm dung
  lượng.

Với batch này, `run-inference` tạo trang cho **mọi** profile trong dữ liệu, còn
`run-retrain` train trên toàn bộ lịch sử. Muốn chạy thử với ít profile hơn, xem
mục "Chạy thử nhanh với ít profile" bên dưới.

### Bước 3. Retrain rồi inference

```bash
python -m recsys_vnpt.batch_cli preflight     --config batch.json --batch-id <batch_id>
python -m recsys_vnpt.batch_cli run-retrain   --config batch.json --batch-id <batch_id>
python -m recsys_vnpt.batch_cli run-inference --config batch.json --batch-id <batch_id>
```

- `run-retrain` luôn fit hai scorer GBDT. Nếu tổ hợp dùng FCN, job train thêm
  FCN tương ứng trên các cửa sổ train_1..3 và validation, rồi ghi tất cả vào một
  bundle. Train FCN nên chạy trên máy có GPU; `hierarchical_fcn` là mô hình train
  lâu nhất.
- Gate của `run-retrain` chấm bundle mới và bundle đang promote theo đúng tổ hợp
  đang cấu hình, rồi chỉ từ chối bundle mới nếu nó kém bundle đang promote quá
  `gate_max_regression` (2%). Gate không còn so với heuristic. Nếu bundle đang
  promote thiếu mô hình mà tổ hợp cần, bước so này được bỏ qua.
- `run-inference` nạp đúng các mô hình mà tổ hợp cần từ bundle đang promote. Nếu
  bundle thiếu mô hình, chẳng hạn vừa đổi `GxGp` sang `GxEp` mà chưa retrain, job
  dừng với `MODEL_UNAVAILABLE` (exit 20); khi đó chạy `run-retrain` với cấu hình
  mới trước.
- `HxH` và `HsegxHseg` không dùng mô hình học được, nên chạy thẳng
  `run-inference`.

Muốn thử nhanh trên dữ liệu giả lập, thêm `"combination": "<tên>"` vào
`batch.json` của mục "Chạy thử trong năm phút" rồi chạy đúng ba lệnh ở đó.

### Chạy thử nhanh với ít profile

| Muốn giới hạn | Key hoặc cờ | Mặc định |
|---|---|---|
| số profile mà `run-inference` tạo trang | `max_profiles` trong `batch.json` | 0, tức mọi profile |
| số profile test mà gate của `run-retrain` chấm | `gate_evaluation_profiles` trong `batch.json` | 5000 |
| số profile test mà `evaluate-baseline` chấm | `--max-eval-profiles` | toàn bộ split |
| số profile của `live-test` | `--num-profiles` | 1000 |

Ví dụ một lượt thử `GxEp` trên dữ liệu thật với 1,000 profile:

```json
{
  "incoming_root": "/srv/recsys/incoming",
  "work_root": "/srv/recsys/trial/work",
  "delivery_root": "/srv/recsys/trial/delivery",
  "model_registry": "/srv/recsys/trial/registry",
  "combination": "GxEp",
  "max_profiles": 1000,
  "gate_evaluation_profiles": 500
}
```

- `max_profiles` lấy mẫu cố định theo SHA-256 của `profile_id`, nên mọi lần chạy
  đều ra cùng một tập profile.
- Bản giao khi đó chỉ chứa các profile được chọn, nên chỉ dùng với
  `delivery_root` riêng cho việc thử, không dùng cho bản giao thật.
- Retrain vẫn train trên toàn bộ dữ liệu. Phần tốn thời gian nhất của nó là gate
  (giảm bằng `gate_evaluation_profiles`) và train FCN.

### Lưu ý khi thử nhiều tổ hợp

- Mỗi tổ hợp nên có `work_root`, `delivery_root` và `model_registry` riêng, như ví
  dụ ở Bước 1, để bundle và bản giao của các tổ hợp không lẫn vào nhau.
  `incoming_root` dùng chung được.
- Có thể đặt từng key thay cho `combination`; khi đã có `combination` mà một key
  lại mang giá trị khác, cấu hình bị từ chối với `CONFIG_INVALID`.

| Key | Giá trị |
|---|---|
| `rule_ordering` | `count` (H), `learned_rrf` (G), `fcn` (Fh), `segment` (Hseg) |
| `content_ordering` | `hand_sum` (H), `learned_rrf` (G), `fcn` (F), `fcn_rrf`, `ensemble` (E), `routed` (R), `segment` (Hseg) |
| `content_rank_scope` | `rule` (trong khối) hoặc `profile` (hậu tố `p`) |
| `content_fcn_family` | FCN dùng cho nội dung: `fcn_official` hoặc `hierarchical_fcn` |
| `routed_fallback` | mô hình cho profile ngoài vocabulary của FCN: `learned_rrf` hoặc `segment` |
| `gbdt_backend`, `gbdt_device` | `xgboost` / `sklearn`; `auto` / `cuda` / `cpu` |
| `fcn_epochs`, `fcn_schedule_epochs` | số epoch train FCN (2) và độ dài lịch learning rate cosine (30) |
| `max_profiles` | số profile mà inference tạo trang; 0 là mọi profile |

### Đánh giá offline cùng một tổ hợp

```bash
python -m recsys_vnpt.pipeline_cli evaluate-baseline --domain all --evaluation-split test --combination GxEp
```

Lệnh này chấm toàn bộ profile của split test; thêm `--max-eval-profiles 300` để
chạy nhanh trên một mẫu cố định.

FCN được nạp từ `data/data/model/all/v2/<family>/`, thư mục mà
`pipeline_cli train` ghi ra, hoặc từ `--fcn-model-dir <thư mục>`. Lệnh `live-test`
và các entry point `all/infer.py`, `all/eval.py` cũng nhận `--combination`.

## Đọc tiếp

| Nếu cần… | Đọc |
|---|---|
| biết chính xác hệ thống giao gì và theo quy tắc nào | [`PRD.md`](PRD.md) |
| cài đặt, lập lịch, giám sát và khôi phục hệ thống | [`OPERATIONS.md`](OPERATIONS.md) |
| tích hợp tại VNPT theo từng bước | [`TODO.md`](TODO.md) |
| hiểu cách hệ thống hoạt động bên trong | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| tra đặc tả chi tiết từng phần: dữ liệu, label, retrieval, scorer, dựng trang | [`SPEC.md`](SPEC.md) |
| tra định nghĩa chỉ số và lệnh đánh giá | [`METRICS.md`](METRICS.md) |
| chạy test GPU trên Kaggle | [`docs/KAGGLE_TESTING.md`](docs/KAGGLE_TESTING.md) |

## Cấu trúc repository

```
recsys_vnpt/   pipeline: chuẩn bị dữ liệu, retrieval, scorer, dựng trang, kiểm tra,
               xuất bản, và các lệnh production batch_cli
configs/       mẫu cấu hình production
tests/         bộ test (python -m pytest -q)
all/ clip/ movie/ music/   entry point theo từng domain; rule_process.py được đóng băng
experiments/   các nghiên cứu và bộ chạy test trên Kaggle
kaggle/        script bootstrap cho Kaggle
rs_output/     file output mẫu theo định dạng của client
```

## Dành cho data scientist

Pipeline nghiên cứu đánh giá mọi thay đổi trên bản sao dữ liệu cục bộ
(`data/data`, hoặc `RECSYS_VNPT_DATA_ROOT`) trước khi đưa lên production:

```bash
python -m recsys_vnpt.pipeline_cli prepare --domain all --skip-behavior
python -m recsys_vnpt.pipeline_cli build-temporal --domain all
python -m recsys_vnpt.pipeline_cli evaluate-baseline --domain all --evaluation-split test
python -m recsys_vnpt.pipeline_cli live-test --domain all --num-profiles 1000 --dry-run
```

Đánh giá và phục vụ dùng chung một bộ cấu hình, nên trang được đo chính là trang
sẽ được phục vụ. `METRICS.md` hướng dẫn cách đọc kết quả.

## Liên hệ

nguyenvietanh010104@gmail.com · luuthienvietcuong01102004@gmail.com
