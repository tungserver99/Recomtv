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
- **Xếp hạng**: hai mô hình gradient boosting, huấn luyện trên lịch sử xem, xếp
  thứ tự các hàng và các nội dung trong từng hàng, rồi kết hợp với một điểm số
  dựa trên độ phổ biến và mức độ quan tâm của profile.
- **Dựng trang**: mỗi hàng có ít nhất 10 nội dung; phần còn lại của trang 600 slot
  được chia dần cho hàng đang có nội dung kế tiếp tốt nhất, và không nội dung nào
  xuất hiện ở hai hàng.
- **Xuất bản**: chỉ giao file khi mọi trang đều đạt hợp đồng output; lượt chạy
  thất bại thì bản giao trước đó vẫn được giữ nguyên.

Hệ thống vận hành bằng hai job định kỳ: **inference** chạy hai lần mỗi ngày để
tạo file; **retrain** chạy 48 giờ một lần để làm mới mô hình, và chỉ thay mô hình
đang dùng khi mô hình mới đạt điểm cao hơn.

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
