# TODO: tích hợp hệ thống gợi ý tại VNPT

Những việc đội VNPT cần làm để đưa repository này lên production. Code cho mọi
bước đã có và đã chạy end-to-end trên toàn bộ dữ liệu; phần còn lại là hạ tầng,
dữ liệu upstream, lập lịch và nghiệm thu. Mỗi bước đều trỏ tới mục tương ứng của
`OPERATIONS.md`.

## 1. Trước staging

- [ ] **Máy chủ.** Chuẩn bị máy chủ retrain và máy chủ inference với Python 3.10+,
  chạy `pip install -r requirements.txt` trên cả hai; scikit-learn và xgboost được
  ghim phiên bản vì model bundle huấn luyện bằng phiên bản khác sẽ bị từ chối. Dự trù khoảng
  4.5 GB RAM cho mỗi worker inference với quy mô dữ liệu hiện tại; con số này tăng
  theo lượng lịch sử. (§2, §9)
- [ ] **Bộ nhớ dùng chung.** Tạo bốn thư mục gốc: `incoming_root`, `work_root`,
  `delivery_root`, `model_registry`. `delivery_root` phải nằm trên một filesystem
  duy nhất và thao tác `rename` trên đó phải atomic: người đọc chỉ thấy
  `CURRENT.json` cũ hoặc mới, không bao giờ thấy file dở dang. Kiểm tra điều này
  trên mount thật. (§3)
- [ ] **Cấu hình.** Sao chép `configs/batch.example.json`, đặt các thư mục gốc,
  `min_free_bytes` và `sla_seconds`; `num_shards` và `workers` lấy từ bước
  qualification bên dưới. (§5)
- [ ] **Bên cung cấp dữ liệu upstream.** Với mỗi lượt chạy, sao chép một snapshot
  đầy đủ của dữ liệu thô vào `incoming_root/<batch_id>/` và ghi `_READY.json` sau
  cùng, liệt kê mọi file kèm kích thước và SHA-256 (`batch_cli make-ready` là bản
  triển khai tham chiếu). Snapshot nội dung phải có các cột liệt kê ở §4, và
  `data_cutoff` phải tăng dần qua từng batch. (§4)
- [ ] **Bên tiêu thụ.** Phía MyTV chỉ đọc bản giao qua
  `delivery_root/CURRENT.json` và kiểm tra hash của manifest cùng từng file
  (`recsys_vnpt.export.read_current_delivery` làm đúng việc này). (§7)
- [ ] **Định danh phiên bản.** Khi triển khai không kèm `.git`, đặt
  `RECSYS_VNPT_CODE_VERSION` bằng tag phát hành. (§2)

## 2. Staging

- [ ] Chạy `python -m pytest -q` trên máy chủ staging, sau đó chạy diễn tập với dữ
  liệu giả lập (`smoke_fixture` → `run-retrain` → `run-inference`) bằng một cấu
  hình staging có thư mục gốc khác với production. (§10)
- [ ] Batch thật đầu tiên: `preflight`, `run-retrain` (kiểm tra `gate.json`),
  `run-inference`; cho phía MyTV nạp generation đó. (§6, §10)
- [ ] **Năng lực xử lý.** Chạy `qualify-hardware` với workload production, tối đa 2
  triệu profile; chọn `num_shards` và `workers` theo báo cáo. Kết luận phải là
  `pass` trước khi cam kết mục tiêu 6 giờ. Nếu thất bại vì bộ nhớ, bước kỹ thuật
  tiếp theo là stream phần lịch sử đang nằm trong serving context của mỗi worker.
  (§6, §9)
- [ ] **Chính sách khi lỗi.** Hiện tại, chỉ cần một trang vi phạm hợp đồng output
  là cả batch thất bại và bản giao trước đó tiếp tục được phục vụ. Xác nhận cách
  này chấp nhận được, hoặc quy định cơ chế bỏ qua và báo cáo. (§8)

## 3. Go-live

- [ ] **Lịch chạy.** Inference hai lần mỗi ngày trên batch mới nhất; retrain 48 giờ
  một lần, ngay sau lượt inference buổi sáng, trên cùng batch đó. (§6)
- [ ] **Giám sát.** Cảnh báo khi có exit code khác 0, khi `report.json` có
  `sla_met: false`, và khi một khung giờ không xuất bản gì; thu thập `log.jsonl`.
  (§8)
- [ ] **Lưu trữ.** Quyết định thời gian giữ các generation cũ, `work/artifacts/`,
  shard và các bundle không được promote, rồi tự động hóa việc dọn dẹp; không bao
  giờ xóa những gì hai con trỏ đang trỏ tới. (§8)
- [ ] **Diễn tập rollback.** Thực hành rollback con trỏ mô hình và con trỏ bản
  giao một lần trước go-live. (§8)
- [ ] **Giai đoạn shadow.** Chạy đúng lịch nhưng client chưa chuyển sang, kiểm tra
  mọi khung giờ đều xuất bản trong SLA, rồi mới chuyển client sang đọc
  `CURRENT.json`. (§10)

## 4. Sau go-live

- [ ] **Đo lường online.** Ghi log impression và dựng luồng A/B test. Chỉ số xếp
  hạng offline không phải CTR hay mức tăng thời lượng xem, và chưa thể khẳng định
  hai chỉ số đó cho đến khi có luồng này.
- [ ] **Dữ liệu theo thời điểm.** Cung cấp snapshot có ngày của thành viên rule và
  metadata nội dung. Hiện cả hai được điền ngược từ một snapshot muộn hơn, nên
  retrieval và đánh giá chưa phản ánh đúng dữ liệu tại từng thời điểm.
