# Yêu cầu sản phẩm — RecSys-VNPT

Những gì hệ thống gợi ý trang chủ MyTV phải giao, và các ràng buộc hệ thống phải
tuân theo. Thiết kế kỹ thuật nằm trong `ARCHITECTURE.md`, định nghĩa chỉ số
trong `METRICS.md`, vận hành trong `OPERATIONS.md`, các bước tích hợp cho VNPT
trong `TODO.md`.

## 1. Bối cảnh

MyTV là nền tảng truyền hình internet và streaming của VNPT, cung cấp clip, phim,
âm nhạc, thể thao, karaoke, nội dung thiếu nhi và các nội dung VOD khác. Dự án này
xây dựng trang chủ cá nhân hóa: với mỗi profile, hệ thống quyết định hiển thị
những hàng nào và, trong mỗi hàng, nội dung nào được xếp trước.

Đây là hệ thống gợi ý **không tuần tự (non-sequential), dựa trên phản hồi ngầm
(implicit feedback)**. Hệ thống chấm điểm từng cặp (profile, nội dung) từ các
thống kê hành vi theo thời gian, metadata nội dung và các target được thiết kế
riêng. Hệ thống không dự đoán item tiếp theo trong một phiên xem, và khác biệt này
chi phối mọi lựa chọn phía sau: label, feature, cách chia split, chỉ số.

## 2. Hợp đồng output

Bắt buộc, được kiểm tra ở mọi lượt chạy.

| Yêu cầu | Giá trị |
|---|---|
| Định danh gợi ý | `profile_id` |
| Số hàng thường mỗi profile | đúng 20 |
| Số nội dung mỗi hàng thường | từ 10 đến 30, phân bổ theo từng hàng |
| Hàng đặc biệt | `112,109,108,110` nối vào cuối theo đúng thứ tự, không kèm nội dung |
| Nội dung trùng trong một hàng | không được phép |
| Nội dung trùng giữa các hàng | xem §3 |

Số nội dung mỗi hàng không cố định trước: mỗi hàng được đảm bảo 10 nội dung, sau
đó phần ngân sách còn lại của trang được phân dần cho hàng có candidate kế tiếp
mạnh nhất, tối đa 30. Hàng hết candidate tốt sẽ được để ngắn hơn thay vì bị lấp
bằng nội dung yếu.

### File bàn giao

Ba file, cùng định dạng với các fixture trong `rs_output/`:

| File | Nội dung |
|---|---|
| `f_personalized_homepage_rule_ms_<domain>_0_700000_DL_MyTV_<date>.json` | mỗi dòng một bản ghi `{pid, d}`, `d` = các hàng theo thứ tự phục vụ |
| `f_personalized_rule_content_ms_0_700000_DL_MyTV_<date>.csv` | `pid\|rule,type#content$type#content;rule,...` |
| `f_profile_grouping_ms_DL_MyTV_0_<date>.json` | các bản ghi `{pid, mid, gid}` |

`<date>` (ngày cutoff dữ liệu của batch) và dải `0_700000` chỉ là định danh batch
mang tính hình thức. Trên production, ba file được xuất bản thành một generation
bất biến và được commit qua `CURRENT.json` (`OPERATIONS.md` §7). Lệnh nghiên cứu
`live-test` ghi cùng ba file này, kèm cặp file tương thích `rulename.json` /
`rule_content.txt` trong `<domain>/result_v2/`.

### Ngữ nghĩa định danh

| Trường | Ý nghĩa | Nguồn |
|---|---|---|
| `pid` | profile_id — định danh gợi ý | log hành vi |
| `mid` | member_id — thuê bao sở hữu profile | join từ `profile_mapping` |
| `gid` | nhóm A/B test | **không có trong dữ liệu được giao** |

Một thuê bao có một username và một member_id nhưng có thể sở hữu nhiều profile;
chỉ profile mặc định dùng member id làm id của chính nó. Vì vậy `mid` **được join,
không bao giờ sao chép từ `pid`**. Trong bộ dữ liệu cung cấp để huấn luyện, hai
trường này trùng nhau ở 99.98% bản ghi — đó là đặc điểm của mẫu dữ liệu, không
phải quy tắc; đã có bảy profile trong đó khác nhau.

`gid` không có nguồn: đã kiểm tra toàn bộ 138 cột của dữ liệu duration, behaviour
và metadata người dùng. `gid` là hằng số do bên gọi truyền vào cho đến khi VNPT
định nghĩa. Cả `mid` lẫn `gid` đều không phải feature huấn luyện; hai trường này
chỉ xuất hiện lúc ghi file.

## 3. Trùng lặp giữa các hàng

VNPT yêu cầu một nội dung không lặp lại giữa các hàng, vì sự lặp lại hiện ra ngay
trên màn hình. Mặc định khi phục vụ là khử trùng lặp (de-duplication) trên toàn
trang theo chiều sâu: mọi hàng lấy nội dung tốt nhất còn lại của mình trước khi
bất kỳ hàng nào lấy nội dung thứ hai.

Có một ngoại lệ bắt buộc: hàng hết candidate mới trước khi đủ 10 nội dung được
phép dùng lại nội dung đã đặt ở hàng khác. **Mức sàn 10 nội dung là cứng; khử
trùng lặp chỉ thực hiện ở mức tốt nhất có thể.** Toàn bộ phần trùng lặp còn lại
đến từ ngoại lệ này.

Khi so sánh các chính sách khử trùng lặp, cần lưu ý `macro_content_ndcg@10` tính
điểm một nội dung đã xem một lần cho mỗi hàng chứa nó, nên chỉ số này giảm khi bỏ
lặp; nên dùng thêm các chỉ số đếm một lần như `overall_recall@page`.

## 4. Dữ liệu hệ thống cần

Dữ liệu thô được đọc từ source root, còn artifact sinh ra được ghi vào data root.
Trên production, source root là thư mục của batch đầu vào (`OPERATIONS.md` §4);
khi nghiên cứu, cả hai mặc định là `data/data` (`RECSYS_VNPT_SOURCE_ROOT`,
`RECSYS_VNPT_DATA_ROOT`).

| Đường dẫn trong source root | Vai trò |
|---|---|
| `duration/` | log playback (CSV hoặc parquet), cần các cột `profile_id`, `username`, `content_id`, `duration`, `create_date` |
| `mytv_vmp_content/<snapshot>/` | metadata nội dung |
| `tags/`, `content_tags/` | metadata tag và ánh xạ nội dung – tag |
| `behavior/` | tương tác ngoài playback, tùy chọn |
| `month_mytv_info.parquet` | metadata người dùng |
| `rule_info.parquet` | metadata rule; bản trong repository là phương án dự phòng |

Snapshot nội dung phải có các cột `content_id`, `type_id`, `content_duration`,
`content_status`, `content_publish_year`, `total_episode`,
`content_category_name`, `content_country`, `createdate` và `publish_date`;
learned content scorer đọc năm cột cuối.

`Dữ-liệu-phục-vụ-Recommend.xlsx` là từ điển schema, không phải dữ liệu theo từng
dòng, và không thể thay thế file parquet người dùng. Các thư mục canonical,
temporal, metrics và model trong `data/data` là artifact được sinh ra. Đầu vào bị
thiếu hoặc bị bỏ qua phải được báo cáo, không bao giờ được lặng lẽ bỏ qua.

## 5. Điều kiện hợp lệ của nội dung

Mọi nội dung được phân loại trước khi huấn luyện và sinh candidate:

| Trạng thái | Ý nghĩa |
|---|---|
| `trainable` | đủ metadata và tín hiệu tương tác để huấn luyện |
| `output-compatible-but-untrained` | có thể xuất hiện trong output cho client, nhưng thiếu dữ liệu để huấn luyện |
| `excluded` | không thể xếp hạng bằng mô hình; không có label đáng tin cậy hoặc thiếu metadata bắt buộc |

Thiếu `data/data/duration` khiến nội dung thành output-compatible-but-untrained.
Thiếu `data/data/mytv_vmp_content` khiến nội dung bị excluded, vì không xác định
được content family. Trong 48,406 nội dung của catalog, 33,404 thuộc ít nhất một
hàng và 15,002 không thuộc hàng nào; nhóm sau không thể đặt lên trang chủ nên bị
loại khỏi mọi pool và mọi phép đo.

## 6. Ranh giới đóng băng

Hành vi output phía client gắn với `rule_process.py`, `rule_info.parquet`,
`result.json`, `rulename.json` và `rule_content.txt`. Các file `rule_process.py`
trong `all/`, `clip/`, `movie/` và `music/` được **đóng băng và không được sửa**.
Code mới có thể gọi chúng như module tương thích; hành vi của rule được test từ
bên ngoài các file này.

## 7. Vận hành production

Đã thống nhất với VNPT cho phần tích hợp batch:

| Yêu cầu | Trạng thái |
|---|---|
| Inference hai lần mỗi ngày, retrain 48 giờ một lần, là hai job độc lập | đã triển khai: `batch_cli run-inference`, `run-retrain` |
| Đầu vào và đầu ra qua filesystem dùng chung | đã triển khai: batch đầu vào + `_READY.json`, generation bản giao + `CURRENT.json` |
| Một batch đầu vào chỉ hoàn chỉnh khi `_READY.json` đã được ghi và mọi file trong đó đều khớp | đã triển khai, fail-closed |
| Mỗi bản giao có đúng ba file `f_*` | đã triển khai; shard chỉ dùng nội bộ và được gộp lại trước khi giao |
| Inference không bao giờ huấn luyện; thiếu mô hình là lỗi, không phải lý do để fit mô hình mới | đã triển khai (`MODEL_UNAVAILABLE`) |
| Lượt chạy thất bại không bao giờ làm thay đổi bản giao đã xuất bản hay mô hình đã promote | đã triển khai: mỗi output có một điểm commit duy nhất, gate chạy trước khi promote |
| Chạy lại batch lỗi thì tiếp tục từ chỗ dừng; chạy lại batch đã xuất bản thì không làm gì | đã triển khai: tái sử dụng các shard đã xác minh |
| 1–2 triệu profile mỗi chu kỳ inference trong vòng 6 giờ | đã thiết kế cho quy mô này (chia shard, worker song song, gộp dạng streaming); **còn thiếu bước qualification trên máy chủ production** |
| Cùng một luồng code trên production, máy cục bộ và Kaggle | đã triển khai; Kaggle kiểm tra chức năng và GPU, không kiểm tra SLA |

Ngoài phạm vi giai đoạn này: API gợi ý online, DAG cho một scheduler cụ thể,
dashboard và tích hợp cảnh báo, tự động dọn dữ liệu cũ (retention), lập lịch trên
nhiều node.

## 8. Không nằm trong mục tiêu

- Không dùng mô hình chuỗi (sequence model). Theo thiết kế, đây là hệ thống gợi ý
  không tuần tự.
- Không thiết kế lại định dạng output của client.
- Không ép gợi ý những nội dung không thể có label đáng tin cậy.
- Bằng chứng xếp hạng offline không phải CTR, không phải mức tăng online, và không
  chứng minh hệ thống đã sẵn sàng cho production. Kiểm chứng online cần log
  impression và luồng A/B test.

## 9. Các vấn đề còn mở với VNPT

| Hạng mục | Trạng thái |
|---|---|
| Định nghĩa `gid` cho phân nhóm A/B | **cần VNPT cung cấp** |
| Mức trùng lặp giữa các hàng có thể chấp nhận | mặc định hiện tại nhắm tới ~0%; xem §3 |
| Năng lực xử lý ở 1–2 triệu profile | cần chạy `qualify-hardware` trên máy chủ production |
| `rename` atomic trên filesystem dùng chung của bản giao | xác nhận trong giai đoạn staging |
| Mức tăng CTR / thời lượng xem online | chưa biết cho đến khi có log impression |
