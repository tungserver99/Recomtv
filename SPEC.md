# Đặc tả hệ thống — RecSys-VNPT

Đặc tả ở mức code của từng phần: dữ liệu, label, retrieval, hai scorer, dựng trang,
model bundle, luồng neural và output. Mỗi mục ghi file và hàm tương ứng ở dòng
**Code**. Tài liệu này trả lời *hệ thống làm chính xác những gì*; lý do thiết kế
nằm trong `ARCHITECTURE.md`, hợp đồng với client trong `PRD.md`, vận hành trong
`OPERATIONS.md`, định nghĩa chỉ số trong `METRICS.md`.

Các bảng feature, hyperparameter và giá trị mặc định được
`tests/test_project_documents.py` đối chiếu với code: đổi code mà không sửa tài
liệu này thì test sẽ fail.

Quy ước dùng trong tài liệu:

- **cutoff**: mốc thời gian mà mọi dữ liệu dùng để tính toán phải có trước hoặc
  bằng nó;
- **lịch sử**: các dòng log xem có `create_date` ≤ cutoff;
- **catalog**: `contents.parquet` sau các bộ lọc ở §2.4;
- **rule thường**: mọi rule trừ bốn rule đặc biệt `112,109,108,110`.

## 1. Luồng tổng thể

1. **Prepare** (§2): dữ liệu thô → artifact canonical.
2. **Temporal** (§3, §4): chia cửa sổ thời gian, dựng lại playback, gắn label.
3. **Serving context**: nạp một lần cho mỗi process. Cutoff phục vụ là
   `create_date` lớn nhất trong log đã lọc theo catalog.
4. **Retrieval** (§5): tối đa 1,000 candidate cho mỗi profile.
5. **Xếp hạng**: rule scorer sắp thứ tự hàng (§6), content scorer sắp thứ tự nội
   dung trong hàng (§7).
6. **Dựng trang** (§8): 20 hàng, 10–30 nội dung mỗi hàng, 600 slot, khử trùng lặp.
7. **Kiểm tra và ghi file** (§11).

Job retrain chạy bước 1–2, fit hai scorer (§6, §7), đánh giá và gate (§9). Job
inference chạy bước 1 rồi 3–7 với bundle đang được promote, không fit gì.

## 2. Dữ liệu thô → canonical

**Code:** `prepare.prepare_canonical_project`, `canonical.py`, `data_sources.py`.

### 2.1 Nguồn đọc

| Nguồn | Cách đọc | Cột bắt buộc | Cột tùy chọn được dùng |
|---|---|---|---|
| `duration/` | mọi `*.csv` và `*.parquet` nằm **trực tiếp** trong thư mục (không đọc thư mục con); CSV đọc theo chunk 500,000 dòng | `profile_id`, `username`, `content_id`, `duration`, và `create_date` (nếu thiếu thì dùng `filename_date` dạng `YYYYMMDD`, chỉ chính xác tới ngày) | `member_id`, `uuid`, `date_time`, `request_id`, `recommendation`, `refer_path` |
| `mytv_vmp_content/` | mọi `*.json` ở mọi cấp thư mục, trừ thư mục `overall_content`; gốc JSON có thể là mảng object, object của các object, hoặc một object | `content_id`, và các cột ở §2.4 | xem §2.4 |
| `tags/` | file `TAG.json` có đường dẫn đứng cuối khi sắp xếp (snapshot mới nhất) | `TAG_ID`, `TAG_NAME` (không phân biệt hoa thường) | `STATUS`: nếu có, chỉ giữ tag có `STATUS` = 1 |
| `content_tags/` | file `CONTENT_TO_TAG.json` mới nhất, cùng cách chọn | `CONTENT_ID`, `TAG_ID` | — |
| `month_mytv_info.parquet` | một file | `username`, `province`, `tentinh`, `sex`, `birthday`, `package_code`; chấp nhận alias `user_name`, `province_code`, `province_name`, `gender`, `birth_date`, `product_code`/`product_id` | — |
| `rule_info.parquet` | bản trong batch nếu có, nếu không thì bản trong repository | `RULE_ID`, `LIST_TAGS_ID` | — |
| `behavior/` | chỉ khi `include_behavior` bật; mọi `*.csv` trực tiếp trong thư mục | `profile_id`, `create_date` (alias `user_name`, `event_time`) | `username`, `content_id`, `action`, `duration` |

Tên file nội dung quyết định content family: `content_movie_2…` → `movie`,
`content_clip_2…` → `clip`, `music_video_2…` → `music`. File có chữ `deleted` trong
tên là danh sách xóa (tombstone). Các file còn lại (album, short, livestream,
series, …) bị bỏ qua và được ghi vào danh sách `skipped` của lượt prepare. Với
domain khác `all`, chỉ đọc family cùng tên.

### 2.2 Định danh

- `normalize_identifier`: ép về chuỗi, bỏ khoảng trắng hai đầu, bỏ đuôi `.0…` của
  số (`6979641.0` → `6979641`). Áp dụng cho `profile_id`, `content_id`,
  `member_id`.
- `normalize_username`: như trên, sau đó chuyển về chữ thường (casefold).
- `profile_mapping.parquet`: mỗi `profile_id` nhận một `username`, là username
  xuất hiện nhiều dòng nhất với profile đó (hòa thì chọn theo thứ tự chữ cái).
  `member_id` chọn theo cùng cách, bỏ giá trị rỗng và `0`.
- Xung đột được ghi lại, không làm dừng lượt chạy: `profile_mapping_conflicts.parquet`
  (một profile nhiều username, hoặc một username nhiều profile),
  `member_mapping_conflicts.parquet`, và `username_case_variants.parquet` (username
  chỉ khác nhau về chữ hoa/thường).
- **Tập profile được phục vụ** là mọi `profile_id` trong `profile_mapping.parquet`,
  tức mọi profile có ít nhất một dòng log kèm username, kể cả profile chỉ xem live
  TV.

### 2.3 Log xem

- `interactions.parquet` giữ **mọi** dòng log (kể cả live TV), ghi thành nhiều
  part; `interactions_by_date/event_date=YYYY-MM-DD/` là bản phân vùng theo ngày
  của `create_date`.
- `duration` ép về số, lỗi thành 0; `create_date` và `date_time` parse thành
  datetime, lỗi thành rỗng.
- Mọi bước phía sau đọc log qua bộ lọc lười (polars): chỉ giữ dòng có `content_id`
  thuộc catalog. Dòng live TV (có `content_id` là id kênh) vì vậy tự bị loại.

### 2.4 Catalog nội dung

1. Gộp mọi snapshot. Mỗi snapshot mang timestamp đọc từ tên file: cụm
   `YYYYMMDD` cuối cùng, kèm `_HHMMSS` nếu có; không có thì coi là cũ nhất.
2. Mỗi `content_id` giữ bản của snapshot mới nhất.
3. Tombstone: nội dung bị loại nếu lần xóa muộn nhất không sớm hơn bản mới nhất
   của nó.
4. `tag_names` được dựng lại từ `CONTENT_TO_TAG.json` và `TAG.json`. Nội dung
   không có trong hai file này giữ `tag_names` của snapshot, nhưng không thuộc
   rule nào (§2.5) nên không lên trang.
5. Bộ lọc: `content_status` phải bằng `1` khi so dưới dạng chuỗi (giá trị `1.0` bị
   loại), có ít nhất một tag, và `content_duration` là số > 0.

Kết quả là `contents.parquet`, kèm các cột thời lượng đơn vị ở §4.2. Các cột mà
phía sau đọc: `content_id`, `type_id`, `content_duration`, `content_status`,
`content_publish_year`, `total_episode`, `content_category_name`,
`content_country`, `createdate`, `publish_date`.

`content_versions.parquet` giữ mọi phiên bản (một dòng cho mỗi cặp nội dung,
timestamp snapshot), dùng cho truy vấn metadata tại một cutoff T
(`select_content_metadata_as_of`):

- lấy phiên bản mới nhất có timestamp ≤ T;
- nếu mọi snapshot của nội dung đều sau T: lấy phiên bản sớm nhất có
  `publish_date` ≤ T, chỉ giữ trường bất biến và trường thay đổi chậm, gắn
  `metadata_backfill=true`; nội dung publish sau T không xuất hiện.

| Nhóm | Trường | Khi điền ngược |
|---|---|---|
| Bất biến | `content_id`, `content_name`, `content_family`, `type_id`, `content_duration`, `content_publish_year`, `publish_date`, `content_publish_date`, `VOD_CODE`, `product_id` | giữ |
| Thay đổi chậm | `tag_names`, `tag_ids`, `content_category_name`, `content_cate_id`, `content_country`, `content_director`, `content_actor`, `movie_cate_name`, `vod_type`, `content_single` | giữ |
| Biến động | `total_episode`, `content_hot`, `content_status`, `locked_level`, `block_platform` | xóa |

Truy vấn as-of được dùng cho phạm vi của rule scorer (§6.1) và cho feature của
luồng neural. Content scorer đọc `contents.parquet` (bản mới nhất), xem §7.1.

### 2.5 Thành viên rule

- Mỗi dòng của `rule_info.parquet` là một cặp (`RULE_ID`, `LIST_TAGS_ID`).
  `LIST_TAGS_ID` là danh sách tag id cách nhau bằng dấu phẩy; `-1` bị bỏ qua.
- Nội dung thuộc dòng đó khi có **tất cả** các tag trong danh sách. Một `RULE_ID`
  có nhiều dòng thì tập nội dung là hợp của các dòng.
- Dòng có một tag không gắn với nội dung nào thì không đóng góp nội dung nào.
- Chỉ giữ nội dung có trong catalog §2.4. Kết quả là
  `content_rule_membership.parquet`.
- Bốn rule đặc biệt `112,109,108,110` không tham gia retrieval hay xếp hạng; chúng
  chỉ được nối vào cuối output (§11).

### 2.6 Người dùng

- `birthday` dạng `YYYYMMDD` → `birth_year` nếu năm nằm trong [1900, năm tham
  chiếu], `age` = năm tham chiếu − `birth_year`; sai định dạng thì để trống và ghi
  vào `users_quarantine.parquet`. Năm tham chiếu là năm của `data_cutoff` trong
  batch, hoặc 2025 khi nghiên cứu.
- `sex` chỉ nhận 0 hoặc 1, giá trị khác để trống.
- `username` được chuẩn hóa; dòng thiếu username bị loại; trùng thì giữ dòng cuối.
- Chỉ luồng neural dùng dữ liệu người dùng; hai scorer phục vụ không đọc.

### 2.7 Artifact và cache

Thư mục canonical chứa `users.parquet`, `interactions.parquet`,
`interactions_by_date/`, `profile_mapping.parquet`, `contents.parquet`,
`content_versions.parquet`, `content_tags.parquet`,
`content_rule_membership.parquet`, `playback_events.parquet`, `sessions.parquet`,
`label_contract_v4.json`, `label_metrics.json` và `data_metrics.json`.

Cache fingerprint là hash của manifest nguồn (đường dẫn, kích thước, hash từng
file) cộng các tùy chọn (domain, giới hạn số dòng, `include_behavior`,
`target_version`, `session_gap_minutes`) và phiên bản schema 10. Fingerprint không
đổi thì artifact được tái sử dụng. Trong batch, artifact nằm ở
`work_root/artifacts/<input_fingerprint>/`, nên inference và retrain trên cùng một
đầu vào chỉ prepare một lần.

## 3. Cửa sổ thời gian và truth

**Code:** `temporal.derive_relative_temporal_windows`,
`temporal.DEFAULT_TEMPORAL_WINDOWS`, `pipeline_v2.build_temporal_project`,
`retrieval.prepare_retrieval_truth`.

Mỗi cửa sổ gồm `history_end` (lịch sử được dùng đến hết mốc này) và khoảng label
[`label_start`, `label_end`]; `history_end` là thời điểm ngay trước `label_start`.

**Tương đối** (`window_mode="relative"`, dùng trong batch): gọi D là ngày của sự
kiện cuối cùng trong log đã lọc theo catalog. Năm cửa sổ 10 ngày liên tiếp kết
thúc ở D:

| Cửa sổ | Khoảng label | Split |
|---|---|---|
| `train_1` | D−49 … D−40 | train |
| `train_2` | D−39 … D−30 | train |
| `train_3` | D−29 … D−20 | train |
| `validation` | D−19 … D−10 | validation |
| `test` | D−9 … D | test |

**Cố định** (mặc định của CLI nghiên cứu, dùng cho mọi số liệu trong
`METRICS.md`): `train_1` 2025-01-01…01-10, `train_2` 01-11…01-20, `train_3`
01-21…01-31, `validation` 02-01…02-10, `test` 02-11…02-20.

| Mục đích | Cửa sổ dùng |
|---|---|
| Fit scorer để phục vụ, hoặc để đánh giá trên test | `train_1`, `train_2`, `train_3`, `validation` |
| Fit scorer để đánh giá trên validation | `train_1`, `train_2`, `train_3` |
| Gate retrain | `test` |
| Fit label contract (§4.3) | sự kiện kết thúc đến hết ngày D−20 |

Truth dùng khi đánh giá một cửa sổ:

1. **positive thô**: mọi cặp (profile, nội dung) có label 1 trong cửa sổ; cặp trùng
   lấy gain lớn nhất (`raw_positive_pairs`);
2. **novel**: bỏ cặp mà profile đã xem trước `history_end`
   (`repeat_positive_pairs_excluded`, `novel_positive_pairs`);
3. **retrievable**: chỉ giữ nội dung thuộc một rule có ít nhất 10 nội dung catalog
   (`retrievable_positive_pairs`).

## 4. Label `engagement_v1`

**Code:** `engagement.build_playback_events`, `engagement.fit_label_contract`,
`engagement.apply_atomic_duration`, `engagement.label_engagement_window`.

### 4.1 Dựng lại playback

Chỉ dùng log thuộc catalog.

1. **Dòng hợp lệ**: `profile_id`, `content_id` khác rỗng, `create_date` parse
   được, `duration` > 0.
2. **Mốc thời gian**: `event_end` = `create_date` (thời điểm **kết thúc** lượt
   xem). `event_start` = `date_time` nếu hợp lệ và không muộn hơn `event_end`,
   ngược lại `event_end` − `duration` giây.
3. **`session_base`**: `uuid` nếu có (bỏ các giá trị rỗng, `0`, `nan`, `None`);
   thiếu thì mọi dòng của profile chung một nhóm.
4. **Khử trùng lần 1**: bỏ dòng trùng hoàn toàn trên (profile, nội dung,
   `event_start`, `event_end`, `duration`, `session_base`).
5. **Khử trùng lần 2**: trong các dòng cùng (profile, nội dung, `session_base`,
   `event_end`), giữ dòng có `duration` lớn nhất.
6. **Phiên xem**: trong mỗi (profile, `session_base`), sắp theo `event_end`; bắt
   đầu phiên mới khi khoảng cách tới dòng trước lớn hơn `session_gap_minutes`
   phút.
7. **Episode**: các dòng liên tiếp cùng một nội dung trong một phiên.
   `watch_seconds` là độ dài phần hợp của các khoảng [`event_start`, `event_end`],
   nên xem lại một đoạn không bị cộng thêm. Episode có `watch_seconds` = 0 bị bỏ.

Kết quả: `playback_events.parquet` (mỗi episode một dòng) và `sessions.parquet`.

### 4.2 Thời lượng đơn vị nội dung

Thời lượng thô = `content_duration` / `total_episode` khi `total_episode` > 1,
ngược lại = `content_duration` (giây). Giá trị này được cắt vào [lower, upper] của
nhóm (`content_family`, `type_id`) trong label contract. Giá trị không hợp lệ
(≤ 0 hoặc rỗng) thay bằng lower và đánh dấu `atomic_duration_fallback`.

### 4.3 Label contract

Fit một lần ở mỗi lượt prepare, lưu thành `label_contract_v4.json` kèm hash. Fit
cutoff = cuối ngày D−20, nên contract không nhìn thấy label của validation và
test.

| Tham số | Giá trị | Cách tính |
|---|---|---|
| `session_gap_minutes` | 30 | cố định |
| `duration_lower_seconds` | theo nhóm | phân vị 1% thời lượng đơn vị của nhóm (cần ≥ 5 nội dung, nếu không dùng giá trị toàn cục); tối thiểu 1; mặc định khi không có dữ liệu: 1.0 |
| `duration_upper_seconds` | theo nhóm | phân vị 99%, cùng quy tắc; mặc định: 86400.0 |
| `watch_cap_seconds` | theo nhóm | phân vị 95% của `watch_seconds` các episode kết thúc trước fit cutoff (cần ≥ 5 episode trong nhóm); tối thiểu 1; mặc định: 3600.0 |

### 4.4 Gain

Với mỗi cặp (profile, nội dung) có episode kết thúc trong [`label_start`,
`label_end`]:

- `completion_gain` = min(1, `max_episode_watch_seconds` / `atomic_duration_seconds`);
- `watch_time_gain` = log1p(min(`total_watch_seconds`, cap)) / log1p(cap), cắt vào
  [0, 1], với cap = `watch_cap_seconds` của nhóm;
- `label` = 1, `sample_type` = `observed_engagement`.

File cửa sổ `temporal/<cửa sổ>.parquet` còn chứa các negative lấy mẫu cho luồng
neural (4 negative cho mỗi positive, xem `ARCHITECTURE.md`). Hai scorer phục vụ chỉ
đọc các dòng `observed_engagement`.

Các cột mang thông tin của chính cặp target (`duration`, `watch_count`,
`total_duration`, `max_duration`, `max_watch_percent`, `percent_duration`,
`watch_percent`, `completion_gain`, `watch_time_gain`, `episode_count`,
`total_watch_seconds`, `max_episode_watch_seconds`) bị cấm làm feature
(`temporal.require_leakage_safe_features`).

## 5. Retrieval

**Code:** `retrieval.prepare_retrieval_index`,
`retrieval.build_personalized_candidate_pools`,
`retrieval.profile_rule_affinity_scores`, `serving.serving_retrieval_config`.

### 5.1 Cấu hình

Cột "Khi phục vụ" là cấu hình mà job batch và `live-test` dùng
(`BatchConfig` → `serving_retrieval_config`).

| Trường | Mặc định thư viện | Khi phục vụ | Ý nghĩa |
|---|---|---|---|
| `global_candidates` | 1000 | 1000 | chỉ dùng cho pool không cá nhân hóa (`build_candidate_pool`) |
| `safety_candidates_per_rule` | 50 | 50 | như trên |
| `minimum_rule_contents` | 10 | 10 | số nội dung catalog tối thiểu để một rule được retrieve |
| `minimum_normal_rules` | 20 | 20 | ít hơn số rule đủ điều kiện này thì báo lỗi |
| `candidate_budget` | 1000 | 1000 | số candidate tối đa mỗi profile |
| `candidate_rules` | 40 | 30 | số rule được chọn cho mỗi profile |
| `reserved_per_rule` | 10 | 33 | số nội dung được giữ chỗ trong mỗi rule được chọn |
| `source_candidates_per_rule` | 20 | 33 | độ sâu kênh độ phổ biến (và kênh tag) của mỗi rule |
| `cowatch_candidates_per_rule` | 0 | 0 | trần kênh co-watch mỗi rule; 0 = dùng `source_candidates_per_rule` |
| `tag_candidates_per_rule` | 0 | 0 | 0 = tắt kênh tag |
| `two_tower_candidates_per_rule` | 200 | 200 | độ sâu kênh `deep_rule_popularity` (tên trường giữ từ bản cũ) |
| `exclude_consumed` | True | True | bỏ nội dung profile đã xem |
| `exact_cowatch_enabled` | False | False | co-watch sparse cosine chính xác, chỉ để chẩn đoán |
| `exact_cowatch_candidates` | 1000 | 1000 | như trên |
| `external_candidate_mode` | rerank | rerank | chỉ dùng khi có ranking từ retriever bên ngoài (nghiên cứu hybrid two-tower) |
| `external_candidate_slots` | 100 | 100 | như trên |
| `external_rrf_weight` | 0.25 | 0.25 | như trên |
| `external_rrf_constant` | 60 | 60 | như trên |
| `external_replacement_requires_history` | True | True | như trên |
| `external_replacement_requires_witness` | False | False | như trên |
| `recency_candidates_per_rule` | 0 | 8 | số nội dung mới nhất lấy từ mỗi rule |
| `recency_reserved_per_rule` | 0 | 8 | trong số đó, bao nhiêu được giữ chỗ |
| `recency_max_age_days` | 0 | 30 | chỉ xét nội dung publish trong số ngày này trước cutoff |

Ràng buộc kiểm tra khi tạo cấu hình: `candidate_rules` ≥ `minimum_normal_rules`;
`candidate_rules` × `reserved_per_rule` ≤ `candidate_budget`; các giá trị recency
không âm; `external_candidate_slots` nằm trong [0, `candidate_budget`].

### 5.2 Chỉ mục dùng chung

Tính một lần cho mỗi cutoff và dùng chung cho mọi shard (`RetrievalIndex`):

- **popularity(c)**: số dòng log của nội dung c có `create_date` ≤ cutoff.
- **Catalog retrieval**: nội dung catalog nhân với thành viên rule thường. Rule đủ
  điều kiện là rule có ≥ `minimum_rule_contents` nội dung; ít hơn
  `minimum_normal_rules` rule đủ điều kiện thì báo lỗi.
- **`_popularity_score`** = popularity / popularity lớn nhất trong catalog.
- **`base_rule_score(r)`** = tổng popularity của các nội dung thuộc r, chia cho giá
  trị lớn nhất giữa các rule.
- **Xếp hạng trong rule**: nội dung của mỗi rule theo `_popularity_score` giảm dần,
  hòa theo `content_id`.
- **Xếp hạng recency**: nội dung của mỗi rule có `publish_date` (hoặc `start_time`)
  trong [cutoff − `recency_max_age_days`, cutoff], mới nhất trước.
- **Co-watch**: mỗi profile giữ tối đa 100 nội dung khác nhau xem gần nhất; mỗi nội
  dung trỏ tới các profile có nó trong 20 nội dung gần nhất của họ.
- **Xếp hạng toàn cục**: toàn catalog theo `_popularity_score`.

### 5.3 Candidate của một profile

1. **Chọn rule**: score(r) = `base_rule_score(r)` + log1p(số dòng log của profile
   trên nội dung thuộc r). Lấy `candidate_rules` rule cao nhất (hòa theo
   `rule_id`); chia cho điểm cao nhất được `normalized_rule_score`.
2. **Co-watch**:
   - lấy các profile láng giềng từ 20 nội dung xem gần nhất của profile, dừng khi
     đủ 200, rồi giữ 200 láng giềng đầu theo thứ tự id;
   - co_watch(c) = số láng giềng có c trong danh sách gần đây của họ, chỉ tính
     nội dung profile chưa xem;
   - chia nội dung co-watch vào các rule được chọn có chứa chúng, mỗi rule tối đa
     `source_candidates_per_rule` nội dung.
3. **Kênh của mỗi rule được chọn**, theo thứ tự ưu tiên nguồn:

   | Kênh | Lấy |
   |---|---|
   | `per_rule_popularity` | top `source_candidates_per_rule` theo độ phổ biến |
   | `profile_tag_affinity` | chỉ khi kênh tag bật |
   | `co_watch` | các nội dung co-watch của rule |
   | `catalog_recency` | top `recency_candidates_per_rule` nội dung mới nhất |
   | `deep_rule_popularity` | top `two_tower_candidates_per_rule` theo độ phổ biến |

4. **Điểm heuristic**: bỏ nội dung đã xem.
   - personalized = `_popularity_score` + tag_score + cowatch_score;
   - cowatch_score = co_watch(c) / co_watch lớn nhất;
   - tag_score = tổng tần suất các tag của nội dung trong lịch sử profile / tần
     suất lớn nhất. Khi kênh tag tắt (mặc định), bảng tag không được dựng nên
     tag_score = 0;
   - `retrieval_score` = personalized + `normalized_rule_score`.
5. **Giữ chỗ** trong mỗi rule: trước hết tối đa `recency_reserved_per_rule` nội
   dung recency, sau đó các nội dung có `retrieval_score` cao nhất, tổng cộng
   `reserved_per_rule`. Như vậy 8 slot recency nằm trong 33 slot giữ chỗ.
6. **Gắn rule**: nội dung xuất hiện ở nhiều rule được chọn giữ dòng của rule cho
   điểm cao nhất (`candidate_rule_id`).
7. **Lấp ngân sách**: nếu chưa đủ `candidate_budget`, lấy tiếp theo xếp hạng toàn
   cục (bỏ nội dung đã xem).
   - Rule gắn là rule được chọn đầu tiên chứa nội dung. Nếu không rule được chọn
     nào chứa nó, gắn vào rule có id nhỏ nhất của nó với điểm rule 0.
   - Nguồn ghi là `global_popularity_fill`.
8. **Sắp và cắt**: nội dung được giữ chỗ đứng trước, sau đó theo `retrieval_score`
   giảm dần; cắt ở `candidate_budget`.

Mỗi candidate có các cột `profile_id`, `content_id`, `retrieval_score`,
`retrieval_source` (nguồn đầu tiên), `retrieval_sources` (mọi nguồn),
`candidate_rule_id` và `rule_score`.

### 5.4 Affinity rule dùng làm incumbent

`profile_rule_affinity_scores` tính điểm cho mọi rule thường của profile theo cùng
công thức ở bước 1 (`affinity="count"`), rồi chia cho điểm cao nhất của profile.
Đây là thứ tự hàng heuristic (`rule_ordering=count`) và là thành phần incumbent mà
rule scorer kết hợp (§6.5).

## 6. Rule scorer

**Code:** `rule_scorer.py` (`fit_rule_scorer`, `train_rule_scorer`,
`rule_serving_scope`, `apply_rule_scorer`).

### 6.1 Phạm vi tại một cutoff

`membership_and_rules_as_of` giữ nội dung có metadata tại cutoff (§2.4) và vẫn
còn trong catalog hiện tại. Rule đủ điều kiện là rule thường có ít nhất 10 nội
dung như vậy.

`rule_index` là mã số nguyên cố định cho mọi rule thường, cấp theo thứ tự chuỗi
của `rule_id`, và được lưu trong model. Khi phục vụ, rule mà model chưa từng thấy
nhận mã rỗng; model coi đó là category thiếu và rule vẫn được chấm điểm.

### 6.2 Feature (18)

Tính từ các dòng log (không phải playback episode) có `create_date` ≤ cutoff trên
nội dung thuộc phạm vi. `duration` là số giây trong log; tuổi của một dòng =
(cutoff − `create_date`) tính theo ngày. Một dòng log được tính cho mọi rule chứa
nội dung của nó. Giá trị thiếu hoặc vô hạn thành 0, kiểu float32.

| Feature | Định nghĩa |
|---|---|
| `rule_index` | mã rule, categorical |
| `historical_rule_events` | số dòng log của profile trên nội dung thuộc rule |
| `historical_rule_unique_contents` | số nội dung khác nhau của rule mà profile đã xem |
| `historical_rule_watch_seconds` | tổng `duration` của các dòng đó |
| `historical_rule_events_30d` | số dòng trong 30 ngày trước cutoff |
| `historical_rule_events_90d` | số dòng trong 90 ngày trước cutoff |
| `historical_rule_watch_seconds_30d` | tổng `duration` trong 30 ngày |
| `historical_rule_watch_seconds_90d` | tổng `duration` trong 90 ngày |
| `historical_rule_recency_days` | số ngày từ lần xem gần nhất trong rule; chưa từng xem = 3650 |
| `historical_profile_events` | tổng số dòng log của profile trên nội dung thuộc phạm vi |
| `historical_profile_unique_contents` | số nội dung khác nhau profile đã xem |
| `historical_profile_watch_seconds` | tổng `duration` của profile |
| `historical_rule_event_share` | `historical_rule_events` / max(`historical_profile_events`, 1) |
| `historical_rule_watch_share` | `historical_rule_watch_seconds` / max(`historical_profile_watch_seconds`, 1) |
| `global_rule_event_share` | tỷ trọng số dòng log (của mọi profile) rơi vào rule |
| `global_rule_watch_share` | tỷ trọng tổng `duration` (của mọi profile) rơi vào rule |
| `global_rule_profiles` | số profile khác nhau từng xem rule |
| `rule_catalog_contents` | số nội dung của rule trong phạm vi |

### 6.3 Mẫu huấn luyện

Với mỗi cửa sổ huấn luyện (§3):

- **Dòng**: mọi profile có mặt trong file cửa sổ × mọi rule đủ điều kiện tại
  `history_end`; feature tính tại `history_end`.
- **Target** của mỗi gain = tổng gain của các positive quan sát được trong cửa sổ
  trên nội dung thuộc rule.
- **Chuẩn hóa `profile_share`**: target chia cho tổng target của profile trong
  cùng cửa sổ.
- **Trọng số mẫu** = `observed_weight` khi target > 0, bằng 1 khi target = 0.

Fit dừng với lỗi (trong batch là `RETRAIN_FAILED`) khi:
- các cửa sổ không chồng lên khoảng thời gian của log;
- không có target dương;
- tổng `historical_profile_events` bằng 0.

### 6.4 Hyperparameter

Hai head `HistGradientBoostingRegressor`, một cho `completion_gain` và một cho
`watch_time_gain`.

| Tham số | Giá trị |
|---|---|
| `seed` | 42 |
| `target_normalization` | `profile_share` |
| `observed_weight` | 3.0 |
| `learning_rate` | 0.06 |
| `max_iter` | 80 |
| `max_leaf_nodes` | 31 |
| `min_samples_leaf` | 50 |
| `l2_regularization` | 1.0 |
| `backend` | sklearn |
| `device` | auto |

`backend` chọn thư viện: `sklearn` (`HistGradientBoostingRegressor`, chỉ chạy CPU) hoặc
`xgboost` (`XGBRegressor`, cây histogram, chạy GPU CUDA khi có). `device` là
`auto`, `cuda` hoặc `cpu`; `auto` dùng GPU nếu có, mô hình train trên GPU vẫn chấm
điểm được trên máy chỉ có CPU (`recsys_vnpt/gbdt.py`).

Cố định trong code: loss `squared_error`, early stopping với
`validation_fraction` 0.1 và `n_iter_no_change` 8, `random_state` = `seed`,
`rule_index` là feature categorical.

### 6.5 Chấm điểm và kết hợp

Gọi RR(x) = 1 / (60 + hạng của x trong profile), với hạng giảm dần và hòa theo thứ
tự xuất hiện.

- learned = RR(dự đoán `completion_gain`) + RR(dự đoán `watch_time_gain`);
- `rule_score` = RR(incumbent §5.4) + w × RR(learned);
- dự đoán theo chunk 200,000 dòng.

| Tham số | Giá trị |
|---|---|
| `rule_scorer_fusion_weight` | 1.25 |

## 7. Content scorer

**Code:** `content_scorer.py` (`fit_content_scorer`, `train_content_scorer`,
`content_serving_scope`, `apply_content_scorer`).

### 7.1 Phạm vi tại một cutoff

`content_scope_as_of` giữ nội dung của `contents.parquet` thỏa cả ba điều kiện:
- thuộc ít nhất một rule thường;
- không có cả `publish_date` lẫn `createdate` sau cutoff (thiếu ngày thì vẫn giữ);
- có trong catalog hiện tại.

Metadata đọc từ `contents.parquet`, tức bản mới nhất chứ không phải bản as-of. Khi
phục vụ, bản mới nhất chính là dữ liệu tại thời điểm phục vụ. Khi fit trên các cửa
sổ cũ, các trường như `total_episode` mang giá trị muộn hơn cửa sổ.

Bảng mã cho bốn feature categorical (`level_maps`):
- lấy 250 giá trị phổ biến nhất (`MAX_CATEGORY_LEVELS`), đánh mã theo thứ tự chữ
  cái;
- mọi giá trị khác dùng chung một mã; rỗng được coi là giá trị `(null)`;
- bảng mã được dựng một lần tại cutoff phục vụ của dữ liệu huấn luyện, lưu trong
  model và dùng lại nguyên vẹn khi phục vụ.

Ma trận tag: nội dung × tag, trọng số idf = log(1 + N / df), với N là số nội dung
trong phạm vi và df là số nội dung mang tag đó.

### 7.2 Feature (15)

Lịch sử ở đây là các dòng log ≤ cutoff trên nội dung thuộc phạm vi; mỗi dòng tính
là 1, không dùng `duration`.

| Feature | Định nghĩa |
|---|---|
| `pop_all_log` | log1p(số dòng log của nội dung đến cutoff) |
| `pop_30d_log` | log1p(số dòng log của nội dung trong 30 ngày trước cutoff) |
| `content_duration_log` | log1p(`content_duration`) |
| `content_publish_year` | `content_publish_year` |
| `total_episode_log` | log1p(`total_episode`) |
| `content_age_days` | (cutoff − `createdate`) tính theo ngày |
| `type_index` | mã của `type_id`, categorical |
| `family_index` | mã của `content_family`, categorical |
| `category_index` | mã của `content_category_name`, categorical |
| `country_index` | mã của `content_country`, categorical |
| `tag_affinity` | cosine giữa vector tag-idf của profile (tổng vector tag-idf của các dòng log, chuẩn hóa L2) và vector tag-idf của nội dung |
| `category_affinity` | tỷ lệ dòng log của profile rơi vào cùng category với nội dung |
| `family_affinity` | tỷ lệ dòng log của profile rơi vào cùng family với nội dung |
| `profile_events_log` | log1p(số dòng log của profile) |
| `profile_unique_contents_log` | log1p(số nội dung khác nhau profile đã xem) |

### 7.3 Mẫu huấn luyện

Với mỗi cửa sổ huấn luyện:

- **Positive**: các dòng `observed_engagement` của file cửa sổ có nội dung thuộc
  phạm vi.
- **Negative** (`negative_strategy` = `rule_head`), cho mỗi positive:
  - chọn ngẫu nhiên một rule chứa nó; bộ sinh số ngẫu nhiên được seed riêng cho
    từng profile bằng sha256(`seed`|`profile_id`);
  - lấy tối đa `negative_ratio` nội dung trong `negative_head_size` nội dung phổ
    biến nhất của rule đó;
  - bỏ qua các nội dung: positive của profile trong cửa sổ, nội dung profile đã
    xem trước `history_end`, và nội dung đã được lấy làm negative;
  - nếu thiếu, lấy tiếp từ toàn bộ rule, rồi từ toàn bộ phạm vi.
  - Negative có gain 0. Chiến lược `band_matched` chỉ giữ lại để tái lập thí
    nghiệm cũ.
- **Target** mỗi gain = gain / tổng gain của profile trong cửa sổ; **trọng số** =
  `observed_weight` khi target > 0, bằng 1 khi target = 0.

Fit dừng với lỗi khi:
- các cửa sổ không chồng lên log;
- không có positive nào;
- mọi dòng đều có `tag_affinity` = 0 và `profile_events_log` = 0.

### 7.4 Hyperparameter

| Tham số | Giá trị |
|---|---|
| `seed` | 42 |
| `negative_ratio` | 4 |
| `negative_strategy` | `rule_head` |
| `negative_head_size` | 200 |
| `observed_weight` | 3.0 |
| `learning_rate` | 0.06 |
| `max_iter` | 400 |
| `max_leaf_nodes` | 31 |
| `min_samples_leaf` | 50 |
| `l2_regularization` | 1.0 |
| `backend` | sklearn |
| `device` | auto |

Cố định trong code: như rule scorer (§6.4), với bốn feature categorical ở §7.2.
`max_iter` là trần số vòng; early stopping có thể dừng sớm hơn.

### 7.5 Chấm điểm và kết hợp

Khối là tập candidate cùng (profile, `candidate_rule_id`). Một khối chỉ được chấm
lại khi mọi nội dung trong khối nằm trong phạm vi §7.1; nếu không, khối giữ nguyên
`retrieval_score`.

Gọi RRk(x) = 1 / (60 + hạng của x trong khối).

- personalized = `retrieval_score` − `rule_score` (phần heuristic của §5.3);
- learned = RRk(dự đoán `completion_gain`) + RRk(dự đoán `watch_time_gain`);
- `retrieval_score` mới = `rule_score` + RRk(personalized) + w × RRk(learned);
- dự đoán theo chunk 250,000 dòng.

| Tham số | Giá trị |
|---|---|
| `content_scorer_fusion_weight` | 5.0 |

## 8. Dựng trang

**Code:** `rule_assembly.py` (`assemble_recommendations`,
`allocate_rule_contents`, `bayesian_rule_popularity`),
`rule_assembly.explode_content_rule_membership`, `serving._assemble_chunk`.

### 8.1 Cấu hình

| Trường | Mặc định thư viện | Khi phục vụ |
|---|---|---|
| `normal_rule_count` | 20 | 20 |
| `contents_per_rule` | 10 | 10 |
| `contents_per_rule_max` | None | 30 |
| `page_content_budget` | None | 600 |
| `page_dedup` | none | balanced_regret |

Mặc định thư viện cho ra trang cố định 20 × 10 không khử trùng lặp.
`page_content_budget` không được nhỏ hơn `normal_rule_count` × `contents_per_rule`.

### 8.2 Điểm dùng cho trang

- Mỗi candidate được nhân ra mọi rule thường chứa nó. Điểm nội dung (`score`) là
  `retrieval_score` sau §7, giống nhau ở mọi rule.
- Điểm hàng là `rule_score` của §6.5, hoặc affinity §5.4 khi không dùng rule
  scorer.
- Pool của mỗi (profile, rule) là các nội dung theo `score` giảm dần, hòa theo
  `content_id`. Rule có ít hơn `contents_per_rule` nội dung bị loại khỏi trang.

### 8.3 Chọn 20 hàng

Sắp các rule của profile theo thứ tự ưu tiên:
1. điểm hàng, giảm dần;
2. trung bình `score` của 10 nội dung đầu, giảm dần;
3. `score` thấp nhất trong 10 nội dung đầu, giảm dần;
4. độ phổ biến Bayes của rule, giảm dần;
5. `rule_id` theo giá trị số, tăng dần.

Lấy 20 rule đầu. Profile không đủ 20 rule hợp lệ gây lỗi
`InsufficientRuleCoverage`; trong batch, shard đó thất bại (`INFERENCE_FAILED`).

Độ phổ biến Bayes của rule r = (tổng popularity các nội dung của r + 10 × trung
bình) / (số nội dung của r + 10), với trung bình lấy trên mọi cặp (nội dung, rule
thường).

### 8.4 Phân bổ nội dung (`balanced_regret`)

1. **Giai đoạn sàn**: 10 lượt. Mỗi lượt, các hàng chưa đủ 10 nội dung lần lượt lấy
   nội dung tốt nhất của mình chưa có trên trang.
   - Thứ tự trong lượt theo regret giảm dần, hòa theo thứ tự hàng.
   - Regret của một hàng = điểm ứng viên kế tiếp − điểm ứng viên sau đó; còn một
     ứng viên thì là điểm của nó; hết ứng viên thì là −1.
2. **Bù sàn**: hàng không còn đủ nội dung chưa dùng thì lấy lại nội dung đã đặt ở
   hàng khác, theo thứ tự `score`, cho đến khi đủ 10. Đây là nguồn trùng lặp duy
   nhất trên trang.
3. **Giai đoạn phần dư**: ngân sách còn lại = `page_content_budget` − số slot đã
   dùng.
   - Một heap giữ ứng viên kế tiếp chưa có trên trang của mỗi hàng, sắp theo điểm
     giảm dần, hòa theo thứ tự hàng.
   - Mỗi lần lấy một slot cho ứng viên đứng đầu heap; hàng dừng khi đạt
     `contents_per_rule_max`. Độ phức tạp O(budget · log số hàng).
4. **Thứ tự hiển thị**: hàng theo §8.3; nội dung trong hàng theo `score` giảm dần.

Các chính sách khác trong `PAGE_DEDUP_POLICIES`, giữ lại để so sánh:

| Chính sách | Cách làm |
|---|---|
| `none` | không khử trùng lặp |
| `greedy_rule_rank` | lấp từng hàng theo thứ hạng, bỏ nội dung đã dùng |
| `relative_fit` | như trên, nhưng hàng có ít nội dung trùng với hàng khác được lấp trước |
| `tail_only` | giữ nguyên 10 nội dung đầu của mỗi hàng, chỉ khử trùng lặp phần vượt quá 10 |
| `balanced_head` | giai đoạn sàn theo chiều sâu như `balanced_regret`, nhưng không sắp theo regret |

## 9. Model bundle và gate

**Code:** `scorer_bundle.py`, `batch_runner.retrain_gate`,
`batch_runner.run_retrain_batch`.

**Bundle**: `work_root/models/<bundle_id>/`, không bao giờ bị sửa sau khi ghi.

| File | Nội dung |
|---|---|
| `scorers.joblib` | hai model: các head, `rule_index` (rule scorer) hoặc `level_maps` (content scorer), và metrics khi fit |
| `bundle.json` | `schema_version` 1, `bundle_id` = `<thời điểm UTC>_<12 ký tự đầu SHA-256>`, `model_sha256`, phiên bản `python`/`scikit-learn`/`numpy`/`pandas`, metrics của hai scorer, `batch_id`, `input_fingerprint`, `data_cutoff`, `serving_cutoff`, `code_version` |

**Nạp bundle**: phải khớp `schema_version`, SHA-256 của `scorers.joblib`, và phiên
bản scikit-learn đang cài. Sai bất kỳ điều kiện nào đều bị từ chối (trong batch
là `MODEL_UNAVAILABLE`).

**Promote**:
- nội dung hiện tại của `promoted_model.json` được sao vào `previous_model.json`;
- sau đó `promoted_model.json` được ghi atomic với `bundle_id`, `bundle_path`,
  `model_sha256`, `promoted_at` và kết quả gate.

**Gate**: ba lượt đánh giá trên cửa sổ `test`, mỗi lượt trong một process riêng,
cùng cấu hình phục vụ và cùng một mẫu `gate_evaluation_profiles` (5000) profile.
Mẫu được chọn tất định: trong các profile có positive, sắp theo
sha256(`seed`|`profile_id`) rồi lấy từ đầu. Ba lượt gồm:
- thứ tự heuristic (`count` + `hand_sum`);
- bundle mới;
- bundle đang promote, nếu có.

Bundle mới đạt khi thỏa mọi điều kiện sau:

- cửa sổ test có ít nhất một novel positive;
- với `rule_ndcg@20` và `macro_content_ndcg@10`: không thấp hơn thứ tự heuristic;
- với cả hai chỉ số đó: không thấp hơn bundle đang promote × (1 −
  `gate_max_regression`), tức mức giảm tương đối tối đa 2%.

## 10. Luồng neural và embedding (chỉ dùng cho nghiên cứu)

**Code:** `feature_contract.py`, `models.py`, `losses.py`, `model_runner.py`.

Luồng này hiện không được dùng để phục vụ trang (`ARCHITECTURE.md`, mục "Họ mô
hình").

**`FeatureContract`** được lưu cùng checkpoint và phải được nạp trước khi chấm.
Các trường:
- `domain`, `schema_version`, `label_policy`;
- `model_family`, `model_config`, `model_config_hash`;
- `numeric_features`: mỗi feature có giá trị mặc định, mean và std;
- `categorical_features` và `multivalue_features`: mỗi feature có từ vựng và dấu
  phân tách.

Giá trị chưa gặp ánh xạ thành `__UNK__`, giá trị thiếu thành `__MISSING__`. Họ mô
hình hoặc hash cấu hình không khớp thì dừng trước khi chấm.

**Cấu hình embedding mặc định** (`EmbeddingModelConfig`):

| Tham số | Giá trị |
|---|---|
| `embedding_dim` | 8 |
| `numeric_projection_dim` | 8 |
| `hidden_dims` | (64, 32) |
| `deep_cross_layers` | 2, nhánh ECN (hàm mũ) |
| `shallow_cross_layers` | 4, nhánh LCN (tuyến tính) |
| `dropout` | 0.1 |
| `num_heads` | 1 |

Feature categorical dùng `nn.Embedding`, feature đa giá trị dùng
`nn.EmbeddingBag` (lấy trung bình). Loss gồm BCE có trọng số, listwise theo nhóm và
pairwise theo nhóm (`losses.py`); nhiều target thì lấy trung bình loss của từng
target. Danh sách họ mô hình nằm trong `ARCHITECTURE.md`.

## 11. Output

**Code:** `export.serialize_assembled_output`, `export.build_profile_grouping`,
`output_validation.validate_client_output`, `batch_runner.merge_shards`.

- Ba file sắp theo `profile_id` tăng dần theo thứ tự chuỗi. Hợp đồng đầy đủ nằm ở
  `PRD.md` §2.
- `f_personalized_homepage_rule_…json`: mảng JSON, mỗi dòng một bản ghi
  `{"pid", "d"}`. `d` là 20 rule thường theo thứ tự hiển thị, nối thêm
  `112,109,108,110`.
- `f_personalized_rule_content_…csv`: mỗi profile một dòng
  `pid|rule,type#content$type#content;rule,...`. Dòng gồm 20 rule thường theo cùng
  thứ tự; nội dung trong rule theo thứ tự hiển thị; `type` là `type_id` của nội
  dung.
- `f_profile_grouping_…json`: `{"pid", "mid", "gid"}`. `mid` lấy từ `member_id`
  của `profile_mapping.parquet`, thiếu thì dùng `pid`; `gid` là giá trị cấu hình.

Kiểm tra bắt buộc trước khi giao (mỗi shard và sau khi gộp):
- đúng tập profile, không thiếu, không thừa, không lặp;
- đúng 20 rule thường khác nhau, phần đuôi đặc biệt đúng thứ tự và không kèm nội
  dung;
- mỗi rule 10–30 nội dung, không trùng trong một rule;
- mọi nội dung có trong catalog, đúng `type_id`, `content_status` = 1, và thực sự
  thuộc rule chứa nó;
- hai file homepage và rule-content khớp nhau về rule.

Vi phạm bất kỳ điều kiện nào thì không giao gì (`OUTPUT_INVALID`).
