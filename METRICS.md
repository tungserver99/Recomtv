# Chỉ số RecSys-VNPT

Các lượt chạy pipeline mới dùng schema chỉ số phiên bản 2, lưu tại:

`data/data/metrics/<run_id>/<domain>/`

Mỗi lượt chạy ghi lại manifest, data, training, evaluation, inference, output,
gate và báo cáo Markdown. Các file phẳng cũ `training_metrics.json` và
`inference_metrics.json` vẫn đọc được như artifact lịch sử, nhưng không phải bằng
chứng để promote.

## Lệnh

```powershell
python -m recsys_vnpt.metrics_cli show --run latest --domain all
python -m recsys_vnpt.metrics_cli compare --baseline RUN_A --candidate RUN_B --domain all
python -m recsys_vnpt.metrics_cli gate --run RUN_B --baseline RUN_A --domain all --phase sprint-3
python -m recsys_vnpt.metrics_cli gate --run RUN_B --baseline RUN_A --domain all --phase sprint-4
python -m recsys_vnpt.metrics_cli report --run latest --domain all --output reports/METRICS_LATEST.md
python -m recsys_vnpt.forecast_cli --domain all --runs RUN_100 RUN_1K RUN_10K RUN_FULL --phase sprint-5 --output reports/RUNTIME_FORECAST_RTX5060.json
```

Gate thất bại sẽ thoát với mã `2`. Không được promote checkpoint khi gate của nó
thất bại.

## Ngữ nghĩa target

Các lượt chạy mới dùng `engagement_v1`, không dùng label CTR. Mỗi lượt playback
hợp lệ quan sát được có hai target có phân cấp: `completion_gain` và
`watch_time_gain`. Các dòng lấy mẫu từ cặp profile–nội dung chưa quan sát là mẫu
xếp hạng tương phản (contrastive), không phải impression negative đã xác minh.
Chính sách label, các ngưỡng chỉ tính trên dữ liệu huấn luyện, chính sách phiên
xem/khử trùng lặp, cutoff khi fit và hash của artifact được lưu trong
`label_contract_v4.json`.

`accuracy`, AUC, precision, F1 và log-loss phân loại không được xuất cho điểm xếp
hạng. Các chỉ số này chỉ hợp lệ với một probability head trong tương lai, được
huấn luyện từ log impression/exposure thật.

## Chỉ số chính

- `rule_ndcg@20`: chất lượng thứ tự của 20 rule thường trên trang chủ.
- `macro_content_ndcg@10`: chất lượng thứ tự nội dung trong các hàng profile–rule
  có liên quan.
- `overall_recall@200`: độ phủ positive tương lai trên 200 vị trí thường.
- `completion_gain_rule_ndcg@20` và `watch_time_gain_rule_ndcg@20`: chất lượng thứ
  tự rule, có trọng số theo mức độ xem hết/thời lượng xem trong tương lai.
- `completion_gain_macro_content_ndcg@10` và
  `watch_time_gain_macro_content_ndcg@10`: chất lượng nội dung trong rule theo cả
  hai mục tiêu engagement, sau khi đã chọn 20 rule thường được phục vụ. Các chỉ số
  này có thể tăng khi việc chọn rule tốt hơn, ngay cả khi thứ tự nội dung không
  đổi.
- `completion_gain_all_rule_macro_content_ndcg@10` và
  `watch_time_gain_all_rule_macro_content_ndcg@10`: cùng phép đo thứ tự trong rule,
  nhưng đánh giá trên mọi nhóm rule candidate hợp lệ. Dùng các chỉ số này làm
  chẩn đoán thứ tự nội dung không phụ thuộc việc chọn rule.
- `candidate_completion_gain_recall@1000` và
  `candidate_watch_time_gain_recall@1000`: phần engagement mà retrieval giữ lại
  được.
- Chỉ số hợp đồng output: đúng 20 rule, 10–30 nội dung mỗi rule, đủ phần đuôi hàng
  đặc biệt, không có nội dung không hợp lệ/không khớp, và nhất quán giữa các file.
- Chỉ số runtime: số giây mỗi bước, cặp/giây, profile/giây, p50/p95/p99, bộ nhớ
  CPU/GPU, mức sử dụng/công suất GPU lấy mẫu, nguồn gốc thiết bị, và lỗi xử lý.

Với `per_rule_popularity`, giá trị xuất ra là điểm thứ hạng đã chuẩn hóa, không
phải xác suất đã hiệu chỉnh. Candidate Recall và Gain Recall cho thấy riêng phần
engagement tương lai bị bỏ lỡ trước khi xếp hạng.

## Lệnh pipeline

```powershell
# Cài dependency cơ bản trước; ghi đè bản CUDA phải làm tường minh.
python -m pip install -r requirements.txt
python -m pip install --force-reinstall --no-deps -r requirements-cuda.txt

# Dữ liệu thô -> canonical. Chỉ thêm giới hạn khi diễn tập có chủ đích.
python -m recsys_vnpt.pipeline_cli prepare --domain all --max-interaction-rows 500000 --skip-behavior --target-version engagement_v1 --session-gap-minutes 30
python -m recsys_vnpt.pipeline_cli build-temporal --domain all --max-profiles 500 --max-contents 500 --target-version engagement_v1 --sequence-mode historical
python -m recsys_vnpt.pipeline_cli evaluate-baseline --domain all --candidate-budget 1000

# Chạy lại một rule scorer học được trên artifact validation/test bất biến. Cutoff
# được đọc từ source manifest, trừ khi truyền --cutoff để đối chiếu.
python -m experiments.rule_scorer_feasibility --source-run RUN_ID --split validation --target-normalization profile_share --fusion-weights 1.25 --output reports/RULE_SCORER_VALIDATION.json

# Cùng luồng v2 dùng chung, gọi qua entrypoint theo domain.
python all/eval.py --model-family per_rule_popularity
python all/train.py --model-family fcn_official --checkpoint-selection hierarchical --batch-size 4096 --require-cuda
python all/eval.py --model-family fcn_official --run-id RUN_ID --eval-serving-candidates --batch-size 4096
python all/infer.py --model-family fcn_official --num-profiles 1000 --candidate-budget 1000 --warm-repetitions 3 --profile-chunk-size 100 --batch-size 4096 --dry-run

# Tăng dần workload, không xuất bản. Cần ít nhất ba kích thước để dự báo holdout.
python -m recsys_vnpt.shadow_cli --domain all --model-family fcn_official --workloads 1000 10000 31468 --candidate-budget 1000 --warm-repetitions 3 --require-cuda
```

`--num-profiles` giới hạn số khóa gợi ý duy nhất. `--num-contents` được áp dụng sau
khi lọc canonical theo active/tagged/đã khử trùng lặp, và giữ lại ít nhất 20 rule
thường hợp lệ khi giới hạn cho phép. `--dry-run` vẫn dựng trang, tuần tự hóa trong
bộ nhớ và kiểm tra output, nhưng không xuất bản file cho client.

## Xuất bản output cho client

Các snapshot đã xuất bản được commit bằng cách thay atomic file
`client_output_manifest.json`. Manifest xác định một generation bất biến trong
`generations/<generation>/`, tham chiếu cả `rulename.json` lẫn `rule_content.txt`,
và ghi SHA-256 cho từng file. Bên đọc cần snapshot nhất quán phải đọc manifest và
kiểm tra cả hai hash được tham chiếu. `rulename.json` và `rule_content.txt` ở cấp
trên cùng vẫn là bản sao tương thích cho các client hiện có, nhưng bên đọc trực
tiếp hai đường dẫn đó sẽ không có ngữ nghĩa snapshot atomic trên nhiều file. Điều
này áp dụng cho output của `live-test` phục vụ nghiên cứu; bản giao production được
commit qua `CURRENT.json` (`OPERATIONS.md` §7).

## Định nghĩa chỉ số

- `candidate_recall@1000`: tỷ lệ positive tương lai hợp lệ được giữ lại bởi một
  pool retrieval có K thực tế tối đa đúng bằng 1,000. Không được báo cáo dưới tên
  này khi pool vượt quá 1,000.
- `candidate_<gain>_recall@1000`: tỷ lệ tổng gain có phân cấp hợp lệ được giữ lại
  bởi cùng pool candidate đó.
- `target_semantics`, `classification_metrics_valid`: hợp đồng target và ràng buộc
  tường minh để không diễn giải điểm xếp hạng như xác suất.
- `evaluation_truth_gain_hash`: hash của các ID và cả hai giá trị gain; so sánh mô
  hình yêu cầu cùng hash cặp và cùng hash gain.
- `raw_interaction_rows`, `exact_duplicate_rows_removed`,
  `same_end_duplicate_rows_removed`, `playback_events`, `playback_sessions`: các số
  đếm audit cho việc dựng lại playback và khử trùng lặp.
- `raw_positive_pairs`, `repeat_positive_pairs_excluded`, `novel_positive_pairs`,
  `retrievable_positive_pairs`: các số đếm liên tiếp theo chính sách truth. Số đếm
  khi phục vụ và số đếm temporal phải khớp nhau.
- `evaluation_truth_pairs_hash`: hash tất định của các cặp positive mới và có thể
  retrieve. Hash này phải khớp ở mọi phép so sánh.
- `candidate_pairs_hash`: hash tất định của các cặp profile–nội dung được retrieve.
  Phải khớp khi so sánh mô hình; challenger retrieval được phép thay đổi.
- `retrieval_candidates_artifact_sha256`, `evaluation_truth_artifact_sha256`,
  `retrieval_profile_metrics_artifact_sha256`: hash theo byte, bảo vệ cả các cột
  thứ hạng, nguồn, gain và cohort. Riêng hash cặp không chứng minh được nguồn gốc
  hay sự tương đương về thứ hạng.
- `candidate_recall_all_positives`: recall tính cả các positive không có trong
  catalog hợp lệ tại thời điểm as-of; chỉ số này tách lỗi retrieval khỏi lỗi do
  nguồn/catalog.
- `candidate_truth_catalog_coverage`, `unavailable_positive_pairs`: tỷ lệ/số lượng
  positive tương lai mà retrieval có thể tiếp cận sau khi lọc catalog theo thời
  điểm as-of.
- `candidate_mean_actual_k`, `candidate_max_actual_k`: kích thước candidate đo được
  sau khi khử trùng lặp và áp ngân sách.
- `preblend_incumbent_*`: chất lượng của pool heuristic có giới hạn, ngay trước khi
  một retriever bên ngoài được phép thay các slot ở phần đuôi.
- `blend_gained_*`, `blend_lost_*`, `blend_net_*`: truth/gain có trong final nhưng
  không có trong preblend, có trong preblend nhưng không có trong final, và hiệu có
  dấu của hai phần đó. Challenger thay thế phải báo cáo cả ba; chỉ báo phần gained
  sẽ phóng đại hiệu quả.
- `external_replacement_requires_witness`: phần gain chỉ đến từ nguồn bên ngoài
  trong final có bắt buộc phải xuất hiện thêm ở một nguồn heuristic độc lập trước
  khi được thay một candidate incumbent ở phần đuôi hay không. Các witness hiện tại
  là độ phổ biến theo rule, co-watch, độ phổ biến sâu theo rule, và recency catalog
  an toàn theo cutoff. Gate này áp dụng sau khi blend; gain bị từ chối sẽ khôi phục
  dòng incumbent thay vì kéo thêm thứ hạng sâu hơn từ nguồn bên ngoài.
- `external_witness_rejected_candidates`,
  `external_witness_rejected_profiles`: số dòng thay thế chỉ có trong hybrid ở
  final bị gate đó loại, và số profile bị ảnh hưởng. Đây là số đếm audit an toàn,
  không phải chỉ số chất lượng; chất lượng vẫn đến từ các so sánh recall/gain ghép
  cặp và NDCG/Recall phía sau trên cùng hash truth.
- `external_incremental_*`: truth từ nguồn bên ngoài nằm ngoài incumbent
  preblend, với `external_incremental_reference=preblend_incumbent`. Oracle union
  tương ứng là dư địa chẩn đoán, không phải chất lượng mô hình đạt được.
- `blend_lost_safety_anchor_*`: các candidate theo rule được bảo vệ nhưng bị loại
  trong lúc thay thế. Giá trị này phải bằng 0. `preblend_incumbent_pair_set_hash`,
  `blend_gained_pair_set_hash`, `blend_lost_pair_set_hash` và
  `blend_union_pair_set_hash` cố định chính xác các tập được dùng trong audit.
- `profiles_with_multiple_accounts`, `profiles_with_username_case_variants`,
  `profiles_with_multiple_member_ids`: xung đột username thực sự sau chuẩn hóa, các
  biến thể chữ hoa/thường của username thô đã được phân giải tất định, và xung đột
  member-ID đi kèm. Không trường nào làm thay đổi khóa gợi ý khỏi `profile_id`.
- `retrieval_ndcg_relative_loss`: mức giảm NDCG Rule/Content tương đối lớn nhất so
  với chấm điểm đầy đủ trên tập candidate cố định.
- `rule_ndcg@20`, `rule_recall@20`, `rule_hit_rate@20`: chất lượng thứ tự của các
  rule thường trên trang chủ.
- `macro_content_ndcg@10`, `macro_content_recall@10`, `macro_content_hit_rate@10`:
  chất lượng macro bên trong mỗi rule có liên quan trong tương lai; rule liên quan
  bị thiếu đóng góp 0.
- `<gain>_all_rule_macro_content_ndcg@10`: thứ tự nội dung khi cho phép mọi nhóm
  rule candidate thường có ít nhất 10 candidate khác nhau, thay vì chỉ 20 rule được
  chọn. Rule liên quan trong tương lai nhưng bị thiếu hoặc không đủ nội dung vẫn
  đóng góp 0. Nếu chỉ số này không đổi trong khi `<gain>_macro_content_ndcg@10` của
  các rule được chọn tăng, thì mức tăng đến từ việc chọn rule.
- `candidate_rule_groups`, `eligible_candidate_rule_groups`,
  `eligible_candidate_rule_group_rate`, `mean_eligible_candidate_rules_per_profile`:
  mẫu số và audit độ hỗ trợ cho chỉ số all-rule. Hợp lệ nghĩa là nhóm profile–rule
  có ít nhất 10 candidate khác nhau.
- `loaded_interaction_rows`, `serving_interaction_rows`,
  `truth_audit_only_interaction_rows`, `truth_audit_only_interaction_contents`:
  audit phạm vi tương tác. Các dòng chỉ dùng cho audit có thể được dùng để dựng
  truth tương lai, nhưng không được đi vào độ phổ biến, affinity, feature mô hình
  hay điểm retrieval.
- `observed_history_profile_oov`, `observed_history_content_oov`: chưa từng xuất
  hiện trong engagement quan sát được trước cutoff. Mẫu tương phản chưa quan sát
  không làm một thực thể trở thành "đã thấy". Đây là cohort theo độ phủ lịch sử,
  không nhất thiết là OOV theo từ vựng của mô hình.
- `historical_popularity_bucket`: nội dung truth được nhóm theo số profile khác
  nhau đã quan sát trong lịch sử (`0`, `1-20`, `21-100`, `101+`).
- `latest_publish_age_bucket`: `publish_date` canonical mới nhất, tính tương đối
  với cutoff lịch sử của lượt đánh giá. Đây là chẩn đoán dựa trên dữ liệu có thể
  thay đổi hoặc bị điền ngược, không phải timestamp bắt đầu khả dụng. Vì vậy
  `latest_after_cutoff` không chứng minh nội dung chưa khả dụng tại cutoff.
  `catalog_recency` vẫn loại timestamp sau cutoff. Kênh này dùng `publish_date`, nên
  phụ thuộc vào chất lượng metadata theo thời điểm (`TODO.md` §4).
- Phân bổ theo nguồn retrieval dùng `primary_source` chỉ như nguồn gốc theo thứ tự
  ưu tiên cố định, `inclusive_witness` như bằng chứng chồng lấn,
  `fractional_credit` như điểm chia đều có thể cộng dồn, và `exclusive_source` cho
  các hit chỉ đến từ một nguồn. Không được cộng các số đếm inclusive với nhau, và
  không số nào là ablation nhân quả.
- `hybrid_item_identity_policy=target_only|context_and_target`: những ID item học
  được nào được dùng khi chấm điểm bằng tower. Item nằm ngoài chính sách quan sát
  đã chọn dùng ID 0 cộng metadata. `topk_tie_policy` ghi lại cách chọn tất định tại
  biên; chính sách này không biến các trường hợp hòa chỉ dựa trên metadata thành cá
  nhân hóa.
- `hybrid_scoring_catalog_policy=all_input|eligible_only`: hybrid tower chấm điểm
  mọi item đầu vào tại thời điểm as-of, hay chỉ các item thỏa chính sách retrieval
  khi phục vụ. `eligible_catalog_hash`, `scoring_catalog_hash`,
  `scored_ineligible_contents` và `retrieved_scored_catalog_coverage` giúp chẩn
  đoán này tái lập được. Chính sách này không thay đổi mục tiêu huấn luyện.
- `overall_ndcg@200`, `overall_recall@200`: chất lượng trên các nội dung đã khử trùng lặp từ 20 rule thường x 10 vị trí.
- `overall_ndcg@page`, `overall_recall@page`: như trên, nhưng trên toàn bộ trang được phục vụ (tối đa 600 slot với cấu hình phục vụ).
- `page_unique_contents`, `page_duplicate_slot_rate`, `page_mean_contents_per_rule`: số nội dung khác nhau mỗi trang, tỷ lệ slot lặp lại một nội dung đã có trên trang, và độ dài trung bình của hàng.
- `flat_candidate_ndcg@200`, `flat_candidate_recall@200`: xếp hạng chẩn đoán theo điểm của cặp trước khi dựng trang theo rule. Không bao giờ được ghi đè lên các chỉ số output phân cấp.
- `expanded_candidate_rule_rows`, `mean_rule_memberships_per_candidate`, `underfilled_rule_groups_excluded`: chi phí mở rộng quan hệ nhiều–nhiều, và số nhóm rule bị loại vì có ít hơn 10 candidate.
- `exact_20_normal_rule_rate`: tỷ lệ profile có đúng 20 rule thường khác nhau.
- `exact_10_content_rate`: tỷ lệ profile có mỗi rule trong 20 rule thường chứa đúng 10 nội dung khác nhau (trang cố định 20x10).
- `content_count_within_range_rate`: tỷ lệ profile có mỗi rule trong 20 rule thường chứa từ 10 đến 30 nội dung khác nhau (trang được phục vụ). Một trong hai tỷ lệ bằng 1.0 là thỏa hợp đồng output.
- `special_rule_complete_rate`, `special_no_content_rate`: phần đuôi `112,109,108,110` đúng thứ tự và không có nội dung gắn vào rule đặc biệt.
- `profile_output_coverage`, `cross_file_consistency_rate`: các PID mong đợi xuất hiện đúng một lần, và tập homepage/rule-content khớp nhau.
- `invalid_content_rate`, `unmatched_rule_rate`, `duplicate_within_rule_rate`, `fallback_rate`: các tỷ lệ sức khỏe output bắt buộc; để promote, tất cả phải bằng 0, trừ fallback đã được chấp nhận tường minh.
- `inference_pairs`: số cặp profile–candidate thực sự được chấm điểm sau retrieval.
- `inference_seconds_per_pair`, `pairs_per_second`, `profiles_per_second`: throughput ở trạng thái ổn định, đo trên cùng workload.
- `load_seconds`, `retrieval_seconds`, `encode_forward_seconds`, `rule_assembly_seconds`, `serialization_seconds`, `validation_seconds`: thời gian theo từng bước. p50 khi warm không tính phần nạp/retrieval chạy một lần; `cold_end_to_end_seconds` có tính.
- `retrieval_stage_timings`, `serving_stage_timings`: thời gian chi tiết của các bước catalog/lịch sử/chỉ mục/vòng lặp profile/materialization, và các bước feature/truth khi phục vụ.
- `p50_seconds`, `p95_seconds`, `p99_seconds`, `warm_seconds`: phân phối độ trễ của workload.
- `peak_observed_rss_bytes`, `peak_gpu_memory_bytes`, `output_bytes`: bộ nhớ và dung lượng output.
- `gpu_utilization_mean_percent`, `gpu_utilization_p95_percent`, `gpu_utilization_max_percent`: mẫu toàn thiết bị lấy định kỳ từ `nvidia-smi` trong lúc chấm điểm neural. Workload nhỏ có thể bị chi phối bởi độ mịn lấy mẫu và các bước CPU; process GPU chạy đồng thời có thể tạo đột biến, nên cần đối chiếu với bộ nhớ allocator của PyTorch và các lượt chạy lặp lại.
- `gpu_device_memory_peak_mib`: bộ nhớ đỉnh toàn thiết bị do `nvidia-smi` báo cáo; cố ý khác với `peak_gpu_memory_bytes`, vốn chỉ tính allocator của PyTorch.
- `gpu_power_available`, `gpu_power_samples`, `gpu_power_mean_watts`: tính khả dụng, số lượng và giá trị trung bình của các mẫu công suất hợp lệ. Trung bình bằng 0 với `gpu_power_available=false` nghĩa là driver không trả về giá trị công suất hợp lệ, không phải thiết bị chạy ở 0 watt.
- `deterministic_warm_hash_rate`: mọi lượt warm cho ra cùng hash output tuần tự hóa trên cùng phần cứng/cấu hình.
- `ramp_warm_reference_run_id`, `ramp_warm_repetitions`, `ramp_deterministic_warm_hash_rate`: tham chiếu warm tường minh của bản tổng hợp nhiều workload; các trường độ trễ của lượt cold đầy đủ không bao giờ bị gắn nhãn lại thành số đo warm.
- `resume_enabled`, `resume_chunks_loaded`: có yêu cầu resume theo chunk hay không, và số chunk đã hoàn thành được nạp lại thay vì chấm điểm lại.
- `forecast_error_rate`: sai số holdout từ chuỗi workload tăng dần đã đo. Giá trị chỉ được xuất khi đã có fit và holdout.
- `confidence_intervals.<metric>`: khoảng tin cậy 95% bằng profile bootstrap cho một chỉ số tuyệt đối.
- `paired_bootstrap.<metric>`: khoảng 95% bằng paired profile bootstrap cho hiệu candidate trừ incumbent. Cả hai cận dưới của chỉ số chính phải lớn hơn 0 để promote mô hình neural.

## Tên họ mô hình

- `embedding_mlp_baseline`: mô hình embedding cộng MLP cũ, trước đây gọi là `embedding_fcn`.
- `legacy_embedding_dcnv3`: bản xấp xỉ dùng cổng kiểu cũ, trước đây gọi là `embedding_dcnv3`.
- `fcn_official`: FCN bám sát bản gốc với các nhánh ECN và LCN, Self-Mask, embedding căn theo field, và Tri-BCE.
- `ecn_official`: ablation chỉ gồm ECN.
- `hierarchical_fcn`: backbone FCN với head riêng cho rule/nội dung và các hàm loss phân cấp.

Alias cũ vẫn đọc được artifact cũ. Checkpoint mô hình mới dùng schema v3 và thất
bại khi họ mô hình, kiến trúc hoặc feature contract không khớp.

## Quy tắc promote

Gate thất bại sẽ thoát với mã khác 0 và giữ artifact ở vai trò challenger.
Sprint-3 so sánh các mô hình neural trên cùng hash candidate và hash truth.
Sprint-4 cho phép hash candidate mới nhưng yêu cầu cùng hash truth, Candidate
Recall@1,000 >=98%, chất lượng phân cấp giảm không quá 1%, và throughput >=5x.

Gate promote `label-l4`/`label-l5` chỉ dựa trên `watch_time_gain` (target phân cấp
công bằng giữa các family): NDCG rule/nội dung tăng tương đối >=3%, cận dưới
paired bootstrap >0, `overall_recall@200` và
`watch_time_gain_all_rule_macro_content_ndcg@10` giữ lại >=99.5%, và đúng 20x10.
`completion_gain` phụ thuộc độ dài nội dung và family, nên được báo cáo như chỉ số
chẩn đoán nhưng không dùng làm gate promote.

Gate retrain production là gate riêng, áp dụng cho từng scorer bundle mới
(`OPERATIONS.md` §6): có positive mới trong cửa sổ test, tốt ít nhất bằng thứ tự
heuristic (hand-built) trên `rule_ndcg@20` và `macro_content_ndcg@10`, và nằm trong
`gate_max_regression` so với bundle đang promote.
