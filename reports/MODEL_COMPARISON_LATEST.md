# Metrics Comparison

- Baseline: 20260926T231649367889Z_all_graded_per_rule_popularity
- Candidate: 20260926T232040855281Z_all_graded_per_rule_popularity

| Metric | Baseline | Candidate |
|---|---:|---:|
| data.evaluation_profile_selection_hash | e9acf94124422b5c99c9518f13cb352b344c512530760af2f5245d303c2cca0d | e9acf94124422b5c99c9518f13cb352b344c512530760af2f5245d303c2cca0d |
| data.evaluation_truth_gain_hash | dc8310528a7ba78c0db7e93df4ca003bdad6cba61d066c9ded804605697f5d01 | dc8310528a7ba78c0db7e93df4ca003bdad6cba61d066c9ded804605697f5d01 |
| data.evaluation_truth_pairs_hash | 58ca25793b5ab1ebb93e6a5a0553ceff37a8117142a0b1f1ba66a082be6a9c2d | 58ca25793b5ab1ebb93e6a5a0553ceff37a8117142a0b1f1ba66a082be6a9c2d |
| evaluation.evaluation_profile_selection_hash | e9acf94124422b5c99c9518f13cb352b344c512530760af2f5245d303c2cca0d | e9acf94124422b5c99c9518f13cb352b344c512530760af2f5245d303c2cca0d |
| evaluation.evaluation_truth_gain_hash | dc8310528a7ba78c0db7e93df4ca003bdad6cba61d066c9ded804605697f5d01 | dc8310528a7ba78c0db7e93df4ca003bdad6cba61d066c9ded804605697f5d01 |
| evaluation.evaluation_truth_pairs_hash | 58ca25793b5ab1ebb93e6a5a0553ceff37a8117142a0b1f1ba66a082be6a9c2d | 58ca25793b5ab1ebb93e6a5a0553ceff37a8117142a0b1f1ba66a082be6a9c2d |
| evaluation.candidate_pairs_hash | 8f7769faf59b18febea22aa437cdd90930575fced6d280fbdfe109245700c8a4 | 8f7769faf59b18febea22aa437cdd90930575fced6d280fbdfe109245700c8a4 |
| evaluation.candidate_profiles | 5000 | 5000 |
| evaluation.retrievable_truth_profiles | 4201 | 4201 |
| evaluation.retrievable_positive_pairs | 12232 | 12232 |
| evaluation.candidate_recall@100 | 0.2995421844342708 | 0.2995421844342708 |
| evaluation.candidate_recall@200 | 0.4040222367560497 | 0.4040222367560497 |
| evaluation.candidate_recall@500 | 0.7093688685415304 | 0.7093688685415304 |
| evaluation.candidate_recall@1000 | 0.8233322432962721 | 0.8233322432962721 |
| evaluation.candidate_completion_gain_recall@1000 | 0.7844047004539019 | 0.7844047004539019 |
| evaluation.candidate_watch_time_gain_recall@1000 | 0.8189739636614081 | 0.8189739636614081 |
| evaluation.rule_ndcg@20 | 0.6813717414813619 | 0.708275742531813 |
| evaluation.macro_content_ndcg@10 | 0.08613142446849903 | 0.10782195479201225 |
| evaluation.overall_recall@200 | 0.4771883983590415 | 0.5659378390222587 |
| evaluation.completion_gain_rule_ndcg@20 | 0.6757132161954257 | 0.704764198122133 |
| evaluation.completion_gain_macro_content_ndcg@10 | 0.08312170998072359 | 0.10490528640752479 |
| evaluation.completion_gain_all_rule_macro_content_ndcg@10 | 0.14804592249993118 | 0.20764473048040855 |
| evaluation.watch_time_gain_rule_ndcg@20 | 0.6878356040592438 | 0.7172413725882409 |
| evaluation.watch_time_gain_macro_content_ndcg@10 | 0.08514152069607547 | 0.106919951122085 |
| evaluation.watch_time_gain_all_rule_macro_content_ndcg@10 | 0.1513121509214731 | 0.21093075802220448 |
| evaluation.candidate_rule_groups | 222109 | 222109 |
| evaluation.eligible_candidate_rule_groups | 170828 | 170828 |
| evaluation.eligible_candidate_rule_group_rate | 0.769117865552499 | 0.769117865552499 |
| evaluation.mean_eligible_candidate_rules_per_profile | 34.1656 | 34.1656 |
| evaluation.loaded_interaction_rows | 563183 | 563183 |
| evaluation.serving_interaction_rows | 559169 | 559169 |
| evaluation.truth_audit_only_interaction_rows | 4014 | 4014 |
| evaluation.truth_audit_only_interaction_contents | 21 | 21 |
| evaluation.retrieval_truth_attribution.cohorts.observed_history_profile_oov.true.candidate_recall | 0.8505224897773739 | 0.8505224897773739 |
| evaluation.retrieval_truth_attribution.cohorts.observed_history_content_oov.true.candidate_recall | 0.6690871369294605 | 0.6690871369294605 |
| evaluation.retrieval_truth_attribution.cohorts.activity_bucket.cold.candidate_recall | 0.8314332247557004 | 0.8314332247557004 |
| evaluation.exact_20_normal_rule_rate | 1.0 | 1.0 |
| evaluation.exact_10_content_rate | 0.06685313020709355 | 0.05833134967864794 |
| evaluation.external_candidate_mode | rerank | rerank |
| evaluation.external_candidate_slots | 100 | 100 |
| evaluation.external_replacement_requires_witness | False | False |
| evaluation.external_witness_rejected_candidates | 0 | 0 |
| evaluation.external_witness_rejected_profiles | 0 | 0 |
| evaluation.preblend_incumbent_truth_hits | 10071 | 10071 |
| evaluation.blend_gained_truth_hits | 0 | 0 |
| evaluation.blend_lost_truth_hits | 0 | 0 |
| evaluation.blend_net_truth_hits | 0 | 0 |
| evaluation.blend_net_completion_gain | 0.0 | 0.0 |
| evaluation.blend_net_watch_time_gain | 0.0 | 0.0 |
| evaluation.mean_incumbent_candidate_retention_rate | 1.0 | 1.0 |
| evaluation.minimum_safety_anchor_retention_rate | 1.0 | 1.0 |
| training.training_time_seconds | 0.0 | 0.0 |
| inference.evaluation_retrieval_time_seconds | 78.03761889995076 | 113.57921110000461 |
| inference.evaluation_ranking_time_seconds | 96.48984490009025 | 98.6407192000188 |
| inference.evaluation_end_to_end_time_seconds | 174.5519115999341 | 212.24370570015162 |
| inference.inference_pairs | 5000000 | 5000000 |
| inference.pairs_per_second | 204517380.78124443 | 210301402.83806327 |

## Paired Bootstrap 95%

| Metric | Mean Delta | Lower | Upper | Profiles |
|---|---:|---:|---:|---:|
| candidate_recall | 0.0 | 0.0 | 0.0 | 4201 |
| candidate_completion_gain_recall | 0.0 | 0.0 | 0.0 | 4201 |
| candidate_watch_time_gain_recall | 0.0 | 0.0 | 0.0 | 4201 |
| rule_ndcg@20 | 0.026904001050451074 | 0.023833425285536048 | 0.029982529304122235 | 4201 |
| macro_content_ndcg@10 | 0.02169053032351321 | 0.018938667568255022 | 0.024206434266875836 | 4201 |
| all_rule_macro_content_ndcg@10 | 0.05993817476635666 | 0.05419075790341176 | 0.06594376657011104 | 4201 |
| overall_recall@200 | 0.08874944066321727 | 0.0783631520452341 | 0.09978402343572106 | 4201 |
| completion_gain_rule_ndcg@20 | 0.02905098192670735 | 0.025989936265828337 | 0.032232708204631956 | 4201 |
| completion_gain_macro_content_ndcg@10 | 0.021783576426801197 | 0.01905477589036471 | 0.024305579711463878 | 4201 |
| completion_gain_all_rule_macro_content_ndcg@10 | 0.059598807980477383 | 0.05389285091054819 | 0.06553582712537159 | 4201 |
| watch_time_gain_rule_ndcg@20 | 0.02940576852899713 | 0.026361092247627407 | 0.03244116853658384 | 4201 |
| watch_time_gain_macro_content_ndcg@10 | 0.021778430426009524 | 0.019071448489781918 | 0.02433250446774829 | 4201 |
| watch_time_gain_all_rule_macro_content_ndcg@10 | 0.059618607100731365 | 0.053977099468103834 | 0.06542662037042017 | 4201 |
