if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from recsys_vnpt.domain_entrypoint import main

    raise SystemExit(main("movie", "infer"))

import os
import argparse
import polars as pl
import gc
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path
from math import ceil
import subprocess
import sys

from torch.utils.data import DataLoader, TensorDataset
from dcnv3 import DCNv3
from rule_process import get_rulename
from processing import process_infer_data
from recsys_vnpt.bootstrap import ensure_checkpoint
from recsys_vnpt.data_sources import resolve_user_metadata_path
from recsys_vnpt.export import build_client_exports
from recsys_vnpt.feature_contract import FeatureContract
from recsys_vnpt.features import FeatureSchema
from recsys_vnpt.metrics import output_health_metrics
from recsys_vnpt.metrics_io import write_metrics_json
from recsys_vnpt.paths import data_artifact_path, domain_work_path, model_artifact_path, project_artifact_path, rule_info_path
from recsys_vnpt.timing import runtime_metrics

def rank_result(data, n):
    reordered_data = {}
    for user_id in data:
        user_scores = []
        for content_id, content in data[user_id]['suggested_content'].items():
            user_scores.append({
                'content_id': content_id,
                'content_name': content['content_name'],
                'tag_names': content['tag_names'],
                'type_id': content['type_id'],
                'score': content['score']
            })
        sorted_user_scores = sorted(user_scores, key=lambda x: x['score'], reverse=True)[:n]
        reordered_data[user_id] = {
            'suggested_content': {
                film['content_id']: {
                    'content_name': film['content_name'],
                    'tag_names': film['tag_names'],
                    'type_id': film['type_id'],
                    'score': film['score']
                } for film in sorted_user_scores
            },
            'user': data[user_id]['user']
        }
    return reordered_data

def infer(model, data, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Inference"):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            predictions.extend(outputs['y_pred'].detach().cpu().numpy())
        torch.cuda.empty_cache()
    return predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Run movie recommendations.")
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train from initialization if the movie checkpoint is missing.",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=-1,
        help="Limit user/profile rows scored; -1 scores all.",
    )
    parser.add_argument(
        "--num-contents",
        type=int,
        default=-1,
        help="Limit content rows scored; -1 scores all.",
    )
    return parser.parse_args()

def train_if_checkpoint_missing(project_root):
    subprocess.run([sys.executable, str(project_root / "movie" / "train.py")], cwd=project_root, check=True)

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    TOP_N = 200

    project_root = Path().resolve()
    os.makedirs(project_root / "movie" / "result", exist_ok=True)
    os.makedirs(project_root / "movie" / "infer_data", exist_ok=True)

    user_data_path = resolve_user_metadata_path(project_root)
    movie_data_path = data_artifact_path("mytv_vmp_content", project_root=project_root)
    processed_user_path = domain_work_path("movie", "infer_data", "user_data.parquet", project_root=project_root)
    processed_item_path = domain_work_path("movie", "infer_data", "movie_item_data.parquet", project_root=project_root)
    content_movie_path = domain_work_path("movie", "infer_data", "merged_content_movies.parquet", project_root=project_root)
    tags_path = data_artifact_path("tags", project_root=project_root)
    rules_path = rule_info_path(project_root)
    result_json_path = project_artifact_path("movie", "result", "result.json", project_root=project_root)
    rulename_json_path = project_artifact_path("movie", "result", "rulename.json", project_root=project_root)
    rule_content_path = project_artifact_path("movie", "result", "rule_content.txt", project_root=project_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = model_artifact_path("movie", "best_model.pth", project_root=project_root)
    try:
        checkpoint_path = ensure_checkpoint(
            "movie",
            checkpoint_path,
            train_if_missing=args.train_if_missing,
            train_command=lambda: train_if_checkpoint_missing(project_root),
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    schema_path = model_artifact_path("movie", "feature_schema.json", project_root=project_root)
    feature_schema = FeatureSchema.load(schema_path) if schema_path.exists() else None
    if feature_schema is not None and len(feature_schema.feature_columns) != expected_input_dim:
        raise ValueError(
            f"Feature schema has {len(feature_schema.feature_columns)} columns, "
            f"but checkpoint expects {expected_input_dim}"
        )
    contract_path = model_artifact_path("movie", "feature_contract.json", project_root=project_root)
    if contract_path.exists() and feature_schema is not None:
        FeatureContract.load(contract_path).assert_model_compatible(
            "dense_mimic_baseline",
            {"input_dim": expected_input_dim, "feature_columns": feature_schema.feature_columns},
        )
    model = DCNv3(expected_input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    user_profile_df, movie_pl, content_dict = process_infer_data(
        processed_user_path,
        user_data_path,
        processed_item_path,
        movie_data_path,
        content_movie_path,
        num_user=args.num_users,
        num_movie=args.num_contents,
    )
    candidate_content_count = movie_pl.height
    
    user_batch_size = 30
    num_users = len(user_profile_df)
    num_chunks = ceil(num_users / user_batch_size)
    total_pairs = 0

    print(f"\nProcessing {num_users} users in {num_chunks} chunks...")
    for i in range(num_chunks):
        chunk_start = time.time()
        print(f"[Chunk {i+1}/{num_chunks}] Processing...")

        start_idx = i * user_batch_size
        end_idx = min((i+1) * user_batch_size, num_users)
        user_chunk = user_profile_df.iloc[start_idx:end_idx]

        cross_start = time.time()
        user_chunk_pl = pl.from_pandas(user_chunk)
        cross_df = user_chunk_pl.join(movie_pl, how="cross")

        interaction_df = cross_df.select(["username", "content_id", "profile_id"]).to_pandas()
        if feature_schema is not None:
            feature_cols = [col for col in feature_schema.feature_columns if col in cross_df.columns]
            features_np = feature_schema.transform(cross_df.select(feature_cols).to_pandas()).to_numpy()
        else:
            exclude = {"username", "content_id", "profile_id"}
            to_convert = [col for col in cross_df.columns if col not in exclude]
            cross_df = cross_df.cast({col: pl.Float32 for col in to_convert})
            features_np = cross_df.select(to_convert).to_numpy()
        print(f"  ↪︎ Cross join: {time.time()-cross_start:.2f} seconds")

        infer_start = time.time()
        infer_tensor = torch.tensor(features_np, dtype=torch.float32)
        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=2048, shuffle=False)
        predictions = infer(model, infer_loader, device)
        total_pairs += len(predictions)
        print(f"  ↪︎ Inference: {time.time()-infer_start:.2f} seconds")

        collect_start = time.time()
        chunk_result = {}
        temp_dict = {}
        for pid, user, cid, score in zip(
            interaction_df['profile_id'],
            interaction_df['username'],
            interaction_df['content_id'],
            predictions
        ):
            temp_dict.setdefault(pid, []).append((cid, float(score)))
        for pid in temp_dict:
            sorted_items = sorted(temp_dict[pid], key=lambda x: x[1], reverse=True)[:TOP_N]
            chunk_result[pid] = {
                'suggested_content': {
                    cid: {
                        'content_name': '',
                        'tag_names': '',
                        'type_id': '',
                        'score': score
                    } for cid, score in sorted_items
                },
                'user': {
                    'username': user_profile_df.loc[user_profile_df['profile_id'] == pid, 'username'].iloc[0],
                    'profile_id': pid
                }
            }
        print(f"  ↪︎ Collect top {TOP_N}: {time.time()-collect_start:.2f} seconds")

        for pid, pdata in chunk_result.items():
            for cid, cdata in pdata['suggested_content'].items():
                meta = content_dict.get(cid)
                if meta:
                    cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)

        rank_start = time.time()
        reordered_chunk = rank_result(chunk_result, TOP_N)
        result_with_rule = get_rulename(reordered_chunk, rules_path, tags_path)
        print(f"  ↪︎ Ranking & rule assignment: {time.time()-rank_start:.2f} seconds")

        save_start = time.time()
        homepage_rule, rule_content = build_client_exports(result_with_rule)

        with open(project_root / f"movie/result/result_chunk_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(result_with_rule, f, indent=4, ensure_ascii=False)
        with open(project_root / f"movie/result/rulename_chunk_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
        with open(project_root / f"movie/result/rule_content_chunk_{i+1}.txt", "w", encoding='utf-8') as f:
            f.write(rule_content)
        print(f"  ↪︎ Saved chunk files in {time.time()-save_start:.2f} seconds")

        del predictions, temp_dict, chunk_result, cross_df, features_np, infer_tensor, infer_loader
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  ↪︎ Total chunk time: {time.time()-chunk_start:.2f} seconds\n")

    merge_start = time.time()
    final_result = {}
    final_rulename = []
    final_rule_content = ""
    for i in range(num_chunks):
        with open(project_root / f"movie/result/result_chunk_{i+1}.json", 'r', encoding='utf-8') as f:
            final_result.update(json.load(f))
        with open(project_root / f"movie/result/rulename_chunk_{i+1}.json", 'r', encoding='utf-8') as f:
            final_rulename.extend(json.load(f))
        with open(project_root / f"movie/result/rule_content_chunk_{i+1}.txt", 'r', encoding='utf-8') as f:
            final_rule_content += f.read()

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    with open(rulename_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_rulename, f, indent=4, ensure_ascii=False)
    with open(rule_content_path, "w", encoding='utf-8') as f:
        f.write(final_rule_content)
    print(f"Merged final files in {time.time()-merge_start:.2f} seconds")

    total_time = time.time() - start_time
    inference_metrics = runtime_metrics(
        model_family="dense_mimic_baseline",
        inference_time_seconds=total_time,
        inference_pairs=total_pairs,
    )
    inference_metrics.update(output_health_metrics(final_result, candidate_content_count, top_n=TOP_N))
    write_metrics_json(
        project_artifact_path("movie", "result", "inference_metrics.json", project_root=project_root),
        inference_metrics,
    )
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Total user-item pairs: {total_pairs:,}")
    print(f"Average inference time per pair: {inference_metrics.get('inference_seconds_per_pair', 0.0):.6f} seconds")
