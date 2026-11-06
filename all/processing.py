from duration_process import merge_parquet_files
from item_process import process_all_item
from user_process import process_user_data
import glob
import time
import os
from pathlib import Path
import pandas as pd
import polars as pl
from recsys_vnpt.labels import finalize_merged_interactions
from recsys_vnpt.data_sources import resolve_user_metadata_path
from recsys_vnpt.paths import data_artifact_path, domain_work_path, project_artifact_path

def process_data(output_filepath):
    project_root = Path().resolve()

    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    duration_folder_path = data_artifact_path("duration", project_root=project_root)
    merged_duration_folder_path = domain_work_path("all", "merged_duration", project_root=project_root)
    user_path = resolve_user_metadata_path(project_root)
    all_data_path = data_artifact_path("mytv_vmp_content", project_root=project_root)

    durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))
    if len(durations)<1:
        merge_parquet_files(duration_folder_path, merged_duration_folder_path)
        durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))

    user_df = process_user_data(user_path, "all/train_data", mode='train')
    all_df = process_all_item(all_data_path, "all/train_data", mode='train')
    all_df['content_id'] = all_df['content_id'].astype(str)

    all_merged_data = []

    for duration in durations:
        try:
            duration_df = pd.read_parquet(duration)
            duration_df['content_id'] = duration_df['content_id'].astype(str)

            print(f"\nProcessing {os.path.basename(duration)}")
            print(f"→ Duration rows: {len(duration_df)}")

            merged_with_user = pd.merge(duration_df, user_df, on='username', how='inner')
            print(f"→ After user merge: {len(merged_with_user)}")

            final_merged = pd.merge(merged_with_user, all_df, on='content_id', how='inner')
            print(f"→ After content merge: {len(final_merged)}")

            all_merged_data.append(final_merged)
        except Exception as e:
            print(f"Error processing {duration}: {str(e)}")

    if all_merged_data:
        combined_df = pd.concat(all_merged_data, ignore_index=True)
        print(f"\nTotal merged rows before drop duplicates: {len(combined_df)}")

        combined_df = combined_df.drop_duplicates()
        print(f"→ After drop_duplicates: {len(combined_df)}")

        combined_df = finalize_merged_interactions(combined_df, domain="all")

        combined_df.to_parquet(output_filepath, index=False)
        return combined_df
    else:
        return
    
def process_infer_data(processed_user_path, user_data_path, processed_item_path, all_data_path, content_all_path, num_user=-1, num_all=-1):

    print("Preprocessing user and item data...")
    preprocess_start = time.time()
    if not os.path.exists(processed_user_path):
        print("  ↪︎ Processing user data...")
        process_user_data(user_data_path, output_dir="all/infer_data", mode='infer')
    else:
        print("  ↪︎ User data already exists, skipping preprocessing.")
    if not os.path.exists(processed_item_path):
        print("  ↪︎ Processing item data...")
        process_all_item(all_data_path, output_dir="all/infer_data", mode='infer')
    else:
        print("  ↪︎ Item data already exists, skipping preprocessing.")
    print(f"Preprocessing completed in {time.time()-preprocess_start:.2f} seconds")

    print("\nLoading user and item data...")
    user_df = pd.read_parquet(processed_user_path)
    all_df = pd.read_parquet(processed_item_path)
    if num_all != -1:
        all_df = all_df.head(num_all)

    project_root = Path().resolve()
    duration_dir = domain_work_path("all", "merged_duration", project_root=project_root)
    durations = glob.glob(os.path.join(duration_dir, "*.parquet"))
    if len(durations) < 1:
        duration_dir = project_artifact_path("all", "merged_duration", project_root=project_root)
        merge_parquet_files(data_artifact_path("duration", project_root=project_root), duration_dir)
        durations = glob.glob(os.path.join(duration_dir, "*.parquet"))
    user_profile_list = []
    for duration in durations:
        df = pd.read_parquet(duration, columns=["username", "profile_id"])
        user_profile_list.append(df.drop_duplicates())
    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")
    if num_user != -1:
        user_profile_df = user_profile_df.head(num_user)
    print(f"Data loaded in {time.time()-preprocess_start:.2f} seconds")

    all_pl = pl.from_pandas(all_df)
    total_contents = all_pl.height
    print(f"\nTotal unique contents: {total_contents:,}")

    content_all_pl = pl.read_parquet(content_all_path)
    content_unique = (
        content_all_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}
    return user_profile_df, all_pl, content_dict
