from recsys_vnpt.data_sources import write_merged_duration_parts

def merge_parquet_files(duration_folder_path, merged_duration_folder_path):
    paths = write_merged_duration_parts(duration_folder_path, merged_duration_folder_path, parts=5)
    for path in paths:
        print(f"Saved merged duration part to {path}")
