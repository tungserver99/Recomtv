import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
from datetime import datetime
from recsys_vnpt.data_sources import load_user_metadata
from recsys_vnpt.paths import model_artifact_path

ENC_DIR = model_artifact_path("music", "encoder")
os.makedirs(ENC_DIR, exist_ok=True)

def fit_user_encoder(user_data, cat_cols):
    user_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    user_ohe.fit(user_data[cat_cols])

    enc_path = os.path.join(ENC_DIR, "user_ohe.joblib")
    joblib.dump(user_ohe, enc_path)  
    return user_ohe

def fit_user_scaler(user_data, cont_cols):
    user_scaler = StandardScaler()
    user_scaler.fit(user_data[cont_cols])

    scaler_path = os.path.join(ENC_DIR, "user_scaler.joblib")
    joblib.dump(user_scaler, scaler_path)
    return user_scaler

def transform_user_data(user_data, cat_cols, cont_cols):
    enc_path = os.path.join(ENC_DIR, "user_ohe.joblib")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Missing encoder at: {enc_path}")
    
    user_ohe = joblib.load(enc_path)
    arr = user_ohe.transform(user_data[cat_cols])
    cols = user_ohe.get_feature_names_out(cat_cols)
    ohe_df = pd.DataFrame(arr, columns=cols, index=user_data.index)

    scaler_path = os.path.join(ENC_DIR, "user_scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler at: {scaler_path}")
    user_scaler = joblib.load(scaler_path)
    scaled_arr = user_scaler.transform(user_data[cont_cols])
    scaled_df = pd.DataFrame(scaled_arr, columns=cont_cols, index=user_data.index)

    user_data = user_data.drop(columns=cont_cols + cat_cols)
    return pd.concat([user_data, ohe_df, scaled_df], axis=1)


def process_user_data(data_path, output_dir, num_user=-1, mode='train'):
    user_data = load_user_metadata(path=data_path)

    user_data = user_data.fillna(0)

    current_year = datetime.now().year
    user_data['birthday'] = user_data['birthday'].astype(str)
    user_data['birthday'] = user_data['birthday'].apply(
        lambda x: x[-4:] if x.isdigit() and len(x) > 3 else str(current_year)
    )
    user_data['birthday'] = user_data['birthday'].astype(int)


    user_data = user_data.drop(columns=["tentinh"])
    user_data = user_data.drop(columns=["province"])
    user_data['sex'] = user_data['sex'].apply(lambda x: x if str(x).isdigit() else 1)
    user_data['sex'] = user_data['sex'].astype(int)
    user_data['sex'] = user_data['sex'].apply(lambda x: x if x in [0, 1] else 1)

    cat_cols = ["package_code"]
    cont_cols = ["birthday"]

    if mode == 'train':
        fit_user_encoder(user_data, cat_cols)
        fit_user_scaler(user_data, cont_cols)
        
    user_data = transform_user_data(user_data, cat_cols, cont_cols)

    if num_user != -1:
        user_data = user_data.head(num_user)

    user_data_path = os.path.join(Path().resolve(), output_dir, "user_data.parquet")
    os.makedirs(os.path.dirname(user_data_path), exist_ok=True)
    user_data.to_parquet(user_data_path, index=False)

    return user_data
