import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from recsys_vnpt.config import DOMAIN_CONFIGS, get_domain_config
from recsys_vnpt.features import FeatureSchema


def test_single_domain_configs_do_not_mix_content_patterns():
    assert get_domain_config("clip").content_patterns == ("content_clip_2*.json",)
    assert get_domain_config("movie").content_patterns == ("content_movie_2*.json",)
    assert get_domain_config("music").content_patterns == ("music_video_2*.json",)

    all_patterns = set(get_domain_config("all").content_patterns)
    single_domain_patterns = set()
    for domain in ("clip", "movie", "music"):
        single_domain_patterns.update(get_domain_config(domain).content_patterns)

    assert all_patterns == single_domain_patterns
    assert get_domain_config("all").uses_family_label_policy is True


def test_domain_configs_have_separate_model_artifact_names():
    assert DOMAIN_CONFIGS["clip"].item_output_name == "clip_item_data.parquet"
    assert DOMAIN_CONFIGS["movie"].item_output_name == "movie_item_data.parquet"
    assert DOMAIN_CONFIGS["music"].item_output_name == "music_item_data.parquet"
    assert DOMAIN_CONFIGS["all"].item_output_name == "all_item_data.parquet"


def test_ohe_unknown_category_keeps_dimension_and_maps_to_zero_vector():
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(pd.DataFrame({"package_code": ["MYTV021", "VASC011"]}))

    transformed = encoder.transform(pd.DataFrame({"package_code": ["NEW_PACKAGE"]}))

    assert transformed.shape == (1, 2)
    assert transformed.tolist() == [[0.0, 0.0]]


def test_mlb_unknown_category_keeps_dimension_and_ignores_unknown_label():
    encoder = MultiLabelBinarizer(sparse_output=False)
    encoder.fit([["sports"], ["movie"]])

    transformed = encoder.transform([["brand_new_category"]])

    assert transformed.shape == (1, 2)
    assert transformed.tolist() == [[0, 0]]


def test_feature_schema_drops_extra_inference_columns_and_restores_missing_train_columns():
    training_frame = pd.DataFrame(
        {
            "label": [1, 0],
            "known_feature": [1.0, 2.0],
            "rare_feature_seen_in_train": [0.0, 1.0],
        }
    )
    inference_frame = pd.DataFrame(
        {
            "known_feature": [5.0],
            "new_ohe_column_from_inference_only": [1.0],
        }
    )

    schema = FeatureSchema.fit(training_frame)
    transformed = schema.transform(inference_frame)

    assert list(transformed.columns) == ["known_feature", "rare_feature_seen_in_train"]
    assert np.allclose(transformed.to_numpy(), [[5.0, 0.0]])
