import pandas as pd
import numpy as np
import os
from ml_scratch import StandardScalerScratch, LabelEncoderScratch, train_test_split_scratch

def load_data(data_path):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [data_path]
    if not os.path.isabs(data_path):
        candidates.append(os.path.abspath(os.path.join(here, data_path)))
    base_dir = os.path.dirname(candidates[-1]) if candidates else here
    candidates.append(os.path.join(base_dir, "train_data.csv"))
    candidates.append(os.path.join(base_dir, "Life_Expectancy.csv"))
    candidates.append(os.path.join(root, "data", "Life_Expectancy.csv"))
    for p in candidates:
        try:
            frame = pd.read_csv(p)
            frame.columns = frame.columns.str.strip()
            return frame
        except FileNotFoundError:
            continue
    return None

def clean_data(frame_in):
    frame = frame_in.copy()
    num_cols = frame.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if frame[c].isnull().any():
            frame[c] = frame[c].fillna(frame[c].median())
    cat_cols = frame.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        if frame[c].isnull().any():
            mode_val = frame[c].mode()
            if not mode_val.empty:
                frame[c] = frame[c].fillna(mode_val.iloc[0])
    return frame

def encode_categorical_features(frame_in, target_name):
    enc_map = {}
    frame = frame_in.copy()
    for c in frame.select_dtypes(include=["object"]).columns:
        if c == target_name:
            continue
        encoder = LabelEncoderScratch()
        frame[c] = encoder.fit_transform(frame[c].astype(str).tolist())
        enc_map[c] = encoder
    return frame, enc_map

def prepare_features_target(frame_in, target_name):
    feat_df = frame_in.drop(columns=[target_name])
    tgt_series = frame_in[target_name]
    return feat_df, tgt_series

def split_data(feat_df, tgt_series, test_size=0.2, random_state=42):
    feat_mat = np.asarray(feat_df, dtype=float)
    tgt_vec = np.asarray(tgt_series, dtype=float)
    feat_fit, feat_hold, tgt_fit, tgt_hold = train_test_split_scratch(
        feat_mat, tgt_vec, test_size=test_size, random_state=random_state
    )
    return feat_fit, feat_hold, tgt_fit, tgt_hold

def scale_features(feat_fit, feat_hold):
    norm = StandardScalerScratch()
    x_fit = norm.fit_transform(feat_fit)
    x_hold = norm.transform(feat_hold)
    return x_fit, x_hold, norm

def preprocess_pipeline(path_in, target_name, test_size=0.2, random_state=42):
    raw = load_data(path_in)
    if raw is None:
        return None
    raw.columns = raw.columns.str.strip()
    if target_name not in raw.columns:
        return None
    base = clean_data(raw)
    num_cols = base.select_dtypes(include=[np.number]).columns
    cat_cols = base.select_dtypes(include=["object"]).columns
    med_map = {c: float(base[c].median()) for c in num_cols}
    mode_map = {c: str(base[c].mode()[0]) for c in cat_cols if base[c].nunique() > 0}
    enc_df, encoders = encode_categorical_features(base, target_name)
    feat_df, tgt_series = prepare_features_target(enc_df, target_name)
    x_tr, x_te, y_tr, y_te = split_data(feat_df, tgt_series, test_size, random_state)
    x_tr_s, x_te_s, scaler = scale_features(x_tr, x_te)
    return {
        "X_train": x_tr_s,
        "X_test": x_te_s,
        "y_train": y_tr,
        "y_test": y_te,
        "scaler": scaler,
        "label_encoders": encoders,
        "feature_names": list(feat_df.columns),
        "original_data": raw,
        "numerical_medians": med_map,
        "categorical_modes": mode_map,
    }

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ds_path = os.path.join(here, "..", "data", "train_data.csv")
    tgt = "Life expectancy"
    _ = preprocess_pipeline(ds_path, tgt)