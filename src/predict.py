   

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

from ml_scratch import mse as mse_s, rmse as rmse_s, r2_score_scratch


def load_pipeline(model_path: str) -> Dict[str, Any]:
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
                                                                                                                      
    return pipeline


def preprocess_with_pipeline(df: pd.DataFrame, pipeline: Dict[str, Any]) -> np.ndarray:
                            
    df = df.copy()
    df.columns = df.columns.str.strip()

    target_col = pipeline.get('target_column')
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

                                           
    feature_names = pipeline['feature_names']
    for col in feature_names:
        if col not in df.columns:
                                                         
            df[col] = np.nan

                                                
    df = df[feature_names]

                               
    num_medians = pipeline.get('numerical_medians', {})
    cat_modes = pipeline.get('categorical_modes', {})
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in num_medians:
                df[col] = df[col].fillna(num_medians[col])
            else:
                df[col] = df[col].fillna(df[col].median())
        else:
            if col in cat_modes:
                df[col] = df[col].fillna(cat_modes[col])
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")

                                              
    encoders = pipeline.get('label_encoders', {})
    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = enc.transform(df[col].astype(str).tolist())

                              
    scaler = pipeline.get('scaler')
    X = df.values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)
    return X


def write_metrics_file(path: str, mse: float, rmse: float, r2: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2:.2f}\n")


def write_predictions_file(path: str, preds: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
                              
    pd.Series(preds).to_csv(path, index=False, header=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved life expectancy model and write metrics/predictions.")
    parser.add_argument('--model_path', required=True, help='Path to the saved model (pipeline) pkl file')
    parser.add_argument('--data_path', required=True, help='Path to the data CSV containing features and true target')
    parser.add_argument('--metrics_output_path', required=True, help='Where to write the metrics txt file')
    parser.add_argument('--predictions_output_path', required=True, help='Where to write the predictions csv file')
    args = parser.parse_args()

    pipeline = load_pipeline(args.model_path)
    target_col = pipeline.get('target_column', 'Life expectancy')

    df = pd.read_csv(args.data_path)
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

                                                                                   
    mask = pd.to_numeric(df[target_col], errors='coerce').replace([np.inf, -np.inf], np.nan).notna()
    df_valid = df.loc[mask].copy()
    y_true = df_valid[target_col].astype(float).values
    X = preprocess_with_pipeline(df_valid, pipeline)

    model = pipeline['model']
    y_pred = model.predict(X)

    mse_v = mse_s(y_true, y_pred)
    rmse_v = rmse_s(y_true, y_pred)
    r2_v = r2_score_scratch(y_true, y_pred)

    write_metrics_file(args.metrics_output_path, mse_v, rmse_v, r2_v)
    write_predictions_file(args.predictions_output_path, y_pred)

    print("Evaluation completed.")
    print(f"Metrics written to: {args.metrics_output_path}")
    print(f"Predictions written to: {args.predictions_output_path}")


if __name__ == '__main__':
    main()

