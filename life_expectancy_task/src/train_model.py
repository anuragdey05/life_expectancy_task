

import pandas as pd
import numpy as np
import pickle
import os
from ml_scratch import (
    LinearRegressionScratch,
    RidgeRegressionScratch,
    LassoRegressionScratch,
    mse as mse_s,
    rmse as rmse_s,
    r2_score_scratch,
    cross_val_score_scratch,
)
from data_preprocessing import preprocess_pipeline


def _atomic_pickle_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class LifeExpectancyModel:
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.target_column = None
        self.numerical_medians = None
        self.categorical_modes = None

    def initialize_models(self):
        self.models = {
            'linear_regression': LinearRegressionScratch(),
            'ridge_regression': RidgeRegressionScratch(alpha=1.0),
            'lasso_regression': LassoRegressionScratch(alpha=0.1, lr=0.001, n_iter=2000),
        }

    def train_models(self, X_train, y_train, X_test, y_test):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)

            cv_mean, cv_std = cross_val_score_scratch(
                type(model), X_train, y_train, cv=5, metric='r2', random_state=42,
                **({} if not isinstance(model, RidgeRegressionScratch) else {'alpha': model.alpha})
            )

            self.model_metrics[name] = {
                'train': train_metrics,
                'test': test_metrics,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': {
                    'train': y_train_pred,
                    'test': y_test_pred
                }
            }

    def calculate_metrics(self, y_true, y_pred):
        mse = mse_s(y_true, y_pred)
        rmse = rmse_s(y_true, y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = r2_score_scratch(y_true, y_pred)
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    def select_best_model(self, metric='r2'):
        best_score = float('-inf') if metric == 'r2' else float('inf')
        for name, metrics in self.model_metrics.items():
            score = metrics['test'][metric]
            if metric == 'r2':
                if score > best_score:
                    best_score = score
                    self.best_model_name = name
                    self.best_model = self.models[name]
            else:
                if score < best_score:
                    best_score = score
                    self.best_model_name = name
                    self.best_model = self.models[name]

    def save_models(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        os.makedirs(models_dir, exist_ok=True)

        for i, (name, model) in enumerate(self.models.items(), 1):
            filename = os.path.join(models_dir, f"regression_model{i}.pkl")
            pipeline = {
                'model': model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'target_column': self.target_column,
                'numerical_medians': self.numerical_medians,
                'categorical_modes': self.categorical_modes,
            }
            _atomic_pickle_dump(pipeline, filename)

        if self.best_model is not None:
            final_filename = os.path.join(models_dir, "regression_model_final.pkl")
            pipeline = {
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'target_column': self.target_column,
                'numerical_medians': self.numerical_medians,
                'categorical_modes': self.categorical_modes,
            }
            _atomic_pickle_dump(pipeline, final_filename)

    def save_metrics(self, metrics_file=None):
        if metrics_file is None:
            metrics_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "train_metrics.txt"))
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

        with open(metrics_file, 'w') as f:
            f.write("LIFE EXPECTANCY PREDICTION - MODEL PERFORMANCE METRICS\n")
            f.write("=" * 60 + "\n\n")
            for name, metrics in self.model_metrics.items():
                f.write(f"MODEL: {name.upper()}\n")
                f.write("-" * 30 + "\n")
                train_metrics = metrics['train']
                f.write(f"Training - R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, MSE: {train_metrics['mse']:.4f}\n")
                test_metrics = metrics['test']
                f.write(f"Test - R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, MSE: {test_metrics['mse']:.4f}\n")
                f.write(f"Cross-Validation R²: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})\n")
                f.write("\n" + "=" * 60 + "\n\n")
            if self.best_model_name:
                f.write("BEST MODEL SUMMARY\n")
                f.write("=" * 30 + "\n")
                f.write(f"Best Model: {self.best_model_name}\n")
                best_metrics = self.model_metrics[self.best_model_name]['test']
                f.write(f"Test R²: {best_metrics['r2']:.4f}, RMSE: {best_metrics['rmse']:.4f}, MAE: {best_metrics['mae']:.4f}\n")

    def save_predictions(self, predictions_file=None):
        if predictions_file is None:
            predictions_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "train_predictions.csv"))
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)

        predictions_data = []
        for name, metrics in self.model_metrics.items():
            train_pred = metrics['predictions']['train']
            test_pred = metrics['predictions']['test']
            for i, pred in enumerate(train_pred):
                predictions_data.append({'model': name, 'dataset': 'train', 'index': i, 'prediction': pred})
            for i, pred in enumerate(test_pred):
                predictions_data.append({'model': name, 'dataset': 'test', 'index': i, 'prediction': pred})

        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(predictions_file, index=False)


def train_life_expectancy_models(data_path="../data/train_data.csv", target_column="Life expectancy"):
    processed_data = preprocess_pipeline(data_path, target_column)
    if processed_data is None:
        return None

    model_trainer = LifeExpectancyModel()
    model_trainer.initialize_models()
    model_trainer.scaler = processed_data['scaler']
    model_trainer.label_encoders = processed_data['label_encoders']
    model_trainer.feature_names = processed_data['feature_names']
    model_trainer.target_column = target_column
    model_trainer.numerical_medians = processed_data.get('numerical_medians')
    model_trainer.categorical_modes = processed_data.get('categorical_modes')

    model_trainer.train_models(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_test'],
        processed_data['y_test']
    )

    model_trainer.select_best_model()
    model_trainer.save_models()
    model_trainer.save_metrics()
    model_trainer.save_predictions()

    return model_trainer


if __name__ == "__main__":
    trainer = train_life_expectancy_models()

