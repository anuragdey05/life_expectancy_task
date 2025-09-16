"""
Model training module for life expectancy prediction.
This module implements linear regression and other models.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import our preprocessing module
from data_preprocessing import preprocess_pipeline

class LifeExpectancyModel:
    """
    A class to handle life expectancy prediction models.
    """
    
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """
        Initialize different regression models.
        """
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        print("Models initialized successfully!")
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate their performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
        """
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred, "train")
            test_metrics = self.calculate_metrics(y_test, y_test_pred, "test")
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Store metrics
            self.model_metrics[name] = {
                'train': train_metrics,
                'test': test_metrics,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': {
                    'train': y_train_pred,
                    'test': y_test_pred
                }
            }
            
            print(f"{name} - R² Score: {test_metrics['r2']:.4f}")
            print(f"{name} - RMSE: {test_metrics['rmse']:.4f}")
            print(f"{name} - MAE: {test_metrics['mae']:.4f}")
            print(f"{name} - CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def calculate_metrics(self, y_true, y_pred, dataset_type):
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_type: 'train' or 'test'
            
        Returns:
            dict: Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def select_best_model(self, metric='r2'):
        """
        Select the best model based on test performance.
        
        Args:
            metric: Metric to use for selection ('r2', 'rmse', 'mae')
        """
        print(f"\nSelecting best model based on {metric}...")
        
        best_score = float('-inf') if metric == 'r2' else float('inf')
        
        for name, metrics in self.model_metrics.items():
            score = metrics['test'][metric]
            
            if metric == 'r2':
                if score > best_score:
                    best_score = score
                    self.best_model_name = name
                    self.best_model = self.models[name]
            else:  # For RMSE and MAE, lower is better
                if score < best_score:
                    best_score = score
                    self.best_model_name = name
                    self.best_model = self.models[name]
        
        print(f"Best model: {self.best_model_name}")
        print(f"Best {metric}: {best_score:.4f}")
    
    def save_models(self, models_dir="../models/"):
        """
        Save all trained models to pickle files.
        
        Args:
            models_dir: Directory to save models
        """
        print(f"\nSaving models to {models_dir}...")
        
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Save individual models
        for i, (name, model) in enumerate(self.models.items(), 1):
            filename = f"{models_dir}regression_model{i}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} as regression_model{i}.pkl")
        
        # Save the best model as final
        if self.best_model is not None:
            final_filename = f"{models_dir}regression_model_final.pkl"
            with open(final_filename, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"Saved best model ({self.best_model_name}) as regression_model_final.pkl")
    
    def save_metrics(self, metrics_file="../results/train_metrics.txt"):
        """
        Save training metrics to a text file.
        
        Args:
            metrics_file: Path to save metrics
        """
        print(f"\nSaving metrics to {metrics_file}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            f.write("LIFE EXPECTANCY PREDICTION - MODEL PERFORMANCE METRICS\n")
            f.write("=" * 60 + "\n\n")
            
            for name, metrics in self.model_metrics.items():
                f.write(f"MODEL: {name.upper()}\n")
                f.write("-" * 30 + "\n")
                
                # Training metrics
                f.write("Training Performance:\n")
                train_metrics = metrics['train']
                f.write(f"  R² Score: {train_metrics['r2']:.4f}\n")
                f.write(f"  RMSE: {train_metrics['rmse']:.4f}\n")
                f.write(f"  MAE: {train_metrics['mae']:.4f}\n")
                f.write(f"  MSE: {train_metrics['mse']:.4f}\n")
                
                # Test metrics
                f.write("Test Performance:\n")
                test_metrics = metrics['test']
                f.write(f"  R² Score: {test_metrics['r2']:.4f}\n")
                f.write(f"  RMSE: {test_metrics['rmse']:.4f}\n")
                f.write(f"  MAE: {test_metrics['mae']:.4f}\n")
                f.write(f"  MSE: {test_metrics['mse']:.4f}\n")
                
                # Cross-validation
                f.write("Cross-Validation:\n")
                f.write(f"  R² Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
            
            # Best model summary
            if self.best_model_name:
                f.write("BEST MODEL SUMMARY\n")
                f.write("=" * 30 + "\n")
                f.write(f"Best Model: {self.best_model_name}\n")
                best_metrics = self.model_metrics[self.best_model_name]['test']
                f.write(f"Test R² Score: {best_metrics['r2']:.4f}\n")
                f.write(f"Test RMSE: {best_metrics['rmse']:.4f}\n")
                f.write(f"Test MAE: {best_metrics['mae']:.4f}\n")
        
        print("Metrics saved successfully!")
    
    def save_predictions(self, predictions_file="../results/train_predictions.csv"):
        """
        Save predictions to CSV file.
        
        Args:
            predictions_file: Path to save predictions
        """
        print(f"\nSaving predictions to {predictions_file}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
        
        # Prepare predictions data
        predictions_data = []
        
        for name, metrics in self.model_metrics.items():
            train_pred = metrics['predictions']['train']
            test_pred = metrics['predictions']['test']
            
            # Add training predictions
            for i, pred in enumerate(train_pred):
                predictions_data.append({
                    'model': name,
                    'dataset': 'train',
                    'index': i,
                    'prediction': pred
                })
            
            # Add test predictions
            for i, pred in enumerate(test_pred):
                predictions_data.append({
                    'model': name,
                    'dataset': 'test',
                    'index': i,
                    'prediction': pred
                })
        
        # Save to CSV
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(predictions_file, index=False)
        
        print("Predictions saved successfully!")
        print(f"Total predictions saved: {len(predictions_data)}")

def train_life_expectancy_models(data_path="../data/train_data.csv", 
                                target_column="life_expectancy"):
    """
    Main function to train life expectancy prediction models.
    
    Args:
        data_path: Path to the training data
        target_column: Name of the target column
    """
    print("Starting Life Expectancy Model Training")
    print("=" * 50)
    
    # Step 1: Preprocess data
    print("\nStep 1: Data Preprocessing")
    processed_data = preprocess_pipeline(data_path, target_column)
    
    if processed_data is None:
        print("Error: Data preprocessing failed!")
        return None
    
    # Step 2: Initialize and train models
    print("\nStep 2: Model Training")
    model_trainer = LifeExpectancyModel()
    model_trainer.initialize_models()
    
    model_trainer.train_models(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    # Step 3: Select best model
    print("\nStep 3: Model Selection")
    model_trainer.select_best_model()
    
    # Step 4: Save everything
    print("\nStep 4: Saving Results")
    model_trainer.save_models()
    model_trainer.save_metrics()
    model_trainer.save_predictions()
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    
    return model_trainer

if __name__ == "__main__":
    # Run the training pipeline
    trainer = train_life_expectancy_models()

