"""
Prediction module for life expectancy.
This module loads trained models and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

class LifeExpectancyPredictor:
    """
    A class to handle life expectancy predictions using trained models.
    """
    
    def __init__(self, model_path="../models/regression_model_final.pkl"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.model_path = model_path
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model from pickle file.
        """
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file {self.model_path} not found.")
            print("Please train a model first using train_model.py")
            self.model = None
    
    def load_preprocessing_objects(self, scaler_path=None, encoders_path=None):
        """
        Load preprocessing objects (scaler and label encoders).
        
        Args:
            scaler_path: Path to the scaler file
            encoders_path: Path to the label encoders file
        """
        # For now, we'll create a simple scaler
        # In a real scenario, you'd save and load these from training
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded successfully")
        else:
            print("Warning: No scaler found. Using default scaling.")
            self.scaler = StandardScaler()
        
        if encoders_path and os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print("Label encoders loaded successfully")
        else:
            print("Warning: No label encoders found.")
            self.label_encoders = {}
    
    def preprocess_new_data(self, data, target_column=None):
        """
        Preprocess new data for prediction.
        
        Args:
            data: New data (DataFrame or dict)
            target_column: Name of target column (if present, will be removed)
            
        Returns:
            np.array: Preprocessed features ready for prediction
        """
        print("Preprocessing new data...")
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Remove target column if present
        if target_column and target_column in data.columns:
            data = data.drop(columns=[target_column])
        
        # Handle missing values
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Encode categorical features
        for col in categorical_cols:
            if col in self.label_encoders:
                # Use existing encoder
                data[col] = self.label_encoders[col].transform(data[col])
            else:
                # Create new encoder for unseen categories
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        # Scale features
        if self.scaler is not None:
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data.values
        
        print("Data preprocessing completed")
        return data_scaled
    
    def predict(self, data, target_column=None):
        """
        Make predictions on new data.
        
        Args:
            data: New data for prediction
            target_column: Name of target column (if present)
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            print("Error: No model loaded. Cannot make predictions.")
            return None
        
        # Preprocess the data
        processed_data = self.preprocess_new_data(data, target_column)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        print(f"Predictions made for {len(predictions)} samples")
        return predictions
    
    def predict_single(self, **kwargs):
        """
        Make a prediction for a single sample.
        
        Args:
            **kwargs: Feature values for prediction
            
        Returns:
            float: Single prediction
        """
        # Convert kwargs to DataFrame
        data = pd.DataFrame([kwargs])
        
        # Make prediction
        predictions = self.predict(data)
        
        if predictions is not None:
            return predictions[0]
        else:
            return None
    
    def predict_batch(self, data_file, output_file=None):
        """
        Make predictions for a batch of data from a CSV file.
        
        Args:
            data_file: Path to CSV file with new data
            output_file: Path to save predictions (optional)
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        print(f"Loading data from {data_file}...")
        
        try:
            data = pd.read_csv(data_file)
            print(f"Loaded {len(data)} samples")
            
            # Make predictions
            predictions = self.predict(data)
            
            if predictions is not None:
                # Add predictions to the original data
                result_df = data.copy()
                result_df['predicted_life_expectancy'] = predictions
                
                # Save to file if specified
                if output_file:
                    result_df.to_csv(output_file, index=False)
                    print(f"Predictions saved to {output_file}")
                
                return result_df
            else:
                return None
                
        except FileNotFoundError:
            print(f"Error: File {data_file} not found.")
            return None

def example_usage():
    """
    Example of how to use the predictor.
    """
    print("Life Expectancy Prediction Example")
    print("=" * 40)
    
    # Initialize predictor
    predictor = LifeExpectancyPredictor()
    
    if predictor.model is None:
        print("No trained model found. Please train a model first.")
        return
    
    # Example 1: Single prediction
    print("\nExample 1: Single Prediction")
    print("-" * 30)
    
    # Example feature values (adjust these based on your actual features)
    sample_data = {
        'gdp_per_capita': 50000,
        'population': 1000000,
        'healthcare_expenditure': 8.5,
        'education_index': 0.8,
        'country': 'United States'  # This would be encoded
    }
    
    prediction = predictor.predict_single(**sample_data)
    if prediction is not None:
        print(f"Predicted life expectancy: {prediction:.2f} years")
    
    # Example 2: Batch prediction
    print("\nExample 2: Batch Prediction")
    print("-" * 30)
    
    # Create sample data for batch prediction
    sample_batch = pd.DataFrame({
        'gdp_per_capita': [50000, 30000, 80000],
        'population': [1000000, 500000, 2000000],
        'healthcare_expenditure': [8.5, 6.2, 9.1],
        'education_index': [0.8, 0.6, 0.9],
        'country': ['United States', 'Canada', 'Germany']
    })
    
    batch_predictions = predictor.predict(sample_batch)
    if batch_predictions is not None:
        print("Batch predictions:")
        for i, pred in enumerate(batch_predictions):
            print(f"Sample {i+1}: {pred:.2f} years")

if __name__ == "__main__":
    # Run example
    example_usage()

