"""
Data preprocessing module for life expectancy prediction.
This module handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Cleaning data...")
    
    # Display basic info about the dataset
    print(f"Original shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"After cleaning shape: {df.shape}")
    return df

def encode_categorical_features(df, target_column):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    print("Encoding categorical features...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Combine features and target
    df_encoded = pd.concat([X, y], axis=1)
    
    print(f"Encoded categorical columns: {list(categorical_cols)}")
    return df_encoded, label_encoders

def prepare_features_target(df, target_column):
    """
    Prepare features (X) and target (y) for modeling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    print(f"Preparing features and target (target: {target_column})...")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting data (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully.")
    return X_train_scaled, X_test_scaled, scaler

def preprocess_pipeline(file_path, target_column, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline.
    
    Args:
        file_path (str): Path to the CSV file
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing all processed data and objects
    """
    print("Starting preprocessing pipeline...")
    
    # Load data
    df = load_data(file_path)
    if df is None:
        return None
    
    # Clean data
    df_clean = clean_data(df)
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(df_clean, target_column)
    
    # Prepare features and target
    X, y = prepare_features_target(df_encoded, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("Preprocessing pipeline completed successfully!")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
        'original_data': df
    }

if __name__ == "__main__":
    # Example usage
    data_path = "../data/train_data.csv"
    target_col = "life_expectancy"  # Adjust this to your actual target column name
    
    # Run preprocessing pipeline
    processed_data = preprocess_pipeline(data_path, target_col)
    
    if processed_data:
        print("\nPreprocessing completed successfully!")
        print(f"Training features shape: {processed_data['X_train'].shape}")
        print(f"Training target shape: {processed_data['y_train'].shape}")

