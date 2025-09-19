   

import pandas as pd
import numpy as np
import os
from ml_scratch import (
    StandardScalerScratch,
    LabelEncoderScratch,
    train_test_split_scratch,
)

def load_data(file_path):
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

                                      
    candidates = []
    candidates.append(file_path)
    if not os.path.isabs(file_path):
        candidates.append(os.path.abspath(os.path.join(script_dir, file_path)))

                                                                               
    base_dir = os.path.dirname(candidates[-1]) if candidates else script_dir
    candidates.append(os.path.join(base_dir, "Life_Expectancy.csv"))

                                               
    candidates.append(os.path.join(repo_root, "data", "Life_Expectancy.csv"))

    last_err = None
    for path in candidates:
        try:
            data = pd.read_csv(path)
            data.columns = data.columns.str.strip()
            print(f"Data loaded successfully from {path}. Shape: {data.shape}")
            return data
        except FileNotFoundError as e:
            last_err = e
            continue

    print("Error: Could not locate the dataset.")
    print("Tried the following paths:")
    for p in candidates:
        print(f" - {p}")
    print("Tip: Place your training CSV at life_expectancy_task/data/train_data.csv")
    print("or at data/Life_Expectancy.csv (temporary fallback).")
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
    
                                          
    print(f"Original shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
                           
                                             
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
                                             
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
    
                                  
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
                                
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoderScratch()
        X[col] = le.fit_transform(X[col].astype(str).tolist())
        label_encoders[col] = le
    
                                 
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
    
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split_scratch(
        X_np, y_np, test_size=test_size, random_state=random_state
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
    
    scaler = StandardScalerScratch()
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
    
               
    df = load_data(file_path)
    if df is None:
        return None
    
                
    df_clean = clean_data(df)

                                                                                 
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    numerical_medians = {col: float(df_clean[col].median()) for col in numerical_cols}
    categorical_modes = {col: str(df_clean[col].mode()[0]) for col in categorical_cols if df_clean[col].nunique() > 0}
    
                                 
    df_encoded, label_encoders = encode_categorical_features(df_clean, target_column)
    
                                 
    X, y = prepare_features_target(df_encoded, target_column)
    
                
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
                    
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
        'original_data': df,
        'numerical_medians': numerical_medians,
        'categorical_modes': categorical_modes,
    }

if __name__ == "__main__":
                   
                                                                                    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "train_data.csv")
    target_col = "Life expectancy"                                    
    
                                
    processed_data = preprocess_pipeline(data_path, target_col)
    
    if processed_data:
        print("\nPreprocessing completed successfully!")
        print(f"Training features shape: {processed_data['X_train'].shape}")
        print(f"Training target shape: {processed_data['y_train'].shape}")

