"""
Data Processing Module
Cleans, encodes, and prepares churn data for modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# Configuration
DATA_RAW_PATH = "C:/Users/afiqu/Desktop/Learning And Evaulating Python/churn-prediction-pipeline/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_PROCESSED_PATH = "C:/Users/afiqu/Desktop/Learning And Evaulating Python/churn-prediction-pipeline/data/processed/"
RANDOM_STATE = 42

print("✅ Data processing module imported successfully")


def load_data(filepath=DATA_RAW_PATH):
    """
    Load raw CSV data

    Args:
        filepath (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def separate_features_target(df, target_col="Churn"):
    """
    Separate target variable from features

    Args:
        df (pd.DataFrame): Full dataset
        target_col (str): Name of target column

    Returns:
        tuple: (features_df, target_series)
    """
    # Remove target column and ID column (not useful for modeling)
    X = df.drop(columns=[target_col, "customerID"])

    # Extract target column
    y = df[target_col]

    print(f"✅ Separated features (X): {X.shape}")
    print(f"✅ Separated target (y): {y.shape}")
    print(f"   Target distribution:\n{y.value_counts()}")

    return X, y


def identify_feature_types(X):
    """
    Identify categorical and numerical columns

    Args:
        X (pd.DataFrame): Features dataframe

    Returns:
        tuple: (categorical_cols, numerical_cols)
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"✅ Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"✅ Numerical columns ({len(numerical_cols)}): {numerical_cols}")

    return categorical_cols, numerical_cols


def encode_categorical_features(X, categorical_cols, fit=True, encoders=None):
    """
    Convert categorical features to numerical

    Args:
        X (pd.DataFrame): Features
        categorical_cols (list): Categorical column names
        fit (bool): If True, fit encoders. If False, use provided encoders.
        encoders (dict): Pre-fitted encoders for transform

    Returns:
        tuple: (X_encoded, encoders_dict)
    """
    X_encoded = X.copy()
    encoders_dict = encoders if encoders else {}

    for col in categorical_cols:
        if fit:
            # Create and fit encoder
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            encoders_dict[col] = le

            # Show what was encoded
            print(
                f"✅ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}"
            )
        else:
            # Use pre-fitted encoder
            X_encoded[col] = encoders_dict[col].transform(X[col])

    return X_encoded, encoders_dict


def scale_numerical_features(X, numerical_cols, fit=True, scaler=None):
    """
    Normalize numerical features to [0, 1] range

    Args:
        X (pd.DataFrame): Features
        numerical_cols (list): Numerical column names
        fit (bool): If True, fit scaler. If False, use provided scaler.
        scaler (StandardScaler): Pre-fitted scaler

    Returns:
        tuple: (X_scaled, scaler_object)
    """
    X_scaled = X.copy()

    if fit:
        # Create and fit scaler
        scaler_obj = StandardScaler()
        X_scaled[numerical_cols] = scaler_obj.fit_transform(X[numerical_cols])

        print(f"✅ Scaled {len(numerical_cols)} numerical features")
        print(f"   Means: {scaler_obj.mean_}")
        print(f"   Stds: {scaler_obj.scale_}")
    else:
        # Use pre-fitted scaler
        X_scaled[numerical_cols] = scaler.transform(X[numerical_cols])

    return X_scaled, scaler_obj if fit else scaler


def create_train_test_split(X, y, test_size=0.15, val_size=0.15):
    """
    Split data into train, validation, and test sets
    Uses stratified split to maintain class distribution

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion for test set (0.15 = 15%)
        val_size (float): Proportion for validation set (0.15 = 15%)

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: 70% train, 30% (test + val)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(test_size + val_size),
        stratify=y,  # Keep same churn ratio in each split
        random_state=RANDOM_STATE,
    )

    # Second split: Split the 30% into equal val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,  # Split 30% into 15% val, 15% test
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print(
        f"✅ Train set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)"
    )
    print(
        f"✅ Val set: {X_val.shape[0]} samples ({X_val.shape[0] / len(X) * 100:.1f}%)"
    )
    print(
        f"✅ Test set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)"
    )

    # Verify stratification
    print("\nChurn distribution:")
    print(f"  Original: {y.value_counts(normalize=True).to_dict()}")
    print(f"  Train: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"  Val: {y_val.value_counts(normalize=True).to_dict()}")
    print(f"  Test: {y_test.value_counts(normalize=True).to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def engineer_features(X, fit=True):
    """
    Create new features from existing ones

    Args:
        X (pd.DataFrame): Features
        fit (bool): If True, return engineered features. If False, just return.

    Returns:
        pd.DataFrame: Features with new columns
    """
    X_eng = X.copy()

    if fit:
        # Example feature engineering ideas:

        # 1. Customer lifetime value indicator
        # (Months as customer × Monthly Charges)
        if "Tenure" in X.columns and "MonthlyCharges" in X.columns:
            X_eng["LifetimeValue"] = X["Tenure"] * X["MonthlyCharges"]
            print("✅ Created LifetimeValue = Tenure × MonthlyCharges")

        # 2. Monthly charge efficiency
        # (Total charges / tenure)
        if "TotalCharges" in X.columns and "Tenure" in X.columns:
            # Avoid division by zero
            X_eng["ChargePerMonth"] = X["TotalCharges"] / (X["Tenure"] + 1)
            print("✅ Created ChargePerMonth = TotalCharges / Tenure")

        # 3. Is long-term customer (tenure > 1 year)
        if "Tenure" in X.columns:
            X_eng["IsLongTermCustomer"] = (X["Tenure"] >= 12).astype(int)
            print("✅ Created IsLongTermCustomer (1 if tenure >= 12 months)")

    return X_eng


def save_processed_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    scaler,
    encoders,
    categorical_cols,
    numerical_cols,
):
    """
    Save all processed data and objects to files

    Args:
        All processed data and preprocessing objects
    """
    # Create processed directory if it doesn't exist
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)

    # Save datasets
    X_train.to_csv(f"{DATA_PROCESSED_PATH}X_train.csv", index=False)
    X_val.to_csv(f"{DATA_PROCESSED_PATH}X_val.csv", index=False)
    X_test.to_csv(f"{DATA_PROCESSED_PATH}X_test.csv", index=False)
    y_train.to_csv(f"{DATA_PROCESSED_PATH}y_train.csv", index=False)
    y_val.to_csv(f"{DATA_PROCESSED_PATH}y_val.csv", index=False)
    y_test.to_csv(f"{DATA_PROCESSED_PATH}y_test.csv", index=False)

    # Save preprocessing objects (for later use)
    with open(f"{DATA_PROCESSED_PATH}scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{DATA_PROCESSED_PATH}encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Save feature names
    with open(f"{DATA_PROCESSED_PATH}categorical_cols.pkl", "wb") as f:
        pickle.dump(categorical_cols, f)

    with open(f"{DATA_PROCESSED_PATH}numerical_cols.pkl", "wb") as f:
        pickle.dump(numerical_cols, f)

    print(f"✅ X_train saved: {X_train.shape}")
    print(f"✅ X_val saved: {X_val.shape}")
    print(f"✅ X_test saved: {X_test.shape}")
    print(f"✅ y_train saved: {y_train.shape}")
    print(f"✅ y_val saved: {y_val.shape}")
    print(f"✅ y_test saved: {y_test.shape}")
    print(f"✅ Scaler saved")
    print(f"✅ Encoders saved")
    print(f"✅ Feature lists saved")


def process_data(filepath=DATA_RAW_PATH):
    """
    Main function: Load → Clean → Encode → Scale → Split

    Args:
        filepath (str): Path to raw data

    Returns:
        dict: Contains all processed data and objects
    """
    print("=" * 70)
    print("DATA PROCESSING PIPELINE")
    print("=" * 70)

    # 1. Load data
    print("\n1️⃣ LOADING DATA")
    print("-" * 70)
    df = load_data(filepath)

    # 2. Separate features and target
    print("\n2️⃣ SEPARATING FEATURES AND TARGET")
    print("-" * 70)
    X, y = separate_features_target(df)

    # 3. Identify feature types
    print("\n3️⃣ IDENTIFYING FEATURE TYPES")
    print("-" * 70)
    categorical_cols, numerical_cols = identify_feature_types(X)

    # 4. Encode categorical features
    print("\n4️⃣ ENCODING CATEGORICAL FEATURES")
    print("-" * 70)
    X_encoded, encoders = encode_categorical_features(X, categorical_cols, fit=True)

    # 5. Scale numerical features
    print("\n5️⃣ SCALING NUMERICAL FEATURES")
    print("-" * 70)
    X_scaled, scaler = scale_numerical_features(X_encoded, numerical_cols, fit=True)

    # 6. Engineer features
    print("\n6️⃣ ENGINEERING FEATURES")
    print("-" * 70)
    X_engineered = engineer_features(X_scaled, fit=True)

    # 7. Create train/val/test split
    print("\n7️⃣ CREATING TRAIN/VAL/TEST SPLIT")
    print("-" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(
        X_engineered, y
    )

    # 8. Save everything
    print("\n8️⃣ SAVING PROCESSED DATA")
    print("-" * 70)
    save_processed_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        encoders,
        categorical_cols,
        numerical_cols,
    )

    print("\n" + "=" * 70)
    print("✅ DATA PROCESSING COMPLETE!")
    print("=" * 70)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
    }


if __name__ == "__main__":
    # Run when script is executed directly
    result = process_data()

    # You now have:
    print(f"\n✅ Available in result dict:")
    print(f"   - X_train, X_val, X_test")
    print(f"   - y_train, y_val, y_test")
    print(f"   - scaler (for scaling)")
    print(f"   - encoders (for encoding)")
    print(f"   - categorical_cols, numerical_cols")
