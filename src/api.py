"""
REST API for Churn Prediction Model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
import os
import traceback
import logging
import warnings
import json
from datetime import datetime
import logging

# Suppress sklearn warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    filename="api_metrics.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn using ML model",
    version="1.0.0",
)

# Global variables
model = None
encoders = None
scaler = None

# Define which columns are categorical
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Define which columns are numerical
NUMERICAL_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges"]

# INPUT SCHEMA


class CustomerData(BaseModel):
    """Customer data for churn prediction"""

    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., description="Months (0-72)")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(
        ..., description="Electronic check, Mailed check, Bank transfer, Credit card"
    )
    MonthlyCharges: float = Field(..., description="Monthly bill (e.g., 65.5)")
    TotalCharges: float = Field(
        ..., description="Total lifetime charges (e.g., 1570.0)"
    )

    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.5,
                "TotalCharges": 1570.0,
            }
        }


# OUTPUT SCHEMA


class PredictionResponse(BaseModel):
    """Response from prediction endpoint"""

    churn: str
    churn_probability: float
    recommendation: str


class HealthResponse(BaseModel):
    """Response from health check"""

    status: str
    model: str
    encoders: str
    scaler: str
    error: str = None


# LOAD MODEL, ENCODERS, AND SCALER AT STARTUP


@app.on_event("startup")
def load_artifacts():
    """Load model, encoders, and scaler at server startup"""
    global model, encoders, scaler

    try:
        cwd = os.getcwd()
        logger.info(f"📂 Working directory: {cwd}")

        api_dir = os.path.dirname(__file__)

        # ===== Load Model =====
        model_path = os.path.abspath(
            os.path.join(api_dir, "../models/logistic_regression_tuned.pkl")
        )
        logger.info(f"🔍 Looking for model at: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info("📦 Loading model...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("✅ Model loaded!")

        # ===== Load Encoders from data/processed/ =====
        encoders_path = os.path.abspath(
            os.path.join(api_dir, "../data/processed/encoders.pkl")
        )
        logger.info(f"🔍 Looking for encoders at: {encoders_path}")

        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders not found at {encoders_path}")

        logger.info("📦 Loading encoders...")
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)
        logger.info(f"✅ Encoders loaded! ({len(encoders)} columns in file)")
        logger.info(f"   Will use categorical columns: {CATEGORICAL_COLUMNS}")

        # ===== Load Scaler from data/processed/ =====
        scaler_path = os.path.abspath(
            os.path.join(api_dir, "../data/processed/scaler.pkl")
        )
        logger.info(f"🔍 Looking for scaler at: {scaler_path}")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        logger.info(f"📦 Loading scaler...")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"✅ Scaler loaded!")
        logger.info(f"   Will scale numerical columns: {NUMERICAL_COLUMNS}")

        # ===== Test Everything Works =====
        logger.info("🧪 Testing with dummy data...")
        test_data = np.zeros((1, 19))
        test_pred = model.predict(test_data)
        logger.info(f"✅ Test prediction successful: {test_pred[0]}")

        logger.info("=" * 60)
        logger.info("✅ ALL ARTIFACTS LOADED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Failed to load artifacts: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# HEALTH CHECK


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if API is healthy and all artifacts are loaded"""
    try:
        status_checks = {
            "model": model is not None,
            "encoders": encoders is not None,
            "scaler": scaler is not None,
        }

        if not all(status_checks.values()):
            return HealthResponse(
                status="unhealthy",
                model="not loaded" if not status_checks["model"] else "loaded",
                encoders="not loaded" if not status_checks["encoders"] else "loaded",
                scaler="not loaded" if not status_checks["scaler"] else "loaded",
                error="Some artifacts not loaded",
            )

        # Test prediction
        test_data = np.zeros((1, 19))
        model.predict(test_data)

        return HealthResponse(
            status="healthy", model="loaded", encoders="loaded", scaler="loaded"
        )

    except Exception as e:
        error = str(e)
        logger.error(f"Health check failed: {error}")
        return HealthResponse(
            status="unhealthy",
            model="error",
            encoders="error",
            scaler="error",
            error=error,
        )


# MAIN PREDICTION ENDPOINT


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Make churn prediction
    """
    try:
        logger.info("Prediction request received")

        # Check all artifacts loaded
        if model is None or encoders is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model artifacts not loaded")

        # Convert to DataFrame
        logger.info("Converting to DataFrame...")
        data_dict = customer.dict()
        df = pd.DataFrame([data_dict])

        logger.info(f"Raw data shape: {df.shape}")

        if df.shape[1] != 19:
            raise ValueError(f"Expected 19 features, got {df.shape[1]}")

        # ===== STEP 1: ENCODE ONLY CATEGORICAL FEATURES =====
        logger.info("Encoding categorical features...")
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns and col in encoders:
                try:
                    original_value = df[col].values[0]
                    df[col] = encoders[col].transform(df[col])
                    encoded_value = df[col].values[0]
                    logger.info(f"  ✅ {col}: '{original_value}' → {encoded_value}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to encode {col}: {str(e)}")
                    raise ValueError(f"Failed to encode {col}: {str(e)}")

        # ===== STEP 2: SCALE ONLY NUMERICAL FEATURES =====
        logger.info("Scaling numerical features...")
        try:
            # Convert all numerical columns to float
            for col in NUMERICAL_COLUMNS:
                df[col] = df[col].astype(float)

            logger.info(f"  Columns to scale: {NUMERICAL_COLUMNS}")
            logger.info(f"  Before scaling: {df[NUMERICAL_COLUMNS].values}")

            # Scale using scaler
            df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])

            logger.info(f"  After scaling: {df[NUMERICAL_COLUMNS].values}")
            logger.info(f"  ✅ Scaled {NUMERICAL_COLUMNS}")
            logger.info(f"  Note: TotalCharges kept as-is (not scaled)")
        except Exception as e:
            logger.error(f"  ❌ Failed to scale: {str(e)}")
            raise ValueError(f"Failed to scale numerical features: {str(e)}")

        # ===== STEP 3: MAKE PREDICTION =====
        logger.info("Making prediction with model...")
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        # LOG THE PREDICTION
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "input_features": customer.dict(),
            "prediction": prediction,
            "probability": float(probability),
            "status": "success",
        }
        logging.info(json.dumps(metrics))
        logger.info(f"✅ Prediction: {prediction}, Probability: {probability:.4f}")

        # ===== STEP 4: GENERATE RECOMMENDATION =====
        if probability >= 0.7:
            rec = "🔴 HIGH RISK (70%+) - Contact immediately with retention offer"
        elif probability >= 0.5:
            rec = "🟡 MEDIUM RISK (50-70%) - Monitor closely, prepare retention plan"
        elif probability >= 0.3:
            rec = "🟠 MEDIUM-LOW RISK (30-50%) - Maintain relationship, gather feedback"
        else:
            rec = "🟢 LOW RISK (<30%) - Customer likely to stay"

        return PredictionResponse(
            churn=prediction,
            churn_probability=round(float(probability), 4),
            recommendation=rec,
        )
    except Exception as e:
        # LOG ERRORS
        error_metrics = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error",
        }
        logging.error(json.dumps(error_metrics))
        raise
    except ValueError as e:
        logger.error(f"Data error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ROOT ENDPOINT


@app.get("/")
def root():
    """Welcome message"""
    return {
        "message": "Churn Prediction API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "docs": "http://localhost:8000/docs (Try it out!)",
            "health": "http://localhost:8000/health (Check status)",
            "predict": "http://localhost:8000/predict (Make prediction)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
