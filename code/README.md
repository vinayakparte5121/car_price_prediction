# Car Price Prediction API 🚗💰

A FastAPI-based machine learning API for predicting car prices using a trained stacking ensemble model. The API supports both JSON input and CSV file uploads for batch predictions.

---

## Features

- Predict car prices using a trained stacking model
- REST API built with FastAPI
- Supports:
  - JSON input predictions
  - CSV batch predictions
- Handles unseen categorical values safely
- Returns downloadable CSV with predictions
- Uses preprocessing artifacts:
  - Scaler
  - Label encoders
  - Feature column mapping

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- XGBoost
- Pandas
- Joblib

---

## Project Structure

```bash
├── app.py
├── stacking_model.pkl
├── requirements.txt
├── eda_and_analysis.ipynb
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/vinayakparte5121/car_price_prediction.git
cd usa_car_price_prediction
```

---

## 2. Create Virtual Environment (Optional but Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the API

```bash
uvicorn app:app --reload
```

API will run at:

```bash
http://127.0.0.1:8000
```

Interactive Swagger Docs:

```bash
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## 1. Health Check

### GET `/`

Returns API status.

### Response

```json
{
  "status": "API is running"
}
```

---

## 2. Predict Using JSON

### POST `/predict`

### Sample Request

```json
[
  {
    "MAKE_MODEL": "toyota corolla",
    "YEAR": 2020,
    "KILOMETERS": 45000,
    "FUEL_TYPE": "petrol"
  }
]
```

### Sample Response

```json
{
  "predictions": [825000.0]
}
```

---

## 3. Predict Using CSV Upload

### POST `/predict_csv`

Upload a CSV file containing input features.

### Response

Returns a downloadable CSV file with an additional column:

```bash
Predicted_Price
```

---

# Model Information

The application loads a serialized artifact file:

```python
stacking_model.pkl
```

This file contains:

- Trained stacking model
- Feature scaler
- Label encoders
- Categorical columns
- Feature column order

The API safely handles unseen categorical values by replacing them with a default known category before encoding.

---

# Example Request Using Python

```python
import requests

url = "http://127.0.0.1:8000/predict"

data = [
    {
        "MAKE_MODEL": "toyota corolla",
        "YEAR": 2020,
        "KILOMETERS": 45000,
        "FUEL_TYPE": "petrol"
    }
]

response = requests.post(url, json=data)

print(response.json())
```

---

# Requirements

```txt
fastapi
requests
python-multipart
uvicorn
pandas
scikit-learn
seaborn
joblib
xgboost
```

---

# Future Improvements

- Add model versioning
- Docker support
- Cloud deployment (AWS/GCP/Azure)
- Input validation with Pydantic
- Authentication & rate limiting
- Model monitoring

---

# Author

**Vinayak Parte**
GitHub: https://github.com/vinayakparte5121

---
