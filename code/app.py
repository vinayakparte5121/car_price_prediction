from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
import io

app = FastAPI(title="Car Price Prediction API")

# Load artifacts
artifacts = joblib.load("stacking_model.pkl")

model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoders = artifacts["label_encoders"]
categorical_columns = artifacts["categorical_columns"]
feature_columns = artifacts["feature_columns"]


def safe_label_transform(df):
    for col in categorical_columns:
        le = label_encoders[col]
        known = set(le.classes_)
        df[col] = df[col].apply(
            lambda x: x if x in known else le.classes_[0]
        )
        df[col] = le.transform(df[col])
    return df


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict")
def predict_json(data: list[dict]):
    df = pd.DataFrame(data)
    df = safe_label_transform(df)
    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return {"predictions": preds.tolist()}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    df = safe_label_transform(df)

    X = df[feature_columns]
    X_scaled = scaler.transform(X)

    df["Predicted_Price"] = model.predict(X_scaled)

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=oot_predictions.csv"
        }
    )