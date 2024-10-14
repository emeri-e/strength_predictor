from fastapi import FastAPI, HTTPException, Request, Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

app = FastAPI()

# Define a mapping for models based on aggregate names
models = {
    "demolished-concrete": {
        "scaler": joblib.load('models/demolished-concrete/scaler.pkl'),
        "linear_regression": joblib.load('models/demolished-concrete/linear_regression.pkl'),
        "polynomial_regression": joblib.load('models/demolished-concrete/polynomial_regression.pkl'),
        "ann": joblib.load('models/demolished-concrete/ann.pkl')
    },
    "laterite": {
        "scaler": joblib.load('models/laterite/scaler.pkl'),
        "linear_regression": joblib.load('models/laterite/linear_regression.pkl'),
        "polynomial_regression": joblib.load('models/laterite/polynomial_regression.pkl'),
        "ann": joblib.load('models/laterite/ann.pkl')
    },
    "rubber": {
        "scaler": joblib.load('models/rubber/scaler.pkl'),
        "linear_regression": joblib.load('models/rubber/linear_regression.pkl'),
        "polynomial_regression": joblib.load('models/rubber/polynomial_regression.pkl'),
        "ann": joblib.load('models/rubber/ann.pkl')
    },
    "lime": {
        "scaler": joblib.load('models/lime/scaler.pkl'),
        "linear_regression": joblib.load('models/lime/linear_regression.pkl'),
        "polynomial_regression": joblib.load('models/lime/polynomial_regression.pkl'),
        "ann": joblib.load('models/lime/ann.pkl')
    }
    # Add more aggregates as needed
}

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PredictionRequest(BaseModel):
    model: str = "polynomial_regression"
    replacement: float
    time: float

@app.get("/{aggregate}", response_class=HTMLResponse)
async def get_home(request: Request, aggregate: str = Path(...)):
    if aggregate not in models:
        raise HTTPException(status_code=404, detail="Aggregate not found")
    return templates.TemplateResponse("index.html", {"request": request, "aggregate": aggregate})

@app.post("/{aggregate}/predict")
def predict(aggregate: str, data: PredictionRequest):
    if aggregate not in models:
        raise HTTPException(status_code=404, detail="Aggregate not found")

    # Validate model
    if data.model not in models[aggregate]:
        raise HTTPException(status_code=400, detail="Invalid model specified")

    # Select model
    model = models[aggregate][data.model]
    scaler = models[aggregate]["scaler"]

    # Scale input
    scaled_input = scaler.transform([[data.replacement, data.time]])
    
    # Predict
    prediction = model.predict(scaled_input)
    
    return {"compressive_strength": prediction[0]}

# Run with `uvicorn main:app --reload`
