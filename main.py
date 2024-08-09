from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
# import numpy as np
# import ssl

# from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain('./cert.pem', keyfile='./key.pem')


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],  
#     allow_headers=["*"], 
# )


# Load models and scaler
scaler = joblib.load('models/scaler.pkl')
linear_model = joblib.load('models/linear_regression.pkl')
polynomial_model = joblib.load('models/polynomial_regression.pkl')
ann_model = joblib.load('models/ann.pkl')

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PredictionRequest(BaseModel):
    model: str = "polynomial_regression"
    replacement: float
    time: float

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: PredictionRequest):
    # Validate model
    if data.model not in ["linear_regression", "polynomial_regression", "ann"]:
        raise HTTPException(status_code=400, detail="Invalid model specified")

    # Select model
    if data.model == "linear_regression":
        model = linear_model
    elif data.model == "polynomial_regression":
        model = polynomial_model
    else:
        model = ann_model

    # Scale input
    scaled_input = scaler.transform([[data.replacement, data.time]])
    
    # Predict
    prediction = model.predict(scaled_input)
    
    return {"compressive_strength": prediction[0]}

# Run with `uvicorn main:app --reload`

