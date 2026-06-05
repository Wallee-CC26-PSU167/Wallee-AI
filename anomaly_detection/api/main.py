from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.anomaly_service import AnomalyService

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Personal Finance — Anomaly Detection API",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = AnomalyService(
    model_path    =str(BASE_DIR / "models" / "anomaly_autoencoder.keras"),
    scaler_path   =str(BASE_DIR / "models" / "scaler.pkl"),
    features_path =str(BASE_DIR / "models" / "feature_columns.pkl"),
    threshold_path=str(BASE_DIR / "models" / "threshold.pkl"),
)


# ── Request models ────────────────────────────────────────

class TransactionItemRequest(BaseModel):
    item_name   : str
    harga       : int
    qty         : int
    subtotal    : int
    category    : Optional[str]   = None
    usual_price : Optional[float] = None


class TransactionRequest(BaseModel):
    id                    : str
    merchant              : str
    amount                : int
    date                  : str
    time                  : str
    item_count            : int
    items                 : List[TransactionItemRequest]
    historical_avg_amount : Optional[float] = None
    usual_hour_min        : Optional[int]   = None
    usual_hour_max        : Optional[int]   = None
    merchant_monthly_freq : int
    merchant_avg_freq     : float


# ── Response models ───────────────────────────────────────

class AnomalyItemResponse(BaseModel):
    id        : str
    type      : str
    message   : str
    detail    : List[Dict[str, Any]]
    dismissed : bool


class TransactionAnomalyResponse(BaseModel):
    id         : str
    merchant   : str
    amount     : int
    date       : str
    time       : str
    item_count : int
    anomalies  : List[AnomalyItemResponse]


# ── Routes ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "anomaly_autoencoder", "version": "2.1.0"}


@app.post("/predict", response_model=TransactionAnomalyResponse)
def predict_transaction(transaction: TransactionRequest):
    try:
        enriched = transaction.model_dump()
        anomalies = service.detect(
            transaction_id=transaction.id,
            enriched=enriched,
        )
        return TransactionAnomalyResponse(
            id        =transaction.id,
            merchant  =transaction.merchant,
            amount    =transaction.amount,
            date      =transaction.date,
            time      =transaction.time,
            item_count=transaction.item_count,
            anomalies =[AnomalyItemResponse(**a) for a in anomalies],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=List[TransactionAnomalyResponse])
def predict_batch(transactions: List[TransactionRequest]):
    results = []
    for trx in transactions:
        try:
            results.append(predict_transaction(trx))
        except HTTPException:
            results.append(
                TransactionAnomalyResponse(
                    id=trx.id, merchant=trx.merchant, amount=trx.amount,
                    date=trx.date, time=trx.time, item_count=trx.item_count,
                    anomalies=[],
                )
            )
    return results