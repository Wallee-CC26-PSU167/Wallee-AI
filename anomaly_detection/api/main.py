# api/main.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI entry point — Personal Finance Anomaly Detection Service
#
# Run:  uvicorn api.main:app --host 0.0.0.0 --port 8001 --workers 2
#
# Architecture boundary:
#   Spring Boot  →  POST /predict  →  FastAPI  →  AnomalyService  →  response
#
# Spring Boot is responsible for:
#   • Fetching raw transaction from DB (transaksi + item_transaksi + kategori)
#   • Computing historical context (rolling averages, usual hour range, etc.)
#   • Sending the enriched payload to this service
#   • Persisting results into notifikasi + item_notifikasi tables
#
# This service is responsible for:
#   • Rule-based anomaly detection (SPENDING_SPIKE, PRICE_SPIKE, etc.)
#   • Autoencoder-based anomaly detection (AUTOENCODER_ANOMALY)
#   • Returning a structured anomaly list per transaction
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.anomaly_service import AnomalyService

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Personal Finance — Anomaly Detection API",
    version="2.1.0",
    description=(
        "AI anomaly detection for personal finance transactions. "
        "Combines rule-based and TensorFlow autoencoder detection."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Service (loaded once at startup) ─────────────────────────────────────────
service = AnomalyService(
    model_path    =str(BASE_DIR / "models" / "anomaly_autoencoder.keras"),
    scaler_path   =str(BASE_DIR / "models" / "scaler.pkl"),
    features_path =str(BASE_DIR / "models" / "feature_columns.pkl"),
    threshold_path=str(BASE_DIR / "models" / "threshold.pkl"),
)


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════

class TransactionItemContext(BaseModel):
    """One item from item_transaksi, enriched with its historical usual price."""
    item_name  : str
    harga      : int                         = Field(..., description="item_transaksi.harga")
    qty        : int                         = Field(..., ge=1)
    subtotal   : int                         = Field(..., description="item_transaksi.subtotal")
    category   : Optional[str]              = None
    usual_price: Optional[float]            = Field(
        None,
        description="Median historical unit price for this item (computed by Spring Boot)",
    )


class TransactionRequest(BaseModel):
    """
    Enriched transaction payload sent by Spring Boot.

    Spring Boot must compute all historical-context fields before calling this
    endpoint.  Raw fields map directly to DB columns; computed fields are
    prefixed with the source of truth in their description.
    """

    # ── Raw DB fields ─────────────────────────────────────────────────────
    id          : str  = Field(..., description="transaksi.id_transaksi")
    merchant    : str  = Field(..., description="transaksi.merchant")
    amount      : int  = Field(..., description="transaksi.total_harga")
    date        : str  = Field(..., description="transaksi.created_at date part  (YYYY-MM-DD)")
    time        : str  = Field(..., description="transaksi.created_at time part  (HH:MM)")
    item_count  : int  = Field(..., ge=1)
    items       : List[TransactionItemContext]

    # ── Context computed by Spring Boot ──────────────────────────────────
    # Amount context
    historical_avg_amount : Optional[float] = Field(
        None,
        description="User's 30-day rolling mean total_harga",
    )
    # amount_ratio          : float = Field(
    #     ...,
    #     description="amount / historical_avg_amount",
    # )

    # Time context
    # hour            : int   = Field(..., ge=0, le=23)
    # month           : int   = Field(..., ge=1, le=12)
    # is_weekend      : int   = Field(..., ge=0, le=1)
    usual_hour_min  : Optional[int] = Field(
        None,
        description="5th  percentile of user's transaction hours"
    )
    usual_hour_max  : Optional[int] = Field(
        None,
        description="95th percentile of user's transaction hours"
    )
    # is_unusual_hour : int   = Field(..., ge=0, le=1,
    #                                 description="1 if hour outside [usual_hour_min, usual_hour_max]")

    # Merchant frequency context
    merchant_monthly_freq : int   = Field(..., description="Times user visited merchant this month")
    merchant_avg_freq     : float = Field(..., description="Historical monthly average visits")
    # merchant_freq_ratio   : float = Field(..., description="merchant_monthly_freq / merchant_avg_freq")

    # Category context
    # category_freq_ratio   : float = Field(..., description="(current month count) / avg monthly count for top category")

    # Payment / item context
    # is_digital_payment    : int   = Field(..., ge=0, le=1)
    # qty_zscore            : float = Field(0.0, description="z-score of total qty vs user history")
    # item_max_price_ratio  : float = Field(0.0, description="max(item.harga) / median historical price")

    # Autoencoder numeric features (mirrors feature_columns.pkl — Spring Boot can
    # send 0.0 for features it cannot compute; they will degrade gracefully)
    # amount_rolling_ratio  : float = Field(0.0)
    # amount_zscore         : float = Field(0.0)
    # user_rolling_mean_amt : float = Field(0.0)
    # user_rolling_std_amt  : float = Field(0.0)
    # hour_deviation        : float = Field(0.0)
    # hour_zscore           : float = Field(0.0)
    # category_share        : float = Field(0.0)
    # user_cat_cumcount     : float = Field(0.0)
    # user_merchant_cumcount: float = Field(0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyItemResponse(BaseModel):
    """Single anomaly finding.  Maps to item_notifikasi in the DB."""
    id       : str
    type     : str
    message  : str
    detail   : List[Dict[str, Any]]
    dismissed: bool


class TransactionAnomalyResponse(BaseModel):
    """
    Full response returned to Spring Boot.

    Spring Boot uses this to:
      - Display anomaly badges in the Flutter UI
      - Persist records into notifikasi + item_notifikasi tables
    """
    id         : str
    merchant   : str
    amount     : int
    date       : str
    time       : str
    item_count : int
    anomalies  : List[AnomalyItemResponse]


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — used by Docker / k8s health checks."""
    return {"status": "ok", "model": "anomaly_autoencoder", "version": "2.1.0"}


@app.post(
    "/predict",
    response_model=TransactionAnomalyResponse,
    status_code=status.HTTP_200_OK,
    tags=["anomaly"],
    summary="Detect anomalies in a single transaction",
)
def predict_transaction(transaction: TransactionRequest):
    """
    Analyse one enriched transaction and return all detected anomalies.

    **Caller:** Spring Boot (after fetching transaction from DB and computing
    historical context fields).

    **Returns:** Transaction mirrored back with an `anomalies` list.
    An empty list means no anomaly was detected.
    """
    try:
        enriched = transaction.model_dump()

        # Flatten items for the service (service reads enriched["items"])
        enriched["items"] = [item.model_dump() for item in transaction.items]

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(exc)}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=List[TransactionAnomalyResponse],
    status_code=status.HTTP_200_OK,
    tags=["anomaly"],
    summary="Detect anomalies in a batch of transactions",
)
def predict_batch(transactions: List[TransactionRequest]):
    """
    Analyse multiple transactions in one call.

    Useful for bulk historical re-analysis or nightly anomaly sweeps.
    Processes each transaction independently.
    """
    results = []
    for trx in transactions:
        try:
            results.append(predict_transaction(trx))
        except HTTPException:
            # Skip failed transactions in batch; don't abort the whole batch
            results.append(
                TransactionAnomalyResponse(
                    id=trx.id, merchant=trx.merchant, amount=trx.amount,
                    date=trx.date, time=trx.time, item_count=trx.item_count,
                    anomalies=[],
                )
            )
    return results