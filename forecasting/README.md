# 💰 Finance Forecast & Budget Tracker API

Proyek ini menyediakan layanan API untuk memprediksi pengeluaran bulanan menggunakan model Ensemble (ARIMA, SARIMAX, Deep Learning).

## 🛠️ Fitur Utama
- **Monthly Forecasting**: Menyediakan prediksi pengeluaran bulan depan beserta Confidence Interval.
- **Health Check**: Endpoint `/health` untuk memeriksa status API.
- **FastAPI Integration**: Siap dideploy sebagai layanan web.

## 📂 Struktur File
- `main.py`: Server FastAPI untuk inference.
- `test_api.py`: Skrip untuk menguji koneksi API.
- `models/`:
  - `forecast_model.keras`: Model Deep Learning yang sudah dilatih.
  - `scaler_X.save` & `scaler_y.save`: File normalisasi fitur.
  - `ensemble_forecast_results.json`: Hasil prediksi ensemble bulanan dan confidence interval.
- `requirements.txt`: Daftar dependensi library.
- `Procfile`: Konfigurasi untuk Railway deployment.

## 🚀 Cara Menjalankan

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan Server (Lokal)
```bash
uvicorn main:app --reload
```

## 📊 Endpoint API

### `GET /`
Returns a welcome message.

### `GET /health`
Returns the health status of the API.

**Response:**
```json
{
  "status": "healthy"
}
```

### `POST /predict`
Returns the pre-calculated monthly ensemble forecast and its confidence interval.

**Request Body (JSON):**
```json
{
  "lag_1": 50000.0,
  "lag_2": 45000.0,
  "lag_3": 60000.0,
  "rolling_mean_7": 52000.0,
  "rolling_mean_30": 50000.0,
  "day_of_week": 2,
  "month": 5,
  "is_weekend": 0,
  "mtd_progress": 0.5,
  "transaction_count": 3
}
```

**Response (JSON):**
```json
{
  "success": true,
  "message": "OK",
  "forecast_next_month": 1250000,
  "confidence_lower": 1187500,
  "confidence_upper": 1312500
}
```

**Note**: The input daily features in the request body are validated but are not used to recompute the monthly forecast in this version of the API. The API returns a pre-calculated monthly ensemble forecast and its confidence interval.
