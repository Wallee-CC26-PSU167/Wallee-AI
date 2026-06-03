# 🤖 Wallee AI Services

Repositori ini berisi dua layanan AI yang digunakan oleh aplikasi Wallee untuk analisis keuangan cerdas.

---

## 📁 Struktur Folder

```
AI/
├── forecast/                        → Layanan prediksi pengeluaran bulanan
│   ├── main.py
│   ├── test_api.py
│   ├── models/
│   │   ├── forecast_model.keras
│   │   ├── scaler_X.save
│   │   ├── scaler_y.save
│   │   └── ensemble_forecast_results.json
│   ├── requirements.txt
│   └── Procfile
│
└── anomaly_detection/               → Layanan deteksi anomali transaksi
    ├── api/
    │   ├── main.py
    │   └── anomaly_service.py
    ├── models/
    │   ├── anomaly_autoencoder.keras
    │   ├── scaler.pkl
    │   ├── feature_columns.pkl
    │   └── threshold.pkl
    ├── requirements.txt
    └── Procfile
```

---

---

# 💰 Finance Forecast API

Layanan API untuk memprediksi pengeluaran bulanan menggunakan model Ensemble (ARIMA, SARIMAX, Deep Learning).

## 🛠️ Fitur Utama
- **Monthly Forecasting**: Prediksi pengeluaran bulan depan beserta Confidence Interval.
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
cd forecast
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

> **Note**: Input features divalidasi namun tidak digunakan untuk menghitung ulang forecast. API mengembalikan hasil ensemble yang sudah dihitung sebelumnya.

---

---

# 🔍 Anomaly Detection API

Layanan AI untuk mendeteksi anomali pada transaksi keuangan menggunakan kombinasi rule-based detection dan TensorFlow Autoencoder.

## 🛠️ Fitur Utama
- **Rule-based Detection**: Mendeteksi `SPENDING_SPIKE`, `PRICE_SPIKE`, dan pola anomali umum lainnya.
- **Autoencoder Detection**: Model deep learning (`AUTOENCODER_ANOMALY`) untuk mendeteksi pola tidak wajar secara otomatis.
- **Batch Processing**: Mendukung analisis banyak transaksi sekaligus dalam satu request.
- **Health Check**: Endpoint `/health` untuk memeriksa status API.
- **FastAPI Integration**: Siap dideploy sebagai layanan web.

## 📂 Struktur File
- `api/main.py`: Server FastAPI untuk inference.
- `api/anomaly_service.py`: Logika deteksi anomali (rule-based + autoencoder).
- `models/`:
  - `anomaly_autoencoder.keras`: Model Autoencoder yang sudah dilatih.
  - `scaler.pkl`: File normalisasi fitur.
  - `feature_columns.pkl`: Daftar kolom fitur yang digunakan model.
  - `threshold.pkl`: Nilai threshold rekonstruksi error untuk klasifikasi anomali.
- `requirements.txt`: Daftar dependensi library.
- `Procfile`: Konfigurasi untuk Railway deployment.

## 🚀 Cara Menjalankan

### 1. Instalasi Dependensi
```bash
cd anomaly_detection
pip install -r requirements.txt
```

### 2. Jalankan Server (Lokal)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

## 📊 Endpoint API

### `GET /health`
Returns the health status of the API.

**Response:**
```json
{
  "status": "ok",
  "model": "anomaly_autoencoder",
  "version": "2.1.0"
}
```

### `POST /predict`
Mendeteksi anomali pada satu transaksi.

**Request Body (JSON):**
```json
{
  "id": "12345",
  "merchant": "Warung Makan ABC",
  "amount": 500000,
  "date": "2026-06-01",
  "time": "12:30",
  "item_count": 2,
  "merchant_monthly_freq": 5,
  "merchant_avg_freq": 2.0,
  "items": [
    {
      "item_name": "Nasi Goreng",
      "harga": 200000,
      "qty": 1,
      "subtotal": 200000,
      "category": "Makanan",
      "usual_price": 25000
    }
  ]
}
```

**Response (JSON):**
```json
{
  "id": "12345",
  "merchant": "Warung Makan ABC",
  "amount": 500000,
  "date": "2026-06-01",
  "time": "12:30",
  "item_count": 2,
  "anomalies": [
    {
      "id": "anomaly-001",
      "type": "PRICE_SPIKE",
      "message": "Harga item jauh di atas biasanya",
      "detail": [],
      "dismissed": false
    }
  ]
}
```

### `POST /predict/batch`
Mendeteksi anomali pada banyak transaksi sekaligus.

**Request Body:** Array dari objek transaksi (sama seperti `/predict`).

**Response:** Array dari hasil deteksi per transaksi.

> **Note**: Transaksi yang gagal diproses dalam batch tidak akan membatalkan seluruh request — transaksi tersebut dikembalikan dengan `anomalies: []`.

---

## 🔗 Tautan Model ML

Model yang digunakan oleh kedua layanan ini perlu diunduh dan ditempatkan di folder `models/` masing-masing sebelum menjalankan server.

| Layanan | File Model | Keterangan |
|---|---|---|
| Forecast | `forecast_model.keras` | Model Deep Learning ensemble |
| Forecast | `scaler_X.save`, `scaler_y.save` | Normalisasi fitur |
| Forecast | `ensemble_forecast_results.json` | Hasil prediksi pre-calculated |
| Anomaly | `anomaly_autoencoder.keras` | Model Autoencoder TensorFlow |
| Anomaly | `scaler.pkl` | Normalisasi fitur |
| Anomaly | `feature_columns.pkl` | Kolom fitur model |
| Anomaly | `threshold.pkl` | Threshold rekonstruksi error |

> Pastikan semua file model sudah tersedia sebelum menjalankan server. Server akan gagal start jika file model tidak ditemukan.
